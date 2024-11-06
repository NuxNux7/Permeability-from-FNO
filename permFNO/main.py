import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os

from learning.scheduler import CosineWithWarmupScheduler
from learning.loss import *

from data.dataSet import DictDataset, analyse_dataset
from data.dataWriter import saveArraysToVTK
from data.normalization import Denormalizer

from models.fno import FNOArch
from models.siren import SirenArch
from models.feedForward import FeedForwardBlock

from learning.trainer import Trainer


# constants
SPHERES = 0
EXTERNAL = 1
IMAGES = 2

DATSET_NAMES = ("spheres", "DRP", "2D")
DATASET_VERSIONS = (("std"),
                    ("std", "filtered_90"),
                    ("std", "full", "full_filtered_99", "noise", "rocks"))



# Main execution
def main(load_checkpoint: bool = False,
         name: str = None,
         evaluation: bool = False,
         experiment: int = 0,
         version: int = 0,
         layers: int = 4,
         factorized: bool = False,
         epochs: int = 50,
         batch_size: int = 8,
         learning_rate: int = 2.5e-3,
         criterion = MaskedMSELoss()):
    """
    The main entry point for the application.
    Here, the learning process can be setup by choosing the model, training parameter and dataset.

    Features:
    - Supports loading a pre-trained model checkpoint for evaluation or continued training.
    - Enables configuring the number of layers in the model, whether the model should be factorized, and the number of training epochs.
    - Provides options for setting the batch size and learning rate.
    - Uses a custom masked MSE loss function (MaskedMSELoss) as the default criterion.

    Args:
        load_checkpoint (bool, optional): Whether to load a pre-trained model checkpoint. Defaults to False.
        name (str, optional): The name of the model, when not, the name is created from the settings. Defaults to None.
        evaluation (bool, optional): Whether to run the model in evaluation mode. Defaults to False.
        experiment (int, optional): The experiment number, specifying the dataset Defaults to SPHERES.
        version (int, optional): The version number, used for organizing the saved models and logs. Defaults to 0.
        layers (int, optional): The number of layers in the model. Defaults to 4.
        factorized (bool, optional): Whether the model should be factorized. Defaults to False.
        epochs (int, optional): The number of training epochs. Defaults to 50.
        batch_size (int, optional): The batch size for training and evaluation. Defaults to 8.
        learning_rate (int, optional): The learning rate for training. Defaults to 2.5e-3.
        criterion (torch.nn.Module, optional): The loss function used for training the model. Defaults to MaskedMSELoss().
    """
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup folder
    if name == None:
        name = name_from_settings(experiment, version, layers, factorized)
    folder = os.getcwd() + "/output/" + name
    if load_checkpoint:
        if not os.path.exists(folder):
            print(f"Error: The folder '{folder}' does not exist and cannopt be loaded.")
            return
    else:
        os.makedirs(folder, exist_ok=True)
    


    # Create data loaders
    dataset_path = os.getcwd() + "/datasets/" + DATSET_NAMES[experiment] + "/" + DATASET_VERSIONS[experiment][version]
    if evaluation:
        test_dataset = DictDataset(dataset_path + "_test.h5", h5=True, masking=True)
        print("Validation dataset loaded successfuly!")
        train_dataset = test_dataset

        #analysis tools
        '''analyse_dataset(test_dataset)
        train_dataset.estimate_by_formula()'''

    else:
        train_dataset = DictDataset(dataset_path + "_train.h5",
                                    h5=True, masking=True)
        print("Training dataset loaded successfuly!")
        test_dataset = DictDataset(dataset_path + "_test.h5", h5=True, masking=True)
        print("Testing dataset loaded successfuly!")
    
    bounds = train_dataset.getBounds()
    denormalizer = Denormalizer(0, 1, bounds[2], 0, 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    

    
    # Initialize the model
    if experiment == IMAGES:
        dimensions = 2
        decoderNet = FeedForwardBlock(  dims=2, fno_layer_size=32, num_blocks=2,
                                        factor=4, activation_fn=None,
                                        use_weight_norm=True)
        modes = [80, 64]
    else:
        dimensions = 3
        decoderNet = SirenArch(in_features=32, out_features=1, layer_size=(4*32), nr_layers=2, weight_norm=True)
        if experiment == SPHERES:
            modes = [32, 16, 16]
        else:
            modes = [24, 16, 16]

    
    model = FNOArch(
        dimension=dimensions,
        nr_fno_layers=layers,
        nr_ff_blocks=2,
        fno_modes=modes,
        padding=8,
        decoder_net=decoderNet,
        coord_features=True,
        factorized=factorized,
        weight_sharing=False,
        weight_norm=True,
        batch_norm=False,
        dropout=False,
    ).to(device)


    # Print model parameters
    '''print("Tunable Parameter: ", sum(p.numel() for p in model.parameters()))
    print(model)
    summary(model, (1, 128, 64, 64))'''

    # Initialize the criterion
    criterion = MaskedMSELoss()

    # Initialize the optimizer with Cosine LR
    gradient_accumulation = 1
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineWithWarmupScheduler(optimizer, 500, ((len(train_loader) // gradient_accumulation) * epochs), min_lambda=0.05)


    # Load checkpoint
    epoch = 0
    if load_checkpoint:
        checkpoint = torch.load(folder + "/checkpoint.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch = checkpoint["epoch"]
        print("Network loaded successfuly!")


    # Start run
    trainer = Trainer(model,
                      train_loader, test_loader,
                      optimizer, scheduler, criterion,
                      epochs, folder,
                      device,
                      False, gradient_accumulation,
                      denormalizer)
    
    if evaluation:
        trainer.evaluate(verbose=True)
    else:
        trainer.trainRun(epoch)



def name_from_settings( experiment: int = 0, 
                        version: int = 0,
                        layers: int = 4,
                        factorized: bool = False,
                        number: int = None):
    
    name = DATSET_NAMES[experiment] + "/" + str(layers) + "l_"
    if factorized:
        name += "factorized_"
    name += DATASET_VERSIONS[experiment][version]
    if number is not None:
        name += "_" + number

    return name



if __name__ == "__main__":

    # Performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    evaluation = True
    main(   load_checkpoint=(False or evaluation),
            name=None,
            evaluation=evaluation,
            experiment=IMAGES,
            version=0,
            layers=4,
            factorized=False,
            )
