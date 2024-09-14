import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os

from learning.scheduler import CosineWithWarmupScheduler
from learning.loss import *

from data.dataSet import DictDataset, analyse_dataset
from data.dataWriter import saveArraysToVTK
from data.normalization import Entnormalizer

from models.fno import FNOArch
from models.layers.convFCLayers import Conv2dFCLayer
from models.feedForward import FeedForwardBlock

from learning.trainer import Trainer


# Main execution
def main(load_checkpoint: bool = False,
         name: str = "runX",
         evaluation: bool = False):
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup folder
    folder = os.getcwd() + "/output/" + name
    cwd = os.getcwd()
    print(cwd)
    if not load_checkpoint:
        os.makedirs(folder, exist_ok=True)

    # Hyperparameters
    batch_size = 30
    learning_rate = 2.5e-3
    epochs = 75

    # Create data loaders
    name_dataset = "2D_full_filtered_99"
    if evaluation:
        test_dataset = DictDataset("/home/woody/iwia/iwia057h/2D/" + name_dataset + "_test.h5",
                                    h5=True, masking=True)
        print("Validation dataset loaded successfuly!")
        train_dataset = test_dataset
        #analyse_dataset(test_dataset)


        #inputs, targets, mask, name = train_dataset[0]
        #saveArraysToVTK(inputs[0], targets[0], targets[0], mask[0], "test.vtk")

        #print(len(test_dataset))
        #return
    else:
        train_dataset = DictDataset("/home/woody/iwia/iwia057h/2D/" + name_dataset + "_train.h5",
                                    h5=True, masking=True)
        print("Training dataset loaded successfuly!")
        test_dataset = DictDataset("/home/woody/iwia/iwia057h/2D/" + name_dataset + "_test.h5",
                                    h5=True, masking=True)
        print("Testing dataset loaded successfuly!")
    
    bounds = train_dataset.getBounds()
    entnormalizer = Entnormalizer(0, 1, bounds[2], bounds[3], bounds[4])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    
    
    # Initialize the model

    decoderNet = FeedForwardBlock(dims=2, fno_layer_size=32, num_blocks=2,
                                  factor=4, activation_fn=None,
                                  use_weight_norm=True)
    #decoderNet = Conv2dFCLayer(in_features=32, out_features=1, use_weight_norm=True)

    model = FNOArch(
        dimension=2,
        nr_fno_layers=8,
        nr_ff_blocks=2,
        fno_modes=[80, 64],
        padding=8,
        decoder_net=decoderNet,
        coord_features=False,           # TESTING
        functional=True,
        weight_sharing=False,
        weight_norm=True,
        batch_norm=False,
        dropout=False,
    ).to(device)

    #print(model)
    #summary(model, (1, 128, 64, 64))


    # Initialize the criterion
    criterion_domain = MaskedMSELoss()
    criterion_inlet = MAELoss()
    criterion = [(criterion_domain, 1.)]    #, (criterion_inlet, 0.)]

    # Initialize the optimizer with Cosine LR
    gradient_accumulation = 1
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineWithWarmupScheduler(optimizer, 250, ((len(train_loader) // gradient_accumulation) * epochs), min_lambda=0.05)

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
                      entnormalizer)
    
    if evaluation:
        val_loss, val_loss_in = trainer.evaluate(True)
        print(f"Val Loss General: {val_loss}, "
        f"Val Loss Inlet: {val_loss_in}")

    else:
       trainer.trainRun(epoch)

    #train_dataset.close()
    #test_dataset.close()



if __name__ == "__main__":

    # Performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    evaluation = True
    main(load_checkpoint=(False or evaluation), name="8l_2D_fulll_nocoord" , evaluation=evaluation)