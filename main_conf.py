import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os

from learning.scheduler import CosineWithWarmupScheduler
from learning.loss import *
from data.dataSet import DictDataset
from models.fno import FNOArch
from models.siren import SirenArch
from trainer import Trainer

def load_cfg(cfg_path):
    with open(cfg_path, 'r') as file:
        return yaml.safe_load(file)

def get_loss_function(loss_name):
    return globals()[loss_name]()

def get_optimizer(optimizer_name, model_params, optimizer_params):
    return getattr(optim, optimizer_name)(model_params, **optimizer_params)


def main(cfg_path: str):
    cfg = load_cfg(cfg_path)
    model_cfg = cfg['model']['params']
    data_cfg = cfg['data']
    runtime_cfg = cfg['runtime']
    train_cfg = cfg['training']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = os.path.join(runtime_cfg['checkpoint']['save_dir'], runtime_cfg['checkpoint']['name'])
    os.makedirs(folder, exist_ok=True)

    # Load data
    dataset_params = data_cfg['dataset_params']
    if runtime_cfg['mode'] == 'eval':
        test_dataset = DictDataset(data_cfg['validation_path'], **dataset_params)
        train_dataset = test_dataset
    else:
        train_dataset = DictDataset(data_cfg['train_path'], **dataset_params)
        test_dataset = DictDataset(data_cfg['test_path'], **dataset_params)

    train_loader = DataLoader(train_dataset, batch_size=data_cfg['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=data_cfg['batch_size'])

    # Initialize the model
    model = FNOArch(
        dimension=model_cfg['dimension'],
        nr_fno_layers=model_cfg['nr_fno_layers'],
        nr_ff_blocks=model_cfg['nr_ff_blocks'],
        fno_modes=model_cfg['modes'],
        decoder_net=SirenArch(32, 1, 32, model_cfg['nr_decoder_layers']),
        **{k: model_cfg[k] for k in ['functional', 'weight_sharing', 'batch_norm', 'dropout']}
    ).to(device)

    # Initialize the criterion and optimizer
    criterion = get_loss_function(train_cfg['loss'])
    optimizer = get_optimizer(train_cfg['optimizer']['name'], 
                              model.parameters(), 
                              train_cfg['optimizer']['params'])

    # Initialize the scheduler
    total_steps = len(train_loader) * runtime_cfg['epochs']
    scheduler_params = train_cfg['scheduler']['params']
    scheduler = CosineWithWarmupScheduler(optimizer, 
                                          scheduler_params['warmup_steps'], 
                                          total_steps,
                                          min_lambda=scheduler_params['min_lambda'])

    # Load checkpoint
    epoch = 0
    if runtime_cfg['checkpoint']['load']:
        checkpoint = torch.load(os.path.join(folder, "checkpoint.pth"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch = checkpoint["epoch"]
        print("Network loaded successfully!")

    # Start run
    trainer = Trainer(model, train_loader, test_loader, optimizer, scheduler, criterion,
                      runtime_cfg['epochs'], folder, device)
    
    if runtime_cfg['mode'] == 'eval':
        val_loss, val_loss_in = trainer.evaluate(True)
        print(f"Val Loss General: {val_loss}, Val Loss Inlet: {val_loss_in}")
    else:
        trainer.trainRun(epoch)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    import sys
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_cfg.yaml>")
        sys.exit(1)
    
    main(sys.argv[1])