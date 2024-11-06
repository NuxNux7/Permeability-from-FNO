import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

import time
from tabulate import tabulate

from permFNO.data.dataWriter import visualize, saveCSV, saveErrorPlot
from permFNO.data.normalization import Denormalizer
from .loss import *


class Trainer():
    """
    A PyTorch Trainer class that handles the training and evaluation of a model.

    Features:
    - Accouts for the additional spatial inlet loss in the evaluation process.
    - Supports training with mixed precision (FP16) using PyTorch's Automatic Mixed Precision (AMP) library.
    - Implements gradient accumulation to reduce GPU memory usage.
    - Allows for the use of a normalization reversion (denormalizer).
    - Provides training and evaluation loops, saving checkpoints, and logging metrics to TensorBoard.
    

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_loader (torch.utils.data.DataLoader): The DataLoader for the training data.
        test_loader (torch.utils.data.DataLoader): The DataLoader for the test/validation data.
        optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler used during training.
        criterion (torch.nn.Module): The loss function used for training the model.
        epochs (int): The number of training epochs.
        folder (str): The folder path to save the training artifacts (checkpoints, logs, etc.).
        device (torch.device): The device (CPU or GPU) to use for training and evaluation.
        amp (bool, optional): Whether to use Automatic Mixed Precision (AMP) for training. Defaults to False.
        gradient_accumulation (int, optional): The number of gradients to accumulate before updating the model's weights. Defaults to 1.
        denorm (denormalizer, optional): A custom ent-normalization technique. Defaults to None.
    """

    def __init__(self,
                 model,
                 train_loader, test_loader,
                 optimizer, scheduler, criterion,
                 epochs,
                 folder, device,
                 amp: bool = False, 
                 gradient_accumulation: int = 1,
                 denorm=Denormalizer()):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.epochs = epochs
        self.folder = folder
        self.device = device
        self.denorm = denorm

        self.amp = amp
        self.gradient_accumulation = gradient_accumulation

        self.scaler = GradScaler(    
            init_scale=65536.0,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
            enabled=self.amp
        )

        self.writer = SummaryWriter(log_dir=folder)



    def trainRun(self, epoch: int = 0):
        for epoch in range(epoch, self.epochs):
            epoch_start_time = time.time()

            train_loss = self.train()
            val_loss, val_loss_in = self.evaluate(((epoch % 10) == 0))

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            self.printProgress(epoch,
                          train_loss, val_loss, val_loss_in,
                          epoch_duration)

            # Save the model checkpoint
            if (((epoch + 1) % 10) == 0) or (epoch == (self.epochs - 1)):
                self.saveCheckpoint(epoch)

        self.writer.flush()



    # Evaluate the model on the validation set
    def evaluateRun(self):
        val_loss, val_loss_in = self.evaluate(True)
        print(f"Val Loss General: {val_loss}, "
              f"Val Loss Inlet: {val_loss_in}")



    # Training function
    def train(self):
        self.model.train()

        total_loss = 0
        self.optimizer.zero_grad()
        for batch_idx, (inputs, targets, mask, _) in enumerate(self.train_loader):

            # load inputs
            inputs, targets, mask = inputs.to(self.device), targets.to(self.device), mask.to(self.device)

            with autocast("cuda", enabled=self.amp):
                outputs = self.model(inputs)

                # loss of preassure filed with set criterion              
                loss = self.criterion(outputs, targets, mask)
                loss = loss / self.gradient_accumulation

            self.scaler.scale(loss).backward()

            # Change weights
            if ((batch_idx + 1) % self.gradient_accumulation == 0) or (batch_idx + 1 == len(self.train_loader)):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.scaler.step(self.optimizer)
                self.optimizer.zero_grad()

                # update progression
                self.scaler.update()

                self.scheduler.step()

            total_loss += loss

        return total_loss / len(self.train_loader)
    


    # Evaluation function
    def evaluate(self, verbose: bool = True):
        self.model.eval()

        if verbose:
            individual_criterium = MaskedIndividualLoss(MaskedMAELoss())
            individual_criterium_mean = IndividualLoss(MAELoss())
            individual_error = {}
            individual_error_mean = {}

        total_loss = 0
        total_mean_loss = 0

        with torch.no_grad():
            outputs_mean = []
            targets_mean = []
            for batch, (inputs, targets, masks, names) in enumerate(self.test_loader):
                inputs, targets, masks  = inputs.to(self.device), targets.to(self.device), masks.to(self.device)

                with autocast("cuda", enabled=self.amp):
                    outputs = self.model(inputs)

                    # reverse power normalization
                    outputs = self.denorm(outputs)
                    targets = self.denorm(targets)

                    total_loss += self.criterion(outputs, targets, masks).item() * outputs.shape[0]

                    #calculate inlet loss
                    if len(outputs.shape) == 5:
                        output_mean = outputs[:, 0, 0:12].mean(dim=(-3, -2, -1))
                        target_mean = targets[:, 0, 0:12].mean(dim=(-3, -2, -1))
                    else:
                        output_mean = outputs[:, 0, 0:12].mean(dim=(-2, -1))
                        target_mean = targets[:, 0, 0:12].mean(dim=(-2, -1))
                    
                    outputs_mean.extend(output_mean.tolist())
                    targets_mean.extend(target_mean.tolist())

                    if verbose:
                        result = individual_criterium(outputs, targets, masks, names)
                        individual_error.update(result)
                        result = individual_criterium_mean(output_mean, target_mean, names)
                        individual_error_mean.update(result)
                    if batch == 0:
                        input_sample = inputs[0][0].cpu().numpy()
                        output_sample = outputs[0][0].cpu().numpy()
                        target_sample = targets[0][0].cpu().numpy()
                        mask_sample = masks[0][0].cpu().numpy()
                        self.writer.add_graph(self.model, torch.unsqueeze(inputs[0], 0))

            # print result
            if verbose:
                visualize(input_sample, output_sample, target_sample, mask_sample, self.folder)
                individual_error = {key: [individual_error[key], individual_error_mean[key]] for key in individual_error}
                saveCSV(individual_error, (self.folder + "/errors.csv"))
                saveErrorPlot(outputs_mean, targets_mean, (self.folder + "/errors.png"))

                mape = MAPELoss()(torch.tensor(outputs_mean), torch.tensor(targets_mean))
                mae = MAELoss()(torch.tensor(outputs_mean), torch.tensor(targets_mean))
                r2 = R2Score()(torch.tensor(outputs_mean), torch.tensor(targets_mean))
                max_mae = -1.
                for _, v in individual_error_mean.items():
                    error = float(v)
                    if error > max_mae:
                        max_mae = error

                print("MAE:", mae, "MAPE:", mape, "R2:", r2, "Max MAE:", max_mae)
            


        return total_loss / len(self.test_loader.dataset), MAPELoss()(torch.tensor(outputs_mean), torch.tensor(targets_mean))
    


    # Print the training progress
    def printProgress(self, epoch, train_loss, val_loss, val_loss_in, epoch_duration):
        # Prepare the data
        headers = ["Epoch", "Train Loss", "Val Loss", "Val Loss In", "LR", "Batch Time"]
        data = [
            [f"{epoch+1}/{self.epochs}",
             f"{train_loss:.2e}",
             f"{val_loss:.2e}",
             f"{val_loss_in:.2e}",
             f"{self.scheduler.get_last_lr()[0]:.2e}",
             f"{epoch_duration:.2f}s"]
        ]

        # Print the table
        if epoch == 0:
            print(tabulate(data, headers=headers, tablefmt="github"))
        else:
            print(tabulate(data, headers="", tablefmt="github"))

        # Write Tensorboard Summary
        self.writer.add_scalar("Training/train_p", train_loss, epoch)
        self.writer.add_scalar("Validation/test_p", val_loss, epoch)
        self.writer.add_scalar("Validation/test_p_in", val_loss_in, epoch)
        self.writer.add_scalar("Training/leatningRate", self.scheduler.get_last_lr()[0], epoch)



    # Save the model checkpoint
    def saveCheckpoint(self, epoch):
        # Save the model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, (self.folder + "/checkpoint.pth"))
        print("Saved model!")