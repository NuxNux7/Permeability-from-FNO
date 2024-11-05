import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

import time
from tabulate import tabulate

'''import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
print(project_root)
sys.path.append(str(project_root))'''

from permFNO.data.dataWriter import visualize, saveCSV, saveErrorPlot
from permFNO.data.normalization import Entnormalizer
from .loss import *


class Trainer():
    def __init__(self,
                 model,
                 train_loader, test_loader,
                 optimizer, scheduler, criterion,
                 epochs,
                 folder, device,
                 amp: bool = False, 
                 gradient_accumulation: int = 4,
                 entnorm=Entnormalizer()):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        if not isinstance(criterion, list):
            self.criterion = [(1., criterion)]
        else:
            self.criterion = criterion
        self.epochs = epochs
        self.folder = folder
        self.device = device
        self.entnorm = entnorm

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

            # Save network
            if (((epoch + 1) % 10) == 0) or (epoch == (self.epochs - 1)):
                torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                }, (self.folder + "/checkpoint.pth"))
                print("Saved model!")

        self.writer.flush()


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
                outputs_entnorm = self.entnorm(outputs)
                targets_entnorm = self.entnorm(targets)
                
                crit, factor = self.criterion[0]
                loss = factor * crit(outputs, targets, mask)

                # loss of preassure at inlet at non normalized datapoints
                if len(self.criterion) > 1:
                    crit, factor = self.criterion[1]

                    output_mean = outputs_entnorm[:, 0, 2].mean(dim=(-2, -1))
                    target_mean = targets_entnorm[:, 0, 2].mean(dim=(-2, -1))

                    inlet_loss = factor * crit(output_mean, target_mean)

                    loss += torch.clip(inlet_loss, 0, 1)

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

        mean_loss_function = MAPELoss

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
                    outputs = self.entnorm(outputs)
                    targets = self.entnorm(targets) #0., 1., 0.3, 0., 0.9215659

                    total_loss += self.criterion[0][0](outputs, targets, masks).item() * outputs.shape[0]

                    #calculate inlet loss
                    if len(outputs.shape) == 5:
                        output_mean = outputs[:, 0, 0:12].mean(dim=(-3, -2, -1))       #TESTING 0:12
                        target_mean = targets[:, 0, 0:12].mean(dim=(-3, -2, -1))       #TESTING 0:12
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
                        print(names[0])
                        input_sample = inputs[0][0].cpu().numpy()
                        output_sample = outputs[0][0].cpu().numpy()
                        target_sample = targets[0][0].cpu().numpy()
                        mask_sample = masks[0][0].cpu().numpy()
                        self.writer.add_graph(self.model, torch.unsqueeze(inputs[0], 0))
                        visualize(input_sample, output_sample, target_sample, mask_sample, self.folder)

            # print result
            if verbose:
                visualize(input_sample, output_sample, target_sample, mask_sample, self.folder)
                individual_error = {key: [individual_error[key], individual_error_mean[key]] for key in individual_error}
                #print("max error mean: ", torch.tensor(individual_error_mean.values()).max())
                saveCSV(individual_error, (self.folder + "/errors.csv"))
                saveErrorPlot(outputs_mean, targets_mean, (self.folder + "/errors.png"))
                #self.writer.add_graph(self.model)

                mare = MAPELoss()(torch.tensor(outputs_mean), torch.tensor(targets_mean))
                mae = MAELoss()(torch.tensor(outputs_mean), torch.tensor(targets_mean))
                r2 = R2Score()(torch.tensor(outputs_mean), torch.tensor(targets_mean))
                max_mae = -1.
                for k, v in individual_error_mean.items():
                    error = float(v)
                    if error > max_mae:
                        max_mae = error

                print("MAE:", mae, "MAPE:", mare, "R2:", r2, "Max MAE:", max_mae)
            


        return total_loss / len(self.test_loader.dataset), MAPELoss()(torch.tensor(outputs_mean), torch.tensor(targets_mean))
    

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