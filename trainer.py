import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import time

from tabulate import tabulate

from data.dataWriter import visualize, saveCSV
from learning.loss import *


class Trainer():
    def __init__(self,
                 model,
                 train_loader, test_loader,
                 optimizer, scheduler, criterion,
                 epochs,
                 folder, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.epochs = epochs
        self.folder = folder
        self.device = device

        self.writer = SummaryWriter(log_dir=folder)


    def trainRun(self, epoch: int = 0):
        for epoch in range(epoch, self.epochs):
            epoch_start_time = time.time()

            train_loss = self.train()
            val_loss, val_loss_in = self.evaluate(((epoch % 5) == 0))

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
        for batch_idx, (inputs, targets, mask, _) in enumerate(self.train_loader):

            inputs, targets, mask = inputs.to(self.device), targets.to(self.device), mask.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            # Calculate loss
            loss = self.criterion(outputs, targets, mask)

            loss.backward()
            #nn.utils.clip_grad_value_(self.model.parameters(), clip_value=0.1)

            # Change weights
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss

        return total_loss / len(self.train_loader)
    


    # Evaluation function
    def evaluate(self, verbose):
        self.model.eval()

        if verbose:
            individual_criterium = MaskedIndividualLoss(MaskedMAELoss())
            individual_criterium_mean = IndividualLoss(MARELoss())
            individual_error = {}
            individual_error_mean = {}

        total_loss = 0
        total_mean_loss = 0

        with torch.no_grad():
            for batch, (inputs, targets, masks, names) in enumerate(self.test_loader):
                inputs, targets, masks  = inputs.to(self.device), targets.to(self.device), masks.to(self.device)

                outputs = self.model(inputs)
                total_loss += self.criterion(outputs, targets, masks).item()
                #calculate inlet loss
                output_mean = outputs[:, 0, 1].mean(dim=(-2, -1))
                target_mean = targets[:, 0, 1].mean(dim=(-2, -1))
                mean_loss = MARELoss()(output_mean, target_mean)
                total_mean_loss += mean_loss.item()
                if verbose:
                    result = individual_criterium(outputs, targets, masks, names)
                    individual_error.update(result)
                    print(outputs.shape)
                    result = individual_criterium_mean(output_mean, target_mean, names)
                    individual_error_mean.update(result)
                if batch == 0:
                    input_sample = inputs[0][0].cpu().numpy()
                    output_sample = outputs[0][0].cpu().numpy()
                    target_sample = targets[0][0].cpu().numpy()
                    mask_sample = masks[0][0].cpu().numpy()
                    self.writer.add_graph(self.model, torch.unsqueeze(inputs[0, :, :, :, :], 0))

            # print result
            if verbose:
                visualize(input_sample, output_sample, target_sample, mask_sample, self.folder)
                individual_error = {key: [individual_error[key], individual_error_mean[key]] for key in individual_error}
                saveCSV(individual_error, (self.folder + "/errors.csv"))
                #self.writer.add_graph(self.model)


        return total_loss / len(self.test_loader), total_mean_loss / len(self.test_loader)
    

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