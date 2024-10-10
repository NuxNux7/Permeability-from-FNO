import torch
import torch.nn as nn

# MSE: Mean Squared Error

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()


    def forward(self, output, target, mask):

        error = (output - target)
        squared_error = error**2
        masked_error = squared_error * mask

        loss = masked_error.sum() / mask.sum()

        # Debugging prints
        '''print(f"Output sum: {output.sum().item()}")
        print(f"Target sum: {target.sum().item()}")
        print(f"Error sum: {error.sum().item()}")
        print(f"Squared error sum: {squared_error.sum().item()}")
        print(f"Masked squared error sum: {masked_error.sum().item()}")
        print(f"Mask sum: {mask.sum().item()}")
        print(f"Computed loss: {loss.item()}")'''

        return loss
    
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output, target):
        squared_error = (output - target)**2

        return squared_error.mean()


# Huber Loss: Mixture of MSE and MAE

class MaskedHuberLoss(nn.Module):
    def __init__(self, delta=1):
        super(MaskedHuberLoss, self).__init__()
        self.delta = delta
    
    def forward(self, output, target, mask):
        error = torch.abs(output - target) * mask

        loss = torch.where(error <= self.delta,
                   0.5 * torch.pow(error, 2),
                   self.delta * (error - 0.5 * self.delta))

        return loss.sum() / mask.sum()

class HuberLoss(nn.Module):
    def __init__(self, delta=1):
        super(HuberLoss, self).__init__()
        self.delta = delta
    
    def forward(self, output, target):
        error = torch.abs(output - target)

        loss = torch.where(error <= self.delta,
                   0.5 * torch.pow(error, 2),
                   self.delta * (error - 0.5 * self.delta))

        return loss.mean()


# MAE: Mean Absolute Error

class MaskedMAELoss(nn.Module):
    def __init__(self):
        super(MaskedMAELoss, self).__init__()

    def forward(self, output, target, mask):
        error = (output - target) * mask
        absolute_error = torch.abs(error)
        return absolute_error.sum() / mask.sum()


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, output, target):
        error = torch.abs(output - target)
        return error.mean()


# MARE: Mean Absolute Relative Error

class MaskedMARELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(MaskedMARELoss, self).__init__()
        self.epsilon = epsilon


    def forward(self, output, target, mask):

        relative_error = (output - target) / (target + self.epsilon)
        squared_error = torch.abs(relative_error)
        masked_error = squared_error * mask

        return masked_error.sum() / mask.sum()

class MARELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(MARELoss, self).__init__()
        self.epsilon = epsilon


    def forward(self, output, target):

        relative_error = (output - target) / (target + self.epsilon)
        squared_error = torch.abs(relative_error)

        return squared_error.mean()
   

# RMSRE: Root Mean Square Relative Error

class MaskedRMSRELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(MaskedRMSRELoss, self).__init__()
        self.epsilon = epsilon


    def forward(self, output, target, mask):

        relative_error = (output - target) / (target + self.epsilon)
        squared_error = torch.clamp(relative_error**2, max=1e8)
        masked_error = squared_error * mask

        return torch.sqrt(masked_error.sum() / mask.sum())

class RMSRELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(RMSRELoss, self).__init__()
        self.epsilon = epsilon


    def forward(self, output, target):

        relative_error = (output - target) / (target + self.epsilon)
        squared_error = torch.clamp(relative_error**2, max=1e8)

        return torch.sqrt(squared_error.mean())
    

# Individual Loss

class MaskedIndividualLoss(nn.Module):
    def __init__(self, criterium):
        super(MaskedIndividualLoss, self).__init__()
        self.criterium = criterium

    def forward(self, output, target, mask, name):
        dict = {}

        for sample in range(output.shape[0]):
            dict[name[sample]] = self.criterium(output[sample], target[sample], mask[sample]).item()
        
        return dict

class IndividualLoss(nn.Module):
    def __init__(self, criterium):
        super(IndividualLoss, self).__init__()
        self.criterium = criterium
    
    def forward(self, output, target, name):
        dict = {}

        for sample in range(output.shape[0]):
            dict[name[sample]] = self.criterium(output[sample], target[sample]).item()

        return dict
    

# R2 Score

class R2Score(nn.Module):
    def __init__(self):
        super(R2Score, self).__init__()
    
    def forward(self, output, target):
        error = output - target
        error = torch.pow(error, 2)

        mean_target = target.mean()
        mean_distance = target - mean_target
        mean_distance = torch.pow(mean_distance, 2)
        
        #print(mean_distance.sum() / len(target))
        if len(target) == 1:      #guess work
            mean_distance = torch.empty_like(target).fill_(0.1)
            print("Warning! Only one Sample in Batch -> guessing mean_distance with 0.1")

        return 1 - (error.sum() / mean_distance.sum())