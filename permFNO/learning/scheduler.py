import math

from torch.optim.lr_scheduler import LambdaLR


class CosineLRLambda:
    """
    Initializes the CosineLRLambda class.

    Args:
        num_warmup_steps (int): The number of warmup steps to use.
        num_training_steps (int): The total number of training steps.
        num_cycles (float): The number of cosine cycles to use.
        min_lambda (float): The minimum percentage of the learning rate multiplier.
    """
    def __init__(self,
                 num_warmup_steps,
                 num_training_steps,
                 num_cycles,
                 min_lambda):
        
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        self.min_lambda = min_lambda
        self.max_amp = 1 - min_lambda

    def __call__(self, current_step):  # the function formerly known as "bar"
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        progress = float(current_step - self.num_warmup_steps) / \
            float(max(1, self.num_training_steps - self.num_warmup_steps))
        amplitude = 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress))
        return max(0.0, self.min_lambda + self.max_amp * amplitude)


class CosineWithWarmupScheduler(LambdaLR):
    """
    Initializes the CosineWithWarmupScheduler class.
    The cosine scheduler is a learning rate scheduling technique that uses a cosine function to adjust the learning rate over the course of training.
    It is often used in conjunction with a warmup period, where the learning rate starts low and gradually increases to the full learning rate.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to use with the scheduler.
        num_warmup_steps (int): The number of warmup steps to use.
        num_training_steps (int): The total number of training steps.
        min_lambda (float, optional): The minimum percentage of the learning rate multiplier. Defaults to 0.
        num_cycles (float, optional): The number of cosine cycles to use. Defaults to 0.5.
        last_epoch (int, optional): The index of the last epoch. Defaults to -1.
    """

    def __init__(self,
                 optimizer,
                 num_warmup_steps: int,
                 num_training_steps: int,
                 min_lambda: float = 0,
                 num_cycles: float = 0.5,
                 last_epoch=-1):
        
        lr_lambda = CosineLRLambda(
            num_warmup_steps, num_training_steps, num_cycles, min_lambda)
        super().__init__(optimizer, lr_lambda, last_epoch)