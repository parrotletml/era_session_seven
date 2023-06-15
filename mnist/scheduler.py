import torch
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np

torch.manual_seed(1)


class CustomOneCycleLR():
    """Custom class for one cycle lr

    """

    def __init__(self, optimizer, schedule, steps_per_epoch):
        """Initialize Scheduler

        Args:
            optimizer (torch.optim): Optimizer to be used for training
            schedule (list): Schedule to be used for training
            steps_per_epoch (int): Number of steps before changing lr value
        """
        self.optimizer = optimizer
        self.schedule = schedule
        self.epoch = 0
        self.steps = 0
        self.steps_per_epoch = steps_per_epoch
        self.optimizer.param_groups[0]['lr'] = self.schedule[self.epoch]

    def step(self):
        """Called every step to set next lr value
        """
        self.optimizer.param_groups[0]['lr'] = self.lr_schedules()

    def lr_schedules(self):
        """Get Next value for lr

        Returns:
            float: LR value to use for next step
        """
        self.steps += 1
        if self.steps % self.steps_per_epoch == 0:
            self.steps = 1
            self.epoch += 1
        return self.schedule[self.epoch]


def one_cycle_lr_pt(optimizer, lr, max_lr, steps_per_epoch, epochs, anneal_strategy='linear'):
    """Create instance of one cycle lr scheduler from python

    Args:
        optimizer (torch.optim): Optimizer to be used for Training
        lr (float): base lr value used
        max_lr (float): max lr value used in one cycle ly
        steps_per_epoch (int): Number of steps in each epochs
        epochs (int): number of epochs for which training is done
        anneal_strategy (str, optional): Defaults to 'linear'.

    Returns:
        OneCycleLR: Instance of one cycle lr scheduler
    """
    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        anneal_strategy='linear'
    )


def one_cycle_lr_custom(optimizer, lr, max_lr, steps_per_epoch, epochs, anneal_strategy='linear'):
    """Create instance of one cycle lr scheduler from python

    Args:
        optimizer (torch.optim): Optimizer to be used for Training
        lr (float): base lr value used
        max_lr (float): max lr value used in one cycle ly
        steps_per_epoch (int): Number of steps in each epochs
        epochs (int): number of epochs for which training is done
        anneal_strategy (str, optional): Defaults to 'linear'.

    Raises:
        Exception: Exception for epoch value < 12

    Returns:
        CustomOneCycleLR: Instance of one cycle lr scheduler
    """
    if epochs < 12:
        raise Exception("Epoch value can not be less than 12")
    schedule = np.interp(
        np.arange(epochs+1), [0, 2, 8, 12, epochs], [lr, max_lr, lr/5.0, lr/10.0, 0])
    return CustomOneCycleLR(optimizer, schedule, steps_per_epoch)
