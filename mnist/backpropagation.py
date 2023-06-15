import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam

torch.manual_seed(1)


def train(use_l1=False, lambda_l1=5e-4):
    """ Function to return train function instance

    Args:
        use_l1 (bool, optional): Enable L1. Defaults to False.
        lambda_l1 (float, optional): L1 Value. Defaults to 5e-4.
    """
    def internal(model, train_loader, optimizer, dropout, device, scheduler=None):
        """ This function is for running backpropagation

        Args:
            model (Net): Model instance to train
            train_loader (Dataset): Dataset used in training
            optimizer (torch.optim): Optimizer used
            dropout (bool): Enable/Disable 
            device (string, cuda/cpu): Device type Values Allowed - cuda/cpu
            scheduler (Scheduler, optional): scheduler instance used for updating lr while training. Defaults to None.

        Returns:
            (float, int): Loss, Number of correct Predictions
        """
        model.train()
        epoch_loss = 0
        correct = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data, dropout)
            loss = F.nll_loss(output, target)
            if use_l1 == True:
                l1 = 0
                for p in model.parameters():
                    l1 = l1 + p.abs().sum()
                loss = loss + lambda_l1 * l1
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            epoch_loss += loss.item()

        return epoch_loss / len(train_loader), correct

    return internal


def test(model, test_loader, device):
    """ Function to perform model validation

    Args:
        model (Net): Model instance to run validation
        test_loader (Dataset): Dataset used in validation
        device (string, cuda/cpu): Device type Values Allowed - cuda/cpu

    Returns:
        (float, int): Loss, Number of correct Predictions
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    return test_loss, correct


def get_sgd_optimizer(model, lr, momentum=0, weight_decay=0):
    return SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


def get_adam_optimizer(model, lr, weight_decay=0):
    return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    get_adam_optimizer
