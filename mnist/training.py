import time
import math
import torch

torch.manual_seed(1)


class Training():
    """ Class to encapsulate model training """

    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 train,
                 test,
                 train_loader,
                 test_loader,
                 lr,
                 epochs,
                 device,
                 dropout
                 ):
        """ Initialize Trainer

        Args:
            model (Net): Instance of the model to be trained
            optimizer (torch.optim): Instance of Optimizer used
            scheduler (Scheduler): Instance of scheduler used
            train (train): Training function for model
            test (test): Validation function for model
            train_loader (DataLoader): Train data loader for model training
            test_loader (DataLoader): Validation data loader for model validation
            lr (float): LR value used for training
            epochs (int): Number of epochs for which training is done
            device (str): Device to use for training cuda/cpu
            dropout (bool): Enable/Disable dropout
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train = train
        self.test = test
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.dropout = dropout
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.list_train_loss = []
        self.list_valid_loss = []
        self.list_train_correct = []
        self.list_valid_correct = []

        self.schedule = []

        self.start_time = 0
        self.end_time = 0

        self.best_perc = 99.2
        self.best_path = ""

    def epoch_time(self):
        """ Calculate epoch time

        Returns:
            (int, int): Time consumed (mins, seconds)
        """
        elapsed_time = self.end_time - self.start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def print_epoch_progress(self, epoch, train_correct, train_loss, valid_correct, valid_loss):
        """ Log status for epoch

        Args:
            epoch (int): epoch number
            train_correct (int): number or correct predictions
            train_loss (float): loss incurred while training
            valid_correct (int): number of correct predictions
            valid_loss (float): loss incurred while validation
        """
        epoch_mins, epoch_secs = self.epoch_time()
        lr = self.schedule[epoch]
        print(f'| {epoch+1:5} | {lr:.6f} | {epoch_mins:02}m {epoch_secs:02}s | {str(round(train_loss, 6)):9} | {train_correct:12} | {str(round(100. * train_correct / len(self.train_loader.dataset), 2)):7}% | {valid_loss:.6f} | {valid_correct:10} | {str(round(100. * valid_correct / len(self.test_loader.dataset), 2)):5}% |')

    def save_best(self, valid_correct):
        """ Save the best model based on validation accuracy

        Args:
            valid_correct (int): number of correct predictions
        """
        valid_perc = (100. * valid_correct / len(self.test_loader.dataset))

        if valid_perc >= self.best_perc:
            self.best_perc = valid_perc
            self.best_path = f'model_weights_{valid_perc:.2f}.pth'
            torch.save(self.model.state_dict(), self.best_path)

    def run(self):
        """ Train training of model """
        print(
            f'| Epoch | {"LR":8} | {"Time":7} | TrainLoss | TrainCorrect | TrainAcc | {"ValLoss":8} | ValCorrect | ValAcc |')
        for epoch in range(self.epochs):
            self.schedule.append(self.optimizer.param_groups[0]['lr'])
#             self.log_epoch_params(epoch)
            self.start_time = time.time()

            train_loss, train_correct = self.train(
                self.model, self.train_loader, self.optimizer, self.dropout, self.device, self.scheduler)
            valid_loss, valid_correct = self.test(
                self.model, self.test_loader, self.device)

            self.list_train_loss.append(train_loss)
            self.list_valid_loss.append(valid_loss)

            self.list_train_correct.append(
                100. * train_correct / len(self.train_loader.dataset))
            self.list_valid_correct.append(
                100. * valid_correct / len(self.test_loader.dataset))

            self.end_time = time.time()

            self.save_best(valid_correct)

            self.print_epoch_progress(
                epoch, train_correct, train_loss, valid_correct, valid_loss)

    def print_best_model(self):
        """ Print the best model """
        self.model.load_state_dict(torch.load(self.best_path))
        self.model.eval()

        valid_loss, valid_correct = self.test(
            self.model, self.test_loader, self.device)

        print(f'Val Accuracy: {valid_correct:4d}/{len(self.test_loader.dataset):5d} | Percent: {(100. * valid_correct / len(self.test_loader.dataset)):.2f}% | Val. Loss: {valid_loss:.6f}')
