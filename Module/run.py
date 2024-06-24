import matplotlib.pyplot as plt
from Module.train import *
from torch.optim import Adam
from torch.utils.data import DataLoader
from Module.dataset import CustomDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Run:
    def __init__(self, X_train, y_train, X_val, y_val, weight_path, config):
        self.config = config
        self.lr = self.config['train'].lr
        self.epochs = self.config['train'].epochs
        self.batch_size = self.config['train'].batch_size
        self.patience = self.config['train'].patience
        self.device = self.config['train'].device

        self.model = self.config['structure']

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.weight_path = weight_path

    def load_data(self):
        print(' ')
        print('Loading data...')
        train_dataset = CustomDataset(self.X_train, self.y_train)
        val_dataset = CustomDataset(self.X_val, self.y_val)

        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False),
            'val': DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False),
        }

        self.dataloaders = dataloaders

        print('Finished loading data!')

    def run_model(self):
        print(' ')
        print('Training model...')

        self.model.to(self.device)
        opt = Adam(self.model.parameters(), lr=self.lr)
        lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.2, patience=self.patience)

        parameters = {
            'num_epochs': self.epochs,
            'weight_path': self.weight_path,

            'train_dl': self.dataloaders['train'],
            'val_dl': self.dataloaders['val'],

            'patience': self.patience,
            'optimizer': opt,
            'lr_scheduler': lr_scheduler,
        }

        TM = Train_Module(self.device)
        self.model, self.loss_hist, self.metric_hist = TM.train_and_eval(self.model, parameters)

        print('Finished training model!')

    def check_validation(self):
        print(' ')
        print('Check loss and RMSE...')

        loss_hist_numpy = self.loss_hist.map(
            lambda x: x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x)
        metric_hist_numpy = self.metric_hist.map(
            lambda x: x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x)
        early_stop_epoch = len(loss_hist_numpy)

        # plot loss progress
        plt.title("Train-Val Loss")
        plt.plot(range(1, early_stop_epoch + 1), loss_hist_numpy.iloc[:, 0], label="train")
        plt.plot(range(1, early_stop_epoch + 1), loss_hist_numpy.iloc[:, 1], label="val")
        plt.ylabel("Loss")
        plt.xlabel("Training Epochs")
        plt.legend()
        plt.show()

        # plot accuracy progress
        plt.title("Train-Val R2")
        plt.plot(range(1, early_stop_epoch + 1), metric_hist_numpy.iloc[:, 0], label="train")
        plt.plot(range(1, early_stop_epoch + 1), metric_hist_numpy.iloc[:, 1], label="val")
        plt.ylabel("R2")
        plt.xlabel("Training Epochs")
        plt.legend()
        plt.show()

        print('Finished checking loss and R2!')
