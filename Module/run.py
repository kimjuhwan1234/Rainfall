import matplotlib.pyplot as plt
from Module.train import *
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from Module.dataset import CustomDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Run:
    def __init__(self, train, val, weight_path, config):
        self.config = config
        self.lr = self.config['train'].lr
        self.epochs = self.config['train'].epochs
        self.batch_size = self.config['train'].batch_size
        self.patience = self.config['train'].patience
        self.device = self.config['train'].device

        self.model = self.config['structure']
        self.model2 = self.config['structure2']

        self.train = train
        self.val = val
        self.weight_path = weight_path

    def load_data(self):
        print(' ')
        print('Loading data...')
        train_dataset = CustomDataset(self.train)
        val_dataset = CustomDataset(self.val)

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
        self.model2.to(self.device)
        opt_VAE = Adam(self.model.parameters(), lr=self.lr)
        lr_VAE = ReduceLROnPlateau(opt_VAE, mode='min', factor=0.2, patience=self.patience)
        opt_GAN = AdamW(self.model2.parameters(), lr=self.lr)

        parameters = {
            'model': self.model,
            'device': self.device,
            'weight_path': self.weight_path,
            'num_epochs': self.epochs,
            'patience': self.patience,

            'train_dl': self.dataloaders['train'],
            'val_dl': self.dataloaders['val'],

            'optimizer': opt_VAE,
            'lr_scheduler': lr_VAE,

            'opt_GAN': opt_GAN,
            'GAN': self.model2,
        }

        TM = Train_Module(parameters)
        self.loss_hist, self.metric_hist = TM.train_and_eval()

        print('Finished training model!')

    def check_validation(self):
        print(' ')
        print('Check loss and Entropy...')

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
        plt.title("Train-Val Entropy")
        plt.plot(range(1, early_stop_epoch + 1), metric_hist_numpy.iloc[:, 0], label="train")
        plt.plot(range(1, early_stop_epoch + 1), metric_hist_numpy.iloc[:, 1], label="val")
        plt.ylabel("Entropy")
        plt.xlabel("Training Epochs")
        plt.legend()
        plt.show()

        print('Finished checking loss and Entropy!')
