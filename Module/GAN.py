import torch
import torch.nn as nn
import torch.nn.functional as F

class GAN(nn.Module):
    def __init__(self, input_dim):
        super(GAN, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        ).double()

        self.softmax = nn.Softmax(dim=1)

    def forward(self, device, x_recon, x=None):
        criterion = nn.BCELoss()
        valid = torch.ones(x_recon.size(0), 1).to(device).double()
        fake = torch.zeros(x_recon.size(0), 1).to(device).double()
        if x != None:
            gt = torch.argmax(self.softmax(x[:, -9:]), dim=1).reshape(-1, 1)
            x2 = torch.cat((x[:, :-9], gt), dim=1)
            real_loss = criterion(self.discriminator(x2), valid)
            fake_loss = criterion(self.discriminator(x_recon), fake)
            d_loss = (real_loss + fake_loss) / 2
            return d_loss

        return criterion(self.discriminator(x_recon), valid)
