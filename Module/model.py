import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()

        self.MLP = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        ).double()

        self.softmax = nn.Softmax(dim=1).double()

    def forward(self, data, gt=None):
        x_recon_target_latent = self.MLP(data)
        x_recon_target_prob = self.softmax(x_recon_target_latent)
        output = torch.argmax(x_recon_target_prob, dim=1).reshape(-1, 1)

        if gt != None:
            loss = nn.functional.cross_entropy(x_recon_target_latent.float(), gt.long())
            return output, loss

        return output


def loss_function(x_recon, x, mu, logvar, method: int):
    if method == 0:
        x = x[:, :12]
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')

    if method == 1:
        x = x[:, 12:-1]
        recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')

    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div


class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.fc_con = nn.Linear(12, 64).double()
        self.fc_con_mu = nn.Linear(64, z_dim).double()
        self.fc_con_logvar = nn.Linear(64, z_dim).double()

        self.fc_one = nn.Linear(45, 64).double()
        self.fc_one_mu = nn.Linear(64, z_dim).double()
        self.fc_one_logvar = nn.Linear(64, z_dim).double()

    def forward(self, x):
        con_x = x[:, :12]
        con_h = self.fc_con(con_x)
        con_mu = self.fc_con_mu(F.leaky_relu(con_h))
        con_logvar = self.fc_con_logvar(F.leaky_relu(con_h))

        dis_x = x[:, 12:-1]
        dis_h = self.fc_one(dis_x)
        dis_mu = self.fc_one_mu(F.leaky_relu(dis_h))
        dis_logvar = self.fc_one_logvar(F.leaky_relu(dis_h))

        return con_mu, con_logvar, dis_mu, dis_logvar


class Decoder(nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        self.fc1_con = nn.Linear(z_dim, 64).double()
        self.fc2_con = nn.Linear(64, 12).double()

        self.fc1_dis = nn.Linear(z_dim, 64).double()
        self.fc2_dis = nn.Linear(64, 45).double()

        self.sigmoid = nn.Sigmoid().double()

    def forward(self, x, method):
        if method == 0:
            h = self.fc1_con(x)
            x_recon = self.fc2_con(F.leaky_relu(h))

        if method == 1:
            h = self.fc1_dis(x)
            x_recon = self.sigmoid(self.fc2_dis(F.leaky_relu(h)))

        return x_recon


class VAE(nn.Module):
    def __init__(self, z_dim, model, num=None):
        super(VAE, self).__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
        self.model = model
        self.z_dim = z_dim
        self.num = num
        self.softmax = nn.Softmax(dim=1).double()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x=None):
        if x != None:
            con_mu, con_logvar, dis_mu, dis_logvar = self.encoder(x)
            con_z = self.reparameterize(con_mu, con_logvar)
            dis_z = self.reparameterize(dis_mu, dis_logvar)
            x_gt = x[:, -1]

            con_x_re = self.decoder(con_z, 0)
            dis_x_re = self.decoder(dis_z, 1)
            dis_x = (dis_x_re > 0.5).float()
            x_recon_input = torch.cat((con_x_re, dis_x), dim=1)
            x_recon_target, target_loss = self.model(x_recon_input, x_gt)
            x_recon = torch.cat((x_recon_input, x_recon_target), dim=1)

            loss1 = loss_function(con_x_re, x, con_mu, con_logvar, 0)
            loss2 = loss_function(dis_x_re, x, dis_mu, dis_logvar, 1)
            total_loss = (loss1 + loss2) + target_loss * 100

            return x_recon, total_loss, target_loss

        if x == None:
            con_z = torch.randn(self.num, self.z_dim).double()
            dis_z = torch.randn(self.num, self.z_dim).double()

            con_x_re = self.decoder(con_z, 0)
            dis_x_re = self.decoder(dis_z, 1)
            dis_x_re = (dis_x_re > 0.5).float()
            x_recon_input = torch.cat((con_x_re, dis_x_re), dim=1)
            x_recon_target = self.model(x_recon_input)
            x_recon = torch.cat((x_recon_input, x_recon_target), dim=1)
            return x_recon
