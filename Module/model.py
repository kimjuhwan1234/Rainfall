import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_function(x_recon, x, mu, logvar, method: int):
    if method == 0:
        x = x[:, :12]
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')

    if method == 1:
        x = x[:, 12:]
        recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')

    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div


class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.fc_con = nn.Linear(12, 128).double()
        self.fc_con_mu = nn.Linear(128, z_dim).double()
        self.fc_con_logvar = nn.Linear(128, z_dim).double()

        self.fc_one = nn.Linear(54, 128).double()
        self.fc_one_mu = nn.Linear(128, z_dim).double()
        self.fc_one_logvar = nn.Linear(128, z_dim).double()

    def forward(self, x):
        con_x = x[:, :12]
        con_h = self.fc_con(con_x)
        con_mu = self.fc_con_mu(F.leaky_relu(con_h,0.2))
        con_logvar = self.fc_con_logvar(F.leaky_relu(con_h, 0.2))

        dis_x = x[:, 12:]
        dis_h = self.fc_one(dis_x)
        dis_mu = self.fc_one_mu(F.leaky_relu(dis_h, 0.2))
        dis_logvar = self.fc_one_logvar(F.leaky_relu(dis_h, 0.2))

        return con_mu, con_logvar, dis_mu, dis_logvar


class Decoder(nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        self.fc1_con = nn.Linear(z_dim, 128).double()
        self.fc2_con = nn.Linear(128, 12).double()

        self.fc1_dis = nn.Linear(z_dim, 128).double()
        self.fc2_dis = nn.Linear(128, 54).double()

        self.sigmoid = nn.Sigmoid().double()

    def forward(self, x, method):
        if method == 0:
            h = self.fc1_con(x)
            x_recon = self.fc2_con(F.leaky_relu(h, 0.2))
            return x_recon

        if method == 1:
            h = self.fc1_dis(x)
            h = self.fc2_dis(F.leaky_relu(h, 0.2))
            h_one = h[:, :-9]
            h_class = h[:, -9:]

            x_recon_one = self.sigmoid(h_one)
            return x_recon_one, h_class


class VAE(nn.Module):
    def __init__(self, z_dim, num=None):
        super(VAE, self).__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
        self.z_dim = z_dim
        self.num = num
        self.sigmoid = nn.Sigmoid().double()
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

            con_x_re = self.decoder(con_z, 0)
            one_x_re, h_class = self.decoder(dis_z, 1)
            dis_x = (one_x_re > 0.5).float()
            class_x = torch.argmax(self.softmax(h_class), dim=1).reshape(-1, 1)
            x_recon = torch.cat((con_x_re, dis_x, class_x), dim=1)

            dis_x_re = torch.cat((one_x_re, self.sigmoid(h_class)), dim=1)
            loss1 = loss_function(con_x_re, x, con_mu, con_logvar, 0)
            loss2 = loss_function(dis_x_re, x, dis_mu, dis_logvar, 1)
            gt = torch.argmax(self.softmax(x[:, -9:]), dim=1)
            target_loss = F.cross_entropy(h_class, gt)
            total_loss = (loss1 + loss2) + target_loss * 100

            return x_recon, total_loss, target_loss

        if x == None:
            con_z = torch.randn(self.num, self.z_dim).double()
            dis_z = torch.randn(self.num, self.z_dim).double()

            con_x_re = self.decoder(con_z, 0)
            one_x_re, h_class = self.decoder(dis_z, 1)
            dis_x = (one_x_re > 0.5).float()
            class_x = torch.argmax(self.softmax(h_class), dim=1).reshape(-1, 1)
            x_recon = torch.cat((con_x_re, dis_x, class_x), dim=1)
            return x_recon
