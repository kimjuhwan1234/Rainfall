from torch.nn.functional import mse_loss
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        self.MLP = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
        ).double()
        self.batch_norm = nn.BatchNorm1d(hidden_size // 2, affine=True, track_running_stats=True).double()
        self.fc = nn.Linear(hidden_size//2, output_size).double()

    def forward(self, train, gt=None):
        output = self.MLP(train)
        output = self.fc(self.batch_norm(output))

        if gt != None:
            # gt = gt.squeeze()
            loss = torch.sqrt(mse_loss(output, gt))
            return output, loss

        return output