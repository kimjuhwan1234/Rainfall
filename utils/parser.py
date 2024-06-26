import argparse
from Module.model import *

parser = argparse.ArgumentParser()
parser.add_argument("--input_dim", type=int, default=57, help="input_size")
parser.add_argument("--output_dim", type=int, default=10, help="hidden_size")
parser.add_argument("--z_dim", type=int, default=32, help="hidden_size")
parser.add_argument("--num_layers", type=int, default=2, help="num_layers")
parser.add_argument("--bidirectional", type=bool, default=True, help="bidirectional")

opt_model = parser.parse_args()
print(opt_model)
# ---------------------------------------------------------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--epochs", type=int, default=50, help="epochs")
parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
parser.add_argument("--patience", type=int, default=15, help="patience")
parser.add_argument("--device", type=str, default='cuda', help="device")
parser.add_argument("--num_workers", type=int, default=4, help="num_workers")

opt_train = parser.parse_args()
print(opt_train)
# ---------------------------------------------------------------------------------------------------------------------#
backbone = MLP(opt_model.input_dim, opt_model.output_dim, opt_model.num_layers, opt_model.bidirectional)
# backbone.load_state_dict(torch.load('Weight/MLP.pth'))
model = VAE(opt_model.z_dim, backbone)
# model.load_state_dict(torch.load('Weight/VAE.pth'))
# ---------------------------------------------------------------------------------------------------------------------#
config = dict()
config['model'] = opt_model
config['train'] = opt_train
config['structure'] = model
