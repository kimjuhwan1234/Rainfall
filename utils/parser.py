import argparse
from Module.model import *

parser = argparse.ArgumentParser()
parser.add_argument("--input_dim", type=int, default=57, help="input_size")
parser.add_argument("--output_dim", type=int, default=10, help="hidden_size")
parser.add_argument("--z_dim", type=int, default=16, help="hidden_size")
parser.add_argument("--num_layers", type=int, default=2, help="num_layers")
parser.add_argument("--bidirectional", type=bool, default=False, help="bidirectional")

opt_model = parser.parse_args()
print(opt_model)
# ---------------------------------------------------------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--epochs", type=int, default=50, help="epochs")
parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
parser.add_argument("--patience", type=int, default=50, help="patience")
parser.add_argument("--device", type=str, default='cuda', help="device")
parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
parser.add_argument("--backbone_weight_path", type=str, default='Weight/meta_model.pth', help="weight")

opt_train = parser.parse_args()
print(opt_train)
# ---------------------------------------------------------------------------------------------------------------------#
backbone = VAE(opt_model.input_dim, opt_model.output_dim, opt_model.z_dim)
# backbone.load_state_dict(torch.load(opt_train.backbone_weight_path))
# ---------------------------------------------------------------------------------------------------------------------#
config = dict()
config['model'] = opt_model
config['train'] = opt_train
config['structure'] = backbone
