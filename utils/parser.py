import argparse
from Module.model import *

parser = argparse.ArgumentParser()
parser.add_argument("--input_size", type=int, default=57, help="input_size")
parser.add_argument("--hidden_size", type=int, default=1024, help="hidden_size")
parser.add_argument("--output_size", type=int, default=1, help="output_size")
parser.add_argument("--num_layers", type=int, default=2, help="num_layers")
parser.add_argument("--bidirectional", type=bool, default=False, help="bidirectional")

opt_model = parser.parse_args()
print(opt_model)
# ---------------------------------------------------------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--epochs", type=int, default=50, help="epochs")
parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
parser.add_argument("--patience", type=int, default=10, help="patience")
parser.add_argument("--device", type=str, default='cuda', help="device")
parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
parser.add_argument("--backbone_weight_path", type=str, default='Weight/STN001.pth', help="weight")

opt_train = parser.parse_args()
print(opt_train)
# ---------------------------------------------------------------------------------------------------------------------#
backbone = MLP(input_size=opt_model.input_size, hidden_size=opt_model.hidden_size, output_size=opt_model.output_size)
# backbone.load_state_dict(torch.load(opt_train.backbone_weight_path))
# ---------------------------------------------------------------------------------------------------------------------#
config = dict()
config['model'] = opt_model
config['train'] = opt_train
config['structure'] = backbone
