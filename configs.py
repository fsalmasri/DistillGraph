import argparse

parser = argparse.ArgumentParser(description="Script to train link prediction in offline graph setting")
parser.add_argument('--exp', type=str, default='32', help="Number of epochs for training")
parser.add_argument('--epochs', type=int, default=3000, help="Number of epochs for training")
parser.add_argument('--lr', type=float, default=3e-3, help="Learning rate training")
parser.add_argument('--node_dim', type=int, default=5, help='Embedding dimension for nodes')
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--hidden_channels', type=int, default=256)

parser.add_argument('--exp_dir', type=str, default="./experiments",
                    help="Path to exp dir for model checkpoints and experiment logs")
args = parser.parse_args()