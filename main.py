import os

import torch
# from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch.optim import optimizer
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch.utils.data import random_split

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from gnn_stack import GNNStack
from train import train
from link_predictor import LinkPredictor
# from evaluate import test
# from utils import print_and_log

import pathlib
from dataset import MyOwnDataset

import pandas as pd
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler


def main():
    from configs import args

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exp = args.exp
    optim_wd = 0
    epochs = args.epochs
    hidden_dim = args.hidden_channels
    dropout = args.dropout
    num_layers = args.num_layers
    lr = args.lr
    node_emb_dim = args.node_dim
    batch_size = args.batch_size
    exp_dir = args.exp_dir

    if exp_dir is None:
        exp_dir = "./experiments"

    model_dir = os.path.join(exp_dir, 'checkpoints')
    logs_dir = os.path.join(exp_dir, 'logs', exp)

    pathlib.Path(exp_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(logs_dir).mkdir(parents=True, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=logs_dir)

    ds = MyOwnDataset(root="DISTILL")

    # Define split sizes
    train_size = int(0.8 * len(ds))  # 80% for training
    valid_size = len(ds) - train_size  # Remaining for validation

    # Randomly split the dataset
    train_dataset, valid_dataset = random_split(ds, [train_size, valid_size])
    torch.save(train_dataset.indices, 'train_indices.pt')
    torch.save(valid_dataset.indices, 'valid_indices.pt')


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = GNNStack(node_emb_dim, hidden_dim, hidden_dim, 4, dropout, emb=True)
    model = model.to(device)

    link_predictor = LinkPredictor(hidden_dim, hidden_dim, 1, 4, dropout)
    link_predictor.to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(link_predictor.parameters()), lr=lr, weight_decay=optim_wd
    )
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=epochs)

    epochs_bar = tqdm(range(epochs))
    acc = 0
    for e in epochs_bar:

        train_loss, auc, ap, accuracy, img = train(e, model, link_predictor, optimizer, train_loader, valid_loader, device)
        scheduler.step()
        epochs_bar.set_description(f'[TRAIN] {e}/{args.epochs + 1} | Loss : {train_loss} | '
                                   f'acc: {accuracy} | lr: {optimizer.param_groups[0]["lr"]}')

        tb_writer.add_scalar('Train', train_loss, e)
        tb_writer.add_scalar('Valid/AUC', auc, e)
        tb_writer.add_scalar('Valid/AP', ap, e)
        tb_writer.add_scalar('Valid/ACC', accuracy, e)

        if img is not None:
            tb_writer.add_figure("ROC Curve", img, global_step=e)

        if accuracy > acc:
            torch.save(model.state_dict(), os.path.join(exp_dir, 'checkpoints', f'model_{args.exp}.pt'))
            torch.save(link_predictor.state_dict(), os.path.join(exp_dir, 'checkpoints', f'link_pred_{args.exp}.pt'))


if __name__ == "__main__":
    main()