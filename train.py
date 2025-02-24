import torch
import torch.nn as nn
from train_utils import evaluate_link_prediction, add_spatial_edges




def train(e, model, link_predictor, optimizer, train_loader, valid_loader, device):
    """
    Runs offline training for model, link_predictor and node embeddings given the message
    edges and supervision edges.
    :param model: Torch Graph model used for updating node embeddings based on message passing
    :param link_predictor: Torch model used for predicting whether edge exists or not
    :param emb: (N, d) Initial node embeddings for all N nodes in graph
    :param edge_index: (2, E) Edge index for all edges in the graph
    :param pos_train_edge: (PE, 2) Positive edges used for training supervision loss
    :param batch_size: Number of positive (and negative) supervision edges to sample per batch
    :param optimizer: Torch Optimizer to update model parameters
    :data_loader: DataLoader object for training
    :return: Average supervision loss over all positive (and correspondingly sampled negative) edges
    """


    model.train()
    link_predictor.train()

    train_losses = []
    # for edge_id in DataLoader(range(pos_train_edge.shape[0]), batch_size, shuffle=True):
    for data in train_loader:
        edge_index = data.edge_index.to(device)
        emb = data.x.to(device)
        batch_indices = data.batch

        neg_edges_index = add_spatial_edges(data.edge_index, data.x, batch_indices)
        neg_edges_index = neg_edges_index.to(device)
        new_edge_index = torch.cat([edge_index, neg_edges_index], dim=-1)

        # from visualize import visualize_graphs
        # visualize_graphs(edge_index, neg_edges_index, data.x, data.batch)

        optimizer.zero_grad()

        # Run message passing on the inital node embeddings to get updated embeddings
        node_emb = model(emb, new_edge_index)  # (N, d)

        # Predict the class probabilities on the batch of positive edges using link_predictor
        # pos_edge = pos_train_edge[edge_id].T  # (2, B)
        pos_edge = edge_index  # (2, B)
        pos_pred = link_predictor(node_emb[pos_edge[0]], node_emb[pos_edge[1]])  # (B, )

        # Sample negative edges (same as number of positive edges) and predict class probabilities
        # neg_edge = negative_sampling(edge_index, num_nodes=emb.shape[0],
        #                              num_neg_samples=edge_index.shape[1], method='dense', force_undirected=True)  # (Ne,2)

        neg_pred = link_predictor(node_emb[neg_edges_index[0]], node_emb[neg_edges_index[1]])  # (Ne,)

        predictions = torch.cat([pos_pred, neg_pred])  # Shape (N,)
        targets = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])
        pos_weight = torch.cat([torch.ones_like(pos_pred) * 4, torch.zeros_like(neg_pred)])

        criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
        loss = criterion(predictions, targets)

        # Compute the corresponding negative log likelihood loss on the positive and negative edges
        # loss = -torch.log(pos_pred + 1e-15).mean() - torch.log(1 - neg_pred + 1e-15).mean()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    pos_preds = []
    neg_preds = []
    for data in valid_loader:
        edge_index = data.edge_index.to(device)
        emb = data.x.to(device)
        batch_indices = data.batch

        pos_edge = edge_index  # (2, B)
        neg_edges_index = add_spatial_edges(data.edge_index, data.x, batch_indices)
        neg_edges_index = neg_edges_index.to(device)
        new_edge_index = torch.cat([edge_index, neg_edges_index], dim=-1)

        with torch.no_grad():
            node_emb = model(emb, new_edge_index)  # (N, d)

            pos_pred = link_predictor.infer(node_emb[pos_edge[0]], node_emb[pos_edge[1]])
            neg_pred = link_predictor.infer(node_emb[neg_edges_index[0]], node_emb[neg_edges_index[1]])


        pos_preds.append(pos_pred)
        neg_preds.append(neg_pred)

    pos_preds = torch.cat(pos_preds)
    neg_preds = torch.cat(neg_preds)

    auc, ap, accuracy, img = evaluate_link_prediction(e, pos_preds, neg_preds)

    return sum(train_losses) / len(train_losses), auc, ap, accuracy, img