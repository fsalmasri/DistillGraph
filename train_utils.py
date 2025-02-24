
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def evaluate_link_prediction(e, pos_scores, neg_scores, threshold=0.5):
    # Labels
    pos_labels = torch.ones(pos_scores.size(0), device=pos_scores.device)
    neg_labels = torch.zeros(neg_scores.size(0), device=neg_scores.device)

    # Combine scores and labels
    all_scores = torch.cat([pos_scores, neg_scores]).squeeze()
    all_labels = torch.cat([pos_labels, neg_labels])

    # Compute metrics
    auc = roc_auc_score(all_labels.cpu().numpy(), all_scores.cpu().detach().numpy())
    ap = average_precision_score(all_labels.cpu().numpy(), all_scores.cpu().detach().numpy())
    fpr, tpr, thresholds = roc_curve(all_labels.cpu().numpy(), all_scores.cpu().detach().numpy())

    img = None
    if e % 100 == 0:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from io import BytesIO
        from PIL import Image

        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")

        # Annotate thresholds
        for i, thresh in enumerate(thresholds):
            if i % (len(thresholds) // 10) == 0:  # Reduce clutter by selecting fewer points
                plt.text(fpr[i], tpr[i], f"{thresh:.2f}", fontsize=8, color="black", alpha=0.7)


        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.grid()

        # buf = BytesIO()
        # plt.savefig(buf, format='PNG')
        # plt.close()
        # buf.seek(0)

        img = plt.gcf() #Image.open(buf)
        # plt.show()

    # Hard predictions (optional)
    predictions = (all_scores > threshold).float()
    accuracy = (predictions == all_labels).float().mean().item()

    return auc, ap, accuracy, img

def add_spatial_edges(edge_index, nodes, batch):

    coords = nodes[:, :2]
    new_edges = []

    for graph_id in torch.unique(batch):
        mask = (batch == graph_id)
        global_indices = torch.nonzero(mask).squeeze()

        # choose class 3 which is CHAR
        global_indices_class_3 = global_indices[nodes[global_indices, -1] == 3]
        global_indices_class_others = global_indices[nodes[global_indices, -1] != 3]

        class_3_coords = coords[global_indices_class_3]
        other_class_coords = coords[global_indices_class_others]

        dist_matrix = torch.cdist(class_3_coords, other_class_coords, p=2)

        k = 5
        for i, distances in enumerate(dist_matrix):
            closest_indices = torch.argsort(distances)[:k]
            closest_global_indices = global_indices_class_others[closest_indices].tolist()

            for idx in closest_global_indices:
                new_edges.append((global_indices_class_3[i].item(), idx))

    new_edges = torch.tensor(new_edges, dtype=torch.long).t().contiguous()

    # remove duplicates edges.
    existing_edges = set(map(tuple, edge_index.t().tolist()))
    unique_new_edges = set(map(tuple, new_edges.t().tolist()))

    filtered_new_edges = [edge for edge in unique_new_edges if edge not in existing_edges]
    neg_edge_index = torch.tensor(filtered_new_edges, dtype=torch.long).t().contiguous()

    return neg_edge_index