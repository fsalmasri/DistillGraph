import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import numpy as np

def visualize_graphs(edge_index, neg_edges_index, emb, batch):
    im_w, im_h = 3509, 2408

    for graph_id in torch.unique(batch):
        mask = (batch == graph_id)
        nodes = emb[mask]
        global_indices = torch.nonzero(mask).squeeze()

        global_indices_class_3 = global_indices[emb[global_indices, -1] == 3]
        global_indices_class_3_set = set(global_indices_class_3.tolist())

        img = np.zeros((im_h, im_w))
        fig, ax = plt.subplots(figsize=(20, 15))
        plt.imshow(img)

        # nodes = emb[global_indices_class_3]
        for n in nodes:
            n = n.numpy()
            xc = (n[0] / 100) * im_w
            yc = (n[1] / 100) * im_h
            width = (n[2] / 100) * im_w
            height = (n[3] / 100) * im_h

            x = xc-width/2
            y = yc-height/2

            color = 'r' if n[-1] != 3 else 'g'
            rect = patches.Rectangle((x, y), width, height,
                                     linewidth=2, edgecolor=color, facecolor='none')

            ax.add_patch(rect)


        filtered_edge_index = edge_index[:, [i for i in range(edge_index.shape[1]) if
                                             edge_index[0, i].item() in global_indices_class_3_set]]

        for e in filtered_edge_index.T:
            from_node = emb[e[0]].numpy()
            to_node = emb[e[1]].numpy()

            fxc = (from_node[0] / 100) * im_w
            fyc = (from_node[1] / 100) * im_h

            exc = (to_node[0] / 100) * im_w
            eyc = (to_node[1] / 100) * im_h


            ax.plot([fxc, exc], [fyc, eyc], 'w-')

        if neg_edges_index is not None:
            filtered_edge_index = neg_edges_index[:, [i for i in range(neg_edges_index.shape[1]) if
                                                 neg_edges_index[0, i].item() in global_indices_class_3_set]]
            for e in filtered_edge_index.T:
                from_node = emb[e[0]].numpy()
                to_node = emb[e[1]].numpy()

                fxc = (from_node[0] / 100) * im_w
                fyc = (from_node[1] / 100) * im_h

                exc = (to_node[0] / 100) * im_w
                eyc = (to_node[1] / 100) * im_h

                ax.plot([fxc, exc], [fyc, eyc], 'y-')


        plt.show()

