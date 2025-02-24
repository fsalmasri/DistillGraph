import json
import os

import pandas as pd



def convert_lsCoords_to_center(coords, im_w, im_h):
    x = (coords[0] / 100) * im_w
    y = (coords[1] / 100) * im_h
    width = (coords[2] / 100) * im_w
    height = (coords[3] / 100) * im_h
    xc, yc = x + width / 2, y + height / 2

    return x, y, width, height, xc, yc

im_w, im_h = 3509, 2408

flst = os.listdir('DISTILL')
# TODO We found issues of compatibility in these file. better to remove. no time to go back and work on them.
# flst.remove('4386')
# flst.remove('4387')
# flst.remove('4459')
# flst.remove('4548')
# flst.remove('4559')
# flst.remove('4568')
# flst.remove('4578')
# flst.remove('4626')

for ex in flst:
    print(ex)

    with open(f"DISTILL/{ex}/test_LC_blocks.json", "r") as json_file:
        blocks = json.load(json_file)

    with open(f"DISTILL/{ex}/nodes.json", "r") as json_file:
        nodes = json.load(json_file)

    csv_nodes = []
    for n in nodes:
        _, _, width, height, xc, yc = convert_lsCoords_to_center(blocks[n][:4],im_w, im_h)

        xc = xc / im_w * 100
        yc = yc / im_h * 100
        width = width / im_w * 100
        height = height / im_h * 100

        csv_nodes.append([n, xc, yc, width, height, blocks[n][4]])

    df = pd.DataFrame(csv_nodes, columns = ['node_id', 'NXC', 'NYC', 'NW', 'NH', 'cls'])
    df.index.name = 'Index'
    df.to_csv(f"DISTILL/{ex}/nodes.csv", index=True)


    with open(f"DISTILL/{ex}/edges.json", "r") as json_file:
        edges = json.load(json_file)

    csv_links = []
    for e in edges:
        source_node_id = df.index[df['node_id'] == e[0]].tolist()
        target_node_id = df.index[df['node_id'] == e[1]].tolist()
        csv_links.append([source_node_id[0], target_node_id[0],
                         df.loc[source_node_id[0], 'cls'], df.loc[target_node_id[0], 'cls']])


    df = pd.DataFrame(csv_links, columns = ['src_id', 'trgt_id', 'src_cls', 'trgt_cls'])
    df.to_csv(f"DISTILL/{ex}/links.csv", index=False)
