import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
import os
import torch
import networkx as nx
# import matplotlib.pyplot as plt
from torch_geometric.data import Data

from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

data1 = torch.load("/blue/lic/huangzihang/repos/PretrainDrugDiscovery-main/data/toy_set/4io7/Graph_GIGN-4io7_5A.pyg")
data2 = torch.load('/blue/lic/huangzihang/repos/PretrainDrugDiscovery-main/data/toy_set/4io7/pyg/Graph_GIGN-4io7_5A.pyg')
# print(data)
# data.edge_index = torch.cat([data.edge_index_inter,data.edge_index_intra],1)
data1.edge_index = data1.edge_index_intra
data2.edge_index = data2.edge_index_intra
print('data1: %s' % data1)
print('data2: %s' % data2)

G = to_networkx(data, to_undirected=True)
x, y, z = data.pos[:, 0].numpy(), data.pos[:, 1].numpy(), data.pos[:, 2].numpy()
# node_values = [0.5046815872192383, 0.19603347778320312, 0.16805362701416016, 0.09709787368774414, 0.23491954803466797, 0.19700384140014648, 0.8571090698242188, -0.09715080261230469, 0.3766045570373535, 0.18902587890625, 1.1495060920715332, 0.2655191421508789, 0.16867637634277344, 0.818871021270752, 0.8456611633300781, 0.8012351989746094, 0.7876300811767578, 0.1583566665649414, 0.061453819274902344, 1.3064508438110352, 0.2883448600769043, 0.22061824798583984, 0.6620264053344727, 0.33239269256591797, -0.014513015747070312, -0.10646438598632812, 0.2072772979736328, 0.3440065383911133, 0.2939329147338867, 1.0787181854248047, 0.40623903274536133, 0.30188465118408203, 0.7321476936340332, 1.1335163116455078, 0.6040439605712891]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


sc = ax.scatter(x, y, z, cmap='viridis', s=100,vmin=-0.4,vmax = 2)

cbar = plt.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label('Affinity Value')
for i, j in data.edge_index.t().numpy():
    ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 'gray') 


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.axis('off')
# plt.title("3D Graph Visualization")
plt.tight_layout()
plt.savefig('graph.jpg',dpi=300)
plt.show()