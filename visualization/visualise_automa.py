import torch
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt

class VisualizeAutoma:

    def __init__(self, env, operation="PARTITION_UPDATE"):
        self.env = env
        self.points = []
        self.operations = []
        self.operation = operation

    def get_breadth_first_nodes(self, root_node):
        '''
        Performs a breadth first search inside the tree.

        Args:
            root_node: tree root node

        Returns:
            list of the tree nodes sorted by depths
        '''
        nodes = []
        stack = [root_node]
        while stack:
            cur_node = stack[0]
            stack = stack[1:]
            nodes.append(cur_node)
            for child in cur_node['childs']:
                stack.append(child)
        return nodes

    def add(self, encoder, node):
        nodes = self.get_breadth_first_nodes(node)

        nodes = list(filter(lambda x: x['visit_count'] > 0, nodes))

        for idx, node in enumerate(nodes):
            node['index'] = idx

        # gather nodes per depth
        max_depth = nodes[-1]['depth']
        nodes_per_depth = {}
        for d in range(0, max_depth + 1):
            nodes_per_depth[d] = list(filter(lambda x: x['depth'] == d, nodes))

        for d in range(0, max_depth + 1):
            nodes_this_depth = nodes_per_depth[d]
            for node in nodes_this_depth:
                if node["selected"]:
                    self.add_point(encoder, node)

    def add_point(self, encoder, node):
        with torch.no_grad():
            if node["program_from_parent_index"] is None:
                self.operations.append(
                    self.operation
                )
            else:
                self.operations.append(
                    self.env.get_program_from_index(node["program_from_parent_index"])
                )
            self.points.append(
                encoder(torch.FloatTensor(node["observation"])).numpy()
            )

    def compute(self):
        print("[*] Executing TSNE")
        self.reduced_points = TSNE(n_components=2,
                                   perplexity=20,
                                   learning_rate=200.0,
                                   n_jobs=-1).fit_transform(self.points)
        #self.reduced_points = PCA(n_components=2).fit_transform(self.points)
        self.reduced_points = pd.DataFrame(self.reduced_points, columns=["x", "y"])
        self.reduced_points["operations"] = self.operations

    def plot(self):
        print("[*] Plot values")
        sns.scatterplot(x="x", y="y", hue="operations", data=self.reduced_points)
        plt.show()

