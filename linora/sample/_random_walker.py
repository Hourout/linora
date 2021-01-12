import random
import itertools
from collections import defaultdict

from joblib import Parallel, delayed

__all__ = ['RandomWalker']

class RandomWalker:
    def __init__(self, G, p=1, q=1):
        """
        :param G: networkx graph instance.
        :param p: Return parameter,controls the likelihood of immediately revisiting a node in the walk.
        :param q: In-out parameter,allows the search to differentiate between “inward” and “outward” nodes
        """
        self.G = G
        self.p = p
        self.q = q
        
        self._nodes_weight_dict = defaultdict(lambda :defaultdict(list))
        for i in self.G.nodes():
            self._nodes_weight_dict[i]['nodes'] = list(self.G.neighbors(i))
            probs = [self.G[i][j].get('weight', 1.0) for j in self._nodes_weight_dict[i]['nodes']]
            probs_sum = sum(probs)
            self._nodes_weight_dict[i]['weights'] = [i/probs_sum for i in probs]
        
#         self._edges_weight_dict = defaultdict(lambda :defaultdict(list))
#         for edge in G.edges():
#             probs = []
#             self._edges_weight_dict[edge]['nodes'] = list(self.G.neighbors(edge[1]))
#             for x in self._edges_weight_dict[edge]['nodes']:
#                 weight = self.G[edge[1]][x].get('weight', 1.0)
#                 if x == edge[0]:
#                     probs.append(weight/self.p)
#                 elif G.has_edge(x, edge[0]):
#                     probs.append(weight)
#                 else:
#                     probs.append(weight/self.q)
#             probs_sum = sum(probs)
#             self._edges_weight_dict[edge]['weights'] = [i/probs_sum for i in probs]

    def deepwalk_walks(self, walk_num, walk_length, walk_prob=False, filter_lenth=0, workers=1, verbose=0):
        return self._parallel_walks(self._deepwalk_walks, walk_num, walk_length, walk_prob, filter_lenth, workers, verbose)
        
    def deepwalk_walk(self, walk_length, start_node, walk_prob=False, filter_lenth=0):
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nodes = self._nodes_weight_dict[cur]['nodes']
            if len(cur_nodes) > 0:
                if walk_prob:
                    walk.append(random.choices(cur_nodes, self._nodes_weight_dict[cur]['weights'])[0])
                else:
                    walk.append(random.choice(cur_nodes))
            else:
                break
        if len(walk)<=filter_lenth:
            walk = []
        return walk
    
    def _deepwalk_walks(self, nodes, walk_num, walk_length, walk_prob, filter_lenth):
        walks = []
        for _ in range(walk_num):
            random.shuffle(nodes)
            walks += [self.deepwalk_walk(walk_length, v, walk_prob, filter_lenth) for v in nodes]
        return walks
    
    def _parallel_walks(self, method, walk_num, walk_length, walk_prob, filter_lenth, workers, verbose):
        nodes = list(self.G.nodes())
        workers_list = [walk_num//workers]*workers + ([walk_num % workers] if walk_num % workers != 0 else [])
        results = Parallel(n_jobs=workers, verbose=verbose)(
            delayed(method)(nodes, num, walk_length, walk_prob, filter_lenth) for num in workers_list)
        walks = itertools.chain(filter(lambda x: len(x) > 0, itertools.chain(*results)))
        return walks
