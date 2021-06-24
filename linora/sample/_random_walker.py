import random
import itertools
from collections import defaultdict

from joblib import Parallel, delayed

__all__ = ['RandomWalker']

class RandomWalker:
    def __init__(self, G):
        """
        :param G: networkx graph instance.
        """
        self.G = G
        
        self._nodes_weight_dict = defaultdict(lambda :defaultdict(list))
        for i in self.G.nodes():
            self._nodes_weight_dict[i]['nodes'] = list(self.G.neighbors(i))
            probs = [self.G[i][j].get('weight', 1.0) for j in self._nodes_weight_dict[i]['nodes']]
            probs_sum = sum(probs)
            self._nodes_weight_dict[i]['weights'] = [i/probs_sum for i in probs]

    def deepwalk_walks(self, walk_num, walk_length, walk_prob=False, filter_lenth=0, workers=1, verbose=0, node_sample=1):
        """Random walk on all nodes
        
        Args:
            walk_num: random walk times, sampling times.
            walk_length: generation sequence length.
            walk_prob: whether each wandering node selection is based on probability.
            filter_lenth: generation sequence length minimum values.
            workers: number of parallel computing processes.
            verbose: whether to enable log output.
            node_sample: (0, 1] is the node random sampling ratio, (1, infinity) is the node random sampling number.
        Return:
            generation sequence list.
        """
        return self._parallel_walks(self._deepwalk_walks, walk_num, walk_length, walk_prob, 
                                    filter_lenth, workers, verbose, node_sample)
        
    def deepwalk_walk(self, walk_length, start_node, walk_prob=False, filter_lenth=0):
        """Random walk on a certain node
        
        Args:
            walk_length: generation sequence length.
            start_node: start wandering node.
            walk_prob: whether each wandering node selection is based on probability.
            filter_lenth: generation sequence length minimum values.
        Return:
            generation sequence list.
        """
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
    
    def _parallel_walks(self, method, walk_num, walk_length, walk_prob, filter_lenth, workers, verbose, node_sample):
        nodes = list(self.G.nodes())
        nodes = random.sample(nodes, k=int(len(nodes)*node_sample) if node_sample<=1 else int(node_sample))
        workers_list = [walk_num//workers]*workers + ([walk_num % workers] if walk_num % workers != 0 else [])
        results = Parallel(n_jobs=workers, verbose=verbose)(
            delayed(method)(nodes, num, walk_length, walk_prob, filter_lenth) for num in workers_list)
        walks = itertools.chain(filter(lambda x: len(x) > 0, itertools.chain(*results)))
        return walks
