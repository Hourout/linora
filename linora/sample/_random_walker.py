import random
import itertools

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

    def deepwalk_walks(self, walk_num, walk_length, walk_prob=False, filter_lenth=0, workers=1, verbose=0):
        return self._parallel_walks(self._deepwalk_walks, walk_num, walk_length, walk_prob, filter_lenth, workers, verbose)
        
    def deepwalk_walk(self, walk_length, start_node, walk_prob=False, filter_lenth=0):
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nodes = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if walk_prob:
                    cur_weights = [G[cur][i]['weight'] for i in cur_nodes]
                    walk.append(random.choices(cur_nodes, cur_weights)[0])
                else:
                    walk.append(random.choice(cur_nbrs))
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
