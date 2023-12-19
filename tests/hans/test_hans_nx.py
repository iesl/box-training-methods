import unittest
import torch
from torch import tensor
import networkx as nx

from box_training_methods.graph_modeling.dataset import HierarchyAwareNegativeEdges


class TestHANS(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):

        edges = [(0,2), (0,3), (1,5), (2,4), (4,5), (5,6)]
        cls.G = nx.DiGraph()
        cls.G.add_edges_from(edges)

        cls.HANS = HierarchyAwareNegativeEdges(edges=torch.tensor(list(cls.G.edges)))

    def test_hans_tail2heads(self):
        assert torch.equal(
            self.HANS.hans_tail2heads_matrix,
            tensor([[1, 2, 3],
                    [0, 7, 7],
                    [1, 3, 4],
                    [1, 2, 7],
                    [1, 3, 7],
                    [3, 6, 7],
                    [3, 7, 7]]))

    def test_hans_head2tails(self):
        assert torch.equal(
            self.HANS.hans_head2tails_matrix,
            tensor([[1, 7, 7, 7, 7],
                    [0, 2, 3, 4, 7],
                    [0, 3, 7, 7, 7],
                    [0, 2, 4, 5, 6],
                    [2, 7, 7, 7, 7],
                    [7, 7, 7, 7, 7],
                    [5, 7, 7, 7, 7]]))

    def test_aggressive_tail2heads(self):
        assert torch.equal(
            self.HANS.aggr_tail2heads_matrix,
            tensor([[7, 7],
                    [0, 7],
                    [4, 7],
                    [1, 2],
                    [1, 7],
                    [6, 7],
                    [3, 7]]))

    def test_aggressive_head2tails(self):
        assert torch.equal(
        self.HANS.aggr_head2tails_matrix,
        tensor([[1, 7],
                [3, 4],
                [3, 7],
                [6, 7],
                [2, 7],
                [7, 7],
                [5, 7]]))


if __name__ == '__main__':
    unittest.main()
