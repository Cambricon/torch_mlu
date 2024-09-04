import torch.fx

from .make_contiguous_clone import make_continuous_clone


def mlu_post_pass(graph: torch.fx.graph.Graph):
    make_continuous_clone(graph)
