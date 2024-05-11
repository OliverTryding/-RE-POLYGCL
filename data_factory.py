from argparse import Namespace

from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork, Actor, HeterophilousGraphDataset
from torch_geometric.transforms import NormalizeFeatures
from ogb.nodeproppred import NodePropPredDataset

from arguments import get_args


def get_dataset(args: Namespace):
    d = args.dataname
    if d in ['cora', 'citeseer', 'pubmed']:
        return Planetoid(root=f'data/{d}', name=d, transform=NormalizeFeatures())

    if d in ['cornell', 'texas', 'wisconsin']:
        return WebKB(root=f'data/{d}', name=d, transform=NormalizeFeatures())

    if d in ['chameleon', 'squirrel']:
        return WikipediaNetwork(root=f'data/{d}', name=d, transform=NormalizeFeatures())

    if d == "actor":
        return Actor(root=f'data/{d}', transform=NormalizeFeatures())

    if d in ['roman_empire', 'amazon_ratings', 'mineseweeper', 'tolokers', 'questions']:
        return HeterophilousGraphDataset(root=f'data/{d}', name=d, transform=NormalizeFeatures())

    if "cSBM" in d:
        from cSBM.cSBM_dataset import dataset_ContextualSBM
        return dataset_ContextualSBM(root="data/cSBM/", name=d)

    if d == "arxiv-year":
        return NodePropPredDataset(name="ogbn-arxiv", root="data/arxiv-year")


if __name__ == "__main__":
    args = get_args()
    for name in [
        "cSBM-1", "cSBM-0.75", "cSBM-0.5", "cSBM-0.25", "cSBM0", "cSBM0.25", "cSBM0.5", "cSBM0.75", "cSBM1",
        "cora", "citeseer", "pubmed", "chameleon", "squirrel", "cornell", "texas", "wisconsin",
        "roman_empire", "amazon_ratings", "minesweeper", "tolokers", "questions", "arxiv-year"
    ]:
        args.dataname = name
        dataset = get_dataset(args)
