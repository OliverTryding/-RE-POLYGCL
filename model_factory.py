import sys
from argparse import Namespace
from torch import nn

from models.PolyGCL_model import PolyGCL


def get_model(args: Namespace) -> nn.Module:
    args.in_dim = args.hid_dim
    args.out_dim = args.hid_dim
    if args.impl == "ours":
        return PolyGCL(in_size=args.hid_dim, hidden_size=args.hid_dim, out_size=args.hid_dim, K=args.K, dropout_p=0.3)
    elif args.impl == "authors":
        if "cSBM" in args.dataname:
            sys.path.append("PolyGCL/cSBM")
            from PolyGCL.cSBM.model import Model
            return Model(**get_kwargs(Model, args))
        if args.dataname in ["roman_empire", "amazon_ratings", "mineseweeper", "tolokers", "questions"]:
            sys.path.append("PolyGCL/HeterophilousGraph")
            from PolyGCL.HeterophilousGraph.model import Model
            return Model(**get_kwargs(Model, args))
        if args.dataname == "arxiv-year":
            import importlib
            sys.path.append("PolyGCL/non-homophilous")
            Model = importlib.import_module("PolyGCL.non-homophilous.model", "Model")
            return Model(**get_kwargs(Model, args))
        if args.dataname in ["cora", "citeseer", "pubmed", "cornell", "texas", "wisconsin", "actor", "chameleon", "squirrel"]:
            sys.path.append("PolyGCL")
            from PolyGCL.model import Model
            return Model(**get_kwargs(Model, args))
        else:
            raise NotImplementedError
    else:
        raise ValueError(f"Unknown implementation: {args.impl}")


def get_kwargs(cls, args: Namespace) -> dict:
    return {k: v for k, v in vars(args).items() if k in cls.__init__.__code__.co_varnames}

('self', 'in_dim', 'out_dim', 'K', 'dprate', 'dropout', 'is_bns', 'act_fn')
if __name__ == '__main__':
    from arguments import get_args
    args = get_args()
    args.dataname = "cSBM"
    args.impl = "authors"
    model = get_model(args)
    # print(model)