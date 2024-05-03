from argparse import ArgumentParser, Namespace
import torch

def get_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument("--impl", type=str, default="ours", choices=["ours", "authors"])

    p.add_argument("--dataname", type=str, default="cora", choices=[
        "cora", "citeseer", "pubmed", "chameleon", "squirrel", "cornell", "texas", "wisconsin",
        "cSBM-1", "cSBM-0.75", "cSBM-0.5", "cSBM-0.25", "cSBM0", "cSBM0.25", "cSBM0.5", "cSBM0.75", "cSBM1",
        "roman_empire", "amazon_ratings", "minesweeper", "tolokers", "questions", "arxiv-year"
    ])

    # From PolyGCL
    p.add_argument('--seed', type=int, default=42, help='Random seed.')  # Default seed same as GCNII
    p.add_argument('--dev', type=int, default=0, help='device id')
    p.add_argument("--gpu", type=int, default=0, help="GPU index. Default: -1, using cpu.")

    p.add_argument("--epochs", type=int, default=500, help="Training epochs.")
    p.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Patient epochs to wait before early stopping.",
    )
    p.add_argument(
        "--lr", type=float, default=0.010, help="Learning rate of prop."
    )
    p.add_argument(
        "--lr1", type=float, default=0.001, help="Learning rate of PolyGCL."
    )
    p.add_argument(
        "--lr2", type=float, default=0.01, help="Learning rate of linear evaluator."
    )
    p.add_argument(
        "--wd", type=float, default=0.0, help="Weight decay of PolyGCL prop."
    )
    p.add_argument(
        "--wd1", type=float, default=0.0, help="Weight decay of PolyGCL."
    )
    p.add_argument(
        "--wd2", type=float, default=0.0, help="Weight decay of linear evaluator."
    )

    p.add_argument(
        "--hid_dim", type=int, default=512, help="Hidden layer dim."
    )

    p.add_argument(
        "--K", type=int, default=10, help="Layer of encoder."
    )

    # From PolyGCL/cSBM
    p.add_argument(
        "--init_low", type=float, default=2.0, help="Initial value for low-pass filter."
    )
    p.add_argument(
        "--init_high", type=float, default=0.0, help="Initial value for high-pass filter."
    )
    p.add_argument(
        "--val_range", type=float, default=2.0, help="The range of filter value."
    )

    p.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')
    p.add_argument('--dprate', type=float, default=0.5, help='dropout for propagation layer.')
    p.add_argument('--is_bns', type=bool, default=False)
    p.add_argument('--act_fn', default='relu',
                        help='activation function')

    return p.parse_args()
