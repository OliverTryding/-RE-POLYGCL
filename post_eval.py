# most code is taken from the authors code and slightly modified to fit the current codebase
# we chose to evaluate our models with the same setup as the authors to ensure a fair comparison
# Original code: https://github.com/ChenJY-Count/PolyGCL/blob/ec246bc176d0a0a8978461be13d70a32a8aedadd/training.py#L157

import numpy as np
import torch as th
import seaborn as sns
from sklearn.metrics import roc_auc_score

from utils2 import random_splits


class LogReg(th.nn.Module):
    def __init__(self, hid_dim, n_classes):
        super(LogReg, self).__init__()

        self.fc = th.nn.Linear(hid_dim, n_classes)

    def forward(self, x):
        ret = self.fc(x)
        return ret


def post_eval(model, dataset, args):
    data = dataset[0]
    label = data.y
    feat = data.x
    n_classes = np.unique(label.cpu().numpy()).shape[0]
    edge_index = data.edge_index

    model.eval()
    with th.no_grad():
        Z_L, Z_H = model(feat, edge_index)
        embeds = model.get_embedding(Z_L, Z_H)

    # Step 5:  Linear evaluation ========================================================== #
    print("=== Evaluation ===")
    results = []
    # 10 fixed seeds for random splits from BernNet
    SEEDS = [1941488137, 4198936517, 983997847, 4023022221, 4019585660, 2108550661, 1648766618, 629014539, 3212139042,
             2424918363]
    train_rate = 0.6
    val_rate = 0.2
    percls_trn = int(round(train_rate * len(label) / n_classes))
    val_lb = int(round(val_rate * len(label)))
    for i in range(10):
        seed = SEEDS[i]
        assert label.shape[0] == dataset[0].x.shape[0]

        if args.dataname in ['roman_empire', 'amazon_ratings', "minesweeper", "tolokers", "questions"]:
            train_mask, val_mask, test_mask = data.train_mask[:, i].to(args.device), data.val_mask[:, i].to(
                args.device), data.test_mask[:, i].to(args.device)
            n_classes = n_classes if args.dataname in ['roman_empire', 'amazon_ratings'] else 1
        else:
            train_mask, val_mask, test_mask = random_splits(label, n_classes, percls_trn, val_lb, seed=seed)

            train_mask = th.BoolTensor(train_mask).to(args.device)
            val_mask = th.BoolTensor(val_mask).to(args.device)
            test_mask = th.BoolTensor(test_mask).to(args.device)

        train_embs = embeds[train_mask]
        val_embs = embeds[val_mask]
        test_embs = embeds[test_mask]

        label = label.to(args.device)
        if args.dataname in ["minesweeper", "tolokers", "questions"]:
            label = label.to(th.float)

        train_labels = label[train_mask]
        val_labels = label[val_mask]
        test_labels = label[test_mask]

        best_val_acc = 0
        eval_acc = 0
        bad_counter = 0

        logreg = LogReg(hid_dim=args.hid_dim, n_classes=n_classes)
        opt = th.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)
        logreg = logreg.to(args.device)

        loss_fn = th.nn.BCEWithLogitsLoss() if args.dataname in ["minesweeper", "tolokers",
                                                                 "questions"] else th.nn.CrossEntropyLoss()
        for epoch in range(2000):
            logreg.train()
            opt.zero_grad()
            logits = logreg(train_embs)
            if args.dataname in ["minesweeper", "tolokers", "questions"]:
                logits = logits.squeeze(-1)
            else:
                preds = th.argmax(logits, dim=1)
                train_acc = th.sum(preds == train_labels).float() / train_labels.shape[0]
            loss = loss_fn(logits, train_labels)
            loss.backward()
            opt.step()

            logreg.eval()
            with th.no_grad():
                val_logits = logreg(val_embs)
                test_logits = logreg(test_embs)

                if args.dataname in ["minesweeper", "tolokers", "questions"]:
                    val_acc = roc_auc_score(y_true=val_labels.cpu().numpy(),
                                            y_score=val_logits.squeeze(-1).cpu().numpy())
                    test_acc = th.tensor(
                        roc_auc_score(y_true=test_labels.cpu().numpy(), y_score=test_logits.squeeze(-1).cpu().numpy()))
                else:
                    val_preds = th.argmax(val_logits, dim=1)
                    test_preds = th.argmax(test_logits, dim=1)

                    val_acc = th.sum(val_preds == val_labels).float() / val_labels.shape[0]
                    test_acc = th.sum(test_preds == test_labels).float() / test_labels.shape[0]

                if val_acc >= best_val_acc:
                    bad_counter = 0
                    best_val_acc = val_acc
                    if test_acc > eval_acc:
                        eval_acc = test_acc
                else:
                    bad_counter += 1

        metric = 'AUC' if args.dataname in ["minesweeper", "tolokers", "questions"] else 'accuracy'

        print(i, f"Linear evaluation {metric}:{eval_acc:.4f}")
        results.append(eval_acc.cpu().data)

    results = [v.item() for v in results]
    test_acc_mean = np.mean(results, axis=0) * 100
    values = np.asarray(results, dtype=object)
    uncertainty = np.max(
        np.abs(sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000), 95) - values.mean()))
    print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty * 100:.4f}')

    return test_acc_mean
