# Reproducibility Challenge for PolyGCL
This repository contains a reimplementation of the PolyGCL model and the results of the reproducing their experiments with their and our implementation.

To run our code please install the authors code as a git submodule and install our requirements:
```bash
git clone https://github.com/ChenJY-Count/PolyGCL.git
pip install -r requirements.txt
```
The log files of the experiments run with the authors code can be found in `reproducibility_results/`.

If you want to run the synthetic dataset experiments, you need create the dataset by running the following command:
```bash
cd PolyGCL/cSBM/
sh create_cSBM.sh
```
and subsequently move each generated dataset into `data/cSBM/`.

In the following table you will find the run configurations to reproduce our results and in `sweep_configs/` the config files for the sweeps we ran to find the below hyperparameters.

| Dataset   | Run config                                                                                                                                                                                    | 
|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Cora      | `python training.py --act_fn=relu --dataname=cora --dprate=0 --dropout=0 --epochs=1000 --hid_dim=1024 --is_bns=False --lr=0.007938083166326508 --lr1=0.00010989151799299788`                  |
| Citeseer  | `python training.py --act_fn=relu --dataname=citeseer --dprate=0.7 --dropout=0 --epochs=1000 --hid_dim=256 --is_bns=False --lr=0.0007851244864509102 --lr1=0.00010538887755758494`            |
| Pubmed    | `python training.py --dataname pubmed --lr 0.0001 --lr1 0.001 --epochs 1000 --wd1 1e-3 --wd 1e-5 --is_bns True --act_fn prelu --dprate 0.6 --dropout 0.0`                                     |
| Cornell   | `python training.py --act_fn=relu --dataname=cornell --dprate=0.7 --dropout=0 --epochs=1000 --hid_dim=1024 --is_bns=False --lr=0.00013469104865846562 --lr1=0.00013881428055279157`           |
| Texas     | `python training.py --act_fn=relu --dataname=texas --dprate=0.5 --dropout=0.7 --epochs=500 --hid_dim=1024 --is_bns=True --lr=0.00015188066295968123 --lr1=0.0002767787558238884`              |
| Wisconsin | `python training.py --act_fn=prelu --dataname=wisconsin --dprate=0 --dropout=0 --epochs=1000 --hid_dim=512 --is_bns=False --lr=0.0030723082480885212 --lr1=0.00011538709732067973`            |
| Actor     | `python training.py --act_fn=prelu --dataname=actor --dprate=0.3 --dropout=0.3 --epochs=1000 --hid_dim=256 --is_bns=False --lr=0.00162357160131385 --lr1=0.0001594048782514113`               |
| Chameleon | `python training.py --act_fn=prelu --dataname=chameleon --dprate=0.3 --dropout=0 --epochs=1000 --hid_dim=512 --is_bns=False --lr=0.0003210130256713253 --lr1=0.0005872189968006944`           |
| Squirrel  | `python training.py --act_fn=relu --dataname=squirrel --dprate=0 --dropout=0.5 --epochs=1000 --hid_dim=512 --is_bns=False --lr=0.00040912123325676305 --lr1=0.0005043587674192926`            |
| Roman_empire | `python training.py --act_fn=prelu --dataname=roman_empire --dprate=0 --dropout=0.5 --epochs=1000 --hid_dim=256 --is_bns=False --lr=0.0002538948746376092 --lr1=0.000242552324155434`      |
| Amazon_ratings | `python training.py --act_fn=prelu --dataname=amazon_ratings --dprate=0 --dropout=0 --epochs=1000 --hid_dim=128 --is_bns=False --lr=0.00037783922239056905 --lr1=0.0002136924185268221`  |
| Minesweeper | `python training.py --act_fn=prelu --dataname=minesweeper --dprate=0.3 --dropout=0.7 --epochs=500 --hid_dim=256 --is_bns=True --lr=0.0009209078560181892 --lr1=0.0002410414215184864`       |
| Tolokers  | `python training.py --act_fn=prelu --dataname=tolokers --dprate=0.5 --dropout=0 --epochs=1000 --hid_dim=256 --is_bns=True --lr=0.0002534550224757627 --lr1=0.0007907434682414036`              |
| Questions | `python training.py --act_fn=prelu --dataname=questions --dprate=0.7 --dropout=0 --epochs=500 --hid_dim=128 --is_bns=True --lr=0.00017711289141459768 --lr1=0.00011690348121058706`           |
| cSBM ϕ = -1 | `python training.py --act_fn=relu --dataname=cSBM-1 --dprate=0.5 --dropout=0.7 --epochs=1000 --hid_dim=512 --is_bns=False --lr=0.001585759327981123 --lr1=0.0005203772328977067`       |
| cSBM ϕ = -0.75 | `python training.py --act_fn=relu --dataname=cSBM-0.75 --dprate=0.3 --dropout=0.3 --epochs=1000 --hid_dim=512 --is_bns=False --lr=0.008092724187933366 --lr1=0.001034359315617839`  |
| cSBM ϕ = -0.5 | `python training.py --act_fn=relu --dataname=cSBM-0.5 --dprate=0.7 --dropout=0 --epochs=500 --hid_dim=512 --is_bns=True --lr=0.0030208516492516674 --lr1=0.0003758238099745588`      |
| cSBM ϕ = -0.25 | `python training.py --act_fn=prelu --dataname=cSBM-0.25 --dprate=0 --dropout=0.3 --epochs=1000 --hid_dim=512 --is_bns=True --lr=0.00012705959052759073 --lr1=0.0002862955992509802` |
| cSBM ϕ = 0 | `python training.py --act_fn=prelu --dataname=cSBM0 --dprate=0.3 --dropout=0.7 --epochs=500 --hid_dim=512 --is_bns=True --lr=0.001175181591599039 --lr1=0.0002707285380266644`          |
| cSBM ϕ = 0.25 | `python training.py --act_fn=relu --dataname=cSBM0.25 --dprate=0.3 --dropout=0.7 --epochs=1000 --hid_dim=512 --is_bns=False --lr=0.0002506346122939143 --lr1=0.00042288887382397`    |
| cSBM ϕ = 0.5 | `python training.py --act_fn=relu --dataname=cSBM0.5 --dprate=0.3 --dropout=0.7 --epochs=500 --hid_dim=512 --is_bns=True --lr=0.002205875161130184 --lr1=0.00010091571075296968`      |
| cSBM ϕ = 0.75 | `python training.py --act_fn=prelu --dataname=cSBM0.75 --dprate=0.7 --dropout=0.5 --epochs=1000 --hid_dim=512 --is_bns=True --lr=0.0011412239349228128 --lr1=0.00016490032359597355` |
| cSBM ϕ = 1 | `python training.py --act_fn=prelu --dataname=cSBM1 --dprate=0.5 --dropout=0.3 --epochs=1000 --hid_dim=128 --is_bns=False --lr=0.006236702905615547 --lr1=0.004193609285600556`         |

