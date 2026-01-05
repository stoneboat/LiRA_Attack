## Membership Inference Attacks From First Principles

This directory contains code to reproduce our paper:

**"Membership Inference Attacks From First Principles"** <br>
https://arxiv.org/abs/2112.03570 <br>
by Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, and Florian Tramèr.

### INSTALLING

You can create a Python virtual environment and install all dependencies from `requirements.txt`:

```bash
# Create virtual environment (using conda or venv)
conda create --prefix /path/to/venv python=3.9 -y
conda activate /path/to/venv

# Install dependencies
pip install -r requirements.txt
```

**For PACE cluster users (Georgia Tech):**
We provide a tested installation script that sets up the environment with GPU support for both TensorFlow and JAX:

```bash
bash scripts/local_scripts/cluster_install_gpu.sh
```

### RUNNING THE CODE

#### 1. Train the models

The first step in our attack is to train shadow models. As a baseline that should give most of the gains in our attack, you should start by training 16 shadow models using the Jupyter notebook:

> Notebooks/lira_attack/train_model.ipynb

**Understanding the parameters:**
- `expid` chooses which subset of the 50,000 training points to train on. Each `expid` value (0-15) corresponds to a different subset selection.
- `num_experiments=16` means you intend to train a family of 16 shadow models (expid 0–15) with a balanced inclusion pattern so every example is "in" about half the time (if you set `pkeep=0.5`).
- `pkeep` is the probability that each sample will be included in the training set.

**To train 16 shadow models:**
Run the notebook 16 times, each time setting `expid` to a different value from 0 to 15. For example:
- First run: set `expid = 0` and `num_experiments = 16`
- Second run: set `expid = 1` and `num_experiments = 16`
- Continue until `expid = 15`

This will train several CIFAR-10 wide ResNet models to ~91% accuracy each, and
will output a bunch of files under the directory `logs/exp/cifar10/` with structure:

```
logs/exp/cifar10/
- experiment_N_of_16
-- hparams.json
-- keep.npy
-- ckpt/
--- 0000000100.npz
-- tb/
```

#### 2. Perform inference

Once the models are trained, now it's necessary to perform inference and save
the output features for each training example for each model in the dataset.

> python3 inference.py --logdir=exp/cifar10/

This will add to the experiment directory a new set of files

```
exp/cifar10/
- experiment_N_of_16
-- logits/
--- 0000000100.npy
```

where this new file has shape (50000, 10) and stores the model's output features
for each example.

#### 3. Compute membership inference scores

Finally we take the output features and generate our logit-scaled membership
inference scores for each example for each model.

> python3 score.py exp/cifar10/

And this in turn generates a new directory

```
exp/cifar10/
- experiment_N_of_16
-- scores/
--- 0000000100.npy
```

with shape (50000,) storing just our scores.

### PLOTTING THE RESULTS

Finally we can generate pretty pictures, and run the plotting code

> python3 plot.py

which should give (something like) the following output

![Log-log ROC Curve for all attacks](fprtpr.png "Log-log ROC Curve")

```
Attack Ours (online)
   AUC 0.6676, Accuracy 0.6077, TPR@0.1%FPR of 0.0169
Attack Ours (online, fixed variance)
   AUC 0.6856, Accuracy 0.6137, TPR@0.1%FPR of 0.0593
Attack Ours (offline)
   AUC 0.5488, Accuracy 0.5500, TPR@0.1%FPR of 0.0130
Attack Ours (offline, fixed variance)
   AUC 0.5549, Accuracy 0.5537, TPR@0.1%FPR of 0.0299
Attack Global threshold
   AUC 0.5921, Accuracy 0.6044, TPR@0.1%FPR of 0.0009
```

where the global threshold attack is the baseline, and our online,
online-with-fixed-variance, offline, and offline-with-fixed-variance attack
variants are the four other curves. Note that because we only train a few
models, the fixed variance variants perform best.

### Citation

You can cite this paper with

```
@article{carlini2021membership,
  title={Membership Inference Attacks From First Principles},
  author={Carlini, Nicholas and Chien, Steve and Nasr, Milad and Song, Shuang and Terzis, Andreas and Tramer, Florian},
  journal={arXiv preprint arXiv:2112.03570},
  year={2021}
}
```
