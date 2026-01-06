## Membership Inference Attacks From First Principles

This repository is based on the original codebase from Google Research for the paper:

**"Membership Inference Attacks From First Principles"** <br>
https://arxiv.org/abs/2112.03570 <br>
by Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, and Florian Tramèr.

The code has been modified and extended to study the Likelihood Ratio Attack (LiRA) methodology described in the paper.

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

### Repduce the LiRA result

#### 1. Train the models

The first step in our attack is to train shadow models. As a baseline that should give most of the gains in our attack, you should start by training 16 shadow models using the Jupyter notebook:

> Notebooks/lira_attack/train_model.ipynb

**Understanding the parameters:**
- `expid` chooses which subset of the 50,000 training points to train on. Each `expid` value (0-15) corresponds to a different subset selection.
- `num_experiments=16` means you intend to train a family of 16 shadow models (expid 0–15) with a balanced inclusion pattern so every example is "in" about half the time (if you set `pkeep=0.5`).
- `pkeep` is the probability that each sample will be included in the training set.

**To train 16 shadow models:**

You can train shadow models using either method:

1. **Using the Jupyter notebook**: Run the notebook 16 times, each time setting `expid` to a different value from 0 to 15. For example:
   - First run: set `expid = 0` and `num_experiments = 16`
   - Second run: set `expid = 1` and `num_experiments = 16`
   - Continue until `expid = 15`

2. **Using the Python script** (recommended for batch training):
   ```bash
   python scripts/lira_attack/train_models.py \
       --num_experiments 16 \
       --expid_list 0,15 \
       --epochs 100 \
       --dataset_size 50000
   ```
   This will train all experiments from 0 to 15 in sequence.

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

#### 2. Inference

Once the models are trained, now it's necessary to perform inference and save the output features for each training example for each model in the dataset. You can run inference using the Jupyter notebook `Notebooks/lira_attack/inference.ipynb`. 

The notebook performs two main tasks:
1. **Generate logits**: Runs inference on each trained shadow model to generate logits for each training example
2. **Compute scores**: Computes membership inference scores for each sample with respect to each shadow model

This will add to the experiment directory two new sets of files:

```
exp/cifar10/
- experiment_N_of_16
-- logits/
--- 0000000100.npy
-- scores/
--- 0000000100.npy
```

- **logits/**: Contains model output logits with shape `(50000, 1, 2, 10)` storing the model's output features for each example with data augmentation
- **scores/**: Contains membership inference scores with shape `(50000,)` - one score per training example indicating how likely it is that the example was in the training set


#### 3. Plot and Analyze Score Distributions

After computing scores, you can visualize the score difference of record in and out under a model using the Jupyter notebook `Notebooks/lira_attack/plot.ipynb`.

1. **Loads scores and masks**: For each shadow model, it loads:
   - `scores/{epoch}.npy`: scores for each imagine, where the score function is a deterministic function parametrized by the training model
   - `keep.npy`: Boolean mask indicating which examples were included in the training set

2. **Separates scores by membership**: 
   - **IN scores**: Scores for examples that were in the training set (`keep == True`)
   - **OUT scores**: Scores for examples that were not in the training set (`keep == False`)

3. **Generates visualizations** for each shadow model: we use two statistics, the histogram and empirical cdf. 

### Citation

We thank the authors for sharing their codebase, which has been invaluable for our study and research.

If you use this codebase or the LiRA attack methodology, please cite the original paper:

```
@article{carlini2021membership,
  title={Membership Inference Attacks From First Principles},
  author={Carlini, Nicholas and Chien, Steve and Nasr, Milad and Song, Shuang and Terzis, Andreas and Tramer, Florian},
  journal={arXiv preprint arXiv:2112.03570},
  year={2021}
}
```
