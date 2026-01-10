## dpsgd-auditbench

This repository provides an end-to-end codebase for training and auditing differentially private SGD (DP-SGD) models.

It includes PyTorch implementations of DP-SGD training pipelines for image classification, including (i) the DP-SGD training procedure from the paper "Unlocking High-Accuracy Differentially Private Image Classification through Scale", and (ii) the auditable DP-SGD training procedure from the paper "Privacy Auditing with One Training Run".

The code also includes privacy auditing workflows, including sequential auditing as described in the paper "Sequential Auditing for f-Differential Privacy".

### INSTALLING

You can create a Python virtual environment and install all dependencies from `requirements.txt`:

```bash
# Create virtual environment (using conda or venv)
conda create --prefix /path/to/venv python=3.12 -y
conda activate /path/to/venv

# Install dependencies
pip install -r requirements.txt
```

**For PACE cluster users (Georgia Tech):**
We provide a tested installation script that sets up the environment with GPU support for PyTorch

```bash
bash scripts/local_scripts/cluster_install_gpu.sh
```

### DP-SGD training 

To train a DP-SGD model, see the notebook `train_original_DP_model.ipynb`. This notebook follows the neural network architectures and training hyperparameters suggested in the paper "Unlocking High-Accuracy Differentially Private Image Classification through Scale" to train a DP image classification model on CIFAR-10. 

### Auditable DP-SGD training

To train an auditable DP-SGD model, see the notebook `train_auditable_DP_model_blackbox.ipynb`. This notebook uses a dataloader that follows the canary insertion procedure suggested in the paper "Privacy Auditing with One Training Run" to train a DP-SGD model. The network architectures and training hyperparameters follow the same settings as the original DP-SGD training, with the only difference that the training dataset is altered to add canaries for auditing purposes. 

We also provide a script `train_auditable_DP_model.py` in the `scripts` directory for long training runs. The script supports checkpoint resuming and tracks privacy loss from previous training sessions.

#### Generating audit observations

To generate observations for auditing, run the notebook `infer_auditable_DP_model.ipynb`. This notebook computes losses and audit scores for all canaries across all checkpoints and saves them in the `logits` and `scores` directories. To visualize basic statistics of these observations, see the notebook `plot_auditable_DP_model.ipynb`. 