## dpsgd-auditbench

This repository supports the rebuttal auditing evaluation for NDIS-based privacy auditing, while also keeping the DP-SGD training and one-run auditing workflows used to generate the experiments.

It includes PyTorch implementations of DP-SGD training pipelines for image classification, including (i) the DP-SGD training procedure from the paper "Unlocking High-Accuracy Differentially Private Image Classification through Scale", and (ii) the auditable DP-SGD training procedure from the paper "Privacy Auditing with One Training Run".

The code also includes privacy auditing workflows, including sequential auditing as described in the paper "Sequential Auditing for f-Differential Privacy".

### Auditing evaluation

The main auditing results for the rebuttal are in the NDIS and one-run auditing notebooks:

- `Notebooks/NDIS/gaussian_whitebox_audit.ipynb`: white-box audit of the Gaussian mechanism with `eps=1` and `delta=1e-5`. This notebook uses the known Gaussian parameters to evaluate the exact pairwise privacy curve, giving a deterministic baseline.
- `Notebooks/NDIS/gaussian_parametric_blackbox_audit.ipynb`: Gaussian parametric black-box audit for the same target privacy point. This notebook estimates the Gaussian parameters from output samples, plugs those estimates into the exact Gaussian-pair formula, and uses the bootstrap upper confidence bound to make the audit decision. It contains the convergence table for sample sizes `n=10^3, 10^4, 10^5`; the conservative upper estimate of `delta` at `eps=1` tightens from `0.000197343` at `n=10^3` to `1.37279e-05` at `n=10^5`, approaching the true value `1e-5`. The pointwise 95% bootstrap interval contains the true value for all three reported sample sizes.
- `Notebooks/white_box/simulation_one_run_auditing.ipynb`: one-run DP-SGD auditing simulation for approximately Gaussian outputs. This is the notebook to inspect for the rebuttal comparison against the "Privacy Auditing with One (1) Training Run" baseline. In the saved 200-epoch output, the theoretical upper bound is `6.91`, the baseline lower bound is `1.86`, the NDIS lower bound in the white-box setting is `6.48`, and the NDIS lower bound in the parametric black-box setting is `5.79`.
- `src/whitebox_auditing/ndis_1d.py`: helper implementation for the 1D Gaussian NDIS calculations, including the closed-form `delta(eps)` routine and the `eps`-from-`delta` root finder used by the one-run auditing workflow.

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

For the rebuttal auditing figures and numbers, use the notebooks listed in the **Auditing evaluation** section above rather than this observation-generation workflow.
