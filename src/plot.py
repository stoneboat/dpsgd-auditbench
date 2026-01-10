import matplotlib.pyplot as plt
import numpy as np
def ecdf(x):
    """Compute empirical cumulative distribution function."""
    x = np.sort(x)
    y = np.arange(1, len(x)+1) / len(x)
    return x, y

def plot_score_distribution(s_in, s_out, exp_dir, epoch):
    """
    Plot score distributions for a single shadow model.
    
    Args:
        s_in: Scores for samples that were IN (trained on)
        s_out: Scores for samples that were OUT (not trained on)
        exp_dir: Experiment directory name
        epoch: Epoch identifier
    """
    # 1) Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(s_out, bins=200, alpha=0.5, density=True, label="OUT (not trained)", color='red')
    plt.hist(s_in,  bins=200, alpha=0.5, density=True, label="IN (trained)", color='blue')
    plt.legend()
    plt.title(f"Score Distribution: IN vs OUT\n{exp_dir} (epoch: {epoch})")
    plt.xlabel("score")
    plt.ylabel("density")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 2) Empirical CDF
    x_out, y_out = ecdf(s_out)
    x_in,  y_in  = ecdf(s_in)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_out, y_out, label="OUT (not trained)", color='red', linewidth=2)
    plt.plot(x_in,  y_in,  label="IN (trained)", color='blue', linewidth=2)
    plt.legend()
    plt.title(f"Empirical CDF: IN vs OUT\n{exp_dir} (epoch: {epoch})")
    plt.xlabel("score")
    plt.ylabel("CDF")
    plt.grid(True, alpha=0.3)
    plt.show()