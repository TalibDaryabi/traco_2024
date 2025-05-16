# utils/metrics.py
# Utility functions for handling and printing metrics.

def print_metrics(metrics, epoch_samples, phase):
    """
    Print metrics for a given phase (e.g., training, validation).
    
    Args:
        metrics (dict): Dictionary containing metrics (e.g., bce, dice, loss).
        epoch_samples (int): Number of samples in the epoch.
        phase (str): Phase name (e.g., 'train', 'val').
    """
    if epoch_samples == 0:
        print(f"{phase}: No samples processed.")
        return

    outputs = []
    for k in metrics.keys():
        value = metrics[k] / epoch_samples
        outputs.append(f"{k}: {value:.4f}")
    print(f"{phase}: {', '.join(outputs)}")