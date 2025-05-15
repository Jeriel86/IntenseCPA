import os
import time
import pandas as pd
import fcntl
import torch


def append_to_csv(data, file_path, max_retries=10, delay=2):
    for attempt in range(max_retries):
        try:
            # Open the file with locking
            with open(file_path, 'a') as f:  # 'a' for append mode
                fcntl.flock(f, fcntl.LOCK_EX)  # Exclusive lock
                # Read existing CSV or create new DataFrame
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    df = pd.read_csv(file_path)
                else:
                    df = pd.DataFrame(columns=[key for key in data])
                # Append data
                data_df = pd.DataFrame([data])
                df = pd.concat([df, data_df], ignore_index=True)
                # Write back (truncate and rewrite to ensure consistency)
                df.to_csv(file_path, index=False)
                fcntl.flock(f, fcntl.LOCK_UN)  # Unlock
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                raise Exception(f"Failed to write to {file_path} after maximum retries")

def measure_training_pl(train_func, *args, **kwargs):
    """
    Measure training time and peak GPU memory for a training function.
    Args:
        train_func: callable, the training function (e.g., model.train)
        *args, **kwargs: arguments to pass to train_func
    Returns:
        wall_time (float): elapsed time in seconds
        peak_memory  (int): peak GPU memory usage in bytes
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    train_func(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
    else:
        peak_memory = 0
    wall_time = time.time() - start
    return wall_time, peak_memory


def measure_inference_pl(model, adata, batch_size=128):
    """
    Measure inference time, throughput, and peak GPU memory using model.get_latent_representation.
    Args:
        model: CPA model instance
        adata: AnnData object for inference
        batch_size (int): batch size for get_latent_representation
    Returns:
        wall_time (float): elapsed inference time in seconds
        throughput (float): samples per second
        peak_memory (int): peak GPU memory usage in bytes
        outputs (dict): latent representations returned by the model
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    outputs = model.get_latent_representation(adata, batch_size=batch_size)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
    else:
        peak_memory = 0
    wall_time = time.time() - start
    n_samples = adata.shape[0]
    throughput = n_samples / wall_time if wall_time > 0 else 0.0
    return wall_time, throughput, peak_memory, outputs
