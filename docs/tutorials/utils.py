import os
import time
import pandas as pd
import fcntl


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