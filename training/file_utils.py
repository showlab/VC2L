import logging
import os
import multiprocessing
import subprocess
import time
import fsspec
import torch
from tqdm import tqdm

def start_sync_process(sync_every, local_dir, remote_dir, protocol):
    p = multiprocessing.Process(target=keep_running_remote_sync, args=(sync_every, local_dir, remote_dir, protocol))
    return p

# Note: we are not currently using this save function.
def pt_save(pt_obj, file_path):
    of = fsspec.open(file_path, "wb")
    with of as f:
        torch.save(pt_obj, file_path)

def pt_load(file_path, map_location=None):
    if file_path.startswith('s3'):
        logging.info('Loading remote checkpoint, which may take a bit.')
    of = fsspec.open(file_path, "rb")
    with of as f:
        out = torch.load(f, map_location=map_location)
    return out

def check_exists(file_path):
    try:
        with fsspec.open(file_path):
            pass
    except FileNotFoundError:
        return False
    return True
