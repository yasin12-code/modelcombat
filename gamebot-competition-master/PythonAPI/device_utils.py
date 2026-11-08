"""
Shared GPU/CPU device selection for the bot models.

Both bot.py (dense/MLP model) and rnn_bot.py (recurrent model) call
get_device() once during Bot.__init__. It:

  1. Looks for a physical GPU. If one is found, memory growth is enabled
     (so TensorFlow doesn't reserve the entire VRAM up front - that alone
     can add multi-second startup stalls and fights the emulator/driver
     for GPU resources) and inference is placed on '/GPU:0'.
  2. If no GPU is found, or GPU setup fails for any reason (missing
     CUDA/cuDNN, driver mismatch, etc.), it falls back to '/CPU:0'
     automatically instead of crashing the bot mid-match.
  3. Can be forced to CPU with the SF2_FORCE_CPU=1 environment variable.
     This matters because this model runs on a single sample (batch
     size 1) every frame - for workloads this small, GPU kernel-launch
     and host<->device transfer overhead can easily exceed the actual
     compute time, so CPU sometimes wins. Benchmark both on your machine
     and force whichever is actually faster:

         set SF2_FORCE_CPU=1        (Windows)
         export SF2_FORCE_CPU=1     (Linux/Mac)
"""
import os
import tensorflow as tf

_DEVICE = None  # cached so we only probe/configure once per process


def get_device(verbose=True):
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE

    if os.environ.get('SF2_FORCE_CPU', '0') == '1':
        if verbose:
            print("[Device] SF2_FORCE_CPU=1 set - using CPU regardless of GPU availability.")
        _DEVICE = '/CPU:0'
        return _DEVICE

    try:
        gpus = tf.config.list_physical_devices('GPU')
    except Exception as e:
        if verbose:
            print(f"[Device] Error probing for GPU ({e}); defaulting to CPU.")
        _DEVICE = '/CPU:0'
        return _DEVICE

    if not gpus:
        if verbose:
            print("[Device] No GPU detected - running inference on CPU.")
        _DEVICE = '/CPU:0'
        return _DEVICE

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if verbose:
            names = ", ".join(g.name for g in gpus)
            print(f"[Device] GPU detected ({names}) - running inference on GPU.")
        _DEVICE = '/GPU:0'
    except RuntimeError as e:
        # set_memory_growth must be called before the GPU is initialized;
        # if that fails for any reason, fall back safely rather than crash.
        if verbose:
            print(f"[Device] GPU found but could not be configured ({e}); falling back to CPU.")
        _DEVICE = '/CPU:0'

    return _DEVICE