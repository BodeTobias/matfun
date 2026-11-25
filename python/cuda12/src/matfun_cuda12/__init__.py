"""
matfun_cuda12 -- CUDA 12 kernel of matfun (libmatfun_cuda) as a plugin for the package matfun.

Not used directly: matfun.jax finds it and registers the kernel for NVIDIA GPUs.  Installed by
    pip install "matfun[cuda12]"
The library links libcudart.so.12 dynamically; it is found via RPATH in the nvidia-cuda-runtime wheel that
jax[cuda12] installs (site-packages/nvidia/cuda_runtime/lib), otherwise via the system loader path.
"""
import glob
import os

__version__ = "0.1.2"
_HERE = os.path.dirname(os.path.abspath(__file__))


def library_path():
    for d in __path__:                  # (all directories of the package: an editable install keeps the library apart)
        found = sorted(glob.glob(os.path.join(d, "libmatfun_cuda.*")))
        if found:
            return found[0]
    return os.path.join(_HERE, "libmatfun_cuda.so")
