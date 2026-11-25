"""
matfun_cuda13 -- CUDA 13 kernel of matfun (libmatfun_cuda) as a plugin for the package matfun.

Not used directly: matfun.jax finds it and registers the kernel for NVIDIA GPUs.  Installed by
    pip install "matfun[cuda13]"
The library links libcudart.so.13 dynamically; it is found via RPATH in the nvidia-cuda-runtime wheel that
jax[cuda13] installs (site-packages/nvidia/cu13/lib), otherwise via the system loader path.
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
