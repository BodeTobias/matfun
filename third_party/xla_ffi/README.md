XLA FFI headers (header-only API for custom calls), copied unchanged from jaxlib 0.6.2
(jaxlib/include/xla/ffi/api/{api.h,c_api.h,ffi.h}, https://github.com/openxla/xla, Apache License 2.0, see LICENSE).
They declare XLA FFI API version 0.1. Handlers built with them were tested with jaxlib 0.6.2, 0.8.3, 0.11.2 and
0.12.0 (FFI API 0.1 to 0.5); matfun.jax compares the versions before registering the handlers. Only needed for the
XLA FFI handlers used by JAX; the plain C/C++ and CUDA APIs do not use them.
