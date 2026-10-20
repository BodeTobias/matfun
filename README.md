# matfun

Matrix functions **exp, log, sqrt and pow** of 3×3 matrices with **first and second derivatives**, for Newton solvers in finite element codes with many integration points:
- `smf`: symmetric matrices, derivatives with respect to the matrix or its Voigt components;
- `gmf`: general (non-symmetric) real matrices, the full derivatives ∂F<sub>ab</sub>/∂A<sub>cd</sub>, principal branch, including complex eigenvalues (rotations) and defective matrices (simple shear).

The derivatives are exact also for repeated eigenvalues. Implementations: C/C++ (SIMD, OpenMP), CUDA, Python (NumPy and JAX), Java and Julia, all tested against the same reference cases.

The methods (closed-form eigendecomposition, Daleckii–Krein formulas with stable divided differences; for general matrices the Newton–Hermite form over the eigenvalues with confluent divided differences) are described in [theory.md](theory.md).

## Installation

**Python**

```bash
pip install matfun                  # NumPy interface, C kernel
pip install "matfun[jax]"           # + JAX (CPU)
pip install "matfun[cuda13]"        # + JAX with CUDA 13 and the CUDA kernel (or "matfun[cuda12]")
```

**C / C++ / CUDA** with CMake (`FetchContent`):

```cmake
include(FetchContent)
FetchContent_Declare(matfun GIT_REPOSITORY https://github.com/BodeTobias/matfun.git GIT_TAG main)
FetchContent_MakeAvailable(matfun)                        # CUDA library too: set(MATFUN_BUILD_CUDA ON) before
target_link_libraries(myapp PRIVATE matfun::matfun)       # CUDA: matfun::matfun_cuda
```

or installed system-wide:

```bash
git clone https://github.com/BodeTobias/matfun.git && cd matfun
cmake -B build && cmake --build build -j && cmake --install build --prefix ~/.local
gcc myapp.c -I ~/.local/include -L ~/.local/lib -lmatfun -Wl,-rpath,$HOME/.local/lib
```

**Java** (Maven):

```xml
<dependency>
  <groupId>io.github.bodetobias</groupId>
  <artifactId>matfun</artifactId>
  <version>0.1.0</version>
</dependency>
```

**Julia**:

```julia
using Pkg; Pkg.add("MatFun")
```

## Usage

`smf` takes one or many symmetric 3×3 matrices, either as matrices or as Voigt vectors v = (A11, A22, A33, A12, A13, A23), and returns the results in the same form; `gmf` takes general 3×3 matrices:

| input | F | J | H |
|---|---|---|---|
| matrix A (3×3) | 3×3 | 3×3×3×3, ∂F<sub>ab</sub>/∂A<sub>cd</sub> | 3×3×3×3×3×3, ∂²F<sub>ab</sub>/∂A<sub>cd</sub>∂A<sub>ef</sub> |
| Voigt vector v (6) | 6 | 6×6, ∂F<sub>i</sub>/∂v<sub>k</sub> | 6×6×6, ∂²F<sub>i</sub>/∂v<sub>p</sub>∂v<sub>q</sub> |

```python
import matfun
F, J, H = matfun.smf("log", A, order=2)      # A: (..., 3, 3) or (..., 6); "exp", "log", "sqrt", "pow" (alpha=...)
F, J, H = matfun.gmf("log", A, order=2)      # A: (..., 3, 3), general real matrices

import jax; jax.config.update("jax_enable_x64", True)
import matfun.jax as mf
F, J = mf.smf("pow", A, alpha=0.3, order=1)   # C kernel on the CPU, CUDA kernel on NVIDIA GPUs
K = jax.jacfwd(lambda a: mf.smf("log", a))(A) # jit, vmap, jacfwd, jacrev, grad, hessian (up to 2nd order)
F = mf.gmf("exp", X)                          # general matrices, the same features
```

```c
#include "matfun/matfun.h"
smf_eval_matrix(SMF_LOG, 0.0, n, A, F, J, H);  /* n matrices; smf_eval: Voigt vectors; J, H may be NULL */
gmf_eval(SMF_EXP, 0.0, n, A, F, J, H);         /* general matrices; CUDA: smf_cuda_eval*, gmf_cuda_eval */
```

```java
import io.github.bodetobias.matfun.SymMatFun;
import io.github.bodetobias.matfun.GenMatFun;
SymMatFun.evalMatrix(SymMatFun.Kind.LOG, 0.0, A, F, J, H);   // eval: Voigt vectors; J, H may be null
GenMatFun.eval(SymMatFun.Kind.EXP, 0.0, A, F, J, H);         // general matrices
```

```julia
using MatFun
F, J, H = smf(:log, A; order = 2)           # A: 3×3, 3×3×n, Voigt vector or 6×n; in place: smf!(F, J, H, :log, A)
F, J, H = gmf(:exp, A; order = 2)           # general matrices, A: 3×3 or 3×3×n; in place: gmf!
```

**Conventions:**
- The indices mean the same in all languages: J[a,b,c,d] = ∂F<sub>ab</sub>/∂A<sub>cd</sub>, as in the table. Only the storage follows each language's array convention, and in all of them every matrix and its results are contiguous: C, Java and Python are row-major and 0-based with the matrix index first (one matrix after the other, e.g. `J[n,a,b,c,d]` in NumPy); Julia is column-major and 1-based with the matrix index last, e.g. `J[a,b,c,d,n]`.
- `smf`: the matrix derivatives are those of A ↦ f((A + Aᵀ)/2) and have the symmetries J<sub>abcd</sub> = J<sub>bacd</sub> = J<sub>abdc</sub> = J<sub>cdab</sub>, likewise for H. The derivative with respect to the Voigt component v₄ = A12 changes A12 and A21 together.
- `smf`: matrices must be symmetric up to round-off, |A<sub>ij</sub> − A<sub>ji</sub>| ≤ 10⁻⁸·max|A|. Otherwise C returns −3, and NumPy, Java and Julia raise an error; in JAX and in the CUDA API the results of that matrix are NaN.
- `smf`: `log` needs a positive definite A. `sqrt` and `pow` need a positive semidefinite A; eigenvalues below zero from round-off are set to 0.
- `gmf`: the full derivatives (no symmetrization), H symmetric in (cd) ↔ (ef). `exp` is defined for every matrix. A matrix has many logarithms, square roots and powers; `gmf` returns the principal one (eigenvalues of log A with imaginary part in (−π, π), of √A and A<sup>α</sup> with argument α·arg λ, arg λ ∈ (−π, π)), which is real for real A. It exists for A without real eigenvalues ≤ 0 (complex eigenvalues with negative real part are fine); otherwise the results of that matrix are NaN.

## Building from source and tests

```bash
cmake -B build -DMATFUN_BUILD_TESTS=ON [-DMATFUN_BUILD_CUDA=ON] && cmake --build build -j && ctest --test-dir build
pip install ".[test]" && pytest python/tests
cd java && mvn test
julia --project=julia/MatFun -e 'using Pkg; Pkg.test()'
```

## Citation and license

See `CITATION.cff` (GitHub: "Cite this repository"). License: MIT (`LICENSE`); `third_party/xla_ffi` is Apache 2.0.
