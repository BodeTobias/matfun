# matfun -- C/C++ library (libmatfun) and CUDA library (libmatfun_cuda).
# Java and Julia are native ports with their own build files (java/, julia/).
#
#   cmake -B build -DMATFUN_BUILD_TESTS=ON && cmake --build build -j && ctest --test-dir build
#   cmake -B build -DMATFUN_BUILD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 ...
cmake_minimum_required(VERSION 3.24)
project(matfun VERSION 0.1.0 LANGUAGES C CXX)

option(MATFUN_BUILD_C "C/C++ library libmatfun" ON)
option(MATFUN_BUILD_CUDA "CUDA library libmatfun_cuda" OFF)
option(MATFUN_WITH_XLA_FFI "add the XLA FFI handlers for JAX (headers in third_party/xla_ffi)" OFF)
# OpenMP: off by default with MSVC (/openmp:llvm needs libomp140.x86_64.dll, which is not part of the VC runtime)
if(MSVC)
  option(MATFUN_OPENMP "parallelise smf_eval over matrices with OpenMP" OFF)
else()
  option(MATFUN_OPENMP "parallelise smf_eval over matrices with OpenMP" ON)
endif()
option(MATFUN_BUILD_TESTS "build the C tests" OFF)
set(MATFUN_PY_INSTALL_DIR "" CACHE STRING "install the libraries into this directory (used by the Python build)")
set(MATFUN_CUDART_RPATH "$ORIGIN/../nvidia/cu13/lib" CACHE STRING
    "RPATH of libmatfun_cuda in the Python build: libcudart of the nvidia wheel (CUDA 12: nvidia/cuda_runtime/lib)")
# kernel variants (instruction sets):
#   dispatch  scalar + AVX2 + AVX-512 in one library, chosen at run time (CPU flags, libmvec loaded with dlopen);
#             portable, e.g. for Python wheels.  Default on Linux x86-64 with GCC / Clang.
#   native    one variant for this CPU (-march=native, libmvec linked directly); not portable
#   scalar    scalar code only (default elsewhere: macOS, Windows, ARM)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND NOT MSVC)
  set(MATFUN_SIMD_DEFAULT dispatch)
else()
  set(MATFUN_SIMD_DEFAULT scalar)
endif()
set(MATFUN_SIMD ${MATFUN_SIMD_DEFAULT} CACHE STRING "kernel variants: dispatch, native or scalar")
set_property(CACHE MATFUN_SIMD PROPERTY STRINGS dispatch native scalar)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(MSVC)
  set(SMF_FP_FLAGS "")
else()
  set(SMF_FP_FLAGS -fno-math-errno -fno-trapping-math)
endif()

set(SMF_XLA_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/xla_ffi/include)

# ---------------------------------------------------------------- C/C++
if(MATFUN_BUILD_C)
  set(SMF_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/c/include)
  set(SMF_LINK_LIBS "")
  # one object library per kernel variant: smf_kernel_<isa> (c/src/kernel_isa.cc with -DSMF_ISA=<isa>)
  function(smf_kernel_variant isa)
    add_library(smf_kernel_${isa} OBJECT c/src/kernel_isa.cc)
    target_include_directories(smf_kernel_${isa} PRIVATE ${SMF_INCLUDE})
    target_compile_definitions(smf_kernel_${isa} PRIVATE SMF_ISA=${isa} ${ARGN})
    target_compile_options(smf_kernel_${isa} PRIVATE ${SMF_FP_FLAGS})
  endfunction()

  if(MATFUN_SIMD STREQUAL "native")
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_FLAGS "-march=native")
    set(CMAKE_REQUIRED_INCLUDES ${SMF_INCLUDE})
    set(CMAKE_REQUIRED_LIBRARIES mvec)
    check_cxx_source_compiles("
      #include \"matfun/smf_simd.hpp\"
      #if SMF_W == 1
      #error no SIMD
      #endif
      int main() { double x[6] = {1,1,1,0,0,0}, F[9]; smf::batch_any(0, 1, x, F, nullptr, 0.0); return F[0] > 0 ? 0 : 1; }"
      SMF_HAVE_NATIVE_SIMD)
    unset(CMAKE_REQUIRED_FLAGS)
    unset(CMAKE_REQUIRED_INCLUDES)
    unset(CMAKE_REQUIRED_LIBRARIES)
    if(NOT SMF_HAVE_NATIVE_SIMD)
      message(WARNING "matfun: -march=native with libmvec does not build here, using MATFUN_SIMD=scalar")
      set(MATFUN_SIMD scalar)
    endif()
  endif()

  smf_kernel_variant(scalar SMF_SCALAR)
  set(SMF_VARIANTS scalar)
  if(MATFUN_SIMD STREQUAL "dispatch")
    set(SMF_AVX2_FLAGS -mavx2 -mfma -mbmi -mbmi2 -mf16c)          # checked at run time in dispatch.cc
    smf_kernel_variant(avx2 SMF_MVEC_DYNAMIC)
    target_compile_options(smf_kernel_avx2 PRIVATE ${SMF_AVX2_FLAGS})
    smf_kernel_variant(avx512 SMF_MVEC_DYNAMIC)
    target_compile_options(smf_kernel_avx512 PRIVATE ${SMF_AVX2_FLAGS} -mavx512f -mavx512bw -mavx512cd -mavx512dq
                                                     -mavx512vl)
    list(APPEND SMF_VARIANTS avx2 avx512)
    list(APPEND SMF_LINK_LIBS ${CMAKE_DL_LIBS})
  elseif(MATFUN_SIMD STREQUAL "native")
    smf_kernel_variant(native)
    target_compile_options(smf_kernel_native PRIVATE -march=native)
    list(APPEND SMF_VARIANTS native)
    list(APPEND SMF_LINK_LIBS mvec)
  endif()

  set(SMF_SOURCES c/src/matfun.cc c/src/dispatch.cc)
  if(MATFUN_WITH_XLA_FFI)
    list(APPEND SMF_SOURCES c/src/matfun_xla.cc)
  endif()
  foreach(v ${SMF_VARIANTS})
    list(APPEND SMF_SOURCES $<TARGET_OBJECTS:smf_kernel_${v}>)
  endforeach()
  add_library(matfun SHARED ${SMF_SOURCES})
  add_library(matfun::matfun ALIAS matfun)
  target_include_directories(matfun PUBLIC $<BUILD_INTERFACE:${SMF_INCLUDE}> $<INSTALL_INTERFACE:include>)
  if(MATFUN_WITH_XLA_FFI)
    target_include_directories(matfun PRIVATE ${SMF_XLA_INCLUDE})
  endif()
  foreach(v ${SMF_VARIANTS})
    string(TOUPPER ${v} V)
    target_compile_definitions(matfun PRIVATE SMF_WITH_${V})
  endforeach()
  target_compile_options(matfun PRIVATE ${SMF_FP_FLAGS})
  target_compile_definitions(matfun PRIVATE SMF_VERSION="${PROJECT_VERSION}")
  target_link_libraries(matfun PRIVATE ${SMF_LINK_LIBS})
  set_target_properties(matfun PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
  if(MATFUN_OPENMP AND MSVC)
    target_compile_options(matfun PRIVATE /openmp:llvm)     # MSVC's default OpenMP 2.0: no int64 loop variable
    set(OpenMP_CXX_FOUND "llvm")
  elseif(MATFUN_OPENMP)
    find_package(OpenMP COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
      target_link_libraries(matfun PRIVATE OpenMP::OpenMP_CXX)
    endif()
  endif()
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(matfun PRIVATE -Wl,--no-undefined)
  endif()
  message(STATUS "matfun: kernel variants ${SMF_VARIANTS}, OpenMP ${OpenMP_CXX_FOUND}, XLA FFI ${MATFUN_WITH_XLA_FFI}")

  if(MATFUN_PY_INSTALL_DIR)
    install(TARGETS matfun LIBRARY DESTINATION ${MATFUN_PY_INSTALL_DIR} RUNTIME DESTINATION ${MATFUN_PY_INSTALL_DIR})
  else()
    include(GNUInstallDirs)
    install(TARGETS matfun)
    install(FILES c/include/matfun/matfun.h c/include/matfun/smf_kernel.hpp c/include/matfun/gmf_kernel.hpp
                  c/include/matfun/smf_simd.hpp
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/matfun)
  endif()

  if(MATFUN_BUILD_TESTS)
    enable_testing()
    add_executable(test_c c/tests/test_c.c)
    target_link_libraries(test_c PRIVATE matfun)
    if(NOT MSVC)
      target_link_libraries(test_c PRIVATE m)
    endif()
    add_test(NAME test_c COMMAND test_c)
    foreach(v ${SMF_VARIANTS})                       # every variant against the same checks
      add_test(NAME test_c_${v} COMMAND test_c)
      set_tests_properties(test_c_${v} PROPERTIES ENVIRONMENT MATFUN_SIMD=${v} SKIP_RETURN_CODE 77)
    endforeach()
  endif()
endif()

# ---------------------------------------------------------------- CUDA
if(MATFUN_BUILD_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  # libcudart: from a CUDA toolkit, or (nvcc from the nvidia-cuda-nvcc wheel) the versioned libcudart.so.<major>
  # next to nvcc, which has no unversioned libcudart.so for FindCUDAToolkit / nvcc -lcudart
  find_package(CUDAToolkit QUIET)
  if(TARGET CUDA::cudart)
    set(SMF_CUDART CUDA::cudart)
  else()
    get_filename_component(SMF_NVCC_DIR ${CMAKE_CUDA_COMPILER} DIRECTORY)
    find_library(SMF_CUDART NAMES cudart libcudart.so.13 libcudart.so.12
                 HINTS ${SMF_NVCC_DIR}/../lib ${SMF_NVCC_DIR}/../lib64 ${SMF_NVCC_DIR}/../../cuda_runtime/lib NO_DEFAULT_PATH)
    if(NOT SMF_CUDART)
      message(FATAL_ERROR "matfun: libcudart not found (CUDA toolkit or the nvidia-cuda-runtime wheel)")
    endif()
  endif()
  message(STATUS "matfun: CUDA ${CMAKE_CUDA_COMPILER_VERSION}, architectures ${CMAKE_CUDA_ARCHITECTURES}, cudart ${SMF_CUDART}")
  add_library(matfun_cuda SHARED cuda/src/matfun_cuda.cu)
  add_library(matfun::matfun_cuda ALIAS matfun_cuda)
  target_include_directories(matfun_cuda PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/c/include>
                                                   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cuda/include>
                                                   $<INSTALL_INTERFACE:include>)
  if(MATFUN_WITH_XLA_FFI)
    target_include_directories(matfun_cuda PRIVATE ${SMF_XLA_INCLUDE})
    target_compile_definitions(matfun_cuda PRIVATE MATFUN_WITH_XLA_FFI)
  endif()
  target_compile_options(matfun_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
  # libcudart as a shared library, linked explicitly (found above); at run time the one of the nvidia runtime wheel
  set_target_properties(matfun_cuda PROPERTIES CUDA_RUNTIME_LIBRARY None)
  target_link_libraries(matfun_cuda PRIVATE ${SMF_CUDART})
  if(MATFUN_PY_INSTALL_DIR)
    # libcudart from the nvidia runtime wheel next to the package, else the system loader path
    set_target_properties(matfun_cuda PROPERTIES INSTALL_RPATH "${MATFUN_CUDART_RPATH}")
    install(TARGETS matfun_cuda LIBRARY DESTINATION ${MATFUN_PY_INSTALL_DIR} RUNTIME DESTINATION ${MATFUN_PY_INSTALL_DIR})
  else()
    include(GNUInstallDirs)
    install(TARGETS matfun_cuda)
    install(FILES cuda/include/matfun/matfun_cuda.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/matfun)
  endif()
  if(MATFUN_BUILD_TESTS AND MATFUN_BUILD_C)
    add_executable(test_cuda_api cuda/tests/test_cuda_api.cu)
    set_target_properties(test_cuda_api PROPERTIES CUDA_RUNTIME_LIBRARY None)
    target_link_libraries(test_cuda_api PRIVATE matfun_cuda matfun ${SMF_CUDART})
    add_test(NAME test_cuda_api COMMAND test_cuda_api)
  endif()
endif()
