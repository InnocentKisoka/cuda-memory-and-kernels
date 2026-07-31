# CUDA Memory Models & High-Performance Kernels

[![CUDA](https://img.shields.io/badge/CUDA-C%2B%2B-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![Language](https://img.shields.io/badge/language-C%2B%2B17-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Portfolio project demonstrating **CUDA Runtime API mastery**: explicit vs managed memory, **cuBLAS** integration, custom multi-block kernels, and **GPU bandwidth / timing** analysis.

> Built from production-style labs completed during the CSCS–USI HPC Summer School (CUDA track), cleaned and packaged for employers.

---

## Skills demonstrated

| Skill | Where |
|-------|--------|
| Host / device memory spaces | `02_*`, `03_*` |
| `cudaMalloc` / `cudaMemcpy` (explicit) | `02_cublas_axpy_explicit.cu` |
| Unified / managed memory | `03_cublas_axpy_managed.cu` |
| Custom `__global__` kernels | `01_custom_axpy.cu` |
| Grid-stride / multi-block launches | `01_custom_axpy.cu` |
| Bounds checks & launch config | `01_custom_axpy.cu` |
| cuBLAS device pointer model | `02_*`, `03_*` |
| H2D / kernel / D2H timing breakdown | all |
| Bandwidth thinking (bytes moved vs time) | `01_*`, `04_*` |

**Resume bullets you can claim:**
- Implemented BLAS-1 **AXPY** on GPU with multi-block CUDA kernels and validated against host results.
- Compared **explicit memory management** vs **CUDA managed memory** with cuBLAS `cublasDaxpy`.
- Measured end-to-end bottlenecks (PCIe transfer vs HBM-bound kernel) for large vectors.

---

## Project layout

```
cuda-memory-and-kernels/
├── include/util.hpp          # error checks, alloc, memcpy helpers
├── src/
│   ├── 01_custom_axpy.cu     # hand-written multi-block AXPY + timing
│   ├── 02_cublas_axpy_explicit.cu
│   ├── 03_cublas_axpy_managed.cu
│   └── 04_bandwidth_probe.cu
├── Makefile
└── README.md
```

---

## Build & run

**Requirements:** NVIDIA GPU, CUDA Toolkit (`nvcc`), C++17.

```bash
# Adjust ARCH to your GPU (sm_70 V100, sm_80 A100, sm_86 RTX 30xx, sm_90 H100)
make ARCH=sm_80
make run-demo ARCH=sm_80

# Individual experiments (argument = log2(N) for vector length)
./bin/01_custom_axpy 22          # N = 2^22 doubles
./bin/02_cublas_axpy_explicit 20
./bin/03_cublas_axpy_managed 20
```

### What to look for in the output

1. **H2D / axpy / D2H times** — for large N, transfers often dominate if you copy every call.
2. **Correctness** — result should satisfy `y = 3 + 2*1.5 = 6` (within FP tolerance).
3. **Managed vs explicit** — convenience of one pointer space vs control/performance of explicit copies.

---

## Design notes (interview talking points)

```
Host DRAM  --cudaMemcpy-->  Device HBM  --kernel/cuBLAS-->  Device HBM  --cudaMemcpy-->  Host
                ↑                                              ↑
         often the bottleneck for BLAS-1              high bandwidth, few FLOPs/byte
```

- **AXPY** is **memory-bandwidth bound** (~3 reads/writes of `double` per element, 1 FMA).
- Optimal block sizes are typically multiples of warp size (32); this project uses 128 threads/block.
- Always use **ceiling division** for grid size: `(n + block - 1) / block` + `if (i < n)`.

---

## Attribution

Educational CUDA patterns adapted from the CSCS–USI Summer School curriculum; structure, documentation, and packaging by **Innocent Kisoka** for portfolio use.
