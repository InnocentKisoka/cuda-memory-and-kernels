# Portable CUDA makefile
# Override architecture for your GPU, e.g. make ARCH=sm_80  (A100) or sm_90 (H100)
ARCH ?= sm_70
NVCC ?= nvcc
CXXFLAGS = -std=c++17 -O3 -lineinfo
NVCCFLAGS = $(CXXFLAGS) -arch=$(ARCH) -Iinclude
LDFLAGS = -lcublas -lcuda

BIN_DIR = bin
SRC_DIR = src

.PHONY: all clean dirs

dirs:
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)

TARGETS = 01_custom_axpy 02_cublas_axpy_explicit 03_cublas_axpy_managed 04_bandwidth_probe

all: dirs $(addprefix $(BIN_DIR)/,$(TARGETS))

$(BIN_DIR)/%: $(SRC_DIR)/%.cu include/util.hpp
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

.PHONY: run-demo
run-demo: all
	@echo "=== Custom AXPY (n=2^20) ==="
	$(BIN_DIR)/01_custom_axpy 20
	@echo "=== cuBLAS explicit memory ==="
	$(BIN_DIR)/02_cublas_axpy_explicit 18
	@echo "=== cuBLAS managed memory ==="
	$(BIN_DIR)/03_cublas_axpy_managed 18
