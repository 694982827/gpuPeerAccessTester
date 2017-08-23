# gpuPeerAccessTester
A quick test written to check the benefit of peer memory access.
```
nvcc -Wno-deprecated-gpu-targets -std=c++11 -O3 memory_tester.cu -o memory_tester.out
./memory_tester.out NUM_GPUs MEMORY_IN_MB_PER_GPU
```

