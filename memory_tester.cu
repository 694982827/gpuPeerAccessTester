#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#define CUDA_CHECK(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code,
                      const char *file,
                      int line,
                      bool abort = true) {
  if(code != cudaSuccess) {
    std::cerr <<"GPUassert: " << cudaGetErrorString(code) << " " << file << " "
                              << line << std::endl;
    exit(1);
  }
}


//Copy from other GPUs to this device
double normalMemCpyOther2This(float ** gpu_ptrs, int num_gpus, int bytes_to_transfer_each, int curr_gpu) {
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_gpus; i++) {
        if (i != curr_gpu) {
            CUDA_CHECK(cudaMemcpy(gpu_ptrs[curr_gpu] + (bytes_to_transfer_each/4/num_gpus)*i,
             gpu_ptrs[i] + (bytes_to_transfer_each/4/num_gpus)*i,
             bytes_to_transfer_each/num_gpus,
                cudaMemcpyDefault));
        }

    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> secs = end-start;
    return secs.count();
}

//From this device to other GPUS, async
double normalMemCpyThis2Other(float ** gpu_ptrs, int num_gpus, int bytes_to_transfer_each, int curr_gpu) {
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_gpus; i++) {
        if (i != curr_gpu) {
            CUDA_CHECK(cudaMemcpy(gpu_ptrs[i] + (bytes_to_transfer_each/4/num_gpus)*i,
             gpu_ptrs[curr_gpu] + (bytes_to_transfer_each/4/num_gpus)*i,
             bytes_to_transfer_each/num_gpus,
                cudaMemcpyDefault));
        }

    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> secs = end-start;
    return secs.count();
}

//Copy from other GPUs to this device, async
double normalMemCpyOther2ThisAsync(float ** gpu_ptrs, int num_gpus, int bytes_to_transfer_each, int curr_gpu) {
    cudaStream_t streams[num_gpus];
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_gpus; i++) {
        if (i != curr_gpu) {
            CUDA_CHECK(cudaMemcpyAsync(gpu_ptrs[curr_gpu] + (bytes_to_transfer_each/4/num_gpus)*i,
             gpu_ptrs[i] + (bytes_to_transfer_each/4/num_gpus)*i,
             bytes_to_transfer_each/num_gpus,
                cudaMemcpyDefault, streams[i]));
        }

    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> secs = end-start;
    return secs.count();
}

//From this device to other GPUS, async
double normalMemCpyThis2OtherAsync(float ** gpu_ptrs, int num_gpus, int bytes_to_transfer_each, int curr_gpu) {
    cudaStream_t streams[num_gpus];
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_gpus; i++) {
        if (i != curr_gpu) {
            CUDA_CHECK(cudaMemcpyAsync(gpu_ptrs[i] + (bytes_to_transfer_each/4/num_gpus)*i,
             gpu_ptrs[curr_gpu] + (bytes_to_transfer_each/4/num_gpus)*i,
             bytes_to_transfer_each/num_gpus,
                cudaMemcpyDefault, streams[i]));
        }

    }

    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> secs = end-start;
    return secs.count();
}

int main(int argc, char * argv[]) {

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " num gpus total_model_memory" << std::endl;
        exit(1);
    }

    int num_gpus = atoi(argv[1]);

    int bytes_to_transfer_each = atoi(argv[2])*1024*1024;

    float ** gpu_ptrs = new float*[num_gpus];

    //Initiate with random memory, we don't care what it is. Also init the streams
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        CUDA_CHECK(cudaMalloc(&gpu_ptrs[i], bytes_to_transfer_each));
    }

    auto other_this = normalMemCpyOther2This(gpu_ptrs, num_gpus, bytes_to_transfer_each, 0);
    auto this_other = normalMemCpyThis2Other(gpu_ptrs, num_gpus, bytes_to_transfer_each, 0);
    auto other_thisAsync = normalMemCpyOther2ThisAsync(gpu_ptrs, num_gpus, bytes_to_transfer_each, 0);
    auto this_otherAsync = normalMemCpyThis2OtherAsync(gpu_ptrs, num_gpus, bytes_to_transfer_each, 0);
    
    for (int i = 1; i < num_gpus; i++) {
      other_this += normalMemCpyOther2This(gpu_ptrs, num_gpus, bytes_to_transfer_each, i);
      this_other += normalMemCpyThis2Other(gpu_ptrs, num_gpus, bytes_to_transfer_each, i);
      other_thisAsync += normalMemCpyOther2ThisAsync(gpu_ptrs, num_gpus, bytes_to_transfer_each, i);
      this_otherAsync += normalMemCpyThis2OtherAsync(gpu_ptrs, num_gpus, bytes_to_transfer_each, i);
    }

    std::cout << std::fixed << "Other to this took: " << other_this << " seconds." << std::endl;
    std::cout << std::fixed << "This to other took: " << this_other << " seconds." << std::endl;
    std::cout << std::fixed << "Other to this Async took: " << other_this << " seconds." << std::endl;
    std::cout << std::fixed << "This to other Async took: " << this_other << " seconds." << std::endl;

    //Attempt to enable peer access
    for (int i = 0; i<num_gpus; i++) {
        for (int j = 0; j<num_gpus; j++) {
            if (i != j) {
                int result;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&result, i, j));
                if (result) {
                    cudaSetDevice(i);
                    cudaDeviceEnablePeerAccess (j, 0);
                } else {
                    std::cout << std::fixed << "Peer access unavailable between devices: " << i << " and " << j << std::endl;
                }
            }
        }
    }

    //Redo the benchmarks, see if it is any different

    other_this = normalMemCpyOther2This(gpu_ptrs, num_gpus, bytes_to_transfer_each, 0);
    this_other = normalMemCpyThis2Other(gpu_ptrs, num_gpus, bytes_to_transfer_each, 0);
    other_thisAsync = normalMemCpyOther2ThisAsync(gpu_ptrs, num_gpus, bytes_to_transfer_each, 0);
    this_otherAsync = normalMemCpyThis2OtherAsync(gpu_ptrs, num_gpus, bytes_to_transfer_each, 0);
    
    for (int i = 1; i < num_gpus; i++) {
      other_this += normalMemCpyOther2This(gpu_ptrs, num_gpus, bytes_to_transfer_each, i);
      this_other += normalMemCpyThis2Other(gpu_ptrs, num_gpus, bytes_to_transfer_each, i);
      other_thisAsync += normalMemCpyOther2ThisAsync(gpu_ptrs, num_gpus, bytes_to_transfer_each, i);
      this_otherAsync += normalMemCpyThis2OtherAsync(gpu_ptrs, num_gpus, bytes_to_transfer_each, i);
    }

    std::cout << std::fixed << "Peer other to this took: " << other_this << " seconds." << std::endl;
    std::cout << std::fixed << "Peer this to other took: " << this_other << " seconds." << std::endl;
    std::cout << std::fixed << "Peer other to this Async took: " << other_this << " seconds." << std::endl;
    std::cout << std::fixed << "Peer this to other Async took: " << this_other << " seconds." << std::endl;

    return 0;

}
