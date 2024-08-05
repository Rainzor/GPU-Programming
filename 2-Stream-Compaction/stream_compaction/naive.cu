#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernelExclusiveScan(int n, int* odata, const int* idata, int d){
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if(index >= n) return;
            int offset = 1 << d;
            if(index >= offset){
                odata[index] = idata[index - offset] + idata[index];
            }
            else{
                odata[index] = idata[index];
            }
        }
        

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_idata;
            int* dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc failed!");
            
            cudaMemset(dev_idata, 0, n * sizeof(int));
            cudaMemset(dev_odata, 0, n * sizeof(int));
            
            // Exclusive scan, first element is 0
            cudaMemcpy(dev_odata+1, idata, (n-1) * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("Input cudaMemcpy failed!");


            /********************************Kernel launch****************************************/
            timer().startGpuTimer();
            int depth = ilog2ceil(n);
            for(int d = 0; d < depth; d++){
                std::swap(dev_idata, dev_odata);
                scan_core<<<(n + blockSize - 1) / blockSize, blockSize>>>(n, dev_odata, dev_idata, d);
            }

            cudaDeviceSynchronize();
            timer().endGpuTimer();
            /**************************************************************************************/

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("Output cudaMemcpy failed!");

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            checkCUDAError("cudaFree failed!");
        }
    }
}
