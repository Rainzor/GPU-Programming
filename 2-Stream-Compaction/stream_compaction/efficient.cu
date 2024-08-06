#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }


        /* Exclusive scan in Block Level using work-efficient algorithm O(n)
        * the size of odata should be BLOCK_SIZE * gird_size, 
        * so that need to padding the last block
        * @parms
        * n: the number of elements in idata
        */
        __global__ void kernInclusiveScanPerBlock(int n, int* odata, const int* idata){
            int tid = threadIdx.x;
            int tid2 = tid << 1;
            int gid2 = (blockIdx.x * blockDim.x + tid) * 2; // Each thread handle 2 elements
            __shared__ int buffer[BLOCK_SIZE];
            buffer[tid2] = (0 < gid2) ? idata[gid2 - 1] : 0;
            buffer[tid2+1] = (gid2 + 1 < n) ? idata[gid2] : 0;       
            __syncthreads();
            
            // Up-Sweep (Reduce)
            int offset = 1;
            for(int stride = blockDim.x; stride > 0; stride >>= 1){
                if(tid < stride){
                    int ai = offset * (tid2 + 1) - 1;
                    int bi = offset * (tid2 + 2) - 1;
                    buffer[bi] += buffer[ai];
                }
                offset <<= 1;
                __syncthreads();
            }

            // Down-Sweep (Distribute)
            int sum = 0;
            if(tid == blockDim.x - 1){
                sum = buffer[BLOCK_SIZE - 1];
                buffer[BLOCK_SIZE - 1] = 0;
            }
            __syncthreads();

            offset = blockDim.x;
            for(int stride = 1; stride <= blockDim.x; stride <<= 1){
                if(tid < stride){
                    int ai = offset * (tid2 + 1) - 1;
                    int bi = offset * (tid2 + 2) - 1;
                    int t = buffer[ai];
                    buffer[ai] = buffer[bi];
                    buffer[bi] += t;
                }
                offset >>= 1;
                __syncthreads();

            }
            // Write the result to the output
            odata[gid2] = buffer[tid2+1];
            odata[gid2+1] = (tid < blockDim.x - 1) ? buffer[tid2+2] : sum;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // Malloc different level of memory
            int level = (ilog2ceil(n)+7)/8;
            int** dev_ptr = new int*[level];
            int* grid_size = new int[level];
            int len = n;
            for(int i = 0; i < level; i++){
                grid_size[i] = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
                cudaMalloc((void**)&dev_ptr[i], grid_size[i] * BLOCK_SIZE * sizeof(int));
                checkCUDAError("cudaMalloc dev_ptr failed!");
                cudaMemset(dev_ptr[i], 0, grid_size[i] * BLOCK_SIZE * sizeof(int));
                len = grid_size[i];
            }
            int* dev_tempbuff;
            int temp_size = n;
            cudaMalloc((void**)&dev_tempbuff, temp_size*sizeof(int));
            checkCUDAError("cudaMalloc dev_tempbuff failed!");
            cudaMemcpy(dev_tempbuff, idata, temp_size*sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata failed!");

            timer().startGpuTimer();

            // Scan each block in different level
            dim3 half_block_size(BLOCK_SIZE/2);
            for(int i = 0; i < level; i++){
                kernInclusiveScanPerBlock<<<grid_size[i], half_block_size>>>(temp_size, dev_ptr[i], dev_tempbuff);
                // Gather the last element of each block
                if(i < level - 1){
                    Common::kernExtractLastElementPerBlock<<<grid_size[i+1], BLOCK_SIZE>>>( grid_size[i], 
                                                                                    BLOCK_SIZE, 
                                                                                    dev_tempbuff, 
                                                                                    dev_ptr[i]);
                }
                temp_size = grid_size[i];
            }
            // Scatter the offset to the original array
            for(int i = level - 2; i >= 0; i--){
                Common::kernAddOffset<<<grid_size[i], BLOCK_SIZE>>>(grid_size[i] * BLOCK_SIZE, dev_ptr[i], dev_ptr[i+1]);
            }

            timer().endGpuTimer();

            // Copy the result to the host
            cudaMemcpy(odata, dev_ptr[0], n*sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed!");
            
            // Free the memory
            for(int i = 0; i < level; i++){
                cudaFree(dev_ptr[i]);
            }
            cudaFree(dev_tempbuff);
            delete[] dev_ptr;
            delete[] grid_size;
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
