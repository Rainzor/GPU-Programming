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
        /* Exclusive scan in Block Level
        * the size of odata should be BLOCK_SIZE * gird_size, 
        * so that need to padding the last block
        * @parms
        * n: the number of elements in idata
        */
        __global__ void kernInclusiveScanPerBlock(int n, int* odata, const int* idata){
            int tid = threadIdx.x;
            int bid = blockIdx.x;
            int gid = bid * blockDim.x + tid;
            __shared__ int buffer[2][BLOCK_SIZE];
            int ping = 0, pong = 1;
            buffer[ping][tid] = (gid > 0 && gid < n) ? idata[gid - 1] : 0;
            __syncthreads();
            for(int d = 1; d < blockDim.x; d <<= 1){
                ping = 1 - ping;
                pong = 1 - pong;
                if(tid >= d){
                    buffer[ping][tid] = buffer[pong][tid] + buffer[pong][tid - d];
                }
                else{
                    buffer[ping][tid] = buffer[pong][tid];
                }
                __syncthreads();
            }
            odata[gid] = buffer[ping][tid];
        }
        

        /*
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
            for(int i = 0; i < level; i++){
                kernInclusiveScanPerBlock<<<grid_size[i], BLOCK_SIZE>>>(temp_size, dev_ptr[i], dev_tempbuff);
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
    }
}
