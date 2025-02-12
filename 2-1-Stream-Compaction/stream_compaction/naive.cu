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
        /* Inclusive scan in Global Level using Naive algorithm O(nlogn)
        * @parms
        * n: the number of elements in idata
        * level: the level of the scan, the size of the block is 2^level
        */
        __global__ void kernInclusiveScanGlobal(uint32_t n, int level, int* odata, const int* idata){
            uint32_t d = 1 << level;
            uint32_t tid = threadIdx.x;
            uint32_t gid = blockIdx.x * blockDim.x + tid;
            if(gid >= n) return;

            if(gid >= d){
                odata[gid] = idata[gid - d] + idata[gid];
            }else{
                odata[gid] = idata[gid];
            }
        }

        void scanOnDeviceGlobal(uint32_t n, int*dev_odata, int*dev_idata){
            // Malloc different level of memory
            uint32_t level = ilog2ceil(n);
            uint32_t grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            for(int i = 0; i < level; i++){
                kernInclusiveScanGlobal<<<grid_size, BLOCK_SIZE>>>(n, i, dev_odata, dev_idata);
                std::swap(dev_odata, dev_idata);
            }

            Common::kernShiftRight<<<grid_size, BLOCK_SIZE>>>(n, dev_odata, dev_idata);
        }

        /* Exclusive scan in Block Level using Hillis algorithm O(nlogn)
          Shared memory is used to store the intermediate result
        * the size of odata should be BLOCK_SIZE * gird_size, 
        * so that need to padding the last block
        * @parms
        * n: the number of blocks in idata
        * s: the stride of the block in idata
        */
        __global__ void kernExclusiveScanPerBlock(uint32_t n, int s, int* odata, const int* idata){
            uint32_t tid = threadIdx.x;
            uint32_t bid = blockIdx.x;
            uint32_t gid = bid * blockDim.x + tid;
            __shared__ int buffer[2][BLOCK_SIZE];
            int ping = 0, pong = 1;
            uint32_t idx_in = gid*s - 1;
            buffer[ping][tid] = (gid > 0 && gid < n) ? idata[idx_in] : 0;
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

        void scanOnDeviceShared(uint32_t n, int*dev_odata, const int*dev_idata){

            // Malloc different level of memory
            uint32_t level = (ilog2ceil(n) + 7) / 8;
            int** dev_ptr = new int* [level];
            int* grid_size = new int[level];
            grid_size[0] = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            dev_ptr[0] = dev_odata;
            for(int i = 1; i < level; i++){
                grid_size[i] = (grid_size[i-1] + BLOCK_SIZE - 1) / BLOCK_SIZE;
                cudaMalloc((void**)&dev_ptr[i], grid_size[i] * BLOCK_SIZE * sizeof(int));
			    checkCUDAError("cudaMalloc dev_ptr failed!");
                cudaMemset(dev_ptr[i], 0, grid_size[i] * BLOCK_SIZE * sizeof(int));
            }

            // Scan each block in different level
            kernExclusiveScanPerBlock << <grid_size[0], BLOCK_SIZE >> > (n, 1, dev_odata, dev_idata);
            for(int i = 1; i < level; i++){
                kernExclusiveScanPerBlock<<<grid_size[i], BLOCK_SIZE>>>(grid_size[i-1], BLOCK_SIZE, dev_ptr[i], dev_ptr[i-1]);
            }

            // Scatter the offset to the original array
            for(int i = level - 1; i > 0; i--){
                Common::kernAddOffset<<<grid_size[i-1], BLOCK_SIZE>>>(grid_size[i-1] * BLOCK_SIZE, dev_ptr[i-1], dev_ptr[i]);
            }

            // Free the memory
            for(int i = 1; i < level; i++){
                cudaFree(dev_ptr[i]);
            }
            delete[] dev_ptr;
            delete[] grid_size;
        } 
    

        /*
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(uint32_t n, int *odata, const int *idata, bool isShared) {
            int* dev_idata, *dev_odata;
            uint32_t grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            uint32_t size = grid_size * BLOCK_SIZE;
            cudaMalloc((void**)&dev_idata, size * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemset(dev_idata, 0, size * sizeof(int));
            cudaMalloc((void**)&dev_odata, size * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMemset(dev_odata, 0, size * sizeof(int));

            cudaMemcpy(dev_idata, idata, n*sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata failed!");

            if(isShared){
                timer().startGpuTimer();
                scanOnDeviceShared(size, dev_odata, dev_idata);
                timer().endGpuTimer();
                cudaMemcpy(odata, dev_odata, n*sizeof(int), cudaMemcpyDeviceToHost);
                checkCUDAError("cudaMemcpy odata failed!");
            }else{
                timer().startGpuTimer();
                scanOnDeviceGlobal(size, dev_odata, dev_idata);
                timer().endGpuTimer();
                int level = ilog2ceil(size);
                if(level % 2 == 1) std::swap(dev_idata, dev_odata);
                cudaMemcpy(odata, dev_odata, n*sizeof(int), cudaMemcpyDeviceToHost);
            }

            cudaFree(dev_idata);
            cudaFree(dev_odata);

        }

        int compact(uint32_t n, int *odata, const int *idata, bool isShared) {
            uint32_t count = 0;
            int final_bool = 0;
            int* dev_idata, *dev_bools, *dev_indices;
            int* dev_odata;

            uint32_t gird_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            uint32_t size = gird_size * BLOCK_SIZE;
            // Malloc the memory
            cudaMalloc((void**)&dev_idata, size * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemset(dev_idata, 0, size * sizeof(int));

            cudaMalloc((void**)&dev_bools, size * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed!");
            cudaMemset(dev_bools, 0, size * sizeof(int));

            cudaMalloc((void**)&dev_indices, size * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed!");
            cudaMemset(dev_indices, 0, size * sizeof(int));

            cudaMalloc((void**)&dev_odata, size * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMemset(dev_odata, 0, size * sizeof(int));

            // Copy the data to the device
            cudaMemcpy(dev_idata, idata, n*sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            // Map to boolean
            Common::kernMapToBoolean<<<gird_size, BLOCK_SIZE>>>(n, dev_bools, dev_idata);

            // Scan the boolean array
            if(isShared)
                scanOnDeviceShared(size, dev_indices, dev_bools);
            else{
                scanOnDeviceGlobal(size, dev_indices, dev_bools);
                int level = ilog2ceil(size);
                if(level % 2 == 1) std::swap(dev_bools, dev_indices);
            }

            // Scatter the data
            Common::kernScatter<<<gird_size, BLOCK_SIZE>>>(n, dev_odata, dev_idata, dev_indices);
            timer().endGpuTimer();

            // Copy the data back to the host
            cudaMemcpy(odata, dev_odata, n*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&count, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            count += idata[n - 1] != 0 ? 1 : 0;
            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_odata);
            return count;
        }
    }
}
