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
        __global__ void kernUpSweepGlobal(uint32_t n, int level, int* data){
            uint32_t offset = 1 << (level);
            uint32_t thread_size = (n + offset*2 - 1) / (offset*2);
            uint32_t tid = threadIdx.x;
            uint32_t gid = (blockIdx.x * blockDim.x + tid);
            if(gid >= thread_size) return;
            uint32_t idx_l = (gid*2 + 1)*offset - 1;
            if(idx_l >= n) return;
            uint32_t idx_r = (gid*2 + 2)*offset - 1;
            if(idx_r >= n) return;
            data[idx_r] += data[idx_l];
        }
        
        __global__ void kernDownSweepGlobal(uint32_t n, int level, int* data){
            uint32_t offset = 1 << (level);
            uint32_t thread_size = (n + offset*2 - 1) / (offset*2);
            uint32_t tid = threadIdx.x;
            uint32_t gid = (blockIdx.x * blockDim.x + tid);
            if(gid >= thread_size) return;
            uint32_t idx_l = (gid*2 + 1)*offset - 1;
            if(idx_l >= n) return;
            uint32_t idx_r = (gid*2 + 2)*offset - 1;
            if(idx_r >= n) return;
            int t = data[idx_l];
            data[idx_l] = data[idx_r];
            data[idx_r] += t;
        }

        // Exclusive scan in Global Level using Blelloch algorithm O(n)
        // Require the size of the array is 2^k
        void scanOnDeviceGlobal(uint32_t n, int*dev_data){
            int level = ilog2ceil(n);
            for(int i = 0; i < level; i++){
                uint32_t stride = 1 << (i + 1);
                uint32_t thread_size = (n + stride - 1) / stride;
                int grid_size = (thread_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernUpSweepGlobal<<<grid_size, BLOCK_SIZE>>>(n, i, dev_data);
            }
            cudaMemset(dev_data + n - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset failed!");
            for(int i = level - 1; i >= 0; i--){
                uint32_t stride = 1 << (i + 1);
                uint32_t thread_size = (n + stride - 1) / stride;
                int grid_size = (thread_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernDownSweepGlobal<<<grid_size, BLOCK_SIZE>>>(n, i, dev_data);
            }

        }

        /* Exclusive scan in Block Level using work-efficient Blelloch algorithm O(n)
        * the size of odata should be BLOCK_SIZE * gird_size, 
        * so that need to padding the last block
        * @parms
        * n: the number of blocks in idata
        * s: the stride of the block in idata
        */
        __global__ void kernExclusiveScanPerBlock(uint32_t n, int s, int* odata, const int* idata){
            uint32_t tid = threadIdx.x;
            uint32_t tid2 = tid*2;
            uint32_t global_base = blockIdx.x * BLOCK_SIZE;
            __shared__ int buffer[BLOCK_SIZE+BLOCK_SIZE/NUM_BANKS];

            // Load the data to the shared memory
            uint32_t ai = tid;
            uint32_t bi = tid + BLOCK_SIZE/2;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            uint32_t gid = global_base + tid;
            uint32_t idx_in = gid * s - 1;// get last element of each pre block
            buffer[ai] = (0 < gid && gid < n) ? idata[idx_in] : 0;

            gid = global_base + tid + BLOCK_SIZE/2;
            idx_in = gid * s - 1;
            buffer[bi] = (0 < gid && gid < n) ? idata[idx_in] : 0;
            __syncthreads();
            
            // Up-Sweep (Reduce)
            int offset = 1;
            for(int stride = blockDim.x; stride > 0; stride >>= 1){
                if(tid < stride){
                    int ai = offset * (tid2 + 1) - 1;
                    int bi = offset * (tid2 + 2) - 1;
                    ai += CONFLICT_FREE_OFFSET(ai);
                    bi += CONFLICT_FREE_OFFSET(bi);
                    buffer[bi] += buffer[ai];
                }
                offset <<= 1;
                __syncthreads();
            }

            // Down-Sweep (Distribute)
            int sum = 0;
            if(tid == blockDim.x - 1){
                ai = BLOCK_SIZE - 1 + CONFLICT_FREE_OFFSET(BLOCK_SIZE - 1);
                sum = buffer[ai];
                buffer[ai] = 0;
            }
            __syncthreads();

            offset = blockDim.x;
            for(int stride = 1; stride <= blockDim.x; stride <<= 1){
                if(tid < stride){
                    int ai = offset * (tid2 + 1) - 1;
                    int bi = offset * (tid2 + 2) - 1;
                    ai += CONFLICT_FREE_OFFSET(ai);
                    bi += CONFLICT_FREE_OFFSET(bi);
                    int t = buffer[ai];
                    buffer[ai] = buffer[bi];
                    buffer[bi] += t;
                }
                offset >>= 1;
                __syncthreads();

            }
            // Write the result to the output
            ai = tid+1;
            bi = tid+1+BLOCK_SIZE/2;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            gid = global_base + tid;
            odata[gid] = buffer[ai];
            gid = global_base + tid + BLOCK_SIZE/2;
            odata[gid] = (tid < blockDim.x - 1) ? buffer[bi] : sum;
        }

        void scanOnDeviceShared(uint32_t n, int*dev_odata,const int*dev_idata){

            // Malloc different level of memory
            int level = (ilog2ceil(n) + 7) / 8;
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
            dim3 half_block_size(BLOCK_SIZE/2);
            kernExclusiveScanPerBlock << <grid_size[0], half_block_size >> > (n, 1, dev_odata, dev_idata);
            for(int i = 1; i < level; i++){
                kernExclusiveScanPerBlock<<<grid_size[i], half_block_size>>>(grid_size[i-1], BLOCK_SIZE, dev_ptr[i], dev_ptr[i-1]);
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

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(uint32_t n, int *odata, const int *idata, bool isShared) {
            if(isShared){
                int* dev_idata, *dev_odata;
                uint32_t size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
                size *= BLOCK_SIZE;
                cudaMalloc((void**)&dev_idata, size * sizeof(int));
                checkCUDAError("cudaMalloc dev_idata failed!");
                cudaMemset(dev_idata, 0, size * sizeof(int));
                cudaMalloc((void**)&dev_odata, size * sizeof(int));
                checkCUDAError("cudaMalloc dev_odata failed!");
                cudaMemset(dev_odata, 0, size * sizeof(int));

                cudaMemcpy(dev_idata, idata, n*sizeof(int), cudaMemcpyHostToDevice);
                checkCUDAError("cudaMemcpy idata failed!");

                timer().startGpuTimer();
                scanOnDeviceShared(size, dev_odata, dev_idata);
                timer().endGpuTimer();

                cudaMemcpy(odata, dev_odata, n*sizeof(int), cudaMemcpyDeviceToHost);
                checkCUDAError("cudaMemcpy odata failed!");

                cudaFree(dev_idata);
                cudaFree(dev_odata);
            }else{
                int level = ilog2ceil(n);
                uint32_t size = 1 << level;
                if(n>size){
                    printf("The size is less than 2^k, need to padding the array\n");
                }
                int* dev_data;
                cudaMalloc((void**)&dev_data, size * sizeof(int));
                checkCUDAError("cudaMalloc dev_data failed!");
                cudaMemset(dev_data, 0, size * sizeof(int));
                cudaMemcpy(dev_data, idata, n*sizeof(int), cudaMemcpyHostToDevice);
                checkCUDAError("cudaMemcpy idata failed!");
                timer().startGpuTimer();
                scanOnDeviceGlobal(size, dev_data);
                timer().endGpuTimer();
                cudaMemcpy(odata, dev_data, n*sizeof(int), cudaMemcpyDeviceToHost);
                checkCUDAError("cudaMemcpy odata failed!");
                cudaFree(dev_data);
            }
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
        int compact(uint32_t n, int *odata, const int *idata, bool isShared) {
            uint32_t count = 0;
            int final_bool = 0;
            int* dev_idata, *dev_bools, *dev_indices;
            int* dev_odata;

            int gird_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            uint32_t size = gird_size * BLOCK_SIZE;

            int level = ilog2ceil(n);
            uint32_t new_size = 1 << level;
            // Malloc the memory
            cudaMalloc((void**)&dev_idata, size * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemset(dev_idata, 0, size * sizeof(int));

            cudaMalloc((void**)&dev_bools, size * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed!");
            cudaMemset(dev_bools, 0, size * sizeof(int));

            if(isShared){
                cudaMalloc((void**)&dev_indices, size * sizeof(int));
                checkCUDAError("cudaMalloc dev_indices failed!");
                cudaMemset(dev_indices, 0, size * sizeof(int));
            }else{
                cudaMalloc((void**)&dev_indices, new_size * sizeof(int));
                checkCUDAError("cudaMalloc dev_indices failed!");
                cudaMemset(dev_indices, 0, new_size * sizeof(int));
            }


            cudaMalloc((void**)&dev_odata, size * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMemset(dev_odata, 0, size * sizeof(int));

            // Copy the data to the device
            cudaMemcpy(dev_idata, idata, n*sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            // Map to boolean
            if(isShared){
                Common::kernMapToBoolean<<<gird_size, BLOCK_SIZE>>>(n, dev_bools, dev_idata);
                scanOnDeviceShared(size, dev_indices, dev_bools);
            }
            else{
                Common::kernMapToBoolean<<<gird_size, BLOCK_SIZE>>>(n, dev_indices, dev_idata);
                scanOnDeviceGlobal(new_size, dev_indices);
            }
            // Scatter the data
            Common::kernScatter<<<gird_size, BLOCK_SIZE>>>(n, dev_odata, dev_idata, dev_indices);
            timer().endGpuTimer();

            // Copy the data back to the host
            cudaMemcpy(odata, dev_odata, n*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&count, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            count += idata[n-1] != 0 ? 1 : 0;
            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_odata);
            return count;
        }
    }
}
