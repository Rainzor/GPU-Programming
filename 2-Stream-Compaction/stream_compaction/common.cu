#include "common.h"

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}


namespace StreamCompaction {
    namespace Common {

        /*
        * Extract the last element of every block  
        * if the size of src array is k*stride, then the size of dst array should be k
        * @parms
        * k: the number of grids for src
        * stride: the stride of the block for src
        * dst: the destination array
        * src: the source array  
        */
        __global__ void kernExtractLastElementPerBlock(int k, int stride, int* dst, const int* src) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx >= k)
                return;
            int lastIdx = (idx + 1) * stride - 1;
            dst[idx] = src[lastIdx];
        }
        /*
        * Add offset to the array, 
        * if size of src arrary is k, then the dst array size should be n = k * stride
        * @parms
        * n: the number of elements in the dst array
        */

        __global__ void kernAddOffset(int n, int* dst, const int* src){
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx >= n)
                return;
            dst[idx] += src[blockIdx.x];
        }

        /**
         * Maps an array to an array of 0s and 1s for stream compaction. Elements
         * which map to 0 will be removed, and elements which map to 1 will be kept.
         */
        __global__ void kernMapToBoolean(int n, int *bools, const int *idata) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx >= n)
				return;
            bools[idx] = idata[idx] != 0 ? 1 : 0;
        }

        /**
         * Performs scatter on an array. That is, for each element in idata,
         * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
         */
        __global__ void kernScatter(int n, int *odata,
                const int *idata, const int *bools, const int *indices) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx >= n)
                return;
            if(bools[idx] == 1){
                odata[indices[idx]] = idata[idx];
			}
        }

    }
}
