#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(uint32_t n, int *odata, const int *idata) {
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

            thrust::device_vector<int> dv_in{idata, idata + n};
            thrust::device_vector<int> dv_out{odata, odata + n};
          
            timer().startGpuTimer();
            thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            timer().endGpuTimer();
          
            thrust::copy(dv_out.begin(), dv_out.end(), odata);
        }
        struct is_zero {
            __host__ __device__
                bool operator()(const int x) {
                return x == 0;
            }
        };

        int compact(uint32_t n, int *odata, const int *idata) {
            // example: for device_vectors dv_in and dv_out:
            // thrust:: remove_if(dv_in.begin(), dv_in.end(), is_zero());
            
            thrust::device_vector<int> dv_in{idata, idata + n};

            timer().startGpuTimer();
            auto new_end = thrust::remove_if(dv_in.begin(), dv_in.end(), is_zero());
            timer().endGpuTimer();

            int new_size = new_end - dv_in.begin();
            thrust::copy(dv_in.begin(), new_end, odata);

            return new_size;

          
        }
    }
}
