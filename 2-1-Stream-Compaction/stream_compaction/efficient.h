#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        __global__ void kernExclusiveScanPerBlock(uint32_t n, int s, int* odata, const int* idata);
        __global__ void kernUpSweepGlobal(uint32_t n, int level, int* data);
        __global__ void kernDownSweepGlobal(uint32_t n, int level, int* data);
        void scanOnDeviceGlobal(uint32_t n, int*dev_data);
        void scanOnDeviceShared(uint32_t n, int*dev_odata,const int*dev_idata);
        void scan(uint32_t n, int *odata, const int *idata, bool isSharedMemory=false);
        int compact(uint32_t n, int *odata, const int *idata, bool isSharedMemory = false);
    }
}
