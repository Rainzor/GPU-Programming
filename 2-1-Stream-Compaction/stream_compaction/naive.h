#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Naive {
        StreamCompaction::Common::PerformanceTimer& timer();
        __global__ void kernInclusiveScanGlobal(uint32_t n, int level, int* odata, const int* idata);
        __global__ void kernExclusiveScanPerBlock(uint32_t n, int s, int* odata, const int* idata);
        void scanOnDeviceShared(uint32_t n, int*dev_odata, const int*dev_idata);
        void scanOnDeviceGlobal(uint32_t n, int*dev_odata, int*dev_idata);
        void scan(uint32_t n, int *odata, const int *idata, bool isShared = false);
        int compact(uint32_t n, int *odata, const int *idata, bool isShared = false);
    }
}
