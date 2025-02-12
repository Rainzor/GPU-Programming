#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(uint32_t n, int *odata, const int *idata);

        int compactWithoutScan(uint32_t n, int *odata, const int *idata);

        int compactWithScan(uint32_t n, int *odata, const int *idata);
    }
}
