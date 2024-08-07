#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Thrust {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(uint32_t n, int *odata, const int *idata);

        int compact(uint32_t n, int *odata, const int *idata);
    }
}
