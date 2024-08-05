#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
        * CPU scan core function.
        * This function runs without starting CPU timer.
        */
        void scan_core(int n, int *odata, const int *idata) {
            odata[0] = 0;
            for (int i = 1; i < n; i++) {
                odata[i] = odata[i - 1] + idata[i - 1];
            }
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            scan_core(n, odata, idata);
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[count] = idata[i];
                    count++;
                }
            }
            timer().endCpuTimer();
            return count;
        }
        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int* boolBuffer = new int[n];
            int* scanBuffer = new int[n];
            // Map to boolean
            for (int i = 0; i < n; i++) {
                boolBuffer[i] = idata[i] != 0 ? 1 : 0;
            }
            // Exclusive scan
            scan_core(n, scanBuffer, boolBuffer);
            int count = scanBuffer[n-1] + boolBuffer[n-1];

            // Scatter
            for (int i = 0; i < n; i++) {
                if (boolBuffer[i] == 1) {
                    odata[scanBuffer[i]] = idata[i];
                }
            }
            
            delete[] boolBuffer;
            delete[] scanBuffer;
            timer().endCpuTimer();
            return count;
        }
    }
}
