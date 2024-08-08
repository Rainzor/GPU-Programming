/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include "testing_helpers.hpp"
#include <cstdio>
#include <fstream>
#include <stream_compaction/cpu.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/thrust.h>

const int SIZE = 1 << 29;      // feel free to change the size of array
const int NPOT = SIZE / 2 + 1; // Non-Power-Of-Two
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];

const std::string filename_scan = "scan_1024.txt";
const std::string filename_compact = "compact_1024.txt";
void printTest() {
    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");
    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
     zeroArray(SIZE, b);
     printDesc("cpu scan, power-of-two");
     StreamCompaction::CPU::scan(SIZE, b, a);
     printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
     printArray(SIZE, b, true);

     zeroArray(SIZE, c);
     printDesc("cpu scan, non-power-of-two");
     StreamCompaction::CPU::scan(NPOT, c, a);
     printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
     printArray(NPOT, b, true);
     printCmpResult(NPOT, b, c);

     zeroArray(SIZE, c);
     printDesc("naive scan, power-of-two, on Global Memory");
     StreamCompaction::Naive::scan(SIZE, c, a, false);
     printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
     printArray(SIZE, c, true);
     printCmpResult(SIZE, b, c);

     zeroArray(SIZE, c);
     printDesc("naive scan, non-power-of-two, on Global Memory");
     StreamCompaction::Naive::scan(NPOT, c, a, false);
     printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
     printArray(NPOT, c, true);
     printCmpResult(NPOT, b, c);

     zeroArray(SIZE, c);
     printDesc("naive scan, power-of-two, on Shared Memory");
     StreamCompaction::Naive::scan(SIZE, c, a, true);
     printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
     printArray(SIZE, c, true);
     printCmpResult(SIZE, b, c);

     zeroArray(SIZE, c);
     printDesc("naive scan, non-power-of-two, on Shared Memory");
     StreamCompaction::Naive::scan(NPOT, c, a, true);
     printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
     printArray(NPOT, c, true);
     printCmpResult(NPOT, b, c);

     zeroArray(SIZE, c);
     printDesc("work-efficient scan, power-of-two, on Global Memory");
     StreamCompaction::Efficient::scan(SIZE, c, a, false);
     printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
     printArray(SIZE, c, true);
     printCmpResult(SIZE, b, c);

     zeroArray(SIZE, c);
     printDesc("work-efficient scan, non-power-of-two, on Global Memory");
     StreamCompaction::Efficient::scan(NPOT, c, a, false);
     printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
     printArray(NPOT, c, true);
     printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two, on Shared Memory");
    StreamCompaction::Efficient::scan(SIZE, c, a, true);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two, on Shared Memory");
    StreamCompaction::Efficient::scan(NPOT, c, a, true);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    // printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    // printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

     // Compaction tests

     // genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
     a[SIZE - 1] = 0;
     printArray(SIZE, a, true);

     int count, expectedCount, expectedNPOT;

     /*initialize b using StreamCompaction::CPU::compactWithoutScan you implement
     We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.*/
     zeroArray(SIZE, b);
     printDesc("cpu compact without scan, power-of-two");
     count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
     printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
     expectedCount = count;
     printArray(count, b, true);
     printCmpLenResult(count, expectedCount, b, b);

     zeroArray(SIZE, c);
     printDesc("cpu compact without scan, non-power-of-two");
     count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
     printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
     expectedNPOT = count;
     printArray(count, c, true);
     printCmpLenResult(count, expectedNPOT, b, c);

     zeroArray(SIZE, c);
     printDesc("cpu compact with scan");
     count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
     printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
     printArray(count, c, true);
     printCmpLenResult(count, expectedCount, b, c);

     zeroArray(SIZE, c);
     printDesc("naive compact, power-of-two, on Global Memory");
     count = StreamCompaction::Naive::compact(SIZE, c, a, false);
     printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
     // printArray(count, c, true);
     printCmpLenResult(count, expectedCount, b, c);

     zeroArray(SIZE, c);
     printDesc("naive compact, non-power-of-two, on Global Memory");
     count = StreamCompaction::Naive::compact(NPOT, c, a, false);
     printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
     // printArray(count, c, true);
     printCmpLenResult(count, expectedNPOT, b, c); 

     zeroArray(SIZE, c);
     printDesc("naive compact, power-of-two, on Shared Memory");
     count = StreamCompaction::Naive::compact(SIZE, c, a, true);
     printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
     // printArray(count, c, true);
     printCmpLenResult(count, expectedCount, b, c);

     zeroArray(SIZE, c);
     printDesc("naive compact, non-power-of-two, on Shared Memory");
     count = StreamCompaction::Naive::compact(NPOT, c, a, true);
     printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
     // printArray(count, c, true);
     printCmpLenResult(count, expectedNPOT, b, c);    

     zeroArray(SIZE, c);
     printDesc("work-efficient compact, power-of-two, on Global Memory");
     count = StreamCompaction::Efficient::compact(SIZE, c, a, false);
     printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
     // printArray(count, c, true);
     printCmpLenResult(count, expectedCount, b, c);

     zeroArray(SIZE, c);
     printDesc("work-efficient compact, non-power-of-two, on Global Memory");
     count = StreamCompaction::Efficient::compact(NPOT, c, a, false);
     printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
     // printArray(count, c, true);
     printCmpLenResult(count, expectedNPOT, b, c);

     zeroArray(SIZE, c);
     printDesc("work-efficient compact, power-of-two, on Shared Memory");
     count = StreamCompaction::Efficient::compact(SIZE, c, a, true);
     printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
     printCmpLenResult(count, expectedCount, b, c);

     zeroArray(SIZE, c);
     printDesc("work-efficient compact, non-power-of-two, on Shared Memory");
     count = StreamCompaction::Efficient::compact(NPOT, c, a, true);
     printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
     printCmpLenResult(count, expectedNPOT, b, c);

     zeroArray(SIZE, c);
     printDesc("thrust compact, power-of-two");
     count = StreamCompaction::Thrust::compact(SIZE, c, a);
     printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
     printCmpLenResult(count, expectedCount, b, c);

     zeroArray(SIZE, c);
     printDesc("thrust compact, non-power-of-two");
     count = StreamCompaction::Thrust::compact(NPOT, c, a);
     printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
     printCmpLenResult(count, expectedNPOT, b, c);
     system("pause"); // stop Win32 console from closing on exit
}

void writeCompact(std::ofstream &ofs, int size) {
    zeroArray(size, b);
    zeroArray(size, c);

    int count = StreamCompaction::CPU::compactWithoutScan(size, b, a);
    ofs << size << "," << StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation() << ",";

    count = StreamCompaction::CPU::compactWithScan(size, c, a);
    ofs << StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation() << ",";

    count = StreamCompaction::Naive::compact(size, c, a, false);
    ofs << StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation() << ",";

    count = StreamCompaction::Naive::compact(size, c, a, true);
    ofs << StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation() << ",";

    count = StreamCompaction::Efficient::compact(size, c, a, false);
    ofs << StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation() << ",";

    count = StreamCompaction::Efficient::compact(size, c, a, true);
    ofs << StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation() << ",";

    count = StreamCompaction::Thrust::compact(size, c, a);
    ofs << StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation() << std::endl;
}

void writeScan(std::ofstream &ofs, int size) {
    zeroArray(size, b);
    zeroArray(size, c);

    StreamCompaction::CPU::scan(size, b, a);
    ofs << size << "," << StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation() << ",";

    StreamCompaction::Naive::scan(size, c, a, false);
    ofs << StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation() << ",";

    StreamCompaction::Naive::scan(size, c, a, true);
    ofs << StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation() << ",";

    StreamCompaction::Efficient::scan(size, c, a, false);
    ofs << StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation() << ",";

    StreamCompaction::Efficient::scan(size, c, a, true);
    ofs << StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation() << ",";

    StreamCompaction::Thrust::scan(size, c, a);
    ofs << StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation() << std::endl;
}

void createDateset() {
    std::ofstream ofs_scan, ofs_compact;
    ofs_scan.open(filename_scan);
    if (!ofs_scan.is_open()) {
        std::cerr << "Failed to open " << filename_scan << std::endl;
        return;
    }

    ofs_compact.open(filename_compact);
    if (!ofs_compact.is_open()) {
        std::cerr << "Failed to open " << filename_compact << std::endl;
        return;
    }

    ofs_scan << "size,cpu,naive,naive_shared,efficient,efficient_shared,thrust" << std::endl;
    ofs_compact << "size,cpu,cpu_scan,naive,naive_shared,efficient,efficient_shared,thrust" << std::endl;
    for (int d = 12; d <= 29; d++) {
        int size = 1 << d;
        int non_power_of_two = size - size / 8*3;
        writeScan(ofs_scan, non_power_of_two);
        writeScan(ofs_scan, size);
        writeCompact(ofs_compact, non_power_of_two);
        writeCompact(ofs_compact, size);
    }
    ofs_scan.close();
}

int main(int argc, char *argv[]) {
    // Scan tests
    genArray(SIZE - 1, a, 50); // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    printTest();
    // createDateset();

    delete[] a;
    delete[] b;
    delete[] c;
}
