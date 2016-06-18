// Copyright (c) 2016, Alexander Merritt
// Test code for NumaAllocator
// clang++ -std=c++11 test.cc -o test -lnuma
// NIBDEBUG=N ./test
// where N is 1,2,3,4,5 (3 enables printing)

#include <vector>

// include last
#include "numa_allocator.hh"

int main() {
    const size_t num_sockets = 2;
    const size_t mask = (1ul<<num_sockets)-1;

    std::vector<int, NumaAllocator<int>>
        v(1ul<<20, 0, NumaAllocator<int>(mask,num_sockets));

    return 0;
}
