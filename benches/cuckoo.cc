#define _GNU_SOURCE
#include <sched.h>

#include <iostream>
#include <cstdlib>
#include <mutex>
#include <cuckoohash_map.hh>

using namespace std;

extern uint64_t CityHash64(const char *buf, size_t len);

class U64Hasher {
    public:
        size_t inline operator()(uint64_t k) const {
            return (size_t)CityHash64((char*)&k, sizeof(k));
        }
};

void pin_cpu(int cpu) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    assert( 0 == sched_setaffinity(0, /*me*/
                sizeof(mask), &mask) );
}

using K = uint64_t;
using V = uint64_t;

int main() {
    pin_cpu(0);
    cuckoohash_map<K,V,U64Hasher> *map =
            new cuckoohash_map<K,V,U64Hasher>(0x1ul, 2ul);
    if (!map) {
        cerr << "Error: OOM" << endl;
        return 1;
    }
    const size_t n = 1ul<<23;
    cout << ">> Reserving " << ((float)n/1e6) << " mil. items" << endl;
    map->reserve(n<<1);
    cout << ">> Inserting items" << endl;
    for (uint64_t i = 0; i < n; i++)
        map->insert(i, i);

    size_t k = 7877ul, ops = (n<<3);
    cout << ">> Warmup" << endl;
    for (uint64_t i = 0; i < ops; i++)
        k = 7877 * map->find(k%n);

    cout << ">> Executing" << endl;
    struct timespec t1,t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    for (uint64_t i = 0; i < ops; i++)
    //while (1)
        k = 7877 * map->find(k%n);
        //k = map->find((rand()+k)%n);
    clock_gettime(CLOCK_MONOTONIC, &t2);

    float sec = ((t2.tv_sec*1e9 - t2.tv_nsec)
        - (t1.tv_sec*1e9 - t1.tv_nsec)) / 1e9;
    cout << "Sec:  " << sec << endl;
    float kops = (((float)ops)/1e3) / sec;
    cout << "Perf: " << kops << endl;
    return 0;
}
