/** \file */

#ifndef _CUCKOOHASH_UTIL_HH
#define _CUCKOOHASH_UTIL_HH

#include <exception>
#include <pthread.h>
#include <thread>
#include <vector>
#include <numaif.h>
#include <sys/mman.h>
#include <assert.h>
#include <stdio.h>
#include "cuckoohash_config.hh" // for LIBCUCKOO_DEBUG

#if LIBCUCKOO_DEBUG
#  define LIBCUCKOO_DBG(fmt, args...)                                   \
     fprintf(stderr, "\x1b[32m""[libcuckoo:%s:%d:%lu] " fmt"" "\x1b[0m", \
             __FILE__,__LINE__, (unsigned long)pthread_self(), ##args)
#else
#  define LIBCUCKOO_DBG(fmt, args...)  do {} while (0)
#endif

// For enabling certain methods based on a condition. Here's an example.
// ENABLE_IF(some_cond, type, static, inline) method() {
//     ...
// }
#define ENABLE_IF(preamble, condition, return_type)                     \
    template <class Bogus=void*>                                        \
    preamble typename std::enable_if<sizeof(Bogus) &&                   \
        condition, return_type>::type

/**
 * Thrown when an automatic expansion is triggered, but the load factor of the
 * table is below a minimum threshold, which can be set by the \ref
 * cuckoohash_map::minimum_load_factor method. This can happen if the hash
 * function does not properly distribute keys, or for certain adversarial
 * workloads.
 */
class libcuckoo_load_factor_too_low : public std::exception {
public:
    /**
     * Constructor
     *
     * @param lf the load factor of the table when the exception was thrown
     */
    libcuckoo_load_factor_too_low(const double lf)
        : load_factor_(lf) {}

    virtual const char* what() const noexcept {
        return "Automatic expansion triggered when load factor was below "
            "minimum threshold";
    }

    /**
     * @return the load factor of the table when the exception was thrown
     */
    double load_factor() {
        return load_factor_;
    }
private:
    const double load_factor_;
};

/**
 * Thrown when an expansion is triggered, but the hashpower specified is greater
 * than the maximum, which can be set with the \ref
 * cuckoohash_map::maximum_hashpower method.
 */
class libcuckoo_maximum_hashpower_exceeded : public std::exception {
public:
    /**
     * Constructor
     *
     * @param hp the hash power we were trying to expand to
     */
    libcuckoo_maximum_hashpower_exceeded(const size_t hp)
        : hashpower_(hp) {}

    virtual const char* what() const noexcept {
        return "Expansion beyond maximum hashpower";
    }

    /**
     * @return the hashpower we were trying to expand to
     */
    size_t hashpower() {
        return hashpower_;
    }
private:
    const size_t hashpower_;
};

// Allocates an array of the given size and value-initializes each element with
// the 0-argument constructor
template <class T, class Alloc>
T* create_array(const size_t size) {
    Alloc allocator;
    //T* arr = allocator.allocate(size);
    size_t bytes = size * sizeof(T);
    T* arr = (T*) mmap(NULL, bytes,
            PROT_READ|PROT_WRITE,
            // don't use hugetlb so we can spread the table out
            MAP_ANONYMOUS|MAP_PRIVATE|MAP_NORESERVE,
            -1, 0);
    if ( arr == MAP_FAILED ) abort();
    // spread memory for hash table across nodes
    const size_t nnodes = 32; // XXX max nodes one may encounter
#if defined(CUCKOO_INTERLEAVE)
    // FIXME find a path to specify which sockets to interleave, like
    // with cuckoohash_map
#warning Spreading lock table among all sockets.
    size_t mask = (1ul<<nnodes)-1;
#elif defined(CUCKOO_BIND0)
    size_t mask = 1ul;
#endif
    printf("%s:%s: array %.2fmB allocated at %p numa mask 0x%lx\n",
            __FILE__, __func__, (float)bytes/(1ul<<20), arr, mask);
    if ( 0 != mbind(arr, bytes, MPOL_INTERLEAVE,
            &mask, nnodes+1, MPOL_MF_STRICT) ) abort();
    memset((void*)arr, 0, bytes);
    // Initialize all the elements, safely deallocating and destroying
    // everything in case of error.
    size_t i;
    try {
        for (i = 0; i < size; ++i) {
            allocator.construct(&arr[i]);
        }
    } catch (...) {
        for (size_t j = 0; j < i; ++j) {
            allocator.destroy(&arr[j]);
        }
        //allocator.deallocate(arr, size);
        munmap(arr, size);
        throw;
    }
    return arr;
}

// Destroys every element of an array of the given size and then deallocates the
// memory.
template <class T, class Alloc>
void destroy_array(T* arr, const size_t size) {
    Alloc allocator;
    for (size_t i = 0; i < size; ++i) {
        allocator.destroy(&arr[i]);
    }
    if (munmap(arr, size * sizeof(T))) abort();
    //allocator.deallocate(arr, size);
}

// executes the function over the given range split over num_threads threads
template <class F>
static void parallel_exec(size_t start, size_t end,
                          size_t num_threads, F func) {
    size_t work_per_thread = (end - start) / num_threads;
    std::vector<std::thread> threads(num_threads);
    for (size_t i = 0; i < num_threads - 1; ++i) {
        threads[i] = std::thread(func, start, start + work_per_thread);
        start += work_per_thread;
    }
    threads[num_threads - 1] = std::thread(func, start, end);
    for (std::thread& t : threads) {
        t.join();
    }
}

#endif // _CUCKOOHASH_UTIL_HH
