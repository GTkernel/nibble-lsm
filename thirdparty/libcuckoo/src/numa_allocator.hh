// Copyright (c) 2016, Alexander Merritt
//
// Use mmap to allocate the backing memory for something.
// We use it here for the Bucket vector in the cuckoo table to
// distribute all pages across sockets (or bind them all to one).
//
// Used for a std::vector, this will allocate/manage the underlying
// array it uses. Class is stateless, so the destructor does nothing.
//
// We assume use of 4KB pages to allow best distribution of memory.

// NOTE: match this with what the Rust Nibble implementation uses. '3'
// is 'info' level.
#define __NIBBLE_DBG_ENV  "NIBDEBUG"
#define __NIBBLE_DBG_LVL  3

#include <numaif.h>     // mbind
#include <stdint.h>     // uintptr_t
#include <stdio.h>      // printf
#include <stdlib.h>     // getenv, strtol
#include <string.h>     // memset
#include <sys/mman.h>   // mmap

#ifndef __unused
#define __unused    __attribute__((unused))
#endif

#if defined(CUCKOO_INTERLEAVE)
#elif defined(CUCKOO_BIND0)
#else
#error Define CUCKOO_INTERLEAVE or _BIND0
#endif

template <typename T>
class NumaAllocator : public std::allocator<T> {
    public:
        typedef size_t size_type;
        typedef off_t offset_type;

        template<typename _Tp1> struct rebind {
            typedef NumaAllocator<_Tp1> other;
        };

        size_t mask, nnodes;
        int lvl; // debug level for printing

        NumaAllocator() : mask(0ul), nnodes(0ul), lvl(0) {
                char *env = getenv(__NIBBLE_DBG_ENV);
                lvl = strtol(env, NULL, 10);
                if (env && lvl >= __NIBBLE_DBG_LVL)
                    printf("%s::%s: default constructor\n",
                            __FILE__, __func__);
            }

#if defined(CUCKOO_INTERLEAVE)
        NumaAllocator(size_t mask_, size_t nnodes_)
            : mask(mask_), nnodes(nnodes_), lvl(0) {
#elif defined(CUCKOO_BIND0)
        NumaAllocator(__unused size_t mask_, size_t nnodes_)
            : mask(1), nnodes(nnodes_), lvl(0) {
#endif
                char *env = getenv(__NIBBLE_DBG_ENV);
                lvl = strtol(env, NULL, 10);
                if (env && lvl >= __NIBBLE_DBG_LVL)
                    printf("%s::%s: mask 0x%lx nnodes %lu\n",
                            __FILE__, __func__, mask, nnodes);
            }

        void set(size_t mask_, size_t nnodes_) {
            mask = mask_;
            nnodes = nnodes_;
            if (lvl >= __NIBBLE_DBG_LVL)
                printf("%s::%s: mask 0x%lx nnodes %lu\n",
                        __FILE__, __func__, mask, nnodes);
        }

        T* allocate(size_type n, __unused const void *hint=0) {
            size_t bytes = n * sizeof(T);
            void *p = mmap(nullptr, bytes,
                    PROT_READ|PROT_WRITE,
                    MAP_ANONYMOUS|MAP_PRIVATE|MAP_NORESERVE,
                    -1, 0);
            if ( p == MAP_FAILED ) abort();
            if (lvl >= __NIBBLE_DBG_LVL)
                printf("%s::%s: %.2fmB at %p\n",
                        __FILE__, __func__,
                        (float)bytes/(1ul<<20), p);
#if defined(CUCKOO_INTERLEAVE)
            if ( 0 != mbind(p, bytes, MPOL_INTERLEAVE,
                        &mask, nnodes+1, MPOL_MF_STRICT) ) abort();
#elif defined(CUCKOO_BIND0)
            assert( mask == 1 );
            if ( 0 != mbind(p, bytes, MPOL_BIND,
                        &mask, nnodes+1, MPOL_MF_STRICT) ) abort();
#endif
            memset(p, 0, bytes);
            return static_cast<T*>(p);
        }
        void deallocate(T* p, size_t n) {
            size_t bytes = n * sizeof(T);
            if (lvl >= __NIBBLE_DBG_LVL)
                printf("%s::%s: %.2fmB at %p\n",
                        __FILE__, __func__,
                        (float)bytes/(1ul<<20), p);
            // simple hack to ensure we're not deallocating at
            // sub-page granularity
            if ((uintptr_t)p & ((1ul<<12)-1)) abort();
            if ( 0 != munmap((void*)p, bytes) ) abort();
        }

        // don't seem to need these, unless we want further
        // construction based on mmap...
#if 0
        void construct(T* p, const T* t) {
            cout << __func__ << "\t" << p << " " << t << endl;
            new ((void*)p) T(t);
        }

        void destroy(T* p) {
            cout << __func__ << "\t\t" << p << endl;
            ((T*)p)->~T();
        }
#endif
};
