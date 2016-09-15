#ifndef __UTIL_H
#define __UTIL_H

#include <linux/mempolicy.h>
#include <numaif.h>
#include <stddef.h>
#include <inttypes.h>
#include <stdbool.h>
#include <assert.h>
#include <sys/mman.h>

typedef unsigned long ul;

extern ul ticks_per_sec;

struct alloc {
    void *addr;
    long bytes;
};

struct mem {
    long vsize, rss;
};

enum {
    MMAP_PROT = PROT_READ|PROT_WRITE,
    MMAP_FLAGS = MAP_PRIVATE|MAP_ANONYMOUS,
};

static inline
void *mmap_alloc(size_t len, int addl_flags) {
    return mmap(NULL, len, MMAP_PROT,
            MMAP_FLAGS|addl_flags, -1, 0);
}

static inline
void interleave(void* addr, size_t len, int sockets) {
    ul mask = (1ul << sockets) - 1;
    int ret = mbind(addr, len,
            MPOL_INTERLEAVE | MPOL_F_STATIC_NODES,
            &mask, sockets+1, MPOL_MF_STRICT);
    if (ret != 0) {
        perror("mbind");
        exit(EXIT_FAILURE);
    }
}

static inline
void fullfence() {
    asm volatile ("sfence" : : : "memory");
}

static inline
uint64_t rdrand() {
    uint64_t r;
    asm ("rdrand %0" : "=r" (r) : : );
    return r;
}

// XXX this is highly Linux-specific
static inline
void rdtscp_id(int *node, int *cpu)
{
    uint64_t ecx = 0;
    __asm__ volatile ("rdtscp"
            : "=c" (ecx)
            : :);
    *node = ecx >> 12;
    *cpu = ecx & ((1<<12)-1);
}

static inline
uint64_t rdtscp(void)
{
    uint64_t rax = 0, rdx = 0;
    __asm__ volatile ("rdtscp"
            : "=a" (rax), "=d" (rdx)
            : : );
    return ((rdx << 32) + rax);
}

static inline
uint64_t rdtsc(void)
{
    uint64_t rax = 0, rdx = 0;
    __asm__ volatile ("rdtsc"
            : "=a" (rax), "=d" (rdx)
            : : );
    return ((rdx << 32) + rax);
}

static inline
uint64_t rdtsc_id(void)
{
    uint64_t rcx = 0;
    __asm__ volatile ("rdtsc"
            : "=c" (rcx)
            : : );
    return rcx & 0x3ff; // max 1k cpus
}

static inline
bool is_pow2(ul value) {
    return 1 == __builtin_popcountl(value);
}

static inline
size_t roundup_align(size_t len, size_t align) {
    assert(is_pow2(align));
    return align*((len + align - 1) / align);
}

struct mem getmem(long less);

void shuffle(struct alloc*, long, struct drand48_data*);
void shuffle_ul(ul*, long, struct drand48_data*);

void print_mem(long less);
void print_mem2(long less, float expected);
void print_mem3(long less, float expected, const char *extra);

size_t now();

void atomic_store(ul *where, ul value);
ul atomic_load(ul *where);
ul atomic_sub_fetch(ul *where, ul value);
ul atomic_add_fetch(ul *where, ul value);

#endif  /* __UTIL_H */
