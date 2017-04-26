/*
 * frag.c
 *
 * Copyright (c) 2016 Alexander Merritt.
 *
 * Measure the RSS of a process which allocates/frees heap data and
 * compare to our tally of bytes allocated via malloc/free.
 *
 * Given max capacity M and object sizes O1 and O2, and a tally of
 * bytes representing how many we've currently allocated with
 * malloc/free:
 *
 *      1. allocate 4*M using objects size O1
 *          1.a free randomly if M exceeded
 *      2. [optional] free objects at random while tally > 0.1*M
 *      3. allocate 4*M using objects size O2
 *          3.a free randomly if M exceeded
 *
 * At the end, read total RSS and display ratio between it and tally.
 * Method matches the USENIX FAST'14 paper authored by Rumble.
 *
 * There are tweaks in the code to disable the free phase, change the
 * percent of objects it releases, and to use ranges of object sizes
 * (instead of two statics).  You may also specify an M' to use during
 * phase 3.
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <sys/mman.h>
#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include "util.h"

// SET THIS APPROPRIATELY
#define NSOCKETS    16ul

// Enable step 2
#define FREE_PHASE
#define FREE_PCT    0.9

// Instead of static sizes, use this range. Command arguments are thus
// ignored for O1 and O2
//#define USE_RANGE
#define RANGE1      50ul,100ul
#define RANGE2      5000ul,15000ul

// Use alternate M for step 3. Specified as a product of the input M.
// e.g. SECOND_M=0.5 -> if M=4 then M'=2
//#define SECOND_M    (0.9)

typedef uint64_t u64;
typedef uint32_t u32;
typedef int32_t  i32;

//
// We do fragmentation testing on socket 0 only, as this does not
// measure performance.
//

// Exposed Rust functions for use to use a LSM object.
extern void kvs_init(size_t cap, size_t nitem);
extern i32 kvs_put(u64 key, u64 bytes);
extern i32 kvs_del(u64 key);

#if defined(USE_RANGE)
// return random value between low,high inclusive
static inline
size_t range(size_t low, size_t high) {
    return (lrand48() % (high-low+1)) + low;
}
#endif

#if 0
void print_mem(const char* text, size_t expected) {
    size_t heap = __atomic_load_n(&total_heap_alloc, __ATOMIC_RELAXED);
    if (!text)
        text = "";
    if (expected == 0)
        printf("%sMEM ratio 1.0\n", text);
    else
        printf("%sMEM ratio %.2f\n", text, (float)heap/(float)expected);
}
#endif

void test(long O1, long O2, long M) {
#if defined(USE_RANGE)
    printf("range: (%lu,%lu) -> (%lu,%lu)\n",
            RANGE1, RANGE2);
#endif

    const long n = 1l<<28;
    struct alloc *entries;
    printf("frag: alloc array bytes %lu\n",
            (size_t)n*sizeof(*entries));
    entries = mmap(NULL, (size_t)n*sizeof(*entries),
            PROT_READ|PROT_WRITE,
            MAP_PRIVATE|MAP_ANONYMOUS|MAP_POPULATE,
            -1, 0);
    if (entries == MAP_FAILED) {
        perror("mmap");
        exit(EXIT_FAILURE);
    }

    struct drand48_data drand_buf;
    srand48_r(((long)rdtscp() << 30) ^ (long)&entries, &drand_buf);

    uint64_t key = 0l;

    long size1 = O1;
    long size2 = O2;

    long at = 0l;
    long max = M;
    long cur = 0l;

    long pushed = 0l;
    const long push = 2 * max / NSOCKETS;
    long keys_inserted = 0l;

    long bytes;
    long pushed_since = 0l;

    printf("push max: %.2f GiB\n",
            (float)push /(float)(1ul<<30));

#if defined(USE_RANGE)
#else
    printf("size1 %lu size2 %lu max %lu\n",
            size1, size2, max);
    fflush(stdout); fflush(stderr);
#endif

    for (int iter = 0; iter < 2; iter++) {
        pushed = 0;

        switch (iter) {
            case 0: bytes = size1; break;
            case 1: bytes = size2; break;
            default: abort();
        }

#if defined(SECOND_M)
        if (iter == 1) {
            max = M * SECOND_M;
            printf("secondary M: %lu\n", max);
        }
#endif

        while (pushed < push) {
#if defined(USE_RANGE)
            // if you want to test ranges of sizes
            bytes = (iter == 0) ? range(RANGE1)
                                : range(RANGE2);
#endif
            bool paused_once = false;
again:;
            key = rdrand();
            //assert( (entries[at].addr = malloc((size_t)bytes)) );
            //memset(entries[at].addr, 0xad, (size_t)bytes);
            while (cur >= max ||
                    (0 != kvs_put((u64)key,(u64)bytes))) {
                if (at <= 0) {
                    if (!paused_once) {
                        sleep(5);
                        paused_once = true;
                        goto again;
                    }
                    printf("oops: at=0\n");
                    fflush(stdout);
                    exit(EXIT_FAILURE);
                }

                long idx = lrand48() % at;
                if ( 0 != kvs_del(entries[idx].key) ) {
                    printf("oops: del failed idx %lu key %lu\n",
                            idx, entries[idx].key);
                    fflush(stdout);
                    exit(EXIT_FAILURE);
                }
                cur -= entries[idx].bytes;
                entries[idx] = entries[--at];
                keys_inserted--;

                // change the key before trying again
                key = rdrand();
            }
            paused_once = false;
            entries[at].bytes = bytes;
            entries[at].key = key;
            cur += bytes;
            keys_inserted++;

            if (++at >= n) {
                puts("need more n");
                fflush(stdout);
                exit(EXIT_FAILURE);
            }

            pushed += bytes;

            if ((pushed - pushed_since) > (1l<<30)) {
                printf("    cur %.2f GiB pushed %.2f GiB\n",
                        (float)cur/(float)(1<<30),
                        (float)pushed/(float)(1<<30)
                        );
                fflush(stdout);
                pushed_since = pushed;
            }
        }
        //print_mem("alloc phase: ", (size_t)cur);

        if (iter == 1)
            break;

#if defined(FREE_PHASE)
        printf("free phase, pct: %.3lf\n", (FREE_PCT));
        fflush(stdout);
        // [optional] free 90% before next allocation round
        shuffle(entries, at, &drand_buf);
        while ((float)cur > ((float)max*(1.-(FREE_PCT)))) {
            at--;
            if ( 0 != kvs_del(entries[at].key) ) {
                printf("oops: del failed\n");
                exit(EXIT_FAILURE);
            }
            cur -= entries[at].bytes;
            keys_inserted--;
        }
        //print_mem("free phase: ", (size_t)cur);
        fflush(stdout);
#endif
    }

    // let compaction catch up
    puts("main pause for compaction");
    fflush(stdout);
    sleep(10);

    // final insertion phase to fill up empty slots
    // pause, insert, pause insert, etc.
    for (int i = 0; i < 8; i++) {
        key = rdrand();
        while (0 == kvs_put((u64)key,(u64)bytes)) {
            if (at <= 0) {
                printf("oops: at=0\n");
                fflush(stdout);
                exit(EXIT_FAILURE);
            }

            entries[at].bytes = bytes;
            entries[at].key = key;
            cur += bytes;
            keys_inserted++;

            if (++at >= n) {
                puts("need more n");
                fflush(stdout);
                exit(EXIT_FAILURE);
            }

            pushed += bytes;
            key = rdrand();

            if ((pushed - pushed_since) > (1l<<27)) {
                printf("    cur %.2f GiB pushed %.2f GiB\n",
                        (float)cur/(float)(1<<30),
                        (float)pushed/(float)(1<<30));
                fflush(stdout);
                pushed_since = pushed;
            }
        }
        puts("pause 5");
        fflush(stdout);
        sleep(5);
    }

    sleep(5);
    printf("keys_inserted %ld\n", keys_inserted);
    printf("cur %.2f MiB\n", (float)cur / (float)(1ul<<20));
}

int main(int narg, char *args[]) {
    if (narg != 5) {
        printf("Usage: %s size1 size2 M capacity\n", *args);
        return EXIT_FAILURE;
    }
    long O1 = strtol(args[1], NULL, 10);
    long O2 = strtol(args[2], NULL, 10);
    long M  = strtol(args[3], NULL, 10);
    long cap = strtol(args[4], NULL, 10);
    kvs_init(cap, 1ul<<30);
    test(O1,O2,M);
    exit(EXIT_SUCCESS);
}

