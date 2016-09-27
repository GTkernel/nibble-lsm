/*
 * This code is verbatim from malloc-tests/src/shuffle/shuffle.c
 * but with some glue to call into Rust (to use Nibble).
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <inttypes.h>
#include <pthread.h>

#include "util.h"
#include "cqueue.h"

#define PRINT_PROGRESS

#if defined(RELEASE_SELF)
#error shuffle benchmark not supported with Nibble
#endif

//////////////////////////////
// Some tunables
//

enum {
    // We need to know this to append to the more performant log.
    CPUS_PER_SOCKET = 6,
    TOTAL_SOCKETS = 2,

    // only begin measuring after this many seconds
    // to wait for test to stabilize
    TEST_SKIP_FIRST_SEC = 10,

    // whole test runs this long
    // make this at least 30 seconds greater than the skip length
    TEST_LEN_SEC = 30 + TEST_SKIP_FIRST_SEC,

    // shift between object sizes every this many seconds
    // make this larger than the test runtime to disable
    SIZE_SHIFT_SEC = 5,

    // total set of batches held by each worker
    NBATCHES = 1<<4,

    // number of objects within each batch
    // large enough to reduce pushing/popping frequency
    NPTRS = 1<<14,
};

// Given command arguments (no need to modify here)
static ul nthreads;
static ul size; // size of object

// Control over object sizes generated. There is a first ('size') and
// a second ('size' * DELTA) with variation around the size in the
// amount of '100*VARIANCE'%.
const float DELTA    = 5.0; // second size relative to first
const float VARIANCE = 0.2; // wiggle room around obj size

//
// End of tunables
//////////////////////////////

ul warmup_barrier;
ul total_ops;
volatile ul stop = 0;

enum status {
    ALLOC_OK = 0,
    FREE_OK,
};

typedef uint64_t u64;
typedef uint32_t u32;

// Exposed Rust functions for use to use a Nibble object.
extern void  nibble_init(size_t cap, size_t nitem);
extern void* nibble_alloc(u64 key, u64 bytes, u32 sock);
// must_exist = 1 -> fail if key not found
extern void  nibble_free(u64 key, int must_exist);

pid_t gettid() {
    return syscall(SYS_gettid);
}

void setup_sizes(ul n, ul *sizes1, ul *sizes2,
        struct drand48_data *rdata) {
    ul size2 = (ul)(size*DELTA);
    ul mod1 = size * VARIANCE;
    ul mod2 = size2 * VARIANCE;
    long s;
    for (ul i = 0; i < n; i++) {
        lrand48_r(rdata, &s);
        sizes1[i] = (s % mod1) + (size - mod1/2);
        lrand48_r(rdata, &s);
        sizes2[i] = (s % mod2) + (size2 - mod2/2);
    }
}

struct batch {
    volatile enum status status;
    // Nibble uses keys, not addresses
    u64 *keys;
};

// nthreads/2 queues
struct worker_args {
    struct cqueue *queues;
    ul id;
};

// nthreads/2 pointers to queues in workers
struct consumer_args {
    struct cqueue **queues;
    ul id;
};

//==--------------------------------------------------------------==//
//      WORKER
//==--------------------------------------------------------------==//

void* worker(void *args_) {
    struct worker_args *args =
        (struct worker_args*)args_;

    // producers get even-numbered CPUs
    int cpu = args->id * 2;
    //printf("producer cpu %lu\n", cpu);
    cpu_set_t mask; CPU_ZERO(&mask); CPU_SET(cpu, &mask);
    assert(0 == sched_setaffinity(0, sizeof(mask), &mask));

    ul many = 0;
    const ul npairs = nthreads / 2;

    // As Nibble requires keys, we must choose a starting key that,
    // when incremented, does not practically lead to overlap amongst
    // the other threads. Keys are u64, so if we assume
    // 1mil. threads, each gets 2**(64-20) keys. Thread IDs on Linux are
    // <64k in value, thus we can safely divide entire keyspace.
    // Thus, the starting key is (thread id) * 2**44.
    // Of course, this assumes threads don't come and go
    // to reuse keys :)
    ul startkey = (u64)gettid() * (1ul << 44);
    ul key = startkey;
    ul maxkeys = 1ul<<18;

    for (ul q = 0; q < npairs; q++) {
        cq_init(&args->queues[q]);
        //printf("worker %lu queue %p\n", args->id, &args->queues[q]);
    }

    //printf("w %lu init batches\n", args->id);
    struct batch *batches = calloc(NBATCHES, sizeof(*batches));
    assert(batches);
    for (ul i = 0; i < NBATCHES; i++) {
        batches[i].status = ALLOC_OK;
        batches[i].keys =
            calloc(NPTRS, sizeof(u64));
        assert(batches[i].keys);
        //printf("worker %lu batches %p\n", args->id, &batches[i]);
    }

    struct drand48_data rdata;
    srand48_r((ul)batches, &rdata);

    //printf("w %lu init sizes arrays\n", args->id);
    const ul nsizes = 256; // keep as power of two
    ul sizes_mask = nsizes-1;
    ul sizes1[nsizes], sizes2[nsizes];
    setup_sizes(nsizes, sizes1, sizes2, &rdata);

    //printf("w %lu shuffle q idxs\n", args->id);
    // offset our starting queue to push to;
    // can also randomize if you wish
    ul *qs = calloc(npairs, sizeof(*qs));
    assert(qs);
    for (ul i = 0; i < npairs; i++)
        qs[i] = i; //(i + 2 * args->id) % npairs;
    shuffle_ul(qs, npairs, &rdata);
    // number of consumer threads to communicate with
    ul nqs = (npairs);// / 4;

    atomic_sub_fetch(&warmup_barrier, 1);
    while (atomic_load(&warmup_barrier) > 0)
        ;
    //printf("w %lu starting!\n", args->id);

    ul next_q = 0;
    ul next_b = 0;
    ul start = rdtsc();

    ul *sizes_[2] = {sizes1, sizes2};
    ul sizesi = 0;
    ul *sizesp = sizes_[sizesi];
    ul last_shift_time = 0ul;

#if defined(PRINT_PROGRESS)
    ul last_print_time = 0ul;
    ul partial_many = 0ul;
#endif

#define BATCH(ii) \
    do { \
        b = &batches[(next_b+(ii)) % NBATCHES]; \
        if (b->status == ALLOC_OK) { \
            next_b = (next_b+(ii)+1) % NBATCHES; \
            goto found; \
        } \
    } while (0)

    while (1) {
        struct batch *b;
        do { // search for a ready batch
            BATCH(0); BATCH(1); BATCH(2); BATCH(3);
            BATCH(4); BATCH(5); BATCH(6); BATCH(7);
            BATCH(8); BATCH(9); BATCH(10); BATCH(11);
            BATCH(12); BATCH(13); BATCH(14); BATCH(15);
        } while (1);
found:;

        if (stop)
            goto out;

        int cpu, socket;
        rdtscp_id(&socket, &cpu);

#define ALLOC(ii) \
    do { \
        nibble_alloc(key, \
            sizesp[(ii) & sizes_mask], socket); \
        b->keys[ii] = key++; \
    } while (0)

        // fill it
        for (ul i = 0; i < NPTRS; i+=16) {
            ALLOC(i+0 ); ALLOC(i+1 );
            ALLOC(i+2 ); ALLOC(i+3 );
            ALLOC(i+4 ); ALLOC(i+5 );
            ALLOC(i+6 ); ALLOC(i+7 );
            ALLOC(i+8 ); ALLOC(i+9 );
            ALLOC(i+10 ); ALLOC(i+11 );
            ALLOC(i+12 ); ALLOC(i+13 );
            ALLOC(i+14 ); ALLOC(i+15 );
        }
        many += NPTRS;
        if (key >= (startkey + maxkeys)) {
            //printf("w %lu restarting keys\n", args->id);
            //fflush(stdout);
            key = startkey;
        }
#if defined(PRINT_PROGRESS)
        partial_many += NPTRS;
#endif

#if 0
        // use it XXX this may inflate performance
        for (ul i = 0; i < NPTRS/4; i++) {
            ul idx;
            //idx = rdrand() % NPTRS;
            lrand48_r(&rdata, (long*)&idx); idx = idx % NPTRS;
            free(b->keys[idx]);
            assert( (b->keys[idx] = malloc(size)) );
            many += 2;
        }
#endif

        // hand it off
        fullfence();
        b->status = FREE_OK;
        cq_push(&args->queues[qs[next_q]], (ul)b);
        next_q = (next_q + 1) % nqs;

        ul sec = (rdtsc()-start) / ticks_per_sec;
        if (sec >= TEST_LEN_SEC || stop)
            break;
        if ((sec-last_shift_time) >= SIZE_SHIFT_SEC) {
            sizesp = sizes_[ (sizesi = !sizesi) ];
#if defined(PRINT_PROGRESS)
            if (args->id == 0)
                printf("# shifting to ca. %lu bytes\n", sizesp[0]);
#endif
            last_shift_time = sec;
        }
#if defined(PRINT_PROGRESS)
        if ((sec - last_print_time) > 5) {
            if (args->id < 2) {
                float ops = (partial_many/1e3) /
                    (sec - last_print_time);
                printf("w %lu kops/sec: %.2f\n", args->id, ops);
                fflush(stdout);
                last_print_time = sec;
                partial_many = 0ul;
            }
        }
#endif
        if (sec < TEST_SKIP_FIRST_SEC)
            many = 0ul;
    }
out:;

    // float sec = (float)(rdtsc()-start) / ticks_per_sec;
    // printf("w %lu many %lu sec %.2f kops/sec %.2f\n",
    //         args->id, many, sec,
    //         ((float)many/1e3)/sec);
    stop = 1;
    atomic_add_fetch(&total_ops, many);

#if 0
    for (ul i = 0; i < NBATCHES; i++)
        free(batches[i].keys);
    free(batches);
#endif

    pthread_exit(NULL);
}

//==--------------------------------------------------------------==//
//      CONSUMER
//==--------------------------------------------------------------==//

void* consumer(void *args_) {
    struct consumer_args *args =
        (struct consumer_args*)args_;
    const ul npairs = nthreads / 2;

    // consumers get odd-numbered CPUs
    int cpu = args->id * 2 + 1;
    //printf("consumer cpu %lu\n", cpu);
    cpu_set_t mask; CPU_ZERO(&mask); CPU_SET(cpu, &mask);
    assert(0 == sched_setaffinity(0, sizeof(mask), &mask));

    for (ul q = 0; q < npairs; q++) {
        //printf("consumer %lu queue %p\n",
                //args->id, args->queues[q]);
    }

    ul many = 0, iters = 0;

    struct drand48_data rdata;
    srand48_r((ul)args, &rdata);

#if defined(PRINT_PROGRESS)
    ul last_print_time = 0ul;
    ul partial_many = 0ul;
#endif

    atomic_sub_fetch(&warmup_barrier, 1);
    while (atomic_load(&warmup_barrier) > 0)
        ;

#define FREE(ii)    nibble_free(b->keys[ii], 1)

    ul start = rdtsc();
    struct batch *b;
    while (1) {
        // find any batch that is ready
        for (ul prod = 0; prod < npairs; prod++) {
            struct cqueue *q = args->queues[prod];
            if (!cq_pop_try(q, (ul*)&b))
                continue;
            //assert(b->status == FREE_OK);
            for (ul i = 0; i < NPTRS; i+=16) {
                FREE(i+0); FREE(i+1);
                FREE(i+2); FREE(i+3);
                FREE(i+4); FREE(i+5);
                FREE(i+6); FREE(i+7);
                FREE(i+8); FREE(i+9);
                FREE(i+10); FREE(i+11);
                FREE(i+12); FREE(i+13);
                FREE(i+14); FREE(i+15);
            }
            memset(b->keys, 0, sizeof(ul)*NPTRS);
            fullfence();
            b->status = ALLOC_OK; // we're done
            many += NPTRS;
#if defined(PRINT_PROGRESS)
            partial_many += NPTRS;
#endif
            if (0 == (++iters & 0xff)) {
                ul sec = (rdtsc()-start) / ticks_per_sec;
                if (sec < TEST_SKIP_FIRST_SEC)
                    many = 0ul;
#if defined(PRINT_PROGRESS)
                if (args->id < 2 && sec >= last_print_time) {
                    float ops = (partial_many/1e3) /
                        (sec - last_print_time);
                    printf("c %lu kops/sec: %.2f\n", args->id, ops);
                    last_print_time = sec;
                    partial_many = 0ul;
                }
#endif
            }
        }
        if (stop)
            goto out;
    }
out:;

    // float sec = (float)(rdtsc()-start) / ticks_per_sec;
    // printf("c %lu many %lu sec %.2f kops/sec %.2f\n",
    //         args->id, many, sec,
    //         ((float)many/1e3)/sec);

    atomic_add_fetch(&total_ops, many);
    pthread_exit(NULL);
}

//==--------------------------------------------------------------==//
//      MAIN
//==--------------------------------------------------------------==//

int main(int narg, char *args[]) {
    if (narg != 3) {
        printf("Usage: %s pairs objsize\n", *args);
        return EXIT_FAILURE;
    }

    printf("my pid %d\n", getpid());
    printf("# CPUS_PER_SOCKET %d and TOTAL_SOCKETS %d\n",
            CPUS_PER_SOCKET, TOTAL_SOCKETS);
    fflush(stdout);

    nthreads = 2 * strtol(args[1], NULL, 10);
    const ul npairs = nthreads / 2;

    size = strtol(args[2], NULL, 10);

    pthread_t *wtids = calloc(npairs, sizeof(*wtids));
    assert(wtids);
    pthread_t *ctids = calloc(npairs, sizeof(*ctids));
    assert(ctids);

    atomic_store(&warmup_barrier, nthreads+1);
    atomic_store(&total_ops, 0);

    printf("# test.sec %d size.shift.sec %d\n"
            "# n.batches %d n.keys %d n.threads %lu\n"
            "# size %lu size2 %lu delta %.2f variance %.2f\n",
            TEST_LEN_SEC, SIZE_SHIFT_SEC,
            NBATCHES, NPTRS, nthreads,
            size, (ul)(size*DELTA), DELTA, VARIANCE);

    struct worker_args *wargs = calloc(npairs,sizeof(*wargs));
    assert(wargs);
    struct consumer_args *cargs = calloc(npairs,sizeof(*cargs));
    assert(cargs);

    for (ul t = 0; t < npairs; t++) {
        wargs[t].queues = malloc(npairs * sizeof(struct cqueue));
        assert(wargs[t].queues);
        wargs[t].id = t;
        assert(0 == pthread_create(&wtids[t], NULL,
                    worker, (void*)&wargs[t]));
        //sched_yield();
    }

#if defined(RELEASE_SELF)
#else
    for (ul t = 0; t < npairs; t++) {
        cargs[t].queues = malloc(npairs * sizeof(struct cqueue*));
        assert(cargs[t].queues);
        for (ul p = 0; p < npairs; p++)
            cargs[t].queues[p] = &wargs[p].queues[t];
        cargs[t].id = t;
        assert(0 == pthread_create(&ctids[t], NULL,
                    consumer, (void*)&cargs[t]));
    }
#endif

    ul nitems = 1ul<<27;
    nitems *= (log2f(nthreads) + 1);
    printf("nthreads %lu nitems %lu\n", nthreads, nitems);
    nibble_init(1ul<<38, nitems);

    atomic_sub_fetch(&warmup_barrier, 1);
    while (atomic_load(&warmup_barrier) > 0)
        ;

    ul start = rdtsc();
    printf("# Test skipping first %d seconds...\n",
            TEST_SKIP_FIRST_SEC);
    sleep(TEST_SKIP_FIRST_SEC);

    start = rdtsc();
    puts("# Test starting");

    for (ul t = 0; t < npairs; t++)
        pthread_join(wtids[t], NULL);
    for (ul t = 0; t < npairs; t++)
        pthread_join(ctids[t], NULL);

    ul end = rdtsc();

    float sec = (end-start)/(float)ticks_per_sec;
    float kops = atomic_load(&total_ops) / 1e3;

    printf("kops/sec %.2f\n", kops/sec);

#if 0
    for (ul t = 0; t < npairs; t++)
        free(wargs[t].queues);
    for (ul t = 0; t < npairs; t++)
        free(cargs[t].queues);
    free(wargs);
    free(cargs);
    free(wtids);
    free(ctids);
#endif

    return 0;
}
