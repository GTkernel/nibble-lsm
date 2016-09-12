#define _GNU_SOURCE

#include <stdlib.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "util.h"

ul ticks_per_sec;

__attribute__((constructor))
void set_ticks() {
    ul t = rdtsc();
    usleep(1000);
    ul t2 = rdtsc();
    ticks_per_sec = 1000*(t2-t);
}

void shuffle(struct alloc *entries, long n, struct drand48_data *d) {
    long r;
    for (long i = 0; i < n-1; i++) {
        struct alloc a = entries[i];
        lrand48_r(d, &r);
        long ii = (i+1) + (r % (n-(i+1)));
        entries[i] = entries[ii];
        entries[ii] = a;
    }
}

void shuffle_ul(ul *entries, long n, struct drand48_data *d) {
    long r;
    for (long i = 0; i < n-1; i++) {
        ul a = entries[i];
        lrand48_r(d, &r);
        long ii = (i+1) + (r % (n-(i+1)));
        entries[i] = entries[ii];
        entries[ii] = a;
    }
}

struct mem getmem(long less) {
    static char *buf = NULL;
    if (!buf)
        assert((buf = (char*)calloc(1, 1024)));
    FILE *fp = fopen("/proc/self/stat", "r");
    if (!fp) {
        perror("fopen");
        exit(EXIT_FAILURE);
    }
    fgets(buf, 1024, fp);
    fclose(fp);
    fp = NULL;
    int field_idx = 22; // max 51
    char *pos = buf;
    while (field_idx-- > 0)
        pos = &strchrnul(pos, ' ')[1];
    struct mem mem;
    mem.vsize = strtol(pos, NULL, 10) - less;
    // next field
    pos = &strchrnul(pos, ' ')[1];
    mem.rss = 4096*strtol(pos, NULL, 10) - less;
    return mem;
}

void print_mem(long less) {
    struct mem mem = getmem(less);
    printf("MEM vsize %.2f MiB rss %.2f MiB\n",
            (float)mem.vsize/(1ul<<20),
            (float)mem.rss/(1ul<<20));
    fflush(stdout); fflush(stderr);
}

void putsf(const char* string) {
    puts(string);
    fflush(stdout); fflush(stderr);
}

void print_mem3(long less, float expected, const char *extra) {
    struct mem mem = getmem(less);
    float vs = (float)mem.vsize/(1ul<<20);
    float rss = (float)mem.rss/(1ul<<20);
    float ratio = expected == 0. ? 1. : (float)mem.rss/expected;
    printf("%sMEM vsize %.2f MiB rss %.2f MiB rss-ratio %.2lf\n",
            extra ? extra : "", vs, rss, ratio);
    fflush(stdout); fflush(stderr);
}

void print_mem2(long less, float expected) {
    print_mem3(less,expected,NULL);
}

size_t now() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec*1e9 + t.tv_nsec);
}

void atomic_store(ul *where, ul value) {
    assert(where);
    __atomic_store_n(where, value, __ATOMIC_SEQ_CST);
}

ul atomic_load(ul *where) {
    assert(where);
    return __atomic_load_n(where, __ATOMIC_SEQ_CST);
}

ul atomic_sub_fetch(ul *where, ul value) {
    assert(where);
    return __atomic_sub_fetch(where, value, __ATOMIC_SEQ_CST);
}

ul atomic_add_fetch(ul *where, ul value) {
    assert(where);
    return __atomic_add_fetch(where, value, __ATOMIC_SEQ_CST);
}
