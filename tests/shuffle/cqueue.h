#pragma once

#include <stdint.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

// TODO put some prefetching of next location of the buffer when
// pushing (not when popping b/c that must be written by the pushing
// side)

//==----------------------------------------------------------------==
//  SOME CONFIGURATION PARAMETERS
//

// TODO put tail and head state into own structures
// and hand that out to the producer and the consumer separately
// They can back-point to the main structure if needed

// Fixed number of entries in a message queue.
#define __CQ_ENTRIES_SHIFT  11ul
#define __CQ_ENTRIES        (1ul << __CQ_ENTRIES_SHIFT)

//
//
//==----------------------------------------------------------------==

#define __ALIGN64 \
    __attribute__((aligned(64)))

#define __UNUSED \
    __attribute__((unused))

#define __CQ_NEXT(ii)       (((ii) + 1) & (__CQ_ENTRIES - 1))

struct cqueue {
    // keep both head and item in same cache line
    uint64_t head __ALIGN64;
    volatile uint64_t array[__CQ_ENTRIES] __ALIGN64;
    uint64_t tail __ALIGN64;
};

_Static_assert(__CQ_ENTRIES >= 2,
        "Need at least two entries in cqueue.array[]:"
        " one for the marker slot, one for data.");


typedef struct cqueue cqueue_t;

static inline
void cq_init(cqueue_t *queue) {
    memset(queue, 0, sizeof(*queue));
    queue->head = 1ul;
}

static inline
bool cq_empty(volatile cqueue_t *queue) {
    return __CQ_NEXT(queue->tail) == queue->head;
}

static inline
bool cq_empty_head(cqueue_t *queue) {
    volatile uint64_t *head = &queue->head;
    return __CQ_NEXT(queue->tail) == *head;
}
static inline
bool cq_empty_tail(cqueue_t *queue) {
    volatile uint64_t *tail = &queue->tail;
    return __CQ_NEXT(*tail) == queue->head;
}

static inline
bool cq_full(volatile cqueue_t *queue) {
    return queue->head == queue->tail;
}

static inline
bool cq_full_tail(cqueue_t *queue) {
    volatile uint64_t *tail = &queue->tail;
    return queue->head == *tail;
}

//                      //
// Private API below    //
//                      //

static inline
void cq_push(cqueue_t *queue, uint64_t val) {
    while (cq_full_tail(queue))
        ;
    assert(queue->head < __CQ_ENTRIES);
    uint64_t prior = queue->head > 0 ? queue->head-1 : __CQ_ENTRIES-1;
    queue->array[prior] = val;
    queue->head = __CQ_NEXT(queue->head);
}

static inline
void __pop(cqueue_t *queue, uint64_t *val) {
    assert(queue->tail < __CQ_ENTRIES);
    *val = queue->array[queue->tail];
    queue->tail = __CQ_NEXT(queue->tail);
}

static inline
uint64_t cq_pop(cqueue_t *queue) {
    uint64_t val;
    while (cq_empty_head(queue))
        ;
    __pop(queue, &val);
    return val;
}

static inline
bool cq_pop_try(cqueue_t *queue, uint64_t *val) {
    bool ret = false;
    if (!cq_empty_head(queue)) {
        __pop(queue, val);
        ret = true;
    }
    return ret;
}
