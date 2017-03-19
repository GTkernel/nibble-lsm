// Implementation of std::queue<long> for use in C.
#include <cassert>
#include <queue>
#include "queue.hh"

using queue_type = std::queue<long>;

void* queue_new() {
    std::queue<long> *q = new std::queue<long>();
    return static_cast<void*>(q);
}

void queue_free(void* q) {
    assert(q);
    auto queue = static_cast<queue_type*>(q);
    delete queue;
}

void queue_push(void* q, long value) {
    assert(q);
    auto queue = static_cast<queue_type*>(q);
    queue->push(value);
}

long queue_pop(void* q) {
    assert(q);
    auto queue = static_cast<queue_type*>(q);
    long v = queue->front();
    queue->pop();
    return v;
}

int queue_empty(void* q) {
    assert(q);
    auto queue = static_cast<queue_type*>(q);
    return queue->empty() ? 1 : 0;
}

