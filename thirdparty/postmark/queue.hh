// Implementation of std::queue<long> for use in C.
#if __cplusplus
#pragma once
extern "C" {
#endif
    void* queue_new();
    void queue_free(void* q);
    void queue_push(void* q, long value);
    long queue_pop(void* q);
    int queue_empty(void* q);
#if __cplusplus
}
#endif
