#ifndef TRACEFMT_H_INCLUDED
#define TRACEFMT_H_INCLUDED

// keep consistent with src/kvs/trace.rs
using OpType = uint8_t;
enum class Op: OpType {
    GET = 0, SET, DEL
};

struct entry {
    uint64_t key;
    uint8_t op;
    uint32_t size;

    entry(uint64_t key, Op op, uint32_t size) :
        key(key), op(static_cast<OpType>(op)), size(size) { }
} __attribute__((packed));

#endif
