#ifndef TRACEFMT_H_INCLUDED
#define TRACEFMT_H_INCLUDED

// keep consistent with src/nibble/trace.rs
enum Op {
    GET = 0, SET, DEL
};

struct entry {
    uint64_t key;
    uint8_t op;
    uint32_t size;
} __attribute__((packed));

#endif
