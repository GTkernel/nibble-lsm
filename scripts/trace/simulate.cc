#include <cstdio>
#include <cassert>
#include <cstdint>
#include <cstdlib>

#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>

#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "tracefmt.h"

void pexit(const char *msg) {
    perror(msg);
    exit(EXIT_FAILURE);
}

typedef struct entry entry_t;

// points to mmap'd file containing the trace
entry_t *trace;
size_t traceN;

// the set of initial objects to load in
uint32_t *objects;
size_t objectsN;

// bit vector - each entry is one object
// indexed by that object's key. true = object
// was inserted and not recently deleted
std::vector<bool> *exists;
std::vector<uint32_t> *sizes;

long totalSize = 0l;
long totalExist = 0l;

// How many of the overall set should we consider inserted during
// setup?
float fraction = 0.4f;

void readInput(char const *path,
        void** addr, size_t &bytes) {
    int fd = open(path, O_RDONLY);
    if (fd < 0)
        pexit("open");
    struct stat s;
    if (fstat(fd, &s))
        pexit("stat");
    size_t len = (size_t)s.st_size;
    void *source = mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (source == MAP_FAILED)
        pexit("mmap");
    if (madvise(source, len, MADV_SEQUENTIAL | MADV_WILLNEED))
        pexit("madvise");
    bytes = len;
    *addr = source;
}

void load() {
    long i = 0l;

    size_t toLoad = objectsN * fraction;
    std::cout << "Loading " << toLoad << " objects ("
        << 100*fraction << " %)..." << std::endl;
    std::generate_n(sizes->begin(), toLoad,
            [&]() mutable {
                if (0 == (i & ((1ul<<29)-1)))
                    std::cout << "    ... " << i/(1<<20)
                        << " mil" << std::endl;
                totalSize += objects[i];
                return objects[i++];
            }
    );

    totalExist = i;
}

// generate histogram of key appearences
void hist(size_t nBuckets, size_t nKeys) {
    std::cout << "Histogram..." << std::endl;
    std::vector<size_t> buckets(nBuckets, 0ul);

    size_t maxKey = 0ul;
    size_t i = 0ul;
    size_t printEvery = 1ul<<27;
    size_t perBucket = nKeys / nBuckets;
    while (true) {
        entry_t &e = trace[i++];
        if (i >= traceN) break;
        ++buckets[ e.key / perBucket ];
        if (e.key > maxKey)
            maxKey = e.key;
        if (0 == (i & (printEvery-1)))
            std::cout << (float)i / (1ul<<20)
                << " mil. ops"
                << std::endl;
    }

    std::cout << "Max key: " << maxKey << std::endl;

    std::cout << "Dumping to hist.dat ..." << std::endl;
    std::ofstream ofile("hist.dat", std::ios_base::out);
    for (auto i : buckets)
        ofile << i << std::endl;
}

void run() {
    std::cout << "Running..." << std::endl;
    long i = 0;
    size_t printEvery = 1ul<<21;
    std::cout << "    printing status every " << printEvery
        << " ops" << std::endl;
    std::cout << "totalSize activeObjects delHit getHit setHit hitRate" << std::endl;
    size_t delHit = 0ul, getHit = 0ul, setHit = 0ul;
    size_t hits = 0ul, total = 0ul;
    while (true) {
        entry_t &e = trace[i++];
        if (i >= traceN) i = 0ul;

        ++total;
        bool isHit = (*exists)[e.key];
        if (isHit)
            ++hits;

        if (e.op == Op::GET) {
            if (isHit)
                ++getHit;
        }
        else if (e.op == Op::SET) {
            long oldSize = 0;
            if (isHit) {
                ++setHit; // becomes an update
                oldSize = (*sizes)[e.key];
            } else {
                // new insertion
                (*exists)[e.key] = true;
                ++totalExist;
            }
            totalSize += ((long)e.size - oldSize);
             (*sizes)[e.key] = e.size;
        }
        else if (e.op == Op::DEL) {
            if (isHit) {
                ++delHit;
                totalSize -= (*sizes)[e.key];
                --totalExist;
                (*exists)[e.key] = false;
            }
        }
        else abort();

        if (0 == (i & (printEvery-1))) {
            std::cout << totalSize << " " << totalExist
                << " " << delHit
                << " " << getHit
                << " " << setHit
                << " " << (float)hits / (float)total
                << std::endl;
            delHit = setHit = getHit = hits = total = 0ul;
        }
    }
}

int main(int narg, char *args[]) {
    if (narg != 3) {
        fprintf(stderr, "Usage: %s tracefile objfile\n", *args);
        exit(EXIT_FAILURE);
    }
    size_t bytes;

    std::cout << "Mapping input files..." << std::endl;
    readInput(args[1], (void**)&trace, bytes);
    traceN = bytes / sizeof(*trace);
    std::cout << "    " << traceN << " trace entries" << std::endl;

    readInput(args[2], (void**)&objects, bytes);
    objectsN = bytes / sizeof(*objects);
    std::cout << "    " << objectsN << " objects" << std::endl;

    size_t loadN = (fraction * objectsN);
    std::cout << "Initializing bit maps... (" << loadN << " objects)" << std::endl;
    exists = new std::vector<bool>(objectsN, false);
    std::fill_n(exists->begin(), loadN, true);
    sizes = new std::vector<uint32_t>(objectsN, 0u);

    load();
    run();
    //hist(512, objectsN);
}
