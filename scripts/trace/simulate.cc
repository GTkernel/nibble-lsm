// Simulate the key-value store trace provided by convert.cc
// We do this to have an initial analysis of the hit rates, and key
// access distribution before doing a full emulation with a real
// key-value store system.
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

// points to mmap'd file containing the trace
entry *trace;
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
        entry &e = trace[i++];
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

void run(bool loop = true) {
    std::cout << "Running..." << std::endl;
    long i = 0;
    size_t printEvery = 1ul<<15;
    std::cout << "    printing status every " << printEvery
        << " ops" << std::endl;
    std::cout << "{GET,PUT,DEL}\% is count of op type / overall ops in a window" << std::endl;
    std::cout << "{ins,del}Rate is, of the PUT/DEL individually, which resulted in new or deleting an object" << std::endl;
    std::cout << "totalSize activeObjects GET\% PUT\% DEL\% insRate delRate" << std::endl;
    size_t dels = 0ul, gets = 0ul, sets = 0ul;
    size_t delHit = 0ul, getHit = 0ul, setHit = 0ul;
    size_t hits = 0ul, total = 0ul;
    size_t overall_total = 0ul;
    while (true) {
        entry &e = trace[i++];
        if (i >= traceN) {
            if (loop) i = 0ul;
            else break;
        }

        ++total;
        ++overall_total;
        bool isHit = (*exists)[e.key];
        if (isHit)
            ++hits;

        if (e.op == static_cast<OpType>(Op::GET)) {
            ++gets;
            if (isHit)
                ++getHit;
        }
        else if (e.op == static_cast<OpType>(Op::SET)) {
             ++sets;
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
        else if (e.op == static_cast<OpType>(Op::DEL)) {
            ++dels;
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
                << " " << 100.f * (float)gets / (float)total
                << " " << 100.f * (float)sets / (float)total
                << " " << 100.f * (float)dels / (float)total
                // SET/PUT that miss will do an insertion
                << " " << 100.f * (float)(sets - setHit) / (float)sets
                // DEL that hit will actually remove
                << " " << 100.f * (float)delHit / (float)dels
                << " " << 100.f * (float)hits / (float)total
                << std::endl;
            delHit = setHit = getHit = hits = total = 0ul;
            dels = sets = gets = 0ul;
        }
    }
    std::cout << totalSize << " " << totalExist
        << " " << 100.f * (float)gets / (float)total
        << " " << 100.f * (float)sets / (float)total
        << " " << 100.f * (float)dels / (float)total
        // SET/PUT that miss will do an insertion
        << " " << 100.f * (float)(sets - setHit) / (float)sets
        // DEL that hit will actually remove
        << " " << 100.f * (float)delHit / (float)dels
        << " " << 100.f * (float)hits / (float)total
        << std::endl;
    delHit = setHit = getHit = hits = total = 0ul;
    dels = sets = gets = 0ul;
    std::cout << "Overall total ops: " << overall_total << std::endl;
    size_t live = std::count(exists->begin(), exists->end(), true);
    std::cout << "Remaining live objects: " << live << std::endl;
}

void run_with_objfile(int narg, char *args[]) {
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

void run_without_objfile(int narg, char *args[]) {
    size_t bytes;

    std::cout << "Mapping input files..." << std::endl;
    readInput(args[1], (void**)&trace, bytes);
    traceN = bytes / sizeof(*trace);
    std::cout << "    " << traceN << " trace entries" << std::endl;

    objectsN = 1ul<<30;
    std::cout << "    " << objectsN << " objects (static assumption)" << std::endl;

    std::cout << "    considering all objects (as not inserted)" << std::endl;
    size_t loadN = objectsN;
    std::cout << "Initializing bit maps... (" << loadN << " objects)" << std::endl;
    exists = new std::vector<bool>(objectsN, false);
    sizes = new std::vector<uint32_t>(objectsN, 0u);

    // load() is skipped because the trace assumes starting from
    // an empty state
    // Pass false to have it stop when the trace concludes
    run(false);
    //hist(512, objectsN);
}

int main(int narg, char *args[]) {
    if (narg < 2) {
        fprintf(stderr, "Usage: %s tracefile [objfile]\n", *args);
        exit(EXIT_FAILURE);
    }

    if (narg == 3)
        run_with_objfile(narg, args);
    if (narg == 2)
        run_without_objfile(narg, args);
}
