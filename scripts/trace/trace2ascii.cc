#include <vector>

#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "tracefmt.h"

void pexit(const char *msg) {
    perror(msg);
    exit(EXIT_FAILURE);
}

int main(int narg, char *args[]) {
    if (narg != 2) {
        fprintf(stderr, "Usage: ./trace2ascii filename\n");
        exit(EXIT_FAILURE);
    }

    // open the input file
    int fd = open(args[1], O_RDONLY);
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

    size_t n = 0ul, max = 0ul;
    char *pos = (char*)source;
    while ((uintptr_t)pos < ((uintptr_t)source + len)) {
        entry *e = (entry*)pos;
        //if (e->key == 0) n++; // count zero keys
        printf("%lu %u %u\n", e->key, e->op, e->size);
        pos += sizeof(*e);
    }
    //printf("n %lu\n", n);

    munmap(source, len);
    close(fd);

    return 0;
}
