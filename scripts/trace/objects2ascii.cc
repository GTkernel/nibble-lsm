// modify this file to print out the entry as-needed
#include <vector>

#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <fcntl.h>

void pexit(const char *msg) {
    perror(msg);
    exit(EXIT_FAILURE);
}

typedef uint32_t entry;

int main(int narg, char *args[]) {
    if (narg != 2) {
        fprintf(stderr, "Usage: %s filename\n", *args);
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

        if (max < *e) { // search for max object size
            max = *e;
            printf("updated %lu\n", max);
        }

        pos += sizeof(*e);
    }
    printf("max %lu\n", max);

    munmap(source, len);
    close(fd);

    return 0;
}
