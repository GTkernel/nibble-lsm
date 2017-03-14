// this file converts a file that has object sizes
// into a binary trace that holds tuples of { key, size }
// each which is 32-bit unsigned integer
// Used for loading the initial data set for the Facebook
// trace from the SIGMETRICS paper
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

static std::vector<uint32_t> vec;

void dump(int fd) {
    size_t expected = vec.size() * sizeof(uint32_t);
    if (expected != write(fd, vec.data(), expected))
        pexit("write");
    vec.clear();
}

int main(int narg, char *args[]) {
    if (narg != 2) {
        fprintf(stderr, "Usage: ./convert filename\n");
        exit(EXIT_FAILURE);
    }

    vec.reserve(1ul<<28);

    // open the input file
    int fd = open(args[1], O_RDWR);
    if (fd < 0)
        pexit("open");
    struct stat s;
    if (fstat(fd, &s))
        pexit("stat");
    size_t len = (size_t)s.st_size;
    void *source = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    if (source == MAP_FAILED)
        pexit("mmap");
    if (madvise(source, len, MADV_SEQUENTIAL | MADV_WILLNEED))
        pexit("madvise");
    puts("Input setup done.");

    // open the output file
    int fd2 = open("output.bin", O_CREAT | O_WRONLY | O_EXCL, S_IRUSR | S_IWUSR);
    if (fd2 < 0)
        pexit("open output.bin");
    puts("Output setup done.");

    // TODO might be faster to mmap and ftruncate as the file grows
    char *pos = (char*)source;
    while ((uintptr_t)pos < ((uintptr_t)source + len)) {
        if (*pos == '#') {
            pos = strchrnul(pos, '\n') + 1;
            continue;
        }
        uint32_t value;
        char *nulpos = strchr(pos, '\n');
        if (nulpos == NULL)
            break;
        *nulpos = '\0';
        int ret = sscanf(pos, "%u", &value);
        pos = nulpos+1; // skip number
        if (ret == 0) {
            fprintf(stderr, "sscanf failed to parse value\n");
            exit(EXIT_FAILURE);
        } else if (ret == EOF)
            break;
        vec.push_back(value);
        if (vec.size() >= (1ul << 27))
            dump(fd2);
    }

    munmap(source, len);
    close(fd);
    close(fd2);

    return 0;
}
