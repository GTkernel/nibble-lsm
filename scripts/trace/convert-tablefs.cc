// this file converts a file that has a trace collected
// from the TableFS source code we added into
// a binary one that LSM can read in
// TableFS uses 16-byte keys, so the first thing is to map these
// to 8-byte keys


#include <vector>
#include <map>
#include <iostream>

#include <cassert>

#include <limits.h>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <cstdlib>

#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "tracefmt.h"

void pexit(const char *msg) {
    perror(msg);
    exit(EXIT_FAILURE);
}

std::vector<entry> trace;
std::map<std::string,uint64_t> key_map;

void dump(int fd) {
    size_t expected = trace.size() * sizeof(std::vector<entry>::value_type);
    if (expected != write(fd, trace.data(), expected))
        pexit("write");
    trace.clear();
}

enum Delim: char {
    Newline = '\n',
    Space = ' ',
    Nul = '\0',
    Hash = '#',
};

// return true if it found 'to' else, it reached the NUL byte
inline bool
skip_to(char const* &p, char to) {
    while (*p != to && *p != Delim::Nul)
        ++p;
    return *p != Delim::Nul;
}

// return true if succeeded and didn't encounter NUL
inline bool
skip_line(char const* &p) {
    if (skip_to(p, Delim::Newline))
        ++p;
    return *p != Delim::Nul;
}

// time_tsc tid op value_len key
// 1767722821910036 6667 get 0 0x01000000000000009b5acf3f2c538253
//
// Want to convert the 16-byte keys into 8-byte keys
// Look for new line, go back to first space, then you have a key
void map_keys(char const *input) {
    using namespace std;

    uint64_t new_key = 1ul;
    char const *p = input;
    char keybuf[64];

    if (!skip_line(p)) {
        cerr << "Error: input has one line?" << endl;
        exit(EXIT_FAILURE);
    }
    assert(p != input);

    // Extract all keys.  At the start of each iteration, we should be
    // at the first character of a line. We then extract the key at
    // the end of the line.
    while (true) {
        // 1767722821910036 6667 get 0 0x01000000000000009b5acf3f2c538253\n
        // ^p    move p to end -->
        if (!skip_to(p, Delim::Newline))
            break;
        --p;
        // p now points to last char before newline
        // 1767722821910036 6667 get 0 0x01000000000000009b5acf3f2c538253\n
        //                                                             p^
        char const *sp = p;
        while (*sp != ' ') --sp;
        // sp now points to first space in front of newline
        // 1767722821910036 6667 get 0 0x01000000000000009b5acf3f2c538253\n
        //                          sp^                                p^
        strncpy(keybuf, sp + 1, p - sp);
        keybuf[p - sp] = Delim::Nul;
        key_map.insert(move( make_pair(string(keybuf), new_key++) ));
        
        // since p is one behind newline, advance to char after.
        // 1234\n5678   1234\n5678
        //    ^p              ^p
        p += 2;
    }

    cout << key_map.size() << " keys" << endl;
}

// code assumes these have 3 characters
char const * const OpPUT = "put";
char const * const OpGET = "get";
char const * const OpDEL = "del";

// column headers: time_tsc tid op value_len key
void convert(char const *input, int fd) {
    using namespace std;
    char const *p = input;
    char buf[256];

    if (!skip_line(p)) {
        cerr << "Error: input has one line?" << endl;
        exit(EXIT_FAILURE);
    }

    size_t total = 0ul;

    // Layout:
    // 1767722821910036 6667 get 0 0x01000000000000009b5acf3f2c538253\n
    while (true) {
        if (*p == Delim::Hash)
            if (!skip_line(p)) break;

        // Skip first two columns in each row.
        for (int i = 0; i < 2; i++, ++p)
            if (!skip_to(p, Delim::Space))
                break;

        // Extract the op.
        // 1767722821910036 6667 get 0 0x01000000000000009b5acf3f2c538253\n
        //                      p^
        Op op;
        if (strncmp(p, OpGET, 3) == 0)
            op = Op::GET;
        else if (strncmp(p, OpDEL, 3) == 0)
            op = Op::DEL;
        else if (strncmp(p, OpPUT, 3) == 0)
            op = Op::SET;
        else {
            strncpy(buf, p, 4);
            buf[4] = Delim::Nul;
            cerr << "Error: input has bogus operation: " << buf << endl;
            exit(EXIT_FAILURE);
        }

        if (!skip_to(p, Delim::Space))
            break;
        ++p;

        // Extract the value length if op is put.
        // 1767722821910036 6667 get 0 0x01000000000000009b5acf3f2c538253\n
        //                          p^
        size_t vlen = 0ul;
        if (op == Op::SET) {
            vlen = strtol(p, NULL, 10);
            if (vlen == LONG_MAX || vlen == LONG_MIN) {
                cerr << "Error: value len has overflowed" << endl;
                exit(EXIT_FAILURE);
            }
        }

        if (!skip_to(p, Delim::Space))
            break;
        ++p;

        // Extract the key and convert to an 8-byte key using our
        // previously constructed map.
        // 1767722821910036 6667 get 0 0x01000000000000009b5acf3f2c538253\n
        //                            p^
        // Same as above code.
        if (!skip_to(p, Delim::Newline))
            break;
        --p;
        char const *sp = p;
        while (*sp != ' ') --sp;
        strncpy(buf, sp + 1, p - sp);
        buf[p - sp] = Delim::Nul;
        auto iter = key_map.find( std::string(buf) );
        if (iter == key_map.end()) {
            cerr << "Error: key not found on conversion: '" << buf << "'" << endl;
            exit(EXIT_FAILURE);
        }
        uint64_t key = iter->second;
        assert( key != 0ul );
        trace.emplace_back(key, op, static_cast<uint32_t>(vlen));

        // 1767722821910036 6667 get 0 0x01000000000000009b5acf3f2c538253\n
        //                                                             p^
        if (!skip_line(p))
            break;

        ++total;
        if (trace.size() >= (1ul << 27)) dump(fd);
    }
    dump(fd);
    std::cout << "total " << total << std::endl;
}

int main(int narg, char *args[]) {
    if (narg != 2) {
        fprintf(stderr, "Usage: ./convert filename\n");
        exit(EXIT_FAILURE);
    }

    trace.reserve(1ul<<28);

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
    puts("Input setup done.");

    // open the output file
    int fd2 = open("output.bin", O_CREAT | O_WRONLY | O_EXCL, S_IRUSR | S_IWUSR);
    if (fd2 < 0)
        pexit("open output.bin");
    puts("Output setup done.");

    map_keys(static_cast<char const *>(source));
    convert(static_cast<char const *>(source), fd2);

    munmap(source, len);
    close(fd);
    close(fd2);

    return 0;
}
