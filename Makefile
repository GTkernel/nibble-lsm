# Build for libcuckoo (src/cuckoo.cc)
# Used only to verify the code compiles independent of cargo build
# Keep this code consistent with build.rs
# If you want to build nibble, use cargo build --release --lib
# If you need to debug this library, change to -O0 and add -ggdb, set
# LD_LIBRARY_PATH to src/ before running with gdb
CXX := g++
CXXFLAGS := -std=c++11 -O3 -I./thirdparty/libcuckoo/src/ -fPIC -Wall -Wextra
LDFLAGS := -shared -Wl,-soname,libcuckoo.so
LIBS := -pthread -lcityhash

all: src/libcuckoo.so

# too many other dependencies
.PHONY:	src/cuckoo.o
src/cuckoo.o:	src/cuckoo.cc
	$(CXX) $^ -o $@ $(CXXFLAGS) -c

src/libcuckoo.so:	src/cuckoo.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)

clean:
	rm -v -f src/cuckoo.o src/libcuckoo.so
