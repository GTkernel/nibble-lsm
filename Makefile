# Build for libcuckoo (src/cuckoo.cc)
# Used only to verify the code compiles independent of cargo build
# Keep this code consistent with build.rs
# If you want to build nibble, use cargo build --release --lib
# If you need to debug this library, change to -O0 and add -ggdb, set
# LD_LIBRARY_PATH to src/ before running with gdb
CXX := g++
CXXFLAGS := -std=c++11 -O3 -I./thirdparty/libcuckoo/src/ -Wall -Wextra
CXXFLAGS += -msse4.2 -mtune=native -march=native -malign-double -ggdb
CXXFLAGS += -Wno-unused-variable
# Configure how the memory backing the hashtable will be allocated:
# CUCKOO_INTERLEAVE or CUCKOO_BIND0
CXXFLAGS += -DCUCKOO_INTERLEAVE
#CXXFLAGS += -DCUCKOO_BIND0
SOFLAGS := -shared -Wl,-soname,libcuckoo.so
LDFLAGS :=  -flto
LIBS := -pthread -lcityhash -lnuma

all: src/libcuckoo.so benches/cuckoo

# too many other dependencies
.PHONY:	src/cuckoo.o
src/cuckoo.o:	src/cuckoo.cc
	$(CXX) $^ -o $@ $(CXXFLAGS) -c -fPIC

src/libcuckoo.so:	src/cuckoo.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(SOFLAGS) $(LIBS)

.PHONY:	benches/cuckoo.o
benches/cuckoo.o:	benches/cuckoo.cc
	$(CXX) $^ -o $@ $(CXXFLAGS) -c

benches/cuckoo:	benches/cuckoo.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS)

clean:
	rm -v -f src/cuckoo.o src/libcuckoo.so \
		benches/cuckoo.o benches/cuckoo
