# Build for cuckoo C++ test code
# Rebuild if you modify any cuckoo header files
CXX := clang++
CXXFLAGS := -std=c++11 -O2 -ggdb -I../thirdparty/libcuckoo/src/ -Wall -Wextra -pthread
LDFLAGS := -L/usr/local/lib -L/usr/local/lib64
LIBS := -pthread -lcityhash

all: cuckoo

# force recompile everytime (other deps not visible to makefile)
.PHONY:	cuckoo.o
cuckoo.o:	cuckoo.cc Makefile
	$(CXX) $< -o $@ $(CXXFLAGS) -c

cuckoo:	cuckoo.o
	$(CXX) $< -o $@ $(LDFLAGS) $(LIBS)

clean:
	rm -v -f cuckoo *.o
