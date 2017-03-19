CC := gcc
CXX := g++
CFLAGS := -std=gnu11 -O2 -g -Wno-unused-result
CXXFLAGS := -std=c++14
LDFLAGS :=
LIBS := -lrt

postmark:	postmark-1_5.o queue.o makefile
	$(CXX) postmark-1_5.o queue.o -o $@ $(LDFLAGS) $(LIBS)

clean:
	rm -f postmark *.o

