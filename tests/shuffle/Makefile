CC := gcc
CFLAGS := -Wall -Wextra -pthread -std=gnu11
CFLAGS += -Ofast
CFLAGS += -floop-optimize -funroll-loops
CFLAGS += -ggdb
LDFLAGS := 
# rustc told me to link against these for libffi.a
LIBS := -lutil -ldl -lpthread -lgcc_s -lc -lm -lrt -lutil -lnuma

all:	shuffle

util.o:	util.c Makefile
	$(CC) $< -o $@ $(CFLAGS) -c

shuffle.o:	shuffle.c Makefile cqueue.h
	$(CC) $< -o $@ $(CFLAGS) -c

# force rebuild each time
.PHONY: shuffle
shuffle:	libffi.a shuffle.o util.o
	cargo build --lib --release
	$(CC) shuffle.o util.o -o $@ -L./ -lffi $(LDFLAGS) $(LIBS)

../../target/release/libnibble.rlib: Makefile
	cargo build --lib --release

# this was non-trivial to figure out :(
.PHONY: libffi.a
libffi.a:	ffi.rs ../../target/release/libnibble.rlib
	cargo build --lib --release
	rustc -O --crate-type=staticlib \
		--extern nibble=../../target/release/libnibble.rlib \
		-L dependency=../../target/release/deps \
		--extern log=../../target/release/deps/liblog-bf16bb9a4912b11d.rlib \
		ffi.rs

clean:
	rm -f *.o shuffle libffi.a
