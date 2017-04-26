CC := gcc
CFLAGS := -Wall -Wextra -pthread -std=gnu11
CFLAGS += -Ofast
CFLAGS += -floop-optimize -funroll-loops
CFLAGS += -ggdb
LDFLAGS := 
# rustc told me to link against these for libffi.a
LIBS := -lutil -ldl -lpthread -lgcc_s -lc -lm -lrt -lutil -lnuma

all:	frag

util.o:	util.c Makefile
	$(CC) $< -o $@ $(CFLAGS) -c

frag.o:	frag.c Makefile
	$(CC) $< -o $@ $(CFLAGS) -c

# force rebuild each time
.PHONY: frag
frag:	libffi.a frag.o util.o
	cargo build --lib --release
	$(CC) frag.o util.o -o $@ -L./ -lffi $(LDFLAGS) $(LIBS)

../../target/release/libkvs.rlib: Makefile
	cargo build --lib --release

# this was non-trivial to figure out :(
.PHONY: libffi.a
libffi.a:	ffi.rs ../../target/release/libkvs.rlib
	cargo build --lib --release
	rustc -O --crate-type=staticlib \
		--extern kvs=../../target/release/libkvs.rlib \
		-L dependency=../../target/release/deps \
		--extern log=../../target/release/deps/liblog-bf16bb9a4912b11d.rlib \
		ffi.rs

clean:
	rm -f *.o frag libffi.a
