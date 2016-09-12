CC := clang
CFLAGS := -O3 -Wall -Wextra -pthread
LDFLAGS := 
# rustc told me to link against these for libffi.a
LIBS := -lutil -ldl -lpthread -lgcc_s -lc -lm -lrt -lutil -lnuma

all:	shuffle

util.o:	util.c Makefile
	$(CC) $< -o $@ $(CFLAGS) -c

shuffle.o:	shuffle.c Makefile
	$(CC) $< -o $@ $(CFLAGS) -c

shuffle:	libffi.a shuffle.o util.o
	$(CC) $^ -o $@ $(LDFLAGS) $(LIBS) -L./ -lffi


# this was non-trivial to figure out :(
libffi.a:	ffi.rs
	rustc -O --crate-type=staticlib \
		--extern nibble=../../target/release/libnibble.rlib \
		-L dependency=../../target/release/deps \
		--extern log=../../target/release/deps/liblog-bf16bb9a4912b11d.rlib \
		$<

clean:
	rm -f *.o shuffle libffi.a