CC := gcc
CFLAGS := -std=gnu11 -O2 -g -Wno-unused-result
LDFLAGS :=
LIBS := -lrt

postmark:	postmark-1_5.o makefile
	$(CC) $< -o $@ $(LDFLAGS) $(LIBS)

clean:
	rm -f postmark *.o

