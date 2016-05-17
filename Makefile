.PHONY:	FORCE
build:	FORCE
	./with-mir cargo build

clean:	FORCE
	./with-mir cargo clean

test: FORCE
	./with-mir cargo test
