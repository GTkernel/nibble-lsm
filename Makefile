.PHONY:	FORCE
build:	FORCE
	./with-mir cargo build

clean:	FORCE
	cargo clean
