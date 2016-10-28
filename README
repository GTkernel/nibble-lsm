----------------------------------------------------------------------
--- Build
----------------------------------------------------------------------

Requires Rust 'nightly'

cargo build --release --lib
cargo build --release --bin scale_test

cargo run --release --bin scale_test

Enable/configure debug messages with environment variable NIBDEBUG=N
where N is 0-5.

You must reserve sufficient amount of 2MB pages:
echo N > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

Currently, we cannot compact multiple segments if their total live
size exceeds the maximum size of one segment.

--- SHUFFLE ---

To run the 'shuffling' benchmark, (which was copied from
malloc-tests.git/src/shuffle/), enter tests/shuffle/ and:

    make

It should build the nibble library and a proxy library that allows C
to invoke Rust code.

--- YCSB ---

Build the ycsb benchmark code with these instructions. You can link it
with Nibble, RAMCloud, or MICA. The latter two require you
build/install the respective shared libraries.

Nibble:

    cargo build --bin ycsb [--release]

RAMCloud:

    (download/compile/install ramcloud-scale-hacks.git)
    cargo build --features "extern_ycsb rc" --bin ycsb [--release]

    RAMCLOUD_ARGV="-t MB -h % [...]" RAMCLOUD_NARG=N \
                   target/[debug|release]/ycsb [args]

MICA:

    (download/compile/install mica-kvs.git)
    cargo build --features "extern_ycsb mica" --bin ycsb [--release]

    TODO execution instructions


----------------------------------------------------------------------
--- Self notes below
----------------------------------------------------------------------

Performance killers
- using strings as keys
    - you need to strncmp on collisions
    - hashes run over arbitrarily long buffers
- too many function indirections
    - inline common paths
- Using crate rand; rand::thread_rng() in parallel apparently has some
kind of lock that slows everything down! Use rdrand instead...

crates to consider

mempool by Andrew Gallant


Note: with a fixed array of Segments (slots) we will need to compact
such that multiple candidates are compressed to maximize a segment's
typical size, else we run out of segment slots and still have many
blocks available!
So, either we fix the compactor to fill in new segments of the
expected size, and the segment table has a fixed number of slots, or
we allocate many slots and ensure the segment allocator checks the
block allocator to ensure there are sufficient blocks (handle the case
that segment slots can be allocated but not blocks)



Compile with MIR support using
RUSTFLAGS="-Z orbit"

Use when running tests to limit concurrency
RUST_TEST_THREADS=1

Remove use of RefCell in these compositons:
Arc<Mutex<RefCell<>>>

Cell, RefCell, UnsafeCell do not share across threads
as they aren't Sync

impl can be split across multple files
might be good to put test-enabled code in separate files
e.g., segment.rs segment_test.rs
or move them to tests/ entirely

doc comments PRECEDE the item they refer to
struct Type {
    var: usize, /// comment actually for var2!
    var2: usize,
}
should be
struct Type {
    /// var is cool
    var: usize,
    /// var2 is cooler
    var2: usize,
}
See http://internals.rust-lang.org/t/any-interest-in-same-line-doc-comments/3212/2


in order to decr live bytes on a segment (in the segment usage table),
need a reverse lookup table of some kind to find segment, given a VA
(from the index).
A) one idea is to use some of the other bits in the
index for the segment ID:
- 48 bits for a VA
- given a terabyte region (1<<40) only need 40 to encode address +base
    - remaining 64-40 = 24 bits could encode segment ID (16 mil.)
        - each must be 1<<19 or 64KiB
- with 32 terabytes (1<<45) need 45 bits of encoding
    - remaning 64-45 = 19 bits for ID (512K)
        - each must be 1<<26 or 64MB
B) another is to convert VA (from index) of old location to block base
address. Use this to get block index (all blocks are contiguous and
thus ordered), then to lookup the block struct, which will have the
segment ID. Maintaining the segment ID this way, means when
adding/removing blocks from segments we must edit this information.

command-line parsing crate
http://kbknapp.github.io/clap-rs/clap/index.html




for the new compiler error output:
    RUST_NEW_ERROR_FORMAT=true


