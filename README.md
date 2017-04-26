# Shoveller: Scalable Parallel Log-structured Memory for Key-Value Stores

**Authors**:  Alexander Merritt (merritt.alex@gatech.edu), Yuan Chen (yuan.chen@hpe.com)

## Description

**Shoveller** is a scalable, memory-capacity efficient key-value store for very large scale machines (e.g., tens of terabytes or more memory and hundreds of CPU cores). Shoveller promotes the use of a concurrent multi-head log-structured memory to attain high performance and resistance to memory fragmentation, together with scalable low-latency synchronization and optimistically concurrent indexing that allow application threads to scale to hundreds of cores. A prototype for single node system has been implemented, and its effectiveness has been evaluated on a HPE SuperDomeX machine with 240 cores and 12 TiB of DRAM across a wide range of workload patterns. 

## Master Source

- HPE Internal: https://github.hpe.com/labs/shoveller
- External: https://github.com/HewlettPackard/shoveller

## Maturity

Research prototype. 

## Dependencies

Shoveller is implemented entirely in the [Rust language](https://www.rust-lang.org/en-US/) and requires the ``nightly'' branch of the compiler.  Installation can be done _without root administration_ from https://rustup.rs.


## Usage

### Build and Test Shoveller

#### Install Rust 'nightly'
https://www.rust-lang.org/en-US/install.html

```
curl https://sh.rustup.rs -sSf | sh 

source ~/.cargo/env

rustup default nightly

```
rustup default nightly’ command may take quite a while due to rust-docs installation known issue: https://github.com/rust-lang-nursery/rustup.rs/issues/763.

Verify the installation.
```
% rustc --version
```
You should see something like 
```
rustc 1.17.0-nightly (e1cec5d4b 2017-03-29)
```

#### Get Shoveller
HPE Internal
```
git clone https://github.hpe.com/labs/shoveller
```
External
```
git clone https://github.com/HewlettPackard/shoveller
```

#### Build Shoveller

The first time build will take a while to update the registry. Just be patient. 

```
cd shoveller
cargo update
cargo build --lib --release
```


#### Test Shoveller

You will need a machine with at least 32GB memory to test Shoveller. 

You must reserve sufficient amount of 2MB pages:

echo N > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages 

<!---
Configure how the cuckoohash_map memory is allocated in Makefile;
either it interleaves itself on all nodes, or binds to socket 0.

Currently, we cannot compact multiple segments if their total live
size exceeds the maximum size of one segment.
-->

--- A Simple Example: create, read and delete an object ---

source code: src/bin/example.rs

Step 1. Reserve sufficient amountof 2MB hugpages
```
sudo su -c 'echo 30000 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages' 
```
Step 2. Compile and run the test example
```
cargo run --bin example --release
```
Environment variable NIBDEBUG configures debug messages. You can set NIBDEBUG=(1-5) to display more or less debug messages.

<!----
**Warning**

The following examples require much larger systems (e.g., HPE SuperDomeX).

--- YCSB ---

Build the ycsb benchmark code with these instructions. You can link it
with Shoveller, RAMCloud, or MICA. The latter two require you
build/install the respective shared libraries.

    cargo build --bin ycsb [--release]

 --- RAMCloud ---

    (download/compile/install ramcloud-scale-hacks.git)
    cargo build --features "extern_ycsb rc" --bin ycsb [--release]

    RAMCLOUD_ARGV="-t MB -h % [...]" RAMCLOUD_NARG=N \
                   target/[debug|release]/ycsb [args]

--- MICA ---

    (download/compile/install mica-kvs.git)
    cargo build --features "extern_ycsb mica" --bin ycsb [--release]

    TODO execution instructions
-->

### Basic API Overview

Shoveller's main API acts like a key-value store (KVS).  It currently
runs as a library within a single process address space, and supports
concurrent access from many threads.  Shoveller requires objects to be
associated with ``keys'' which are fixed at 8 bytes (2^64 values).
Objects can be any size up to the size of Shoveller's internal segment
length (default is 32 MiB).  Keys exist within a single namespace.


One creates an instance of Shoveller and invokes methods directly on
that object instance:

```
let mut kvs = Shoveller::default();
```

All public methods return a Status object, which is an alias of type
[std::result::Result](https://doc.rust-lang.org/nightly/std/result/enum.Result.html).
To create or update an existing object, use `put_where`:

```
fn put_where(obj: &ObjDesc, socket: u32) -> Status
let key: u64 = 1;
let v: Vec<u64> = vec![1u64,2,3,4,5];
let p = Pointer(value.as_ptr() as *const u8);
let obj = ObjDesc::new(key, p, v.len()*8);
assert!(kvs.put_where(&obj, 0).is_ok());
```

`ObjDesc` is a metatype that simplifies the argument list:

```
struct ObjDesc {
    key: u64,
    value: *const u8,
    vlen: u64,
}
```

To read an object from the store use the following:

```
fn get_object(key: u64, buf: &mut [u8]) -> Status
let key: u64 = 1;
let mut buf: Vec<u64> = Vec::with_capacity(8);
assert!(kvs.get_object(key, &mut buf).is_ok());
// Use 'buf' as desired
```

It will write the object to the provided "buf" input parameter, which
is allocated by the caller.

To remove an object from the store, or check if it exists:

```
fn del_object(key: u64) -> Status
fn exists(key: u64) -> Status
let key: u64 = 1;
assert!(kvs.del_object(key).is_ok());
assert!(kvs.exists(key).is_err());
```

Compaction is enabled manually by invoking the appropriate methods
(below).  By default, eight threads are spawned on each processor
socket to provide compaction for the local memory.  They will only
engage once 20% of remaining space is free.

```
for node in 0..numa::NODE_MAP.sockets() {
    kvs.enable_compaction(NodeId(node));
}
```

## Notes
<!--
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
-->
##### Shoveller currently does not support the following:
- Networked environments.
- Persistent data (e.g., NVM, or disk).  Topic of future work.

##### Shoveller requires systems with a minimum of 32 GiB of memory, reserved as 2MiB mappings with Linux, which it reserves on startup:
```
echo 16384 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
```
Debugging messages are supported via environment variables:
```
# N is a value in the inclusive range [0,5]
NIB_DEBUG=N cargo run --bin ycsb --release
```

Binaries are written to ``target/release/`` or
``target/debug/``. Omitting ``--release`` on build will generate the
latter, and enable easy debugging with GDB.  There are many tunable
parameters found at the top of many source files.

## See Also

- HPE integrity SuperdomeX. https://www.hpe.com/ us/en/servers/superdome.html, January 2016.
