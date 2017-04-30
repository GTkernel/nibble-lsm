# Shoveller: Scalable Parallel Log-structured Memory for Key-Value Stores

**Authors**:  Alexander Merritt (merritt.alex@gatech.edu), Yuan Chen (yuan.chen@hpe.com)

## Description

**Shoveller** is a scalable, memory-capacity efficient key-value store for very large scale machines (e.g., tens of terabytes or more memory and hundreds of CPU cores). Shoveller promotes the use of a concurrent multi-head log-structured memory to attain high performance and resistance to memory fragmentation, together with scalable low-latency synchronization and optimistically concurrent indexing that allow application threads to scale to hundreds of cores. A prototype for single node system has been implemented, and its effectiveness has been evaluated on a HPE SuperDomeX machine with 240 cores and 12 TiB of DRAM across a wide range of workload patterns. 

## Source

- HPE Internal: https://github.hpe.com/labs/shoveller
- External: https://github.com/HewlettPackard/shoveller

## Maturity

Research prototype. 

## Dependencies

Shoveller is implemented entirely in the [Rust language](https://www.rust-lang.org/en-US/) and requires the ``nightly'' branch of the compiler.  Installation can be done _without root administration_ from [rustup.rs](https://rustup.rs).


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

```
echo N > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
```

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
Environment variable NIBDEBUG configures debug messages. You can set NIBDEBUG=(1-5) to display more or fewer debug messages.

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
let capacity = 1_usize << 38;
let mut kvs = LSM::new(capacity);
```

All public methods return a Status object, which is an alias of type
[std::result::Result](https://doc.rust-lang.org/nightly/std/result/enum.Result.html).
To create or update an existing object, use `put_object`:

```
// function signature
fn put_object(obj: &ObjDesc) -> Status

let key: u64 = 1;
let v: Vec<u64> = vec![1_u64,2,3,4,5];
let p = Pointer(value.as_ptr() as *const u8);
let obj = ObjDesc::new(key, p, v.len()*8);
assert!(kvs.put_object(&obj).is_ok());
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
// function signature
fn get_object(key: u64, buf: &mut [u8]) -> Status

let key: u64 = 1;
let mut buf: Vec<u64> = Vec::with_capacity(8);
assert!(kvs.get_object(key, &mut buf).is_ok());
// Use 'buf' as desired
```

It will write the object to the provided input parameter buffer, which is allocated by the caller.

To remove an object from the store, or check if it exists:

```
// function signatures
fn del_object(key: u64) -> Status
fn exists(key: u64) -> Status

let key: u64 = 1;
assert!(kvs.del_object(key).is_ok());
assert!(kvs.exists(key).is_err());
```

Compaction is enabled manually by invoking the appropriate methods (below).  By default, eight threads are spawned on each processor socket to provide compaction for the local memory.  They will only engage once 20% of remaining space is free. Worker threads are pinned to one specific socket, and only compact the memory for that socket.

```
for node in 0..numa::NODE_MAP.sockets() {
    kvs.enable_compaction(NodeId(node));
}
```

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
