# Nibble: Scalable Large-Memory Key-Value Store
## Description
**Nibble** is a scalable, capacity efficient key-value store for
large in-memory data. Nibble promotes the use of a concurrent
multi-head log-structured approach to attain high performance and
resistance to memory fragmentation, together with methods for scalable
low-latency synchronization that benefits data-intensive applications
on large scale machines with massive memory (e.g., tens of terabytes
or more) and hundreds of CPU cores.

## Master Source
(HPE) https://github.hpe.com/labs/nibble/
(GA Tech) https://github.gatech.edu/kernel/nibble-lsm

## Maturity
Research prototype, approximately 7.5k lines of code. It has been
measured to run with up to 288 threads on 12 TiB of data.

With 240 threads on 1TiB of data with random access to 1KiB objects:
- 40 mil. GET/sec for pure reads (38 GiBps)
- 90 mil. PUT/sec for pure writes (85 GiBps)

PUT throughput is higher due to NUMA effects; a 16-socket platform
will have higher random-read latencies, than writing to a socket's
local memory (the default behavior).

For high-locality workloads (zipfian 0.99):
- 2500 mil. GET/sec for pure reads
- 2 mil. PUT/sec for pure writes

GET throughput is extremely high due to the small working set size; it
is able to fit within each processor's LLC to avoid memory latencies.

PUT throughput is low, as a large number of concurrent writers to a
small number of objects will result in significant synchronization
contention. The current design does not allow for concurrent updates
to an individual object.

## Dependencies
Nibble is implemented entirely in the [Rust
language](https://www.rust-lang.org/en-US/) and requires the
``nightly'' branch of the compiler. Installation can be done to
_without root administration_ from https://rustup.rs

Rust is typically installed via non-root user into the user's home
directory. Please visit [rustup](https://rustup.rs) and execute the
given shell command. Configure 'nightly' to be the default toolchain.

Verify the installation (example below).
```
% rustc --version
rustc 1.16.0-nightly (4ce7accaa 2017-01-17)
```

## Building Nibble
Building Nibble is done via the Cargo package manager (provided with
the rustup script above):

```
% cargo build --lib [--release]
```

To build a specific executable (found in src/bin/):

```
% cargo build --bin <name> [--release]
```

Omitting '--release' will create a debug binary. All binaries are
written to either target/release/ or target/debug/.

TODO how to add flags for rdrand

## Usage 
Nibble's main API acts like a key-value store (KVS). It currently runs
as a library within a single process address space, and supports
concurrent access from many threads. Nibble requires objects to be
associated with ``keys'' which are fixed at 8 bytes (2^64 values).
Objects can be any size up to the size of Nibble's internal segment
length (default is 32 MiB). Keys exist within a single namespace.

#### Basic API Overview
One creates an instance of Nibble and invokes methods directly on that
object instance:

```
let mut nib = Nibble::default();
```

All public methods return a Status object, which is an alias of type
[std::result::Result](https://doc.rust-lang.org/nightly/std/result/enum.Result.html).

To create or update an existing object, use put_where:
```
fn put_where(obj: &ObjDesc, socket: u32) -> Status

let key: u64 = 1;
let v: Vec<u64> = vec![1u64,2,3,4,5];
let p = Pointer(value.as_ptr() as *const u8);
let obj = ObjDesc::new(key, p, v.len()*8);
assert!(nib.put_where(&obj, 0).is_ok());
```
ObjDesc is a metatype that simplifies the argument list:
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
assert!(nib.get_object(key, &mut buf).is_ok());
// Use 'buf' as desired
```
It will write the object to the provided “buf” input parameter,
which is allocated by the caller.

To remove an object from the store, or check if it exists:
```
fn del_object(key: u64) -> Status
fn exists(key: u64) -> Status

let key: u64 = 1;
assert!(nib.del_object(key).is_ok());
assert!(nib.exists(key).is_err());
```

Compaction is enabled manually by invoking the appropriate methods
(below). By default, eight threads are spawned on each processor
socket to provide compaction for the local memory. They will only
engage once 20% of remaining space is free.

```
for node in 0..numa::NODE_MAP.sockets() {
    nibble.enable_compaction(NodeId(node));
}
```

Nibble can provide runtime and diagnostic information via an
environment variable:

```
NIBDEBUG=N cargo run --bin <name>
```

N is a value from 1-5 for errors (least verbose), warnings, general information,
debug, and trace messages (most verbose). The recommended is 3.

## Notes

Nibble currently does not support the following:
- Networked environments .
- Persistent data (e.g., NVM, or disk). Topic of future work.

Nibble requires systems with a minimum of 32 GiB of memory, reserved
as 2MiB mappings with Linux, which it reserves on startup:

```
echo 16384 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
```

Debugging messages are supported via environment variables:

```
# N is a value in the inclusive range [0,5]
NIB_DEBUG=N cargo run --bin ycsb --release
```

Binaries are written to ``target/release/`` or ``target/debug/``.
Omitting ``--release`` on build will generate the latter, and enable
easy debugging with GDB.

There are many tunable parameters found at the top of many source
files.

## See Also

We have submitted various invention disclosures associated with this
work and a research paper is currently in-submission.
