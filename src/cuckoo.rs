// The Rust interface to the cuckoo hashtable. We link against
// cuckoo.a (not the actual cityhasher or cuckoohash_map.hh files)
// which gets built by build.rs from cuckoo.cc
// See: https://doc.rust-lang.org/book/ffi.html

use libc::{size_t};

// Repeat the cuckoo.cc interface but in Rust syntax.
// We don't invoke the interface in cuckoo.cc directly; instead, use
// the Rust-based methods below this extern block.
#[link(name = "cuckoo")]
extern {
    fn libcuckoo_init();
    // TODO add the rest of the functions
}

pub fn init() {
    unsafe {
        libcuckoo_init();
    }
}

#[cfg(tests)]
mod test {
    fn init() {
        super::init();
    }
}
