// The Rust interface to the cuckoo hashtable. We link against
// cuckoo.a (not the actual cityhasher or cuckoohash_map.hh files)
// which gets built by build.rs from cuckoo.cc
// See: https://doc.rust-lang.org/book/ffi.html

use std::ffi;
use libc::c_char;

pub fn init() {
    unsafe {
        libcuckoo_init();
    }
}

pub fn clear() {
    unsafe {
        libcuckoo_clear();
    }
}

pub fn size() -> usize {
    unsafe {
        libcuckoo_size()
    }
}

pub fn empty() -> bool {
    unsafe {
        libcuckoo_empty()
    }
}

pub fn insert(key: u64, value: usize) -> bool {
    unsafe {
        libcuckoo_insert(key, value)
    }
}

pub fn contains(key: u64) -> bool {
    unsafe {
        libcuckoo_contains(key)
    }
}

pub fn erase(key: u64, value: &mut usize) -> bool {
    unsafe {
        libcuckoo_erase(key, value)
    }
}

pub fn find(key: u64) -> Option<usize> {
    let mut value: usize = 0;
    unsafe {
        match libcuckoo_find(key, &mut value) {
            true => Some(value),
            false => None,
        }
    }
}

pub fn update(key: u64, value: usize) -> Option<usize> {
    let mut ret: usize = 0;
    unsafe {
        match libcuckoo_update(key, value, &mut ret) {
            false => None, // insertion
            true => Some(ret), // updated existing
        }
    }
}

/// The lower-level raw interface called by the above.
/// Repeat the cuckoo.cc interface but in Rust syntax.
/// We don't invoke the interface in cuckoo.cc directly; instead, use
/// the Rust-based methods below this extern block.
#[link(name = "cuckoo")]
extern {
    // FIXME instead of &mut use *mut
    fn libcuckoo_init();
    fn libcuckoo_clear();
    fn libcuckoo_size() -> usize;
    fn libcuckoo_empty() -> bool;
    fn libcuckoo_insert(key: u64, value: usize) -> bool;
    fn libcuckoo_contains(key: u64) -> bool;
    fn libcuckoo_find(key: u64, value: &mut usize) -> bool;
    fn libcuckoo_erase(key: u64, value: &mut usize) -> bool;
    fn libcuckoo_update(key: u64, value: usize,
                        old: &mut usize) -> bool;
}

#[cfg(tests)]
mod test {
    fn init() {
        super::init();
    }
}
