// The Rust interface to the cuckoo hashtable. We link against
// cuckoo.a (not the actual cityhasher or cuckoohash_map.hh files)
// which gets built by build.rs from cuckoo.cc
// See: https://doc.rust-lang.org/book/ffi.html

//use std::ffi;
use std::os::raw::c_void;
use std::ptr;

pub type CVoidPointer = *const c_void;

pub fn init(numa_mask: usize, nnodes: usize) {
    unsafe {
        libcuckoo_init(numa_mask, nnodes);
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

#[inline(always)]
pub fn insert(key: u64, value: usize) -> bool {
    unsafe {
        libcuckoo_insert(key, value)
    }
}

#[inline(always)]
pub fn contains(key: u64) -> bool {
    unsafe {
        libcuckoo_contains(key)
    }
}

#[inline(always)]
pub fn erase(key: u64, value: &mut usize) -> bool {
    unsafe {
        libcuckoo_erase(key, value)
    }
}

#[inline(always)]
pub fn find(key: u64) -> Option<usize> {
    let mut value: usize = 0;
    unsafe {
        match libcuckoo_find(key, &mut value) {
            true => Some(value),
            false => None,
        }
    }
}

#[inline(always)]
pub fn update(key: u64, value: usize) -> Option<usize> {
    let mut ret: usize = 0;
    unsafe {
        match libcuckoo_update(key, value, &mut ret) {
            false => None, // insertion
            true => Some(ret), // updated existing
        }
    }
}

#[inline(always)]
pub fn update_hold_ifeq(key: u64, value: usize, cmp: usize)
    -> Option<CVoidPointer> {
    unsafe {
        let p: CVoidPointer;
        p = libcuckoo_update_hold_ifeq(key, value, cmp);
        if p.is_null() {
            None
        } else {
            Some(p)
        }
    }
}

#[inline(always)]
pub fn update_release(obj: *const c_void) {
    unsafe {
        libcuckoo_update_release(obj);
    }
}

pub fn print_conflicts(pct: usize) {
    unsafe {
        libcuckoo_print_conflicts(pct);
    }
}

/// The lower-level raw interface called by the above.
/// Repeat the cuckoo.cc interface but in Rust syntax.
/// We don't invoke the interface in cuckoo.cc directly; instead, use
/// the Rust-based methods above this extern block.
#[link(name = "cuckoo")]
extern {
    // FIXME instead of &mut use *mut
    fn libcuckoo_init(numa_mask: usize, nnodes: usize);
    fn libcuckoo_clear();
    fn libcuckoo_size() -> usize;
    fn libcuckoo_empty() -> bool;
    fn libcuckoo_insert(key: u64, value: usize) -> bool;
    fn libcuckoo_contains(key: u64) -> bool;
    fn libcuckoo_find(key: u64, value: &mut usize) -> bool;
    fn libcuckoo_erase(key: u64, value: &mut usize) -> bool;
    fn libcuckoo_update(key: u64, value: usize,
                        old: &mut usize) -> bool;
    fn libcuckoo_print_conflicts(pct: usize);
    fn libcuckoo_update_hold_ifeq(key: u64, value: usize,
                                  cmp: usize) -> CVoidPointer;
    fn libcuckoo_update_release(obj: CVoidPointer);
}

#[cfg(tests)]
mod test {
    fn init() {
        super::init();
    }
}
