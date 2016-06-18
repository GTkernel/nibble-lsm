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

pub fn insert(key: &str, value: usize) -> bool {
    let k = key.to_owned(); // TODO can we avoid a copy just for the NUL?
    unsafe {
        let cstr = ffi::CString::from_vec_unchecked(k.into_bytes());
        let p = cstr.into_raw(); // transfer ownership to cuckoo
        libcuckoo_insert(p, value)
    }
}

pub fn contains(key: &str) -> bool {
    let k = key.to_owned(); // TODO see above
    unsafe {
        let cstr = ffi::CString::from_vec_unchecked(k.into_bytes());
        libcuckoo_contains(cstr.as_ptr())
    }
}

pub fn erase(key: &str, value: &mut usize) -> bool {
    let k = key.to_owned(); // TODO see above
    unsafe {
        let cstr = ffi::CString::from_vec_unchecked(k.into_bytes());
        libcuckoo_erase(cstr.as_ptr(), value)
    }
    // FIXME we have to take the value removed as the key and free it
}

pub fn find(key: &mut String) -> Option<usize> {
    let mut value: usize = 0;
    //key.push('\0');
    unsafe {
        match libcuckoo_find(key.as_ptr(), &mut value) {
            true => Some(value),
            false => None,
        }
    }
}

pub fn update(key: &str, value: usize) -> Option<usize> {
    let k = key.to_owned(); // TODO see above
    let mut ret: usize = 0;
    unsafe {
        let cstr = ffi::CString::from_vec_unchecked(k.into_bytes());
        match libcuckoo_update(cstr.as_ptr(), value, &mut ret) {
            false => { // insertion
                cstr.into_raw(); // cuckoo owns the string now
                None
            },
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
    fn libcuckoo_insert(key: *const c_char, value: usize) -> bool;
    fn libcuckoo_contains(key: *const c_char) -> bool;
    fn libcuckoo_find(key: *const u8, value: &mut usize) -> bool;
    fn libcuckoo_erase(key: *const c_char, value: &mut usize) -> bool;
    fn libcuckoo_update(key: *const c_char, value: usize,
                        old: &mut usize) -> bool;
}

#[cfg(tests)]
mod test {
    fn init() {
        super::init();
    }
}
