use libc;

use std::mem;
use std::ptr;
use std::ptr::copy;
use std::ptr::copy_nonoverlapping;

//==----------------------------------------------------==//
//      Heap allocation
//==----------------------------------------------------==//

// Use Vec to dynamically manage memory for us.
// NOTE this is a hack.

pub fn allocate<T>(count: usize) -> *mut T {
    let mut v = Vec::with_capacity(count);
    let ptr = v.as_mut_ptr();
    mem::forget(v);
    ptr
}

pub unsafe fn deallocate<T>(ptr: *mut T, count: usize) {
    mem::drop(Vec::from_raw_parts(ptr, 0, count));
}

//==----------------------------------------------------==//
//      Buffer
//==----------------------------------------------------==//

/// Generic heap buffer that uses malloc underneath. Might be better
/// to back Buffers with a slab allocator to avoid object sizes
/// becoming exposed to the heap allocator.
pub struct Buffer {
    addr: usize,
    len: usize,
}

impl Buffer {

    pub fn new(len: usize) -> Self {
        let va: usize = unsafe { libc::malloc(len) as usize };
        assert!(va != 0);
        Buffer { addr: va, len: len }
    }

    pub fn getaddr(&self) -> usize { self.addr }
    pub fn getlen(&self) -> usize { self.len }

    // TODO append()
}

impl Drop for Buffer {
    fn drop (&mut self) {
        if self.addr > 0 {
            unsafe { libc::free(self.addr as *mut libc::c_void); }
        }
    }
}

//==----------------------------------------------------==//
//      Memory map
//==----------------------------------------------------==//

/// Memory mapped region in our address space.
pub struct MemMap {
    addr: usize,
    len: usize,
}

/// Create anonymous private memory mapped region.
impl MemMap {

    pub fn new(len: usize) -> Self {
        // TODO fault on local socket
        let prot: libc::c_int = libc::PROT_READ | libc::PROT_WRITE;
        let flags: libc::c_int = libc::MAP_ANON |
            libc::MAP_PRIVATE | libc::MAP_NORESERVE;
        let addr: usize = unsafe {
            let p = 0 as *mut libc::c_void;
            libc::mmap(p, len, prot, flags, 0, 0) as usize
        };
        info!("mmap 0x{:x}-0x{:x} {} MiB",
              addr, (addr+len), len>>20);
        assert!(addr != libc::MAP_FAILED as usize);
        MemMap { addr: addr, len: len }
    }

    pub fn addr(&self) -> usize { self.addr }
    pub fn len(&self) -> usize { self.len }

    /// Wipe the memory region to zeros.
    pub unsafe fn clear(&mut self) {
        ptr::write_bytes(self.addr as *mut u8, 0 as u8, self.len);
    }
}

/// Prevent dangling regions by unmapping it.
impl Drop for MemMap {

    fn drop (&mut self) {
        info!("unmapping 0x{:x}", self.addr);
        let p = self.addr as *mut libc::c_void;
        unsafe { libc::munmap(p, self.len); }
    }
}

//==----------------------------------------------------==//
//      Unit tests
//==----------------------------------------------------==//

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::mem::size_of;
    use std::mem::transmute;
    use std::ptr::copy;
    use std::ptr::copy_nonoverlapping;
    use std::rc::Rc;
    use std::sync::Arc;

    use test::Bencher;

    #[test]
    fn memory_map_init() {
        let len = 1<<26;
        let mm = MemMap::new(len);
        assert_eq!(mm.len, len);
        assert!(mm.addr != 0 as usize);
        // TODO touch the memory somehow
        // TODO verify mmap region is unmapped
    }

}
