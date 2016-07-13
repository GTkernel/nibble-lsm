use libc;
use syscall;

use std::mem;
use std::ptr;
use std::time::Instant;

use numa::{self,NodeId};
use sched;
use common::{Pointer,errno};

//==----------------------------------------------------==//
//      Alignment
//==----------------------------------------------------==//

/// 64-byte type used for aligning data structures.
/// Put a zero-sized array into your structure to align it and have
/// auto-padding.
#[repr(simd)]
#[derive(Debug,Copy,Clone)]
pub struct align64(u64, u64, u64, u64, u64, u64, u64, u64);

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
    addr: Pointer<u8>,
    len: usize,
}

impl Buffer {

    pub fn new(len: usize) -> Self {
        Buffer {
            addr: Pointer(allocate::<u8>(len)),
            len: len
        }
    }

    pub fn getaddr(&self) -> usize {
        self.addr.0 as usize
    }

    pub fn getlen(&self) -> usize {
        self.len
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.addr.0 as *mut u8
    }

    // TODO append()
}

impl Drop for Buffer {
    fn drop (&mut self) {
        if !self.addr.0 .is_null() {
            unsafe { deallocate(self.addr.0 as *mut u8, self.len); }
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

    // map and allocate anon memory, bound to a socket
    // shm segments + hugepg + numa on Linux asinine to get working
    // FIXME until mbind is available, we change the cpu mask before
    // faulting in all pages, then restore mask
    pub fn numa(len: usize, node: NodeId) -> Self {
        debug!("len {} node {}", len, node.0);
        let prot: libc::c_int = libc::PROT_READ | libc::PROT_WRITE;
        let flags: libc::c_int = libc::MAP_ANON |
            libc::MAP_PRIVATE | libc::MAP_NORESERVE |
            libc::MAP_HUGETLB;
        let addr: usize = unsafe {
            let p = 0 as *mut libc::c_void;
            libc::mmap(p, len, prot, flags, -1, 0) as usize
        };
        if addr == libc::MAP_FAILED as usize {
            panic!("mmap: {}", unsafe{errno()});
        }
        info!("mmap 0x{:x}-0x{:x} {} MiB",
              addr, (addr+len), len>>20);

        // bind the memory to a socket
        let nnodes = numa::NODE_MAP.sockets();
        let mask = 1usize << node.0;
        unsafe {
            let maskaddr: usize = mem::transmute(&mask);
            assert!(0usize == syscall::syscall6(syscall::nr::MBIND,
                addr, len, numa::MPOL_BIND,
                maskaddr, nnodes+1, numa::MPOL_MF_STRICT),
                "mbind failed");
        }

        // allocate pages by faulting
        let now = Instant::now();
        unsafe {
            for pg in 0..(len>>12) {
                let pos: usize = addr + (pg<<12);
                ptr::write(pos as *mut usize, 42usize);
            }
        }
        info!("alloc node {}: {} sec", node, now.elapsed().as_secs());
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

#[cfg(IGNORE)]
mod tests {
    use super::*;
    use super::super::logger;
    use super::super::numa::{self,NodeId};
    use std::cmp;

    #[test]
    fn memory_map_init() {
        logger::enable();
        let len = 1<<26;
        let mm = MemMap::new(len);
        assert_eq!(mm.len, len);
        assert!(mm.addr != 0 as usize);
        // TODO touch the memory somehow
        // TODO verify mmap region is unmapped
    }

    #[test]
    fn numa() {
        logger::enable();
        let nnodes = numa::nodes();
        let len = 1<<30;

        for i in 0..cmp::min(1,nnodes) {
            // allocate on socket and verify
            let node = NodeId(i);
            let mm = MemMap::numa(len,node);
            assert_eq!(mm.len, len);
            assert!(mm.addr != 0 as usize);
            let v = numa::numa_allocated();
            let n = v[0];
            println!("allocated: {:?}", v);
            assert!(n >= (len>>21), "n {} len {}", n, len);

            // release the memory and verify
            drop(mm);
            let v = numa::numa_allocated();
            assert!(v[0] < n);
        }
    }
}
