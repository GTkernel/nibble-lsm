use libc;
use syscall;
use crossbeam;

use std::mem;
use std::ptr;
use std::time::Instant;
use std::slice;
use std::sync::atomic::{AtomicUsize, Ordering};

use numa::{self,NodeId};
use common::{Pointer,errno};

//==----------------------------------------------------==//
//      Memory copying
//==----------------------------------------------------==//

#[inline(always)] pub
unsafe fn copy(dst: *const u8, src: *const u8, len: usize) {
    asm!("rep movsb"
         :
         : "{rcx}" (len), "{rdi}" (dst), "{rsi}" (src)
         : "rcx", "rsi", "rdi"
         : "volatile");
}

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
#[derive(Debug)]
pub struct Buffer {
    pub addr: Pointer<u8>,
    pub len: usize,
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

    pub unsafe fn as_slice(&mut self) -> &mut [u8] {
        slice::from_raw_parts_mut(self.addr.0 as *mut u8,
                                  self.len)
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
/// This needs to be aligned on BLOCK_SIZE boundary as we mask virtual
/// addresses within this map to test for contiguity of objects.
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
        debug!("mmap 0x{:x}-0x{:x} {} MiB",
              addr, (addr+len), len>>20);
        assert!(addr != libc::MAP_FAILED as usize);
        MemMap { addr: addr, len: len }
    }

    // map and allocate anon memory, bound to a socket
    // shm segments + hugepg + numa on Linux asinine to get working
    // alignment must be power of two
    pub fn numa(len: usize, node: NodeId,
                align: usize, alloc: bool) -> Self {

        assert!(align.is_power_of_two());
        assert!(align >= 4096);
        assert!(align < len);

        debug!("len {} node {} align {}", len, node.0, align);
        let prot: libc::c_int = libc::PROT_READ | libc::PROT_WRITE;
        let flags: libc::c_int = libc::MAP_ANON |
            libc::MAP_PRIVATE | libc::MAP_NORESERVE |
            libc::MAP_HUGETLB;
        let mut addr: usize = unsafe {
            let p = 0 as *mut libc::c_void;
            libc::mmap(p, len + align, prot, flags, -1, 0) as usize
        };
        if addr == libc::MAP_FAILED as usize {
            panic!("mmap: {}", unsafe{errno()});
        }
        debug!("mmap    0x{:x}-0x{:x} {} MiB",
              addr, (addr+len), len>>20);

        // fix the alignment
        // TODO release unused memory
        if addr & (align - 1)  > 0 {
            addr = (addr + align) & !(align-1);
        }
        debug!("aligned 0x{:x}-0x{:x} to {}",
              addr, (addr+len), align);

        // bind the memory to a socket
        // mbind has no wrapper, currently
        let nnodes = numa::NODE_MAP.sockets();
        let mask = 1usize << node.0;
        unsafe {
            let maskaddr = &mask as *const usize as usize;
            // syscall returns usize, but it's really an i32
            let ret: i32 = syscall::syscall6(syscall::nr::MBIND,
                addr, len, numa::MPOL_BIND,
                maskaddr, nnodes+1, numa::MPOL_MF_STRICT) as i32;
            assert_eq!(ret, 0, "mbind: {}", ret);
        }

        // fault pages in
        if alloc {
            let mut guards = vec![];
            let idx = AtomicUsize::new(0);
            let now = Instant::now();
            let nthreads = 8;
            debug!("Using {} threads to fault in MemMap region",
                  nthreads);
            crossbeam::scope(|scope| {
                for _ in 0..nthreads {
                    let guard = scope.spawn(|| {
                        let amt = len / nthreads;
                        let o = Ordering::Relaxed;
                        let i = idx.fetch_add(1, o);
                        let offset = amt * i;
                        unsafe {
                            let startpg = offset>>12;
                            let endpg   = (offset+amt)>>12;
                            for pg in startpg..endpg {
                                let pos: usize = addr + (pg<<12);
                                ptr::write_volatile(pos as *mut usize, 0usize);
                            }
                        }
                    });
                    guards.push(guard);
                }
            });
            for guard in guards {
                guard.join();
            }
            info!("alloc node {}: {} sec", node, now.elapsed().as_secs());
        }
        MemMap { addr: addr, len: len }
    }

    /// Tell the operating system that, upon a crash, to exclude this
    /// memory mapping from a core dump file.
    pub fn exclude_corefile(&self) {
        unsafe {
            let addr = self.addr as *const usize as *mut usize
                as *mut libc::c_void;
            let ret = libc::madvise(addr, self.len,
                                    libc::MADV_DONTDUMP);
            assert_eq!(ret, 0, "madvise: {}", ret);
        }
    }

    pub fn addr(&self) -> usize { self.addr }
    pub fn len(&self) -> usize { self.len }

    /// Wipe the memory region to zeros.
    pub unsafe fn clear(&mut self) {
        ptr::write_bytes(self.addr as *mut u8, 0 as u8, self.len);
    }

    /// Wipe a subset of the map with zeros
    pub unsafe fn clear_region(&self, offset: usize, len: usize) {
        let addr: *mut u8 = (self.addr + offset)
            as *const usize as *mut usize as *mut u8;
        ptr::write_bytes(addr, 0u8, len);
    }
}

/// Prevent dangling regions by unmapping it.
impl Drop for MemMap {

    fn drop (&mut self) {
        debug!("unmapping 0x{:x}", self.addr);
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
