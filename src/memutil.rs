use libc;

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
        info!("mmap 0x{:x} {} MiB", addr, len>>20);
        assert!(addr != libc::MAP_FAILED as usize);
        MemMap { addr: addr, len: len }
    }
    pub fn addr(&self) -> usize { self.addr }
    pub fn len(&self) -> usize { self.len }
}

/// Prevent dangling regions by unmapping it.
impl Drop for MemMap {
    fn drop (&mut self) {
        unsafe {
            let p = self.addr as *mut libc::c_void;
            libc::munmap(p, self.len);
        }
    }
}

// -----------------------------------------------
// Test Code
// -----------------------------------------------

#[test]
fn map() {
    let len = 1<<26;
    let mm = MemMap::new(len);
    assert_eq!(mm.len, len);
    assert!(mm.addr != 0 as usize);
    // TODO touch the memory somehow
    // TODO verify mmap region is unmapped
}

