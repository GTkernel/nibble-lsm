use libc;
use std::ptr;

//==----------------------------------------------------==//
//      General types
//==----------------------------------------------------==//

/// Size of a cache line in bytes.
pub const CACHE_LINE: usize = 64;

pub type Pointer = Option<*const u8>;
pub type PointerMut = Option<*mut u8>;

//==----------------------------------------------------==//
//      Random stuff
//==----------------------------------------------------==//

/// Generate 32-bit random numbers via the CPU's rdrand instruction.
#[inline(always)]
#[allow(unused_mut)]
pub unsafe fn rdrand() -> u32 {
    let mut r: u32;
    asm!("rdrand $0" : "=r" (r));
    r
}

/// Generate 64-bit random numbers via the CPU's rdrand instruction.
#[inline(always)]
#[allow(unused_mut)]
pub unsafe fn rdrandq() -> u64 {
    let mut r: u64;
    asm!("rdrand $0" : "=r" (r));
    r
}


//==----------------------------------------------------==//
//      Error handling
//==----------------------------------------------------==//

#[allow(dead_code)]
pub unsafe fn errno() -> i32 {
    let loc = libc::__errno_location();
    ptr::read(loc)
}

#[allow(dead_code)]
pub unsafe fn set_errno(val: i32) {
    let loc = libc::__errno_location();
    ptr::write(loc, val);
}

#[derive(Debug)]
pub enum ErrorCode {

    SegmentFull,
    SegmentClosed,

    OutOfMemory,

    KeyNotExist,
    InvalidSocket,

    EmptyObject,

    ObjectTooBig,
}

pub fn err2str(code: ErrorCode) -> &'static str {
    match code {
        ErrorCode::SegmentFull   => { "Segment is full" },
        ErrorCode::SegmentClosed => { "Segment is closed" },
        ErrorCode::OutOfMemory   => { "Out of memory" },
        ErrorCode::KeyNotExist   => { "Key does not exist" },
        ErrorCode::InvalidSocket => { "Invalid socket ID" },
        ErrorCode::EmptyObject   => { "Object is empty" },
        ErrorCode::ObjectTooBig  => { "Object too big" },
    }
}

pub type Status = Result<(usize), ErrorCode>;

//==----------------------------------------------------==//
//      System info
//==----------------------------------------------------==//

#[cfg(target_os="linux")]
pub fn get_tid() -> i32 {
    unsafe {
        let id = libc::SYS_gettid;
        libc::syscall(id) as i32
    }
}

#[cfg(target_os="macos")]
pub fn get_tid() -> i32 {
    unsafe {
        // let id = libc::SYS_thread_selfid;
        let id = 372; // XXX
        libc::syscall(id) as i32
    }
}

