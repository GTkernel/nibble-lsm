use libc;
use num;
use std::ptr;
use std::slice;

use std::intrinsics;

//==----------------------------------------------------==//
//      General types
//==----------------------------------------------------==//

/// Size of a cache line in bytes.
pub const CACHE_LINE: usize = 64;

/// An unsafe type to share pointers across threads >:)
#[derive(Copy,Clone,Debug)]
pub struct Pointer<T>(pub *const T);
unsafe impl<T> Send for Pointer<T> {}
unsafe impl<T> Sync for Pointer<T> {}

/// A 'slice' skirting the borrow-checker (until I can find a sane
/// design to avoid the need for this).
#[derive(Copy,Clone,Debug)]
pub struct uslice<T>( usize, Pointer<T> );
impl<T> uslice<T> {
    pub fn null() -> Self {
        uslice( 0, Pointer(ptr::null::<T>()) )
    }
    pub fn make(v: &Vec<T>) -> Self {
        uslice( v.len(), Pointer(v.as_ptr()) )
    }
    pub fn ptr(&self) -> *const T { self.1 .0 }
    pub fn len(&self) -> usize { self.0 }
    pub unsafe fn slice(&self) -> &[T] {
        slice::from_raw_parts(self.1 .0, self.0)
    }
}
unsafe impl<T> Send for uslice<T> {}
unsafe impl<T> Sync for uslice<T> {}

//==----------------------------------------------------==//
//      Atomics and numbers and hashing
//==----------------------------------------------------==//

#[inline] pub unsafe
fn atomic_add<T: num::Integer>(loc: *mut T, amt: T) {
    intrinsics::atomic_xadd(loc, amt);
}

#[inline] pub unsafe
fn atomic_cas<T: num::Integer>(loc: *mut T, old: T, val: T) -> (T,bool) {
    intrinsics::atomic_cxchg(loc, old, val)
    // intrinsics::atomic_cxchg_failrelaxed
}

#[inline] pub unsafe
fn volatile_add<T: num::Integer>(loc: *mut T, amt: T) {
    ptr::write_volatile(loc,
        ptr::read_volatile(loc) + amt);
}

#[inline] pub
fn is_even<T: num::Integer>(value: T) -> bool {
    value.is_even()
}

#[inline] pub
fn is_odd<T: num::Integer>(value: T) -> bool {
    value.is_odd()
}

pub const FNV_OFFSET_BASIS_64: u64 = 0xcbf29ce484222325_u64;
pub const FNV_PRIME_64: u64 = 0x100000001b3_u64;

#[inline] pub
fn fnv1a(value: u64) -> u64 {
    let mut hash: u64 = FNV_OFFSET_BASIS_64;
    let p = &value as *const u64 as *const u8;
    let bytes: &[u8] =
        unsafe {
            slice::from_raw_parts(p, 8)
        };
    for b in bytes {
        hash = (hash ^ (*b as u64)).wrapping_mul(FNV_PRIME_64);
    }
    hash
}

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

