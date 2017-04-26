use libc;
use num;
use rand::{self,Rng};
use std::ptr;
use std::slice;

use std::intrinsics;

//==----------------------------------------------------==//
//      General types
//==----------------------------------------------------==//

/// Keys are fixed 8-byte unsigned values.
/// TODO use this everywhere we have size_of::<u64>() hard-coded
pub type KeyType = u64;

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

#[inline(always)] pub
fn prefetchw(cacheline: *const u8) {
    unsafe { asm!("prefetchw $0" : : "m" (cacheline) : ) }
}

#[inline(always)] pub
fn prefetch(cacheline: *const u8) {
    unsafe { asm!("prefetch $0" : : "m" (cacheline) : ) }
}

#[inline] pub unsafe
fn atomic_add<T: num::Integer>(loc: *mut T, amt: T) -> T {
    intrinsics::atomic_xadd(loc, amt)
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

/// Even: the lowest bit is zero.
/// FIXME how to make this generic? can't seem to use `&` on generic `T`
#[inline(always)] pub
fn is_even(value: u64) -> bool {
    (value & 1u64) == 0u64
}

/// Odd: not even
#[inline(always)] pub
fn is_odd(value: u64) -> bool {
    !is_even(value)
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

/// Knuth or FY shuffle
pub
fn shuffle<T: num::Integer>(vec: &mut Vec<T>) {
    if vec.is_empty() { return; }
    //let mut rng = rand::thread_rng(); // doesn't scale!
    let n = vec.len();
    for i in 0..(n-1) {
        let ii = (i+1) + (unsafe { rdrand() } as usize % (n-i-1));
        vec.swap(i, ii);
    }
}

//==----------------------------------------------------==//
//      Random stuff
//==----------------------------------------------------==//

// TODO use <T> for rdrand so we don't have two functions.

/// Generate 32-bit random numbers via the CPU's rdrand instruction.
#[inline(always)]
#[allow(unused_mut)]
#[cfg(feature="rdrand")]
pub unsafe fn rdrand() -> u32 {
    let mut r: u32;
    let mut eflags: u8;
    loop {
        asm!("rdrand $0; lahf"
             : "=r" (r), "={ah}"(eflags)
             :
             :
             : "volatile"
            );
        if intrinsics::likely(1 == (eflags & 1)) {
            break;
        }
        warn!("rdrand CF=0");
    }
    r
}

/// Generate 64-bit random numbers via the CPU's rdrand instruction.
#[inline(always)]
#[allow(unused_mut)]
#[cfg(feature="rdrand")]
pub unsafe fn rdrandq() -> u64 {
    let mut r: u64;
    let mut eflags: u8;
    loop {
        asm!("rdrand $0; lahf"
             : "=r" (r), "={ah}"(eflags)
             :
             :
             : "volatile"
            );
        if intrinsics::likely(1 == (eflags & 1)) {
            break;
        }
        warn!("rdrandq CF=0");
    }
    r
}

/// Slower and potentially less-scalable method for
/// random number generation using urandom.
#[cfg(not(feature="rdrand"))]
pub unsafe fn rdrand() -> u32 {
    let mut rng = rand::thread_rng();
    rng.gen::<u32>()
}

/// Slower and potentially less-scalable method for
/// random no. generation.
#[cfg(not(feature="rdrand"))]
pub unsafe fn rdrandq() -> u64 {
    let mut rng = rand::thread_rng();
    rng.gen::<u64>()
}

//==----------------------------------------------------==//
//      CPU Feature Detection (CPUID)
//==----------------------------------------------------==//

pub struct CPUIDRegs {
    eax: u32,
    ebx: u32,
    ecx: u32,
    edx: u32
}

impl CPUIDRegs {
    pub fn new() -> CPUIDRegs {
        CPUIDRegs { eax: 0u32,  ebx: 0u32, ecx: 0u32, edx: 0u32 }
    }
}

#[allow(unused_mut)]
pub unsafe fn cpuid(id: u32, subid: u32) -> CPUIDRegs {
    let mut regs = CPUIDRegs::new();
    asm!("cpuid"
         : "={eax}" (regs.eax),
           "={ebx}" (regs.ebx),
           "={ecx}" (regs.ecx),
           "={edx}" (regs.edx)
         : "{eax}" (id), "{ecx}" (subid)
         :
         : "volatile"
         );
    regs
}

/// Determine whether the CPU has the 'rdrand' instruction.
/// From the manual: CPUID.01H:ECX.RDRAND[bit 30] = 1
#[cfg(feature="rdrand")]
pub fn kvs_rdrand_compile_flags() -> bool {
    info!("rand: using rdrand instruction");
    let id = 1_u32;
    let regs = unsafe { cpuid(id, 0u32) };
    ((regs.ecx >> 30) & 0x1) == 1
}

/// If compiled with 'rdrand' this will always return true.
/// The above rdrand methods will fall back to
/// an implementation without the instruction.
#[cfg(not(feature="rdrand"))]
pub fn kvs_rdrand_compile_flags() -> bool {
    info!("rand: using urandom");
    true
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
    TableFull,

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
        ErrorCode::TableFull     => { "Table is full" },
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

