use std::time::Instant;

//==----------------------------------------------------==//
//      Cycles information / state
//==----------------------------------------------------==//

static mut CYCLES_PER_SECOND: u64 = 0u64;

/// Read CPU's time-stamp counter.
/// TODO make safe
#[inline(always)]
#[allow(unused_mut)]
pub unsafe fn rdtsc() -> u64 {
    let mut low: u32;
    let mut high: u32;
    asm!("rdtsc" : "={eax}" (low),
                    "={edx}" (high));
    ((high as u64) << 32) | (low as u64)
}

/// Read the CPU ID using RDTSCP. Returns (socket,core) where core is
/// the global core ID, not the socket-local ID.
/// ca. 30 cycles overhead
/// On Linux, IA32_TSC_AUX is used to hold the core and node
/// identifiers; first 12 bits hold the CPU, and bit 12+ hold the NUMA
/// node. The state of this MSR depends on the implementation of the
/// OS kernel (it must initialize this value).
#[inline(always)]
#[allow(unused_mut)]
pub fn rdtscp_id() -> (u32,u32) {
    let mut ecx: u32;
    unsafe {
        asm!("rdtscp" : "={ecx}" (ecx));
    }
    (ecx >> 12, ecx & ((1<<12)-1))
}

// TODO when RDPID is available, use that instead of RDTSCP

/// Same as calling rdtsc but we internalize the unsafe block
#[inline(always)]
pub fn now() -> u64 {
    unsafe { rdtsc() }
}

fn init() {
    let now = Instant::now();
    let start = unsafe { rdtsc() };
    loop {
        let dur = now.elapsed();
        if dur.as_secs() >= 1u64 {
            break;
        }
    }
    let dur = now.elapsed();
    let end = unsafe { rdtsc() };

    let tim = dur.as_secs() * 10u64.pow(9) +
        dur.subsec_nanos() as u64;
    let sec = (tim as f64)/1e9;

    let cycles = (end-start) as f64 / sec;
    unsafe {
        CYCLES_PER_SECOND = cycles as u64;
    }
}

macro_rules! do_init {
    () => { unsafe {
        if CYCLES_PER_SECOND == 0u64 {
            init();
        }
    }}
}

/// Return the above global but hide the unsafe block. We only write
/// it during init() so shouldn't matter.
#[inline(always)]
pub fn per_second() -> u64 {
    do_init!();
    unsafe { CYCLES_PER_SECOND }
}

#[inline(always)]
pub fn per_nano() -> u64 {
    do_init!();
    unsafe { CYCLES_PER_SECOND / 1_000_000_000u64 }
}

#[inline(always)]
pub fn to_seconds(cycles: u64) -> u64 {
    do_init!();
    cycles / unsafe { CYCLES_PER_SECOND }
}

#[inline(always)]
pub fn to_secondsf(cycles: u64) -> f64 {
    do_init!();
    cycles as f64 / unsafe { CYCLES_PER_SECOND as f64 }
}

#[inline(always)]
pub fn to_msec(cycles: u64) -> u64 {
    do_init!();
    cycles / unsafe { CYCLES_PER_SECOND / 1_000u64 }
}

#[inline(always)]
pub fn to_usec(cycles: u64) -> u64 {
    do_init!();
    cycles / unsafe { CYCLES_PER_SECOND / 1_000_000u64 }
}

#[inline(always)]
pub fn to_nano(cycles: u64) -> u64 {
    do_init!();
    cycles / unsafe { CYCLES_PER_SECOND / 1_000_000_000u64 }
}

#[inline(always)]
pub fn from_nano(nanos: u64) -> u64 {
    do_init!();
    (unsafe { CYCLES_PER_SECOND as f64
        * (nanos as f64 / 1e9) }) as u64
}

