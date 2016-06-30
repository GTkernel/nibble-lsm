use std::time::{Instant,Duration};

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
    asm!("rdtsc" : "={eax}" (low), "={edx}" (high));
    ((high as u64) << 32) | (low as u64)
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

