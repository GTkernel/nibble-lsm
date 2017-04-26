use libc;
use std::mem;
use std::thread;
use std::time::{Duration};
use kvs::numa;
use kvs::common;

// TODO make afunction that runs a closure while pinned. that way we
// can set and then unset the cpu mask within the function itself...

/// Pins calling thread to cpu.
#[allow(dead_code)]
pub unsafe fn pin_cpu(cpu: usize) {
    let mut mask:     libc::cpu_set_t = mem::zeroed();
    let tid: libc::pid_t = 0;
    libc::CPU_ZERO(&mut mask);
    libc::CPU_SET(cpu, &mut mask);
    let ret: libc::c_int;
    let size = mem::size_of::<libc::cpu_set_t>();
    ret = libc::sched_setaffinity(tid, size, &mut mask);
    assert_eq!(ret, 0);
}

/// Pins calling thread to all cores of the given socket.
#[allow(dead_code)]
pub unsafe fn pin_socket(sock: usize) {
    let mut mask:     libc::cpu_set_t = mem::zeroed();
    let tid: libc::pid_t = 0;
    let cpus = numa::NODE_MAP.cpus_of(numa::NodeId(sock)).get();
    libc::CPU_ZERO(&mut mask);
    for cpu in cpus {
        libc::CPU_SET(cpu, &mut mask);
    }
    let ret: libc::c_int;
    let size = mem::size_of::<libc::cpu_set_t>();
    ret = libc::sched_setaffinity(tid, size, &mut mask);
    assert_eq!(ret, 0);
}

/// Run a closure on a specific cpu, restoring original CPU mask prior
/// to returning.
pub unsafe fn pin_map<F>(cpu: usize, func: F) where F: Fn() {
    let mut mask:     libc::cpu_set_t = mem::zeroed();
    let mut tmpmask:  libc::cpu_set_t = mem::zeroed();
    let tid: libc::pid_t = 0;

    let mut ret: libc::c_int;
    let size = mem::size_of::<libc::cpu_set_t>();

    // save old mask
    libc::CPU_ZERO(&mut mask);
    ret = libc::sched_getaffinity(tid, size, &mut mask);
    assert_eq!(ret, 0);

    // set temporary mask
    libc::CPU_ZERO(&mut tmpmask);
    libc::CPU_SET(cpu, &mut tmpmask);
    ret = libc::sched_setaffinity(tid, size, &mut tmpmask);
    assert_eq!(ret, 0);
    thread::yield_now();

    func();

    // restore mask
    ret = libc::sched_setaffinity(tid, size, &mut mask);
    assert_eq!(ret, 0);
}

/// Sleep the calling thread for a random amount of time bounded above
/// and below by low_ms and high_ms.
pub fn sleep_rand(low_ms: usize, high_ms: usize) {
    debug_assert!(high_ms > low_ms);
    let d = high_ms - low_ms;
    let t = unsafe { common::rdrandq() % d as u64 } + low_ms as u64;
    let dur = Duration::from_millis(t);
    thread::sleep(dur);
}

/// Helper function to sleep a random amount of time, sub-second.
pub fn sleep_short() {
    sleep_rand(200, 500);
}

