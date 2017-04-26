/*
 * Ticket lock implementation.
 *
 * John M. Mellor-Crummey and Michael L. Scott. 1991. Algorithms for
 * scalable synchronization on shared-memory multiprocessors. ACM
 * Trans. Comput. Syst. 9, 1 (February 1991), 21-65.
 * DOI=http://dx.doi.org/10.1145/103727.103729
 *
 */

use std::sync::atomic::{self, Ordering};
use std::intrinsics;
use std::ptr;

// TODO make next,now u32
pub struct TicketLock {
    next: u64,
    now:  u64,
}

// Allow sharing among threads
unsafe impl Send for TicketLock {}
unsafe impl Sync for TicketLock {}

impl TicketLock {

    pub fn new() -> Self {
        TicketLock {
            next: 0_u64,
            now:  0_u64,
        }
    }

    fn incr_next(&self) -> u64 {
        let nextp = &self.next as *const u64 as *mut u64;
        unsafe {
            intrinsics::atomic_xadd_acq(nextp, 1)
        }
    }

    fn incr_now(&self) {
        let nowp = &self.now as *const u64 as *mut u64;
        unsafe {
            let v = ptr::read_volatile(nowp);
            ptr::write_volatile(nowp, v+1);
        }
    }

    fn read_now(&self) -> u64 {
        let nowp = &self.now as *const u64;
        unsafe {
            atomic::fence(Ordering::Acquire);
            ptr::read_volatile(nowp)
        }
    }

    fn eq_now_next(&self) -> bool {
        let nowp  = &self.now  as *const u64 as *mut u64;
        let nextp = &self.next as *const u64 as *mut u64;
        unsafe {
            ptr::read_volatile(nowp) ==
                ptr::read_volatile(nextp)
        }
    }

    pub fn lock(&self) {
        let mine = self.incr_next();
        loop {
            // TODO pause
            if mine == self.read_now() {
                break;
            }
        }
    }

    pub fn try_lock(&self) -> bool {
        unimplemented!();
    }

    pub fn unlock(&self) {
        if self.eq_now_next() { return; }
        self.incr_now();
    }
}

mod tests {
    use super::*;
    use std::thread::{self,JoinHandle};
    use std::ptr;

    #[test]
    fn init() {
        let t = TicketLock::new();
        assert_eq!(t.next, 0);
        assert_eq!(t.now, 0);
    }

    #[test]
    fn lock_unlock() {
        let t = TicketLock::new();

        t.lock();
        assert_eq!(t.next, 1);
        assert_eq!(t.now, 0);
        t.unlock();
        assert_eq!(t.next, 1);
        assert_eq!(t.now, 1);

        t.lock();
        assert_eq!(t.next, 2);
        assert_eq!(t.now, 1);
        t.unlock();
        assert_eq!(t.next, 2);
        assert_eq!(t.now, 2);
    }

    #[test]
    fn multi_unlock() {
        let t = TicketLock::new();
        t.unlock();
        assert_eq!(t.next, 0);
        assert_eq!(t.now, 0);
        t.lock();
        t.unlock();
        assert_eq!(t.next, 1);
        assert_eq!(t.now, 1);
        t.unlock();
        assert_eq!(t.next, 1);
        assert_eq!(t.now, 1);
    }

    #[test]
    fn threads() {
        let b = Box::new(TicketLock::new());
        let u = Box::into_raw(b) as u64;

        let shared: u64 = 0;
        let mut tids: Vec<JoinHandle<()>> = Vec::new();

        let nthreads = 8;
        let niters = 1_000_000;

        for _ in 0..nthreads {
            let a = &shared as *const u64 as u64;
            let t = thread::spawn( move || {
                let p = a as *const u64 as *mut u64;
                let tlock = unsafe { &*(u as *const TicketLock) };
                for _ in 0..niters {
                    tlock.lock();
                    unsafe {
                        let v = ptr::read_volatile(p);
                        ptr::write_volatile(p, v+1);
                    }
                    tlock.unlock();
                }
            });
            tids.push(t);
        }

        for t in tids {
            let _ = t.join();
        }
        assert_eq!(shared, nthreads*niters);

        let tlock = unsafe {
            Box::from_raw(u as *const TicketLock
                            as *mut TicketLock)
        };

        assert_eq!(tlock.next, nthreads*niters);
        assert_eq!(tlock.now, nthreads*niters);
    }
}
