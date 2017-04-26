/// Implementation of the MCS lock.
/// Create a shared variable of type AtomicPtr<McsQnode> as the main
/// lock.  Each threads creates a private McsQnode and passes that
/// into each of the methods lock and unlock.
use std::ptr;
use std::intrinsics;
use std::thread;
use std::sync::atomic::*;

#[inline(always)]
unsafe fn sfence() {
    asm!("sfence" : : : "memory");
}

pub struct McsQnode {
    next: AtomicPtr<McsQnode>,
    locked: AtomicBool,
}

impl McsQnode {

    pub fn new() -> Self {
        McsQnode {
            next: AtomicPtr::new(0usize as *mut Self),
            locked: AtomicBool::new(false),
        }
    }

    #[inline(always)]
    pub unsafe fn set_next(&self, next: *const Self) {
        self.next.store(next as *mut _, Ordering::SeqCst);
    }

    #[inline(always)]
    pub unsafe fn clear_next(&self) {
        self.next.store(0usize as *mut Self, Ordering::SeqCst);
    }

    #[inline(always)]
    pub unsafe fn get_next(&self) -> *mut Self {
        self.next.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub unsafe fn set_locked(&self, status: bool) {
        self.locked.store(status, Ordering::SeqCst);
    }

    #[inline(always)]
    pub unsafe fn is_locked(&self) -> bool {
        self.locked.load(Ordering::Relaxed)
    }

    pub unsafe fn lock(glob: &AtomicPtr<Self>, slot: &mut Self) {
        slot.clear_next();
        sfence(); // TODO necessary?
        let next_qnode = glob.swap(slot as *mut Self, Ordering::SeqCst);
        if !next_qnode.is_null() {
            slot.set_locked(true);
            sfence();
            (&*next_qnode).set_next(slot as *mut Self);
            while slot.is_locked() { ; }
        }
    }

    pub unsafe fn unlock(glob: &AtomicPtr<Self>, slot: &mut Self) {
        let mut next_qnode = slot.get_next();
        if next_qnode.is_null() {
            let expected = slot as *mut Self;
            let ok = expected == glob.compare_and_swap(expected,
                                0usize as *mut Self, Ordering::SeqCst);
            if ok {
                return;
            } else {
                next_qnode = slot.get_next();
                while next_qnode.is_null() {
                    next_qnode = slot.get_next();
                }
            }
        }
        (&*next_qnode).set_locked(false);
    }
}
