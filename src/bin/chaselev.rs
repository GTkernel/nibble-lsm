/// Simple multi-threaded test for the Chase-Lev work stealing deque
/// https://github.com/aturon/crossbeam
/// http://aturon.github.io/crossbeam-doc/crossbeam/
///
/// chase_lev allows one worker that pushes to the front but N
/// "stealers" that pop from the end.

use std::thread;

extern crate crossbeam;
use crossbeam::sync::chase_lev;
use crossbeam::sync::chase_lev::{Steal};

fn main() {
    let (mut worker, stealer) = chase_lev::deque::<usize>();

    let many: usize = 12;
    let mut threads = Vec::with_capacity(12);

    let until = 1usize << 20;

    for i in 0..((many*until)>>1) {
        worker.push(i);
    }

    for _ in 0..many {
        let s = stealer.clone();
        let t = thread::spawn( move || {
            let mut count = 0;
            loop {
                if let Steal::Data(_) = s.steal() {
                    count += 1;
                } else {
                    continue;
                }
                if count >= until {
                    break;
                }
            }
        });
        threads.push(t);
    }

    for i in 0..((many*until)>>1) {
        worker.push(i);
    }

    for entry in threads {
        let _ = entry.join();
    }
}
