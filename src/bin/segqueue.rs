/// Simple multi-threaded test for a Michael-Scott lock-free queue
/// https://github.com/aturon/crossbeam
/// http://aturon.github.io/crossbeam-doc/crossbeam/
///
/// SegQueue can have any number of threads pushing and popping from
/// either end.

use std::thread;

extern crate crossbeam;
use crossbeam::sync::SegQueue;

use std::sync::Arc;

fn main() {
    let queue: SegQueue<usize> = SegQueue::new();
    let arc = Arc::new(queue);

    let many: usize = 12;
    let mut threads = Vec::with_capacity(12);

    let until = 1usize << 20;

    for i in 0..((many*until)>>1) {
        arc.push(i);
    }

    for _ in 0..many {
        let arc = arc.clone();
        let t = thread::spawn( move || {
            let mut count = 0;
            loop {
                if let Some(_) = arc.try_pop() {
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
        arc.push(i);
    }

    for entry in threads {
        let _ = entry.join();
    }
}
