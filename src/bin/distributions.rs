extern crate kvs;
use kvs::distributions::*;
use std::time::Instant;

fn main() {
	let mut d: Box<DistGenerator> = Box::new(Zipfian::new(10000, 0.99));

	let now = Instant::now();
	let mut hist: [usize; 10000] = [0usize; 10000];
	for _ in 0..10_000_000_usize {
		hist[d.next() as usize] += 1;
		hist[d.next() as usize] += 1;
		hist[d.next() as usize] += 1;
		hist[d.next() as usize] += 1;
		hist[d.next() as usize] += 1;
		hist[d.next() as usize] += 1;
		hist[d.next() as usize] += 1;
		hist[d.next() as usize] += 1;
		hist[d.next() as usize] += 1;
		hist[d.next() as usize] += 1;
	}
	let el = now.elapsed();
	let t = el.as_secs() as f64 +
				el.subsec_nanos() as f64 / 1_000_000_000f64;

	for i in 0..10000 {
		println!("{}: {}", i, hist[i]);
	}

	println!("generates {} numbers/sec", 100_000_000f64 / t);
}
