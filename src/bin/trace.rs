extern crate nibble;
use nibble::trace::*;
use std::time::Instant;

fn main() {
	let path = "trace.in";
	let t = Trace::new(path);
	t.print();
}