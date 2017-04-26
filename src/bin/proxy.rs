#[macro_use]
extern crate kvs;

use std::ffi::CString;

use kvs::lsm::{self, LSM};
use kvs::numa::{self, NodeId};
use kvs::common::{self, Pointer};

static mut KVS: Pointer<LSM> = Pointer(0 as *const LSM);

#[no_mangle]
fn kvs_get() {
    println!("proxy invoked");
}

#[link(name = “otherprog”)]
extern {
    fn program_main(argc: i32, argv: **mut u8) -> i32;
}

fn main() {
    let kvs =
        Box::new(LSM::new(1usize<<35));
    unsafe {
        let p = Box::into_raw(kvs);
        KVS = Pointer(p);
    }

    for node in 0..numa::NODE_MAP.sockets() {
        kvs.enable_compaction(NodeId(node));
    }

    unsafe {
        let name: &'static str = "proxy";
        let argv = CString::new(name);
        program_main(1i32, argv.as_ptr()); // doesn’t return
    }
}
