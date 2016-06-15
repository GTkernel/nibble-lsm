#![feature(test)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(dead_code)]

#[macro_use]
extern crate log;
extern crate test;
extern crate nibble;

use std::time::Duration;
use std::thread;
use log::LogLevel;

use nibble::logger;
use nibble::nib::Nibble;
use nibble::segment::{ObjDesc,SEGMENT_SIZE};
use nibble::common::ErrorCode;
use nibble::cuckoo;

#[test]
fn direct_insert() {
    logger::enable();
    cuckoo::init();
    assert_eq!(cuckoo::empty(), true);
    assert_eq!(cuckoo::size(), 0usize);

    assert_eq!(cuckoo::contains("na"), false);

    assert_eq!(cuckoo::update("na", 11usize), None);
    assert_eq!(cuckoo::update("na", 12usize), Some(11usize));

    assert_eq!(cuckoo::insert("hello", 42usize), true);
    assert_eq!(cuckoo::contains("na"), true);

    assert_eq!(cuckoo::insert("hello", 42usize), false);
    assert_eq!(cuckoo::empty(), false);
    assert_eq!(cuckoo::contains("hello"), true);
    assert_eq!(cuckoo::find("hello"), Some(42usize));

    assert_eq!(cuckoo::update("na", 11usize), Some(12usize));

    assert_eq!(cuckoo::update("hello", 99usize), Some(42usize));
    assert_eq!(cuckoo::find("hello"), Some(99usize));
    let mut old: usize = 0;
    assert_eq!(cuckoo::erase("hello", &mut old), true);
    assert_eq!(old, 99usize);
    old = 0usize;
    assert_eq!(cuckoo::erase("hello", &mut old), false);
    assert_eq!(old, 0usize);
}

#[test]
fn nibble_insert() {
    logger::enable();

    let mut nib = Nibble::default();
    let nkeys: usize = 20;

    for counter in 0..nkeys {
        let key = counter.to_string();
        let obj = ObjDesc::new2(&key, &key);
        if let Err(code) = nib.put_object(&obj) {
            match code {
                ErrorCode::OutOfMemory => break,
                _ => panic!("put failed"),
            }
        }
    }

    for counter in 0..nkeys {
        let key = counter.to_string();
        match nib.get_object(&key) {
            (Ok(_),Some(buf)) => {
                let mut v: Vec<u8> = Vec::with_capacity(buf.getlen());
                unsafe {
                    v.set_len(buf.getlen());
                    std::ptr::copy(buf.getaddr() as *const u8,
                        v.as_mut_ptr(), buf.getlen());
                }
                match String::from_utf8(v) {
                    Ok(string) => {
                        assert_eq!(string, key);
                    },
                    Err(code) => {
                        panic!("utf8 error: {:?}", code);
                    },
                }
            },
            (Err(code),None) => panic!("error: {:?}", code),
            _ => panic!("unhandled case"),
        }
    }

    // free some (TODO random picking)
//    let many: usize = ((counter as f32) * pct_to_free) as usize;
//    for key in allkeys.iter().take(many) {
//        assert!(nib.del_object(key).is_ok());
//    }
}
