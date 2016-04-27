use common::*;
use thelog::*;
use memory::*;
use segment::*;
use index::*;

use std::ptr;
use std::sync::{Arc,Mutex};
use std::cell::RefCell;

//==----------------------------------------------------==//
//      Nibble interface
//==----------------------------------------------------==//

pub struct Nibble<'a> {
    index: Index<'a>,
    manager: SegmentManagerRef,
    log: Log,
}

impl<'a> Nibble<'a> {

    pub fn new(capacity: usize) -> Self {
        let manager_ref = segmgr_ref!(0, SEGMENT_SIZE, capacity);
        Nibble {
            index: Index::new(),
            manager: manager_ref.clone(),
            log: Log::new(manager_ref.clone()),
        }
    }

    // TODO add some locking
    pub fn put_object(&mut self, obj: &ObjDesc<'a>) -> Status {
        let va: usize;
        // 1. add object to log
        match self.log.append(obj) {
            Err(code) => return Err(code),
            Ok(v) => va = v,
        }
        // 2. update reference to object
        self.index.update(obj.getkey(), va);
        Ok(1)
    }

    pub fn get_object(&self, key: &'a str) -> (Status,Option<Buffer>) {
        // TODO lock the object? need to make sure it isn't relocated
        // or deleted while we read it
        let va: usize;
        match self.index.get(key) {
            None => return (Err(ErrorCode::KeyNotExist),None),
            Some(v) => va = v,
        }
        let mut header: EntryHeader;
        unsafe {
            header = ptr::read(va as *const EntryHeader);
        }
        let buf = Buffer::new(header.getdatalen() as usize);
        unsafe {
            let src = header.data_address(va);
            ptr::copy(src, buf.getaddr() as *mut u8, buf.getlen());
        }
        (Ok(1),Some(buf))
    }

    pub fn del_object(&mut self) -> Status {
        unimplemented!();
    }

    #[cfg(test)]
    pub fn nlive(&self) -> usize {
        self.index.len()
    }
}

//==----------------------------------------------------==//
//      Unit tests
//==----------------------------------------------------==//

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::mem::size_of;
    use std::mem::transmute;
    use std::rc::Rc;
    use std::sync::Arc;

    use test::Bencher;

    use segment::*;
    use common::*;

    #[test]
    fn nibble_single_small_object() {
        let mem = 1 << 23;
        let mut nib = Nibble::new(mem);

        // insert initial object
        let key: &'static str = "keykeykeykey";
        let val: &'static str = "valuevaluevalue";
        let obj = ObjDesc::new(key, Some(val.as_ptr()), val.len() as u32);
        match nib.put_object(&obj) {
            Ok(ign) => {},
            Err(code) => panic!("{:?}", code),
        }

        // verify what we wrote is correct FIXME reduce copy/paste
        {
            let status: Status;
            let ret = nib.get_object(key); // XXX
            let string: String;
            match ret {
                (Err(code),_) => panic!("key should exist: {:?}", code),
                (Ok(_),Some(buf)) => {
                    // convert buf to vec to string for comparison
                    // FIXME faster method?
                    let mut v: Vec<u8> = Vec::with_capacity(buf.getlen());
                    for i in 0..buf.getlen() {
                        let addr = buf.getaddr() + i;
                        unsafe { v.push( *(addr as *const u8) ); }
                    }
                    match String::from_utf8(v) {
                        Ok(string) => {
                            let mut compareto = String::new();
                            compareto.push_str(val);
                            assert_eq!(compareto, string);
                        },
                        Err(code) => {
                            panic!("error converting utf8 from log: {:?}", code);
                        },
                    }
                },
                _ => panic!("unhandled return combo"),
            }
        }

        // shove in the object multiple times to cross many blocks
        let val2: &'static str = "VALUEVALUEVALUE";
        let obj2 = ObjDesc::new(key, Some(val2.as_ptr()), val2.len() as u32);
        for i in 0..100000 {
            match nib.put_object(&obj2) {
                Ok(ign) => {},
                Err(code) => panic!("{:?}", code),
            }
        }

        {
            let status: Status;
            let ret = nib.get_object(key);
            let string: String;
            match ret {
                (Err(code),_) => panic!("key should exist: {:?}", code),
                (Ok(_),Some(buf)) => {
                    // convert buf to vec to string for comparison
                    // FIXME faster method?
                    let mut v: Vec<u8> = Vec::with_capacity(buf.getlen());
                    for i in 0..buf.getlen() {
                        let addr = buf.getaddr() + i;
                        unsafe { v.push( *(addr as *const u8) ); }
                    }
                    match String::from_utf8(v) {
                        Ok(string) => {
                            let mut compareto = String::new();
                            compareto.push_str(val2);
                            assert_eq!(compareto, string);
                        },
                        Err(code) => {
                            panic!("error converting utf8 from log: {:?}", code);
                        },
                    }
                },
                _ => panic!("unhandled return combo"),
            }
        }

        assert_eq!(nib.nlive(), 1);
    }

}
