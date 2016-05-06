use common::*;
use thelog::*;
use memory::*;
use segment::*;
use index::*;

use std::ptr;
use std::sync::{Arc,Mutex};

//==----------------------------------------------------==//
//      Nibble interface
//==----------------------------------------------------==//

pub struct Nibble {
    index: IndexRef,
    manager: SegmentManagerRef,
    log: Log,
    epochs: EpochTableRef,
}

impl Nibble {

    pub fn new(capacity: usize) -> Self {
        let manager = SegmentManager::new(0, SEGMENT_SIZE, capacity);
        let epochs = manager.epochs();
        let mref = Arc::new(Mutex::new(manager));
        Nibble {
            index: index_ref!(),
            manager: mref.clone(),
            log: Log::new(mref.clone()),
            epochs: epochs,
        }
    }

    // TODO add some locking
    pub fn put_object(&mut self, obj: &ObjDesc) -> Status {
        let va: usize;
        // 1. add object to log
        match self.log.append(obj) {
            Err(code) => return Err(code),
            Ok(v) => va = v,
        }
        // 2. update reference to object
        let opt = match self.index.lock() {
            Err(_) => panic!("lock poison"),
            Ok(mut index) => 
                index.update(&String::from(obj.getkey()), va),
        };
        // 3. decrement live size of segment if we overwrite object
        if let Some(old) = opt {
            // FIXME this shouldn't need a lock..
            let idx: usize = match self.manager.lock() {
                Err(_) => panic!("lock poison"),
                Ok(manager) =>  {
                    // should not fail
                    let opt = manager.segment_of(old);
                    assert_eq!(opt.is_some(), true);
                    opt.unwrap()
                },
            };
            self.epochs.decr_live(idx, obj.len_with_header());
        }
        Ok(1)
    }

    pub fn get_object(&self, key: &String) -> (Status,Option<Buffer>) {
        let va: usize;
        match self.index.lock() {
            Ok(index) => {
                match index.get(key) {
                    None => return (Err(ErrorCode::KeyNotExist),None),
                    Some(v) => va = v,
                }
            },
            Err(poison) => panic!("index lock poisoned"),
        }
        //
        // XXX XXX use EntryReference to copy out data!! XXX XXX
        //
        let header: EntryHeader;
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

    pub fn del_object(&mut self, key: &String) -> Status {
        let va: usize;
        match self.index.lock() {
            Err(poison) => panic!("index lock poisoned"),
            Ok(mut guard) => {
                match guard.remove(key) {
                    None => return Err(ErrorCode::KeyNotExist),
                    Some(v) => va = v,
                }
            },
        } // index lock

        //
        // XXX XXX use EntryReference to copy out data!! XXX XXX
        //

        // determine object size (to decr epoch table)
        // TODO maybe keep this in the index?
        let header: EntryHeader;
        unsafe {
            header = ptr::read(va as *const EntryHeader);
        }

        // get segment this object belongs to
        // XXX need to make sure delete and object cleaning don't
        // result in decrementing twice!
        let idx: usize = match self.manager.lock() {
            Err(_) => panic!("lock poison"),
            Ok(guard) =>  {
                // should not fail
                let opt = guard.segment_of(va);
                assert_eq!(opt.is_some(), true);
                opt.unwrap()
            },
        };

        // update epoch table
        self.epochs.decr_live(idx, header.len_with_header());
        Ok(1)
    }

    #[cfg(test)]
    pub fn nlive(&self) -> usize {
        match self.index.lock() {
            Ok(index) => index.len(),
            Err(poison) => panic!("index lock poisoned"),
        }
    }
}

//==----------------------------------------------------==//
//      Unit tests
//==----------------------------------------------------==//

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;
    use std::mem::size_of;
    use std::mem::transmute;
    use std::rc::Rc;
    use std::sync::Arc;

    use test::Bencher;

    use segment::*;
    use common::*;

    use super::super::logger;

    #[test]
    fn nibble_single_small_object() {
        logger::enable();
        let mem = 1 << 23;
        let mut nib = Nibble::new(mem);

        // insert initial object
        let key = String::from("keykeykeykey");
        let val = String::from("valuevaluevalue");
        let obj = ObjDesc::new(key.as_str(),
                        Some(val.as_ptr()), val.len() as u32);
        match nib.put_object(&obj) {
            Ok(ign) => {},
            Err(code) => panic!("{:?}", code),
        }

        // verify what we wrote is correct FIXME reduce copy/paste
        {
            let status: Status;
            let ret = nib.get_object(&key);
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
                            compareto.push_str(val.as_str());
                            assert_eq!(compareto, string);
                        },
                        Err(code) => {
                            panic!("utf8 error: {:?}", code);
                        },
                    }
                },
                _ => panic!("unhandled return combo"),
            }
        }

        // shove in the object multiple times to cross many blocks
        let val2: &'static str = "VALUEVALUEVALUE";
        let obj2 = ObjDesc::new(key.as_str(),
                            Some(val2.as_ptr()), val2.len() as u32);
        for i in 0..100000 {
            match nib.put_object(&obj2) {
                Ok(ign) => {},
                Err(code) => panic!("{:?}", code),
            }
        }

        {
            let status: Status;
            let ret = nib.get_object(&key);
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
                            panic!("utf8 error: {:?}", code);
                        },
                    }
                },
                _ => panic!("unhandled return combo"),
            }
        }

        assert_eq!(nib.nlive(), 1);
    }

    /// Give the segment index of the specified key (as String)
    fn segment_of(nib: &Nibble, key: &String) -> usize {
        logger::enable();

        // look up virtual address
        let va: usize = match nib.index.lock() {
            Err(_) => panic!("lock poison"),
            Ok(guard) => match guard.get(key.as_str()) {
                None => panic!("key should exist"),
                Some(va_) => va_,
            },
        };

        // associate with segment and return
        match nib.manager.lock() {
            Err(_) => panic!("lock poison"),
            Ok(guard) => match guard.segment_of(va) {
                None => panic!("segment should exist"),
                Some(idx) => idx,
            },
        }
    }

    /// on init, epoch table should be zero
    #[test]
    fn epoch_0() {
        logger::enable();
        let mem = 1 << 23;
        let mut nib = Nibble::new(mem);

        for idx in 0..nib.epochs.len() {
            assert_eq!(nib.epochs.get_live(idx), 0usize);
            assert_eq!(nib.epochs.get_epoch(idx), 0usize);
        }
    }

    /// add one item repeatedly and observe the live size of the
    /// segment remains constant. upon rolling the head, check the
    /// segment live size is zero and the new is updated
    #[test]
    fn epoch_1() {
        logger::enable();
        let mut nib = Nibble::new(1<<23);

        let key = String::from("keykeykeykey");
        let val = String::from("valuevaluevalue");
        let obj = ObjDesc::new2(&key, &val);
        let size = obj.len_with_header();

        if let Err(code) = nib.put_object(&obj) {
            panic!("{:?}", code)
        }

        // do first insertion, grab head idx used
        if let Err(code) = nib.put_object(&obj) {
            panic!("{:?}", code)
        }
        let mut head = segment_of(&nib, &key);
        assert_eq!(nib.epochs.get_live(head), size);

        // insert until the head rolls
        loop {
            if let Err(code) = nib.put_object(&obj) {
                panic!("{:?}", code)
            }
            let segidx = segment_of(&nib, &key);
            assert_eq!(nib.epochs.get_live(segidx), size);
            if head != segidx {
                // head rolled. let's check prior segment live size
                assert_eq!(nib.epochs.get_live(head), 0usize);
                break;
            }
        }
    }

    /// add unique items, observe the live size of the segment grows
    #[test]
    fn epoch_2() {
        logger::enable();
        let mut nib = Nibble::new(1<<23);
        let mut keyv = 0usize;
        let value = String::from("sldkfslkfjlsdjflksdjfksjddfdfdf");

        // do first insertion, grab head idx used
        let key = keyv.to_string();
        let obj = ObjDesc::new2(&key, &value);
        let mut len = obj.len_with_header();
        if let Err(code) = nib.put_object(&obj) {
            panic!("{:?}", code)
        }
        let head = segment_of(&nib, &key);
        assert_eq!(nib.epochs.get_live(head), len);
        keyv += 1;

        let mut total = len; // accumulator excluding current obj

        // insert until the head rolls
        loop {
            let key = keyv.to_string();
            let obj = ObjDesc::new2(&key, &value);
            len = obj.len_with_header();
            if let Err(code) = nib.put_object(&obj) {
                panic!("{:?}", code)
            }
            let segidx = segment_of(&nib, &key);
            if head == segidx {
                assert_eq!(nib.epochs.get_live(segidx), total+len);
            } else {
                // head rolled. check old and new live sizes
                assert_eq!(nib.epochs.get_live(head), total);
                assert_eq!(nib.epochs.get_live(segidx), len);
                break;
            }
            keyv += 1;
            total += len;
        }
    }

    /// add/remove one item and observe the live size is set then zero
    #[test]
    fn epoch_3() {
        logger::enable();
        let mut nib = Nibble::new(1<<23);

        let key = String::from("lsfkjlksdjflks");
        let value = String::from("sldkfslkfjlsdjflksdjfksjddfdfdf");

        // insert, then delete
        let obj = ObjDesc::new2(&key, &value);
        if let Err(code) = nib.put_object(&obj) {
            panic!("{:?}", code)
        }

        let idx = segment_of(&nib, &key);
        let mut len = obj.len_with_header();
        assert_eq!(nib.epochs.get_live(idx), len);

        if let Err(code) = nib.del_object(&key) {
            panic!("{:?}", code)
        }
        assert_eq!(nib.epochs.get_live(idx), 0usize);
    }
}
