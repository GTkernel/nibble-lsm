use common::*;
use thelog::*;
use memory::*;
use segment::*;
use index::*;
use compaction::*;
use numa::{self,NodeId};
use epoch;

use std::ptr;
use std::sync::{Arc,Mutex};

//==----------------------------------------------------==//
//      Nibble interface
//==----------------------------------------------------==//

pub struct NibblePerNode {
    index: IndexRef,
    manager: SegmentManagerRef,
    log: Log,
    seginfo: epoch::SegmentInfoTableRef,
    compactor: CompactorRef, // TODO move to segmgr instead?
}

pub struct Nibble {
    /// Indexed per socket
    nodes: Vec<NibblePerNode>,
}

impl Nibble {

    // TODO allocate pages for this code to that node
    pub fn new(capacity: usize) -> Self {
        let nnodes = numa::NODE_MAP.sockets();
        let mut nodes: Vec<NibblePerNode>;
        nodes = Vec::with_capacity(nnodes);
        info!("sockets:  {}", nnodes);
        let persock = capacity/nnodes;
        info!("capacity: {:.2} GiB",
              (capacity as f64)/(2f64.powi(30)));
        info!("socket:   {:.2} GiB",
              (persock as f64)/(2f64.powi(30)));
        let index = index_ref!();
        for node in 0..nnodes {
            let n = NodeId(node);
            let manager =
                SegmentManager::numa(SEGMENT_SIZE, persock, n);
            let seginfo = manager.seginfo();
            let mref = Arc::new(Mutex::new(manager));
            nodes.push( NibblePerNode {
                index: index.clone(),
                manager: mref.clone(),
                log: Log::new(mref.clone()),
                seginfo: seginfo,
                compactor: comp_ref!(&mref, &index),
            } );
        }
        Nibble { nodes: nodes, }
    }

    pub fn enable_compaction(&mut self, node: NodeId) {
        let mut comp = self.nodes[node.0].compactor.lock().unwrap();
        comp.spawn(WorkerRole::Reclaim);
        comp.spawn(WorkerRole::Compact);
    }

    #[allow(unused_variables)]
    pub fn disable_compaction(&mut self, node: NodeId) {
        unimplemented!();
    }

    // TODO add some locking
    pub fn put_object(&mut self, obj: &ObjDesc) -> Status {
        epoch::pin();
        let va: usize;
        // XXX pick the node to append to
        // 1. add object to log
        match self.nodes[0].log.append(obj) {
            Err(code) => return Err(code),
            Ok(v) => va = v,
        }
        // 2. update reference to object
        let opt = self.nodes[0].index.update(&String::from(obj.getkey()), va);
        // 3. decrement live size of segment if we overwrite object
        if let Some(old) = opt {
            // FIXME this shouldn't need a lock..
            let idx: usize = match self.nodes[0].manager.lock() {
                Err(_) => panic!("lock poison"),
                Ok(manager) =>  {
                    // should not fail
                    let opt = manager.segment_of(old);
                    assert_eq!(opt.is_some(), true);
                    opt.unwrap()
                },
            };
            self.nodes[0].seginfo.decr_live(idx, obj.len_with_header());
        }
        epoch::quiesce();
        Ok(1)
    }

    pub fn get_object(&self, key: &String) -> (Status,Option<Buffer>) {
        epoch::pin();
        let va: usize;
        match self.nodes[0].index.get(key) {
            None => return (Err(ErrorCode::KeyNotExist),None),
            Some(v) => va = v,
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
        epoch::quiesce();
        (Ok(1),Some(buf))
    }

    pub fn del_object(&mut self, key: &String) -> Status {
        epoch::pin();
        let va: usize;
        match self.nodes[0].index.remove(key) {
            None => return Err(ErrorCode::KeyNotExist),
            Some(v) => va = v,
        }

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
        let idx: usize = match self.nodes[0].manager.lock() {
            Err(_) => panic!("lock poison"),
            Ok(guard) =>  {
                // should not fail
                let opt = guard.segment_of(va);
                assert_eq!(opt.is_some(), true);
                opt.unwrap()
            },
        };

        // update epoch table
        self.nodes[0].seginfo.decr_live(idx, header.len_with_header());

        epoch::quiesce();
        Ok(1)
    }

    #[cfg(test)]
    pub fn nlive(&self) -> usize {
        self.nodes[0].index.len()
    }
}

//==----------------------------------------------------==//
//      Unit tests
//==----------------------------------------------------==//

#[cfg(test)]
mod tests {
    use super::*;

    use segment::*;

    use super::super::logger;
    use super::super::memory;
    use super::super::segment;

    #[test]
    fn nibble_single_small_object() {
        logger::enable();
        let mem = 1 << 30;
        let mut nib = Nibble::new(mem);

        // insert initial object
        let key = String::from("keykeykeykey");
        let val = String::from("valuevaluevalue");
        let obj = ObjDesc::new(key.as_str(),
                        Some(val.as_ptr()), val.len() as u32);
        if let Err(code) = nib.put_object(&obj) {
            panic!("{:?}", code);
        }

        // verify what we wrote is correct FIXME reduce copy/paste
        {
            let ret = nib.get_object(&key);
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
        for _ in 0..100000 {
            if let Err(code) = nib.put_object(&obj2) {
                panic!("{:?}", code);
            }
        }

        {
            let ret = nib.get_object(&key);
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
        let va: usize = match nib.nodes[0].index.get(key.as_str()) {
            None => panic!("key should exist"),
            Some(va_) => va_,
        };

        // associate with segment and return
        match nib.nodes[0].manager.lock() {
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
        let mem = 1 << 30;
        let nib = Nibble::new(mem);

        for idx in 0..nib.nodes[0].seginfo.len() {
            assert_eq!(nib.nodes[0].seginfo.get_live(idx), 0usize);
            assert_eq!(nib.nodes[0].seginfo.get_epoch(idx), 0usize);
        }
    }

    /// add one item repeatedly and observe the live size of the
    /// segment remains constant. upon rolling the head, check the
    /// segment live size is zero and the new is updated
    #[test]
    fn epoch_1() {
        logger::enable();
        let mut nib = Nibble::new(1<<30);

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
        let head = segment_of(&nib, &key);
        assert_eq!(nib.nodes[0].seginfo.get_live(head), size);

        // insert until the head rolls
        loop {
            if let Err(code) = nib.put_object(&obj) {
                panic!("{:?}", code)
            }
            let segidx = segment_of(&nib, &key);
            assert_eq!(nib.nodes[0].seginfo.get_live(segidx), size);
            if head != segidx {
                // head rolled. let's check prior segment live size
                assert_eq!(nib.nodes[0].seginfo.get_live(head), 0usize);
                break;
            }
        }
    }

    /// add unique items, observe the live size of the segment grows
    #[test]
    fn epoch_2() {
        logger::enable();
        let mut nib = Nibble::new(1<<30);
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
        assert_eq!(nib.nodes[0].seginfo.get_live(head), len);
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
                assert_eq!(nib.nodes[0].seginfo.get_live(segidx), total+len);
            } else {
                // head rolled. check old and new live sizes
                assert_eq!(nib.nodes[0].seginfo.get_live(head), total);
                assert_eq!(nib.nodes[0].seginfo.get_live(segidx), len);
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
        let mut nib = Nibble::new(1<<30);

        let key = String::from("lsfkjlksdjflks");
        let value = String::from("sldkfslkfjlsdjflksdjfksjddfdfdf");

        // insert, then delete
        let obj = ObjDesc::new2(&key, &value);
        if let Err(code) = nib.put_object(&obj) {
            panic!("{:?}", code)
        }

        let idx = segment_of(&nib, &key);
        let len = obj.len_with_header();
        assert_eq!(nib.nodes[0].seginfo.get_live(idx), len);

        if let Err(code) = nib.del_object(&key) {
            panic!("{:?}", code)
        }
        assert_eq!(nib.nodes[0].seginfo.get_live(idx), 0usize);
    }

    #[test]
    #[should_panic(expected = "larger than segment")]
    fn obj_too_large() {
        logger::enable();
        let mut nib = Nibble::new(1<<30);

        let key = String::from("lsfkjlksdjflks");
        let len = segment::SEGMENT_SIZE;
        let value = memory::allocate::<u8>(len);

        let v = Some(value as *const u8);
        let obj = ObjDesc::new(key.as_str(), v, len as u32);
        if let Err(code) = nib.put_object(&obj) {
            panic!("{:?}", code)
        }
        unsafe { memory::deallocate(value, len); }
    }

    #[test]
    fn large_objs() {
        logger::enable();
        let mut nib = Nibble::new(1<<30);

        let key = String::from("lsfkjlksdjflks");
        let len = segment::SEGMENT_SIZE - segment::BLOCK_SIZE;
        let value = memory::allocate::<u8>(len);

        let v = Some(value as *const u8);
        let obj = ObjDesc::new(key.as_str(), v, len as u32);
        for _ in 0..4 {
            if let Err(code) = nib.put_object(&obj) {
                panic!("{:?}", code)
            }
        }
        unsafe { memory::deallocate(value, len); }
    }
}
