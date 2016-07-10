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
use std::thread::{self,JoinHandle};
use rand::{self,Rng};

//==----------------------------------------------------==//
//      Constants
//==----------------------------------------------------==//

const MIN_SEG_PER_SOCKET: usize = 4;

macro_rules! min_log_size {
    ( $nsockets:expr ) => {
        ((NUM_LOG_HEADS as usize) * MIN_SEG_PER_SOCKET)
            * SEGMENT_SIZE * $nsockets
            + COMPACTION_RESERVE_SEGMENTS
    }
}

//==----------------------------------------------------==//
//      Nibble interface
//==----------------------------------------------------==//

pub struct NibblePerNode {
    socket: usize,
    manager: SegmentManagerRef,
    log: Log,
    seginfo: epoch::SegmentInfoTableRef,
    compactor: CompactorRef, // TODO move to segmgr instead?
}

pub struct Nibble {
    /// Indexed per socket
    nodes: Vec<NibblePerNode>,
    nnodes: u32,
    index: IndexRef,
    capacity: usize,
}

#[derive(Copy,Clone,Debug)]
pub enum PutPolicy {
    Specific(usize),
    Interleave,
}

impl Nibble {

    /// Create new instance of Nibble. It partitions itself across the
    /// sockets. You must create an instance with at least enough
    /// memory per-socket to hold some minimum of segments.
    pub fn new(capacity: usize) -> Self {
        let nnodes = numa::NODE_MAP.sockets();
        let mincap = min_log_size!(nnodes);
        assert!(capacity >= mincap,
                "nibble requires more memory: {} but have {}",
                mincap, capacity);
        info!("sockets:  {}", nnodes);
        let persock = capacity/nnodes;
        info!("capacity: {:.2} GiB",
              (capacity as f64)/(2f64.powi(30)));
        info!("socket:   {:.2} GiB",
              (persock as f64)/(2f64.powi(30)));

        // Create all per-socket elements with threads.
        let nodes: Arc<Mutex<Vec<NibblePerNode>>>;
        let index = index_ref!();
        nodes = Arc::new(Mutex::new(Vec::with_capacity(nnodes)));
        {
            let mut handles: Vec<JoinHandle<()>> = Vec::new();
            for node in 0..nnodes {
                let i = index.clone();
                let nodes = nodes.clone();
                let n = NodeId(node);
                handles.push( thread::spawn( move || {
                    let s = SEGMENT_SIZE;
                    let manager = SegmentManager::numa(s,persock,n);
                    let seginfo = manager.seginfo();
                    let mref = Arc::new(Mutex::new(manager));
                    let per = NibblePerNode {
                        socket: node,
                        manager: mref.clone(),
                        log: Log::new(mref.clone()),
                        seginfo: seginfo,
                        compactor: comp_ref!(&mref, &i),
                    };
                    let mut guard = nodes.lock().unwrap();
                    guard.push(per);
                }));
            }
            for handle in handles {
                let _ = handle.join();
            }
        }
        // consume the Arc and Mutex, then put back into order
        let mut nodes = match Arc::try_unwrap(nodes) {
            Err(_) => panic!("glug glug glug"),
            Ok(lock) => match lock.into_inner() {
                Err(_) => panic!("glug glug glug"),
                Ok(n) => n,
            },
        };
        nodes.sort_by( | a, b | {
            a.socket.cmp(&b.socket)
        });
        Nibble {
            nodes: nodes,
            nnodes: nnodes as u32,
            index: index,
            capacity: capacity
        }
    }

    pub fn default_capacity() -> usize {
        min_log_size!( numa::NODE_MAP.sockets() )
    }

    /// Allocate Nibble with a default (small) amount of memory.
    pub fn default() -> Self {
        Nibble::new( Self::default_capacity() )
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn nnodes(&self) -> usize {
        self.nnodes as usize
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

    #[inline(always)]
    fn __put(&self, obj: &ObjDesc, hint: PutPolicy) -> Status {
        epoch::pin();
        let va: usize;

        let socket: usize = match hint {
            PutPolicy::Specific(id) => id,
            PutPolicy::Interleave =>
                (unsafe { rdrand() } % self.nnodes) as usize,
        };
        trace!("put socket {:?}", socket);

        if socket >= self.nodes.len() {
            return Err(ErrorCode::InvalidSocket);
        }

        // 1. add object to log
        match self.nodes[socket].log.append(obj) {
            Err(code) => {
                warn!("log full: {} bytes",
                      self.nodes[0].seginfo.live_bytes());
                return Err(code);
            },
            Ok(v) => va = v,
        }
        trace!("key {} va 0x{:x}", obj.getkey(), va);
        // 2. update reference to object
        let opt = self.index.update(obj.getkey(), va);
        // 3. decrement live size of segment if we overwrite object
        if let Some(old) = opt {
            // FIXME this shouldn't need a lock..
            let idx: usize = match self.nodes[socket].manager.lock() {
                Err(_) => panic!("lock poison"),
                Ok(manager) =>  {
                    // should not fail
                    let opt = manager.segment_of(old);
                    assert_eq!(opt.is_some(), true);
                    opt.unwrap()
                },
            };
            self.nodes[socket].seginfo
                .decr_live(idx, obj.len_with_header());
        }
        epoch::quiesce();
        Ok(1)
    }

    #[inline(always)]
    pub fn put_where(&self, obj: &ObjDesc,
                     hint: PutPolicy) -> Status {
        self.__put(obj, hint)
    }

    #[inline(always)]
    pub fn put_object(&self, obj: &ObjDesc) -> Status {
        self.__put(obj, PutPolicy::Specific(0))
    }

    #[inline(always)]
    pub fn get_object(&self, key: u64) -> (Status,Option<Buffer>) {
        epoch::pin();
        let va: usize;
        match self.index.get(key) {
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
        trace!("key {} va 0x{:x} datalen {}",
               key, va, header.getdatalen());
        let buf = Buffer::new(header.getdatalen() as usize);
        unsafe {
            let src = header.data_address(va);
            ptr::copy_nonoverlapping(src,
                buf.getaddr() as *mut u8, buf.getlen());
        }
        epoch::quiesce();
        (Ok(1),Some(buf))
    }

    #[inline(always)]
    pub fn del_object(&mut self, key: u64) -> Status {
        epoch::pin();
        let va: usize;
        match self.index.remove(key) {
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
        self.index.len()
    }

    #[cfg(IGNORE)]
    pub fn dump_seg_info(&self) {
        for node in &self.nodes {
            let mgr = node.manager.lock().unwrap();
            mgr.dump_seg_info();
        }
    }

    // hack
    #[cfg(IGNORE)]
    #[inline(always)]
    pub fn seg_of(&mut self, key: u64) -> Option<usize> {
        epoch::pin();
        let va: usize;
        match self.index.get(key) {
            None => return None,
            Some(v) => va = v,
        }
        match self.nodes[0].manager.lock() {
            Err(_) => panic!("lock poison"),
            Ok(manager) => manager.segment_of(va),
        }
    }

}

//==----------------------------------------------------==//
//      Unit tests
//==----------------------------------------------------==//

#[cfg(IGNORE)]
mod tests {
    use super::*;

    use segment::*;

    use super::super::logger;
    use super::super::memory;
    use super::super::segment;
    use super::super::common;

    #[test]
    fn nibble_single_small_object() {
        logger::enable();
        let mut nib = Nibble::default();

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
                match code {
                    common::ErrorCode::OutOfMemory => break,
                    _ => panic!("{:?}", code),
                }
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
        let va: usize = match nib.index.get(key.as_str()) {
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
        let nib = Nibble::default();

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
        let mut nib = Nibble::default();

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
        let mut nib = Nibble::default();
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
        let mut nib = Nibble::default();

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
        let mut nib = Nibble::default();

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
        let mut nib = Nibble::default();

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
