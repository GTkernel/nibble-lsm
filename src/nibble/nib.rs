use common::*;
use thelog::*;
use memory::*;
use segment::*;
use index::*;
use compaction::*;
use numa::{self,NodeId};
use epoch;

use std::sync::Arc;
use std::thread::{self,JoinHandle};
use parking_lot as pl;

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
        let nodes: Arc<pl::Mutex<Vec<NibblePerNode>>>;
        let index = Arc::new(Index::new(1, 1usize<<20));
        nodes = Arc::new(pl::Mutex::new(Vec::with_capacity(nnodes)));
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
                    let mref = Arc::new(manager);
                    let per = NibblePerNode {
                        socket: node,
                        manager: mref.clone(),
                        log: Log::new(mref.clone()),
                        seginfo: seginfo,
                        compactor: comp_ref!(&mref, &i),
                    };
                    nodes.lock().push(per);
                }));
            }
            for handle in handles {
                let _ = handle.join();
            }
        }
        // consume the Arc and Mutex, then put back into order
        let mut nodes = match Arc::try_unwrap(nodes) {
            Err(_) => panic!("glug glug glug"),
            Ok(lock) => lock.into_inner(),
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
        let mut comp = self.nodes[node.0].compactor.lock();
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
        trace!("PUT key {} socket {:?}", obj.getkey(),socket);

        if socket >= self.nodes.len() {
            return Err(ErrorCode::InvalidSocket);
        }

        // 1. add object to log
        // FIXME if full for this socket, should we attempt to append
        // elsewhere?
        match self.nodes[socket].log.append(obj) {
            Err(code) => {
                trace!("log full: {} bytes",
                      self.nodes[socket].seginfo.live_bytes());
                return Err(code);
            },
            Ok(v) => va = v,
        }
        let ientry = merge(socket as u16, va as u64);
        trace!("key {} va 0x{:x} ientry 0x{:x}",
               obj.getkey(), va, ientry);

        // 2. update reference to object
        let (ok,opt) = self.index.update(obj.getkey(), ientry);
        if !ok {
            // no need to undo the log append;
            // entries are stale until we update the index
            return Err(ErrorCode::TableFull);
        }

        // 3. decrement live size of segment if we overwrite object
        if let Some(old_ientry) = opt {
            let (old_sock,old_va) = extract(old_ientry);
            let idx: usize = self.nodes[old_sock as usize]
                                 .manager.segment_of(old_va as usize);
            // XXX is this working?? XXX
            self.nodes[old_sock as usize].seginfo
                .decr_live(idx, obj.len_with_header());
        }
        epoch::quiesce();
        Ok(1)
    }

    /// Put an object according to a specific policy. If a node is
    /// specified and an error status returned as OOM, that only
    /// applies to that node and the caller is free to choose another
    /// to put to, or to invoke put_object and let the system decide.
    #[inline(always)]
    pub fn put_where(&self, obj: &ObjDesc,
                     hint: PutPolicy) -> Status {
        self.__put(obj, hint)
    }

    /// FIXME shouldn't hard-code to Node 0
    /// TODO Here is where logic should decide that, when an append to
    /// a specific node fails, to try allocation to another node. Only
    /// when all fail, return error to user.
    #[inline(always)]
    pub fn put_object(&self, obj: &ObjDesc) -> Status {
        self.__put(obj, PutPolicy::Specific(0))
    }

    /// TODO use Result<Buffer> as return value
    /// FIXME invoke some method in thelog that copies out data
    #[inline(always)]
    pub fn get_object(&self, key: u64) -> (Status,Option<Buffer>) {
        epoch::pin();

        // 1. lookup the key and get the entry
        let ientry: IndexEntry = match self.index.get(key) {
            None => {
                return (Err(ErrorCode::KeyNotExist),None);
            },
            Some(entry) => entry,
        };
        let (socket,va) = extract(ientry);

        trace!("GET key {:x}: ientry {:x} -> socket 0x{:x} va 0x{:x}",
               key, ientry, socket, va);

        // 2. ask Log to give us the object
        let buf = self.nodes[socket as usize]
                            .log.get_entry(va as usize);

        epoch::quiesce();
        (Ok(1),Some(buf))
    }

    #[cfg(IGNORE)]
    #[inline(always)]
    pub fn del_object(&mut self, key: u64) -> Status {
        unimplemented!();
    }

    #[cfg(test)]
    pub fn nlive(&self) -> usize {
        self.index.len()
    }

    #[cfg(IGNORE)]
    pub fn dump_seg_info(&self) {
        for node in &self.nodes {
            node.manager.read().dump_seg_info();
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
