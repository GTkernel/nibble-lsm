use common::*;
use thelog::*;
use memory::*;
use segment::*;
use index::*;
use compaction::*;
use numa::{self,NodeId};
use meta;

use std::process;
use std::sync::Arc;
use std::thread::{self,JoinHandle};
use parking_lot as pl;
use std::mem;

//==----------------------------------------------------==//
//      Constants
//==----------------------------------------------------==//

const MIN_SEG_PER_SOCKET: usize = 4;

macro_rules! min_log_size {
    ( $nsockets:expr ) => {
        (num_log_heads() * MIN_SEG_PER_SOCKET)
            * SEGMENT_SIZE * $nsockets
            + RESERVE_SEGS
    }
}

//==----------------------------------------------------==//
//      LSM interface
//==----------------------------------------------------==//

/// Threads use this to automatically pin their epoch values,
/// and release them when exiting each method.
struct PinnedEpoch { }
impl PinnedEpoch {
    pub fn new() -> Self {
        meta::pin();
        // info!("Pinning epoch");
        PinnedEpoch { }
    }
}
impl Drop for PinnedEpoch {
    fn drop(&mut self) {
        meta::quiesce();
        // info!("Unpinning epoch");
    }
}

pub struct LSMPerNode {
    socket: usize,
    manager: SegmentManagerRef,
    log: Log,
    seginfo: meta::SegmentInfoTableRef,
    compactor: CompactorRef, // TODO move to segmgr instead?
}

pub struct LSM {
    /// Indexed per socket
    nodes: Vec<LSMPerNode>,
    nnodes: u32,
    index: IndexRef,
    capacity: usize,
}

#[derive(Copy,Clone,Debug)]
pub enum PutPolicy {
    Specific(usize),
    Interleave,
}

impl LSM {

    #[cfg(IGNORE)]
    pub fn dump_segments(&self, node: usize) {
        println!("LSM: DUMPING SEGMENT INFO NODE {}", node);
        self.nodes[node].manager.dump_segments();
    }

    /// Create new instance of LSM. It partitions itself across the
    /// sockets. You must create an instance with at least enough
    /// memory per-socket to hold some minimum of segments.
    pub fn new(capacity: usize) -> Self {
        Self::__new(capacity, Self::default_ht_nitems() )
    }

    pub fn new2(capacity: usize, ht_nitems: usize) -> Self {
        Self::__new(capacity, ht_nitems)
    }

    /// Allocate LSM with a default (small) amount of memory.
    pub fn default() -> Self {
        Self::new2(
            Self::default_capacity(),
            Self::default_ht_nitems() )
    }

    /// Verify LSM's current compilation either supports rdrand, or
    /// was compiled with appropriate fallback implementation.
    fn __check_rdrand() -> bool {
        if !kvs_rdrand_compile_flags() {
            println!(">> Oops: your CPU probably doesn't support 'rdrand'.");
            println!(">> Please recompile LSM with --nordrand");
            println!("");
            false
        } else {
            true
        }
    }

    fn __new(capacity: usize, ht_nitems: usize) -> Self {
        if !LSM::__check_rdrand() {
            println!("Cannot initialize LSM.");
            println!("Please verify above messages.");
            process::exit(1);
        }

        let nnodes = numa::NODE_MAP.sockets();
        let mincap = min_log_size!(nnodes);
        let persock = capacity/nnodes;

        assert!(capacity >= mincap,
                "LSM requires more memory: {} GiB",
                mincap / (1usize<<30));

        //let ntables = numa::NODE_MAP.ncpus();
        //let ntables: usize = 64;

        let nsock = numa::NODE_MAP.sockets();
        let ntables: usize = 8 * nsock;
        let nitems = ht_nitems; //1usize << 30;
        let n_per  = nitems / ntables;

        assert!(ntables.is_power_of_two(),
            "ntables is not power of two: {:?}", ntables);

        info!("    sockets:     {}", nnodes);
        info!("   capacity:     {:.2} GiB",
              (capacity as f64)/(2f64.powi(30)));
        info!("   cap/sock:     {:.2} GiB",
              (persock as f64)/(2f64.powi(30)));

        info!("    index n:     {}", nitems);
        info!("    #tables:     {}", ntables);

        info!("   seg size:     {}", SEGMENT_SIZE);
        info!(" block size:     {}", BLOCK_SIZE);
        info!("  block/seg:     {}", BLOCKS_PER_SEG);
        info!("   #blk var:     {}", ALLOC_NBLKS_VAR);

        info!(" resrv segs:     {} x{}", RESERVE_SEGS, nsock);
        info!(" resrv size:     {:.2} GiB x{}",
              (RESERVE_SEGS * SEGMENT_SIZE) as f64 /
                (2f64.powi(30)), nsock);
        info!(" resrv as %:     {:.2}",
              (nsock * RESERVE_SEGS * SEGMENT_SIZE) as f64 /
              (capacity as f64));

        let index = Arc::new(Index::new(ntables, n_per));

        // Create all per-socket elements with threads.
        let nodes: Arc<pl::Mutex<Vec<LSMPerNode>>>;
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
                    let per = LSMPerNode {
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
        LSM {
            nodes: nodes,
            nnodes: nnodes as u32,
            index: index,
            capacity: capacity
        }
    }

    pub fn default_capacity() -> usize {
        min_log_size!( numa::NODE_MAP.sockets() )
    }

    pub fn default_ht_nitems() -> usize {
        1usize << 25
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn nnodes(&self) -> usize {
        self.nnodes as usize
    }

    pub fn enable_compaction(&self, node: NodeId) {
        info!("Enabling compaction on node {}", node.0);
        let mut comp = self.nodes[node.0].compactor.lock();
        comp.spawn();
    }

    #[allow(unused_variables)]
    pub fn disable_compaction(&mut self, node: NodeId) {
        unimplemented!();
    }

    //
    // Get/Put/Del API
    //

    #[inline(always)]
    fn __put(&self, obj: &ObjDesc, hint: PutPolicy) -> Status {
        // NOTE DO NOT pin the epoch during a PUT. It will stall
        // the compaction logic.

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
                return Err(code);
            },
            Ok(v) => va = v,
        }
        let ientry = merge(socket as u16, va as u64);
        trace!("key {} va 0x{:x} ientry 0x{:x}",
               obj.getkey(), va, ientry);

        // 2. add to index; if we are updating it, remove live state from
        // prior segment. running a lambda while we hold the item's
        // lock avoids race conditions with the cleaner

        let key = obj.getkey();
        let ok: bool = self.index.update_map(key, ientry as u64, |old| {
            // decrement live size of segment if we overwrite object
            // old=None if this was an insertion
            if let Some(ientry) = old {
                let (socket,va) = extract(ientry);
                let node = &self.nodes[socket as usize];
                let idx: usize = node.manager.segment_of(va as usize);
                let head = node.log.copy_header(va as usize);
                // decrement live bytes
                self.nodes[socket as usize].seginfo
                    .decr_live(idx, head.len_with_header());
            }
        });
        if !ok {
            // no need to undo the log append;
            // entries are stale until we update the index
            warn!("index update returned false");
            return Err(ErrorCode::TableFull);
        }

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

    #[inline(always)]
    pub fn exists(&self, key: u64) -> bool {
        self.index.get(key).is_some()
    }

    /// FIXME why don't we return the length... ?
    #[inline(always)]
    pub fn get_object(&self, key: u64, buf: &mut [u8]) -> Status {
        let ep = PinnedEpoch::new();

        // 1. lookup the key and get the entry
        let ientry: IndexEntry = match self.index.get(key) {
            None => return Err(ErrorCode::KeyNotExist),
            Some(entry) => entry,
        };
        let (socket,va) = extract(ientry);

        // 2. ask Log to give us the object
        self.nodes[socket as usize]
            .log.get_entry(va as usize, buf);

        Ok(1)
    }

    #[inline(always)]
    pub fn del_object(&self, key: u64) -> Status {
        let ep = PinnedEpoch::new();

        // 1. remove key and acquire old
        let r = self.index.remove_map(key, |entry| {
            if let Some(ientry) = entry {
                let (socket,va) = extract(ientry);

                // 2. read the size of the object
                let node = &self.nodes[socket as usize];
                let head = node.log.copy_header(va as usize);

                // 3. decrement live size of segment
                let idx: usize = node.manager.segment_of(va as usize);
                self.nodes[socket as usize].seginfo
                    .decr_live(idx, head.len_with_header());
            }
        });

        if r { Ok(1) }
        else { Err(ErrorCode::KeyNotExist) }
    }

    //
    // Lower-level allocation API
    //

    // Insert key into index; write object header into log, but
    // do not write actual object itself.
    // This version of alloc acts like an explicit insert.
    // NOTE: this is only used for the shuffle benchmark for now.
    // It will block if key already exists, and fail if key wasn't
    // able to insert, or was an update
    #[cfg(IGNORE)]
    pub fn alloc(&self, key: u64, len: u64, sock: u32) -> Pointer<u8> {
        // meta::pin();
        let va: usize;

        let obj = ObjDesc::null(key, len as usize);

        // make sure we wait for key to be deleted before adding again
        while let Some(_) = self.index.get(obj.key) {
            ;
        }

        let res = self.nodes[sock as usize].log.append(&obj);
        debug_assert_eq!(res.is_ok(), true);
        va = res.unwrap();

        let ientry = merge(sock as u16, va as u64);

        // 2. update reference to object
        let (ok,opt) = self.index.update(obj.key, ientry);
        // verify this inserted, and was not an update
        debug_assert_eq!(ok, true,
                        "Table insert failed; can it resize?");
        // XXX
        debug_assert_eq!(opt.is_none(), true);

        // meta::quiesce();
        Pointer(va as *const u8)
    }

    // Delete key from index.
    #[cfg(IGNORE)]
    pub fn free(&self, key: u64) -> bool {
        // we need to pin the epoch, because we use the VA to lookup
        // the containing segment
        meta::pin();

        // 1. remove key and acquire old
        let res = self.index.remove(key);
        if res.is_none() {
            return false;
        }
        let ientry: IndexEntry = res.unwrap();
        let (socket,va) = extract(ientry);

        // 2. read the size of the object
        let node = &self.nodes[socket as usize];
        let head = node.log.copy_header(va as usize);

        // 3. decrement live size of segment
        let idx: usize = node.manager.segment_of(va as usize);
        node.seginfo.decr_live(idx, head.len_with_header());

        meta::quiesce();
        true
    }

    pub fn pin(key: u64) {
        unimplemented!();
    }

    pub fn unpin(key: u64) {
        unimplemented!();
    }

    //
    // Misc. methods
    //

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
        meta::pin();
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

mod tests {
    use super::*;
    use common::{self,ErrorCode};
    use logger;
    use memory;
    use index;
    use rand::{self,Rng};
    use segment;
    use segment::*;
    use std::slice;
    use sched;

    // test with one simple object
    #[test]
    fn simple() {
        logger::enable();
        let mut kvs = LSM::default();

        let key: u64 = 1;
        let value: Vec<u64> = vec![1u64,2,3,4,5];
        let vptr = common::Pointer(value.as_ptr() as *const u8);

        let obj = ObjDesc::new(key, vptr, value.len()*8);
        assert!(kvs.put_object(&obj).is_ok());

        let (st,opt) = kvs.get_object(key);
        assert!(st.is_ok());
        assert!(opt.is_some());

        let buf = opt.unwrap();
        let addr = buf.addr.0 as *const u64;
        let sl: &[u64] = unsafe {
            slice::from_raw_parts(addr, buf.len/8)
        };
        assert_eq!(sl.iter().sum::<u64>(),
                    value.iter().sum::<u64>());

        assert!(kvs.del_object(key).is_ok());
        let ret = kvs.del_object(key);
        assert!(ret.is_err());
    }

    // shove in the object multiple times to cross many blocks
    #[test]
    fn many_objects() {
        logger::enable();
        let mut kvs = LSM::default();
        let mut rng = rand::thread_rng();

        let mut value: Vec<u64> = Vec::with_capacity(200);
        for i in 0..200 {
            // avoid overflow during summation later
            value.push(rng.gen::<u32>() as u64);
        }
        let sum = value.iter().sum::<u64>();
        let vptr = common::Pointer(value.as_ptr() as *const u8);
        let vlen = value.len() * 8;

        let nobj = 2 * segment::SEGMENT_SIZE / vlen;
        info!("nobj {}", nobj);

        for i in 0..nobj {
            let key = (i+1) as u64;
            let obj = ObjDesc::new(key, vptr, vlen);
            assert!(kvs.put_object(&obj).is_ok());
        }

        for i in 0..nobj {
            let key = (i+1) as u64;
            let (st,opt) = kvs.get_object(key);
            assert!(st.is_ok());
            assert!(opt.is_some());
            let buf = opt.unwrap();
            let addr = buf.addr.0 as *const u64;
            let sl: &[u64] = unsafe {
                slice::from_raw_parts(addr, buf.len/8)
            };
            assert_eq!(sl.iter().sum::<u64>(), sum);
        }

        for i in 0..nobj {
            let key = (i+1) as u64;
            assert!(kvs.del_object(key).is_ok());
            let ret = kvs.del_object(key);
            assert!(ret.is_err());
        }

        for i in 0..nobj {
            let key = (i+1) as u64;
            let (st,opt) = kvs.get_object(key);
            assert!(st.is_err());
            assert!(opt.is_none());
        }
    }

    /// Give the segment index of the specified key (as String)
    fn segment_of(kvs: &LSM, key: u64) -> usize {
        logger::enable();

        // look up virtual address
        let opt = kvs.index.get(key);
        assert!(opt.is_some(), "key {:x} not in index", key);
        let ientry: index::IndexEntry = opt.unwrap();
        let (socket,va) = index::extract(ientry);
        let socket = socket as usize;

        // associate with segment and return
        let mgr: &SegmentManager = &kvs.nodes[socket].manager;
        mgr.segment_of(va as usize)
    }

    /// on init, epoch table should be zero
    #[test]
    fn epoch_0() {
        logger::enable();
        let kvs = LSM::default();

        for idx in 0..kvs.nodes[0].seginfo.len() {
            assert_eq!(kvs.nodes[0].seginfo.get_live(idx), 0usize);
            assert_eq!(kvs.nodes[0].seginfo.get_epoch(idx), 0usize);
        }
    }

    /// add one item repeatedly and observe the live size of the
    /// segment remains constant. upon rolling the head, check the
    /// segment live size is zero and the new is updated
    #[test]
    fn epoch_1() {
        logger::enable();
        let mut kvs = LSM::default();

        let key: u64 = 1;
        let value: Vec<u64> = vec![1,2,3,4,5];
        let vptr = common::Pointer(value.as_ptr() as *const u8);
        let vlen = value.len() * 8;
        let obj = ObjDesc::new(key, vptr, vlen);
        let size = obj.len_with_header();

        unsafe { sched::pin_cpu(0); }

        // do first insertion, grab head idx used
        assert!(kvs.put_where(&obj, PutPolicy::Specific(0)).is_ok());
        let head = segment_of(&kvs, key);
        assert_eq!(kvs.nodes[0].seginfo.get_live(head), size);

        // insert until the head rolls
        loop {
            assert!(kvs.put_where(&obj, PutPolicy::Specific(0)).is_ok());
            // FIXME assumes the segment index we compare to doesn't
            // change sockets
            let segidx = segment_of(&kvs, key);
            assert_eq!(kvs.nodes[0].seginfo.get_live(segidx), size);
            if head != segidx {
                // head rolled. let's check prior segment live size
                assert_eq!(kvs.nodes[0].seginfo.get_live(head), 0usize);
                break;
            }
        }
    }

    /// add unique items, observe the live size of the segment grows
    #[test]
    fn epoch_2() {
        logger::enable();
        let mut kvs = LSM::default();
        let mut rng = rand::thread_rng();

        // do first insertion, grab head idx used
        let mut key: u64 = 1;
        let mut value: Vec<u64> = Vec::with_capacity(200);
        for i in 0..200 {
            value.push(rng.gen::<u32>() as u64);
        }
        let vptr = common::Pointer(value.as_ptr() as *const u8);
        let vlen = value.len() * 8;
        let obj = ObjDesc::new(key, vptr, vlen);
        let mut len = obj.len_with_header();
        assert!(kvs.put_object(&obj).is_ok());

        let head = segment_of(&kvs, key);
        // XXX the socket may be different
        assert_eq!(kvs.nodes[0].seginfo.get_live(head), len);
        key += 1;

        let mut total = len; // accumulator excluding current obj

        // insert until the head rolls
        loop {
            let obj = ObjDesc::new(key, vptr, vlen);
            len = obj.len_with_header();
            assert!(kvs.put_object(&obj).is_ok());
            // FIXME assumes the segment index we compare to doesn't
            // change sockets
            let segidx = segment_of(&kvs, key);
            if head == segidx {
                assert_eq!(kvs.nodes[0].seginfo.get_live(segidx), total+len);
            } else {
                // head rolled. check old and new live sizes
                assert_eq!(kvs.nodes[0].seginfo.get_live(head), total);
                assert_eq!(kvs.nodes[0].seginfo.get_live(segidx), len);
                break;
            }
            key += 1;
            total += len;
        }
    }

    /// add/remove one item and observe the live size is set then zero
    #[test]
    fn epoch_3() {
        logger::enable();
        let mut kvs = LSM::default();

        let key: u64 = 1;
        let value: Vec<u64> = vec![1,2,3,4,5];
        let vptr = common::Pointer(value.as_ptr() as *const u8);
        let vlen = value.len() * 8;
        let obj = ObjDesc::new(key, vptr, vlen);
        let size = obj.len_with_header();

        assert!(kvs.put_object(&obj).is_ok());

        let idx = segment_of(&kvs, key);
        let len = obj.len_with_header();
        assert_eq!(kvs.nodes[0].seginfo.get_live(idx), len);

        assert!(kvs.del_object(key).is_ok());
        assert_eq!(kvs.nodes[0].seginfo.get_live(idx), 0usize);
    }

    #[test]
    #[should_panic(expected = "larger than segment")]
    fn obj_too_large() {
        logger::enable();
        let mut kvs = LSM::default();

        let key: u64 = 1;
        let len = 2 * segment::SEGMENT_SIZE;
        let value = memory::allocate::<u8>(len);

        let v = common::Pointer(value as *const u8);
        let obj = ObjDesc::new(key, v, len);
        if let Err(code) = kvs.put_object(&obj) {
            panic!("{:?}", code); // <--
        }
    }

    #[test]
    fn large_objs() {
        logger::enable();
        let mut kvs = LSM::default();

        let key: u64 = 1;
        let len = segment::SEGMENT_SIZE - segment::BLOCK_SIZE;
        let value = memory::allocate::<u8>(len);

        let v = common::Pointer(value as *const u8);
        let obj = ObjDesc::new(key, v, len);
        for _ in 0..4 {
            assert!(kvs.put_object(&obj).is_ok());
        }
        unsafe { memory::deallocate(value, len); }
    }
}
