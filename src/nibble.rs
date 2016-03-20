use libc;

use std::cell::RefCell;
use std::cmp;
use std::collections::HashMap;
use std::mem::size_of;
use std::mem::transmute;
use std::ptr::copy;
use std::ptr::copy_nonoverlapping;
use std::rc::Rc;
use std::sync::Arc;

const BLOCK_SIZE: usize = 1 << 16;
const SEGMENT_SIZE: usize = 1 << 20;

// -------------------------------------------------------------------
// General utilities, macros, etc.
// -------------------------------------------------------------------

/// Extract reference &T from Option<T>
macro_rules! r {
    ( $obj:expr ) => { $obj.as_ref().unwrap() }
}

/// Borrow on a reference &T from a RefCell<Option<T>>
macro_rules! rb {
    ( $obj:expr ) => { r!($obj).borrow() }
}

/// Same as rb! but mutable
macro_rules! rbm {
    ( $obj:expr ) => { r!($obj).borrow_mut() }
}

// -------------------------------------------------------------------
// Error handling
// -------------------------------------------------------------------

#[derive(Debug)]
pub enum ErrorCode {

    SegmentFull,
    SegmentClosed,

    OutOfMemory,

    KeyNotExist,
}

pub fn err2str(code: ErrorCode) -> &'static str {
    match code {
        ErrorCode::SegmentFull   => { "Segment is full" },
        ErrorCode::SegmentClosed => { "Segment is closed" },
        ErrorCode::OutOfMemory   => { "Out of memory" },
        ErrorCode::KeyNotExist   => { "Key does not exist" },
    }
}

pub type Status = Result<(usize), ErrorCode>;

// -------------------------------------------------------------------
// Block strutures
// -------------------------------------------------------------------

pub struct Block {
    addr: usize,
    len: usize,
    // TODO owning Segment
}

impl Block {

    pub fn new(addr: usize, len: usize) -> Self {
        debug!("new Block 0x{:x} {}", addr, len);
        Block { addr: addr, len: len }
    }
}

pub type BlockRef = Arc<Block>;
pub type BlockRefPool = Vec<BlockRef>;

pub struct BlockAllocator {
    block_size: usize,
    pool: BlockRefPool,
    freepool: BlockRefPool,
    mmap: MemMap,
}

impl BlockAllocator {

    pub fn new(block_size: usize, bytes: usize) -> Self {
        let mmap = MemMap::new(bytes);
        let count = bytes / block_size;
        let mut pool: Vec<Arc<Block>> = Vec::new();
        let mut freepool: Vec<Arc<Block>> = Vec::new();
        for b in 0..count {
            let addr = mmap.addr() + b*block_size;
            let b = Arc::new(Block::new(addr, block_size));
            pool.push(b.clone());
            freepool.push(b.clone());
        }
        BlockAllocator { block_size: block_size,
            pool: pool, freepool: freepool, mmap: mmap,
        }
    }

    pub fn alloc(&mut self, count: usize) -> Option<BlockRefPool> {
        // TODO lock
        match self.freepool.len() {
            0 => None,
            len =>
                if count <= len {
                    Some(self.freepool.split_off(len-count))
                } else { None },
        }
    }

    pub fn len(&self) -> usize { self.pool.len() }
    pub fn freelen(&self) -> usize { self.freepool.len() }
}

// -------------------------------------------------------------------
// Segment structures
// -------------------------------------------------------------------

pub type SegmentRef = Arc<RefCell<Segment>>;
pub type SegmentManagerRef = Arc<RefCell<SegmentManager>>;

/// Make a new SegmentRef
macro_rules! seg_ref {
    ( $id:expr, $blocks:expr ) => {
        Arc::new( RefCell::new(
                Segment::new($id, $blocks)
                ))
    }
}

/// Make a new SegmentManagerRef
macro_rules! segmgr_ref {
    ( $id:expr, $segsz:expr, $blocks:expr ) => {
        Arc::new( RefCell::new(
                SegmentManager::new( $id, $segsz, $blocks)
                ))
    }
}

/// Use this to describe an object. User is responsible for
/// transmuting their objects into byte arrays, and for ensuring the
/// lifetime of the originating buffers exceeds that of an instance of
/// ObjDesc used to refer to them.
pub struct ObjDesc<'a> {
    key: &'a str,
    value: *const u8,
    vlen: u32,
}

impl<'a> ObjDesc<'a> {

    pub fn new(key: &'a str, value: *const u8, vlen: u32) -> Self {
        ObjDesc { key: key, value: value, vlen: vlen }
    }

    pub fn len(&self) -> usize {
        self.key.len() + self.vlen as usize
    }

    pub fn len_with_header(&self) -> usize {
        size_of::<EntryHeader>() + self.len()
    }
}

/// Describe entry in the log. Format is:
///     | EntryHeader | Key bytes | Data bytes |
/// An invalid object may exist if it was deleted or a new version was
/// created in the log.
#[repr(packed)]
pub struct EntryHeader {
    valid: u16, /// 1: valid 0: invalid
    keylen: u16,
    datalen: u32,
}

impl EntryHeader {

    pub fn new(desc: &ObjDesc) -> Self {
        assert!(desc.key.len() <= usize::max_value());
        EntryHeader {
            valid: 1 as u16,
            keylen: desc.key.len() as u16,
            datalen: desc.vlen
        }
    }

    pub fn empty() -> Self {
        EntryHeader {
            valid: 0 as u16,
            keylen: 0 as u16,
            datalen: 0 as u32
        }
    }

    /// Overwrite ourself with an entry somewhere in memory.
    pub fn read(&mut self, va: usize) {
        assert!(va > 0);
        let len = size_of::<EntryHeader>();
        unsafe {
            let src: *const u8 = transmute(va);
            let dst: *mut u8 = transmute(self);
            copy(src, dst, len);
        }
    }

    /// Store ourself to memory.
    pub fn write(&self, va: usize) {
        assert!(va > 0);
        let len = size_of::<EntryHeader>();
        unsafe {
            let src: *const u8 = transmute(self);
            let dst: *mut u8 = transmute(va);
            copy(src, dst, len);
        }
    }

    /// Mark an EntryHeader invalid in memory.
    pub fn invalidate(va: usize) {
        assert!(va > 0);
        let mut header = EntryHeader::empty();
        header.read(va);
        assert_eq!(header.valid, 1 as u16);
        header.valid = 0 as u16;
        header.write(va);
        // TODO need memory fence?
    }

    pub fn is_valid(&self) -> bool {
        self.valid == 1 as u16
    }

    /// Size of this entry in the log. Same as ObjDesc's methods.
    pub fn len(&self) -> usize {
        size_of::<EntryHeader>() +
            self.keylen as usize +
            self.datalen as usize
    }

    pub fn as_ptr(&self) -> *const u8 {
        let addr: *const u8;
        unsafe { addr = transmute(self); }
        addr
    }
}

pub struct Segment {
    id: usize,
    closed: bool,
    head: usize, /// Virtual address of head TODO atomic
    len: usize, /// Total capacity TODO atomic
    rem: usize, /// Remaining capacity TODO atomic
    curblk: usize,
    blocks: BlockRefPool,
}

/// A logically contiguous chunk of memory. Inside, divided into
/// virtually contiguous blocks. For now, objects cannot be broken
/// across a block boundary.
impl Segment {

    pub fn new(id: usize, blocks: BlockRefPool) -> Self {
        let mut len: usize = 0;
        for b in blocks.iter() {
            len += b.len;
        }
        let blk = 0;
        let start: usize = blocks[blk].addr;
        Segment {
            id: id, closed: false, head: start,
            len: len, rem: len, curblk: blk, blocks: blocks,
        }
    }

    /// Append an object with header to the log. Let append_safe
    /// handle the actual work.  If successful, returns virtual
    /// address in Ok().
    pub fn append(&mut self, buf: &ObjDesc) -> Status {
        if !self.closed {
            if self.can_hold(buf) {
                let va = self.headref() as usize;
                let header = EntryHeader::new(buf);
                let hlen = size_of::<EntryHeader>();
                self.append_safe(header.as_ptr(), hlen);
                self.append_safe(buf.key.as_ptr(), buf.key.len());
                self.append_safe(buf.value, buf.vlen as usize);
                Ok(va)
            } else { Err(ErrorCode::SegmentFull) }
        } else { Err(ErrorCode::SegmentClosed) }
    }

    pub fn close(&mut self) {
        self.closed = true;
    }

    /// Increment the head offset into the start of the next block.
    /// Return false if we cannot roll because we're already at the
    /// final block.
    fn next_block(&mut self) -> bool {
        let numblks: usize = self.blocks.len();
        if self.curblk < (numblks - 1) {
            self.curblk += 1;
            self.head = self.blocks[self.curblk].addr;
            true
        } else {
            warn!("block roll asked on last block");
            false
        }
    }

    /// Scan segment for all live entries (deep scan).
    pub fn num_live(&self) -> usize {
        let mut count: usize = 0;
        let mut header = EntryHeader::empty();
        let mut offset: usize = 0; // offset into segment (logical)
        let mut curblk: usize = 0; // Block that offset refers to
        // Manually cross block boundaries searching for entries.
        while offset < self.len {
            let base: usize = self.blocks[curblk].addr;
            let addr: usize = base + (offset % BLOCK_SIZE);
            header.read(addr);
            count += if header.is_valid() { 1 } else { 0 };
            // Determine next location to jump to
            offset += header.len();
            curblk = offset / BLOCK_SIZE;
        }
        count
    }

    pub fn can_hold(&self, buf: &ObjDesc) -> bool {
        self.rem >= buf.len_with_header()
    }

    //
    // --- Private methods ---
    //

    fn headref(&mut self) -> *mut u8 {
        self.head as *mut u8
    }

    /// Append some buffer safely across block boundaries (if needed).
    /// Caller must ensure the containing segment has sufficient
    /// capacity.
    fn append_safe(&mut self, from: *const u8, len: usize) {
        assert!(len <= self.rem);
        let mut remblk = self.rem_in_block();
        // 1. If buffer fits in remainder of block, just copy it
        if len <= remblk {
            unsafe {
                copy_nonoverlapping(from,self.headref(),len);
            }
            self.head += len;
            if len == remblk {
                self.next_block();
            }
        }
        // 2. If it spills over, perform two (or more) copies.
        else {
            let mut loc = from;
            let mut rem = len;
            // len may exceeed capacity of one block. Copy and roll in
            // pieces until the input is consumed.
            while rem > 0 {
                remblk = self.rem_in_block();
                let amt = cmp::min(remblk,rem);
                unsafe {
                    copy_nonoverlapping(loc,self.headref(), amt);
                }
                self.head += amt;
                rem -= amt;
                loc = (from as usize + amt) as *const u8;
                // If we exceeded the block, get the next one
                if remblk == amt {
                    assert_eq!(self.next_block(), true);
                }
            }
        }
        self.rem -= len;
    }

    fn rem_in_block(&self) -> usize {
        let blk = &self.blocks[self.curblk];
        blk.len - (self.head - blk.addr)
    }
}

impl Drop for Segment {

    fn drop(&mut self) {
        for b in self.blocks.iter() {
            // TODO push .clone() to BlockAllocator
        }
    }
}

pub struct SegmentManager {
    id: usize,
    size: usize, /// Total memory
    next_seg_id: usize, // FIXME atomic
    segment_size: usize,
    allocator: BlockAllocator,
    segments: Vec<Option<SegmentRef>>,
    free_slots: Vec<u32>,
}

impl SegmentManager {

    pub fn new(id: usize, segsz: usize, len: usize) -> Self {
        let b = BlockAllocator::new(BLOCK_SIZE, len);
        let num = len / segsz;
        let mut v: Vec<Option<SegmentRef>> = Vec::new();
        for i in 0..num {
            v.push(None);
        }
        let mut s: Vec<u32> = vec![0; num];
        for i in 0..num {
            s[i] = i as u32;
        }
        SegmentManager {
            id: id, size: len,
            next_seg_id: 0,
            segment_size: segsz,
            allocator: b,
            segments: v,
            free_slots: s,
        }
    }

    pub fn alloc(&mut self) -> Option<SegmentRef> {
        // TODO lock, unlock
        if self.free_slots.is_empty() {
            None
        } else {
            let slot: usize;
            match self.free_slots.pop() {
                None => panic!("No free slots"),
                Some(v) => slot = v as usize,
            };
            match self.segments[slot] {
                None => {},
                _ => panic!("Alloc from non-empty slot"),
            };
            // FIXME use config obj
            let num = SEGMENT_SIZE / BLOCK_SIZE;
            let blocks = self.allocator.alloc(num);
            match blocks {
                None => panic!("Could not allocate blocks"),
                _ => {},
            };
            self.next_seg_id += 1;
            self.segments[slot] = Some(seg_ref!(self.next_seg_id,
                                                  blocks.unwrap()));
            self.segments[slot].clone()
        }
    }

    pub fn free(&self, segment: SegmentRef) {
        unimplemented!();
    }

    //
    // --- Internal methods used for testing only ---
    //

    #[cfg(test)]
    pub fn test_all_segrefs_allocated(&mut self) -> bool {
        for i in 0..self.segments.len() {
            match self.segments[i] {
                None => return false,
                _ => {},
            }
        }
        true
    }

    /// Returns #live objects
    #[cfg(test)]
    pub fn test_count_live_objects(&self) -> usize {
        let mut count: usize = 0;
        for opt in &self.segments {
            match *opt {
                None => {},
                Some(ref seg) => {
                    count += seg.borrow().num_live();
                },
            }
        }
        count
    }
}

// -------------------------------------------------------------------
// The log
// -------------------------------------------------------------------

// TODO i definitely need a concurrent atomic append/pop list..

pub type LogHeadRef = Arc<RefCell<LogHead>>;

pub struct LogHead {
    segment: Option<SegmentRef>,
    manager: SegmentManagerRef,
}

impl LogHead {

    pub fn new(manager: SegmentManagerRef) -> Self {
        LogHead { segment: None, manager: manager }
    }

    pub fn append(&mut self, buf: &ObjDesc) -> Status {
        // allocate if head not exist
        match self.segment {
            None => { match self.roll() {
                Err(code) => return Err(code),
                Ok(ign) => {},
            }},
            _ => {},
        }
        if !rbm!(self.segment).can_hold(buf) {
            match self.roll() {
                Err(code) => return Err(code),
                Ok(ign) => {},
            }
        }
        let mut seg = rbm!(self.segment);
        match seg.append(buf) {
            Ok(va) => Ok(va),
            Err(code) => panic!("has space but append failed"),
        }
    }

    //
    // --- Private methods ---
    //

    /// Roll head. Close current and allocate new.
    fn roll(&mut self) -> Status {
        match self.segment.clone() {
            None => {
                self.segment = self.manager.borrow_mut().alloc();
                match self.segment {
                    None => Err(ErrorCode::OutOfMemory),
                    _ => Ok(1),
                }
            },
            Some(segref) => {
                segref.borrow_mut().close();
                // TODO add segment to 'closed' list
                self.segment = self.manager.borrow_mut().alloc();
                match self.segment {
                    None => Err(ErrorCode::OutOfMemory),
                    _ => Ok(1),
                }
            },
        }
    }

}

pub struct Log {
    head: LogHeadRef, // TODO make multiple
    manager: SegmentManagerRef,
}

impl Log {
    pub fn new(manager: SegmentManagerRef) -> Self {
        Log {
            head: Arc::new(RefCell::new(LogHead::new(manager.clone()))),
            manager: manager.clone(),
        }
    }

    /// Append an object to the log. If successful, returns the
    /// virtual address within the log inside Ok().
    pub fn append(&mut self, buf: &ObjDesc) -> Status {
        // 1. determine log head to use
        let head = &self.head;
        // 2. call append on the log head
        head.borrow_mut().append(buf) // returns address if ok
    }

    /// Toggle the valid bit within an entry's header. Typically used
    /// to invalidate an object that was overwritten or removed.
    pub fn invalidate_entry(&mut self, va: usize) -> Status {
        assert!(va > 0);
        // XXX lock the segment? lock something else?
        EntryHeader::invalidate(va);
        Ok(1)
    }

    pub fn enable_cleaning(&mut self) {
        unimplemented!();
    }

    pub fn disable_cleaning(&mut self) {
        unimplemented!();
    }
}

// -------------------------------------------------------------------
// TODO Cleaning
// -------------------------------------------------------------------

// -------------------------------------------------------------------
// Index structure
// -------------------------------------------------------------------

/// Index structure that allows us to retreive objects from the log.
/// It is just a simple wrapper over whatever data structure we wish
/// to eventually use.
pub struct Index<'a> {
    table: HashMap<&'a str, usize>,
}

impl<'a> Index<'a> {

    pub fn new() -> Self {
        Index {
            table: HashMap::new(), // also ::with_capacity(N)
        }
    }

    /// Return value of object if it exists, else None.
    pub fn get(&self, key: &'a str) -> Option<usize> {
        self.table.get(key).map(|r| *r) // &usize -> usize
    }

    /// Update location of object in the index. Returns None if object
    /// was newly inserted, or the virtual address of the prior
    /// object.
    pub fn update(&mut self, key: &'a str, value: usize) -> Option<usize> {
        self.table.insert(key, value)
    }

    /// Remove an entry. If it existed, return value, else return
    /// None.
    pub fn remove(&mut self, key: &'a str) -> Option<usize> {
        self.table.remove(key)
    }
}

// -------------------------------------------------------------------
// TODO RPC or SHM interface
// -------------------------------------------------------------------

// -------------------------------------------------------------------
// Main Nibble interface
// -------------------------------------------------------------------

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

    pub fn put_object(&mut self, obj: &ObjDesc<'a>) -> Status {
        let va: usize;
        // 1. add object to log
        match self.log.append(obj) {
            Err(code) => return Err(code),
            Ok(v) => va = v,
        }
        // 2. update reference to object, and if the object already
        // exists, 3. invalidate old entry
        match self.index.update(obj.key, va) {
            None => {},
            Some(old) => {
                match self.log.invalidate_entry(old) {
                    Err(code) => {
                        panic!("Error marking old entry at 0x{:x}: {:?}",
                               old, code);
                    },
                    Ok(v) => {},
                }
            },
        }
        Ok(1)
    }

    pub fn get_object(&self) -> Status {
        unimplemented!();
    }

    pub fn del_object(&mut self) -> Status {
        unimplemented!();
    }
}

// -------------------------------------------------------------------
// Memory utilities
// -------------------------------------------------------------------

/// Memory mapped region in our address space.
pub struct MemMap {
    addr: usize,
    len: usize,
}

/// Create anonymous private memory mapped region.
impl MemMap {

    pub fn new(len: usize) -> Self {
        // TODO fault on local socket
        let prot: libc::c_int = libc::PROT_READ | libc::PROT_WRITE;
        let flags: libc::c_int = libc::MAP_ANON |
            libc::MAP_PRIVATE | libc::MAP_NORESERVE;
        let addr: usize = unsafe {
            let p = 0 as *mut libc::c_void;
            libc::mmap(p, len, prot, flags, 0, 0) as usize
        };
        info!("mmap 0x{:x} {} MiB", addr, len>>20);
        assert!(addr != libc::MAP_FAILED as usize);
        MemMap { addr: addr, len: len }
    }

    pub fn addr(&self) -> usize { self.addr }
    pub fn len(&self) -> usize { self.len }
}

/// Prevent dangling regions by unmapping it.
impl Drop for MemMap {

    fn drop (&mut self) {
        unsafe {
            let p = self.addr as *mut libc::c_void;
            libc::munmap(p, self.len);
        }
    }
}

// -------------------------------------------------------------------
// Test Code
// -------------------------------------------------------------------

/// Used to save us from copy/paste in the tests.
#[cfg(test)]
fn test_create_segment_manager() -> SegmentManager {
    let memlen = 1<<23;
    let numseg = memlen / SEGMENT_SIZE;
    segmgr_ref!(0, SEGMENT_SIZE, memlen)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::mem::size_of;
    use std::mem::transmute;
    use std::ptr::copy;
    use std::ptr::copy_nonoverlapping;
    use std::rc::Rc;
    use std::sync::Arc;

    const BLOCK_SIZE: usize = 1 << 16;
    const SEGMENT_SIZE: usize = 1 << 20;

    #[test]
    fn memory_map_init() {
        let len = 1<<26;
        let mm = MemMap::new(len);
        assert_eq!(mm.len, len);
        assert!(mm.addr != 0 as usize);
        // TODO touch the memory somehow
        // TODO verify mmap region is unmapped
    }

    #[test]
    fn block() {
        let b = Block::new(42, 37);
        assert_eq!(b.addr, 42);
        assert_eq!(b.len, 37);
    }

    #[test]
    fn block_allocator_alloc_all() {
        let num = 64;
        let bytes = num * BLOCK_SIZE;
        let mut ba = BlockAllocator::new(BLOCK_SIZE, bytes);
        assert_eq!(ba.len(), num);
        assert_eq!(ba.freelen(), num);
        assert_eq!(ba.block_size, BLOCK_SIZE);

        let mut count = 0;
        while ba.freelen() > 0 {
            match ba.alloc(2) {
                None => break,
                opt => {
                    for b in opt.unwrap() {
                        count += 1;
                    }
                },
            }
        }
        assert_eq!(count, num);
        assert_eq!(ba.len(), num);
        assert_eq!(ba.freelen(), 0);
    }

    #[test]
    fn alloc_segment() {
        let num = 64;
        let bytes = num * BLOCK_SIZE;
        let mut ba = BlockAllocator::new(BLOCK_SIZE, bytes);
        let set;
        match ba.alloc(64) {
            None => panic!("Alloc should not fail"),
            opt => set = opt.unwrap(),
        }
        let id = 42;
        let seg = Segment::new(id, set);
        assert_eq!(seg.closed, false);
        assert!(seg.head != 0);
        assert_eq!(seg.len, bytes);
        // TODO verify blocks?
    }

    #[test]
    fn segment_manager() {
        let memlen = 1<<23;
        let numseg = memlen / SEGMENT_SIZE;
        let mut mgr = SegmentManager::new(0, SEGMENT_SIZE, memlen);
        for i in 0..numseg {
            match mgr.alloc() {
                None => panic!("segment alloc failed"),
                _ => {},
            }
        }
        assert_eq!(mgr.test_all_segrefs_allocated(), true);
    }

    /// Insert one unique object repeatedly;
    /// there should be only one live object in the log.
    /// lots of copy/paste...
    #[test]
    fn segment_manager_one_obj_overwrite() {
        let manager = test_create_segment_manager();
        let mut log = Log::new(manager.clone());

        let key: &'static str = "onlyone";
        let val: &'static str = "valuevaluevalue";
        let obj = ObjDesc::new(key, val.as_ptr(), val.len() as u32);
        let mut old_va: usize = 0; // address of prior object
        // fill up the log
        let mut count: usize = 0;
        loop {
            match log.append(&obj) {
                Ok(va) => {
                    count += 1;
                    // we must manually nix the old object (the Nibble
                    // class would otherwise handle this for us)
                    if old_va > 0 {
                        EntryHeader::invalidate(old_va);
                    }
                    old_va = va;
                },
                Err(code) => match code {
                    ErrorCode::OutOfMemory => break,
                    _ => panic!("filling log returned {:?}", code),
                },
            }
        }
        println!("-- appends: {}", count);
        assert_eq!(manager.borrow().test_count_live_objects(), 1);
    }

    #[test]
    fn log_alloc_until_full() {
        let mut log = Log::new(test_create_segment_mangager());
        let key: &'static str = "keykeykeykey";
        let val: &'static str = "valuevaluevalue";
        let obj = ObjDesc::new(key, val.as_ptr(), val.len() as u32);
        loop {
            match log.append(&obj) {
                Ok(ign) => {},
                Err(code) => match code {
                    ErrorCode::OutOfMemory => break,
                    _ => panic!("filling log returned {:?}", code),
                },
            }
        }
    }

    #[test]
    fn index_basic() {
        let mut index = Index::new();

        match index.update("alex", 42) {
            None => {}, // expected
            Some(v) => panic!("key should not exist"),
        }
        match index.update("alex", 24) {
            None => panic!("key should exist"),
            Some(v) => assert_eq!(v, 42),
        }

        match index.get("notexist") {
            None => {}, // ok
            Some(v) => panic!("get on nonexistent key"),
        }

        match index.get("alex") {
            None => panic!("key should exist"),
            Some(vref) => {}, // ok
        }
    }

    #[test]
    fn entry_header_readwrite_raw() {
        // get some raw memory
        let mem: Box<[u8;32]> = Box::new([0 as u8; 32]);
        let ptr = Box::into_raw(mem);
        // put a header into it with known values
        let mut header = EntryHeader::empty();
        assert_eq!(header.valid, 0 as u16);
        header.valid = 1 as u16;
        let len = size_of::<EntryHeader>();
        unsafe {
            let src: *const u8 = transmute(&header);
            let dst: *mut u8 = transmute(ptr);
            copy(src, dst, len);
        }
        // reset our copy, and re-read from raw memory
        header = EntryHeader::empty();
        assert_eq!(header.valid, 0 as u16);
        unsafe {
            let src: *const u8 = transmute(ptr);
            let dst: *mut u8 = transmute(&header);
            copy(src, dst, len);
        }
        // verify what we did worked
        assert_eq!(header.valid, 1 as u16);
        // free the original memory again
        let mem = unsafe { Box::from_raw(ptr) };
    }

    #[test]
    fn entry_header_readwrite() {
        // get some raw memory
        let mem: Box<[u8;32]> = Box::new([0 as u8; 32]);
        let ptr = Box::into_raw(mem);
        // put a header into it with known values
        let mut header = EntryHeader::empty();
        header.valid = 1 as u16;
        header.keylen = 47 as u16;
        header.datalen = 1025 as u32;
        header.write(ptr as usize);
        // reset our copy and verify
        header = EntryHeader::empty();
        header.read(ptr as usize);
        assert_eq!(header.valid, 1 as u16);
        assert_eq!(header.keylen, 47 as u16);
        assert_eq!(header.datalen, 1025 as u32);
        // invalidate, reset, verify
        EntryHeader::invalidate(ptr as usize);
        header = EntryHeader::empty();
        header.read(ptr as usize);
        assert_eq!(header.valid, 0 as u16);
        assert_eq!(header.keylen, 47 as u16);
        assert_eq!(header.datalen, 1025 as u32);
    }

    #[test]
    fn nibble() {
        let mem = 1 << 23;
        let mut nib = Nibble::new(mem);

        // insert initial object
        let key: &'static str = "keykeykeykey";
        let val: &'static str = "valuevaluevalue";
        let obj = ObjDesc::new(key, val.as_ptr(), val.len() as u32);
        match nib.put_object(&obj) {
            Ok(ign) => {},
            Err(code) => panic!("{:?}", code),
        }
        // change the value, keep key, check object is updated
        let val2: &'static str = "VALUEVALUEVALUE";
        let obj2 = ObjDesc::new(key, val2.as_ptr(), val2.len() as u32);
        match nib.put_object(&obj2) {
            Ok(ign) => {},
            Err(code) => panic!("{:?}", code),
        }
    }

    // TODO test objects larger than block, and segment
    // TODO test we can determine live vs dead entries in segment
}
