use common::*;
use segment;
use segment::*;
use meta::*;
use memory::*;
use clock;
use numa::{self,NodeId};

use std::mem::{self,size_of};
use std::sync::Arc;
use std::ptr;
use std::intrinsics;

use parking_lot as pl;

/// 2^this number of blocks to vary new segment allocations
pub const ALLOC_NBLKS_VAR:  usize = BLOCKS_PER_SEG >> 2;

/// If this breaks compilation, you need to make segments larger or
/// blocks smaller.
#[allow(dead_code)]
fn __assert_Check_nblks_Not_Zero() {
    1/ALLOC_NBLKS_VAR;
}

/// Acquire read lock on SegmentRef
macro_rules! rlock {
    ( $segref:expr ) => {
        $segref.unwrap().read().unwrap()
    }
}

/// Acquire write lock on SegmentRef
macro_rules! wlock {
    ( $segref:expr ) => {
        $segref.unwrap().write().unwrap()
    }
}

//==----------------------------------------------------==//
//      Entry header
//==----------------------------------------------------==//

/// Describe entry in the log. Format is:
///     | EntryHeader | Key bytes | Data bytes |
/// This struct MUST NOT contain any pointers.
#[derive(Debug)]
#[repr(C,packed)]
pub struct EntryHeader {
    //keylen: u32, // TODO don't need this; keys are fixed-size at 8B
    datalen: u32,
}

// TODO can I get rid of most of this?
// e.g. use std::ptr::read / write instead?
impl EntryHeader {

    pub fn new(desc: &ObjDesc) -> Self {
        debug_assert!(desc.valuelen() > 0usize);
        // NOTE an ObjDesc may have a null value pointer,
        // as it may originate from an alloc instead of a PUT.
        // assert!(!desc.getvalue().0 .is_null());
        EntryHeader {
            //keylen: desc.keylen() as u32,
            datalen: desc.valuelen() as u32,
        }
    }

    pub fn empty() -> Self {
        EntryHeader {
            //keylen: 0 as u32,
            datalen: 0 as u32,
        }
    }

    #[inline(always)]
    pub fn getdatalen(&self) -> u32 { self.datalen }
    #[inline(always)]
    pub fn object_length(&self) -> u32 {
        self.datalen + size_of::<KeyType>() as u32
    }
    #[inline(always)]
    pub fn len_with_header(&self) -> usize {
        (self.object_length() as usize) + size_of::<EntryHeader>()
    }

    /// Size of this (entire) entry in the log.
    pub fn len(&self) -> usize {
        size_of::<EntryHeader>() +
            self.datalen as usize +
            size_of::<KeyType>()
    }

    pub fn as_ptr(&self) -> *const u8 {
        self as *const Self as *const u8
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self as *mut Self as *mut u8
    }

    //#[cfg(test)]
    //pub fn set_key_len(&mut self, l: u32) { self.keylen = l; }

    #[cfg(test)]
    pub fn set_data_len(&mut self, l: u32) { self.datalen = l; }
}


//==----------------------------------------------------==//
//      Log head
//==----------------------------------------------------==//

pub type LogHeadRef = Arc<pl::Mutex<LogHead>>;

macro_rules! loghead_ref {
    ( $manager:expr ) => {
        Arc::new( pl::Mutex::new(
                LogHead::new($manager)
                ))
    }
}

pub struct LogHead {
    segment: Option<SegmentRef>,
    manager: SegmentManagerRef,
}

// TODO when head is rolled, don't want to contend with other threads
// when handing it off to the compactor. we could keep a 'closed
// segment pool' with each log head. then periodically merge them into
// the compactor. this pool could be a concurrent queue with atomic
// push/pop. for now we just shove it into the compactor directly.

impl LogHead {

    pub fn new(manager: SegmentManagerRef) -> Self {
        LogHead { segment: None, manager: manager }
    }

    pub fn append(&mut self, buf: &ObjDesc) -> Status {
        debug_assert!(buf.len_with_header() <
                (SEGMENT_SIZE-size_of::<SegmentHeader>()),
                "object {} larger than segment {}",
                buf.len_with_header(), SEGMENT_SIZE);

        let roll: bool;

        // check if head exists
        if unlikely!(self.segment.is_none()) {
            roll = true;
        }
        // check if the object can fit in remaining space
        else {
            roll = self.segment.as_ref().map(|seg|
                !seg.read().can_hold(buf)
            ).unwrap();
        }
        if unlikely!(roll) {
            trace!("rolling head, socket {:?}",
                   self.manager.socket());
            if let Err(code) = self.roll() {
                return Err(code);
            }
        }

        self.segment.as_ref().map(|seg| {
            match seg.write().append(buf) {
                Err(s) => panic!("has space but append failed: {:?}",s),
                va @ Ok(_) => va,
            }
        }).unwrap()
    }

    //
    // --- Private methods ---
    //

    /// Replace the head segment.
    /// Vary the size slightly to avoid contention on future
    /// allocations (in case many threads roll all at once).
    fn replace(&mut self) -> Status {
        //self.segment = self.manager.alloc();
        let nblks: usize = unsafe {
            (rdrand() as usize % ALLOC_NBLKS_VAR) +
                BLOCKS_PER_SEG
        };
        self.segment = self.manager.alloc_size(nblks);
        match self.segment {
            None => Err(ErrorCode::OutOfMemory),
            _ => Ok(1),
        }
    }

    /// Upon closing a head segment, add reference to the recently
    /// closed list for the compaction code to pick up.
    /// TODO move to local head-specific pool to avoid locking
    fn add_closed(&mut self) {
        if let Some(segref) = self.segment.clone() {
            self.manager.add_closed(&segref);
        }
    }

    /// Roll head. Close current and allocate new.
    fn roll(&mut self) -> Status {
        let segref = self.segment.clone();
        if let Some(seg) = segref {
            seg.write().close();
            self.add_closed();
        }
        self.replace()
    }

}

//==----------------------------------------------------==//
//      The log
//==----------------------------------------------------==//

pub fn num_log_heads() -> usize {
    numa::NODE_MAP.cpus_in(NodeId(0))
}

pub struct Log {
    heads: Vec<LogHeadRef>,
    manager: SegmentManagerRef,
    seginfo: SegmentInfoTableRef,
    // TODO track current capacity?
    /// Number of log heads per instance of the log (per socket).
    /// Allocate as many as there are cores on the socket.
    nheads: usize,

}

impl Log {

    pub fn new(manager: SegmentManagerRef) -> Self {
        let nheads: usize = num_log_heads();
        let seginfo = manager.seginfo();
        let mut heads: Vec<LogHeadRef>;
        heads = Vec::with_capacity(nheads);
        for _ in 0..nheads {
            heads.push(loghead_ref!(manager.clone()));
        }
        info!("new, heads {} len {}",
              nheads,
              size_of::<Vec<LogHeadRef>>() *
              size_of::<LogHeadRef>() * nheads);
        Log {
            heads: heads,
            manager: manager.clone(),
            seginfo: seginfo,
            nheads: nheads,
        }
    }

    /// Append an object to the log. If successful, returns the
    /// virtual address within the log inside Ok().
    pub fn append(&self, buf: &ObjDesc) -> Status {
        let va: usize;

        // fast quasi-randomness (TODO might always be zero?)
        //let mut i = (buf as *const _ as usize) & LOG_HEADS_MASK;
        //let mut i = clock::now() as usize & LOG_HEADS_MASK;

        // using processor ID should hopefully avoid conflicts
        // compared to random assignment and hoping for luck
        let (sockID,coreID) = clock::rdtscp_id();
        let i = coreID as usize % self.nheads;

        // TODO keep track of #times we had to iterate for avail head

        // 1. pick a log head and append
        let mut opt;
        loop {
            opt = self.heads[i].try_lock();
            if likely!(opt.is_some()) {
                break;
            }
        }
        let mut head = opt.unwrap();
        match head.append(buf) {
            e @ Err(_) => return e,
            Ok(v) => va = v,
        }

        // 2. update segment info table
        let idx = self.manager.segment_of(va);
        let len = buf.len_with_header();
        debug_assert!(len < SEGMENT_SIZE);
        self.seginfo.incr_live(idx, len);

        // 3. return virtual address of new object
        Ok(va)
    }

    /// Pull out the value for an entry within the log (not the entire
    /// object). DO NOT do any buffer allocations on this fast path.
    #[inline(always)]
    pub fn get_entry(&self, va: usize, buf: &mut [u8]) {
        let head_len = mem::size_of::<EntryHeader>();
        let key_len = mem::size_of::<KeyType>();
        let block_addr: usize = va & !BLOCK_OFF_MASK;
        let remain: usize = BLOCK_SIZE - (va - block_addr);

        // If object lands squarely within a single block, just memcpy
        // that out. else, figure out the segment and thus the
        // block list, and do a slowpath extraction
        if likely!(remain > (head_len + key_len)) {
            let entry: &EntryHeader = unsafe {
                &* (va as *const usize as *const EntryHeader)
            };
            let size: usize = entry.len_with_header();
            if likely!(size <= remain) {
                let mut value_len = entry.datalen as usize;
                // if unlikely!(buf.len() < value_len) {
                //     warn!("Buffer len {} GET is smaller than object {}",
                //          buf.len(), value_len);
                //     value_len = buf.len();
                // }
                let valuep = (va + head_len + key_len)
                    as *const usize as *const u8;
                unsafe {
                    copy(buf.as_mut_ptr(), valuep, value_len);
                    //ptr::copy_nonoverlapping(valuep,
                        //buf.as_mut_ptr(), value_len);
                }
            }
        }

        // gotta assemble the object (even if it lies fully in a
        // block, but just not the one the entry header is in, we
        // still don't know which block that is without the segment
        else {
            let block: Block = self.manager.block_of(va);
            debug_assert_eq!(block.list().ptr().is_null(), false);
            let usl = block.list();
            debug_assert!(block.blk_idx() < usl.len(),
                "block idx {} out of bounds for uslice {}",
                block.blk_idx(), usl.len());
            let list: &[BlockRef] = unsafe { usl.slice() };
            let entry = get_ref(list, block.blk_idx(), va);
            unsafe { entry.get_buf(buf); }
        }
    }

    /// Only pull out the entry header. Useful to know the object size
    /// when deleting, updating, or compacting. Unlike get_ref, we
    /// make a real copy of the header. get_ref only does so if the
    /// header happens to be split across two blocks.
    #[inline(always)]
    pub fn copy_header(&self, va: usize) -> EntryHeader {
        let mut header: EntryHeader = EntryHeader::empty();
        let offset = va & BLOCK_OFF_MASK;
        let blk_tail = BLOCK_SIZE - offset;
        let len = size_of::<EntryHeader>();

        if blk_tail >= len {
            let head_addr = va as *const usize as *const EntryHeader;
            header = unsafe { ptr::read(head_addr) };
        }
        // header is split.. gotta do more work
        // some copy/paste from get_entry above
        else {
            let block: Block = self.manager.block_of(va);
            debug_assert_eq!(block.list().ptr().is_null(), false);
            let usl = block.list();
            debug_assert!(block.blk_idx() < usl.len(),
                "block idx {} out of bounds for uslice {}",
                block.blk_idx(), usl.len());
            let list: &[BlockRef] = unsafe { usl.slice() };
            unsafe {
                copy_out(&list[block.blk_idx()..], offset,
                         header.as_mut_ptr(), len);
            }
        }

        header
    }

    //
    // --- Internal methods used for testing only ---
    //

    #[cfg(test)]
    pub fn seginfo(&self) -> SegmentInfoTableRef { self.seginfo.clone() }
}

//==----------------------------------------------------==//
//      Entry reference
//==----------------------------------------------------==//

/// Reference to entry in the log. Used by Segment iterators since i)
/// items in memory don't have an associated language type (this
/// provides that function) and ii) we want to avoid copying objects
/// each time a reference is passed around; we lazily copy the object
/// from the log only when a client asks for it
#[derive(Debug)]
pub struct EntryReference<'a> {
    pub offset: usize, // into first block
    pub len: usize, /// header + key + data
    pub datalen: u32,
    /// TODO can we avoid cloning the Arcs?
    pub blocks: &'a [BlockRef]
}

// TODO optimize for cases where the blocks are contiguous
// copying directly, or avoid copying (provide reference to it)
impl<'a> EntryReference<'a> {

    pub fn get_loc(&self) -> usize {
        self.offset + self.blocks[0].addr()
    }

    /// Copy out the key
    pub unsafe fn get_key(&self) -> u64 {
        let offset = self.offset + size_of::<EntryHeader>();
        let mut key: u64 = 0;
        // TODO optimize if contiguous
        // hm, lots of overhead for copying 8 bytes
        segment::copy_out(&self.blocks, offset,
                          &mut key as *mut u64 as *mut u8,
                          size_of::<u64>());
        key
    }

    /// Copy out the value
    #[inline(always)]
    pub unsafe fn get_data(&self, out: *mut u8) {
        let offset = self.offset + self.len
                            - self.datalen as usize;
        // TODO optimize if contiguous
        segment::copy_out(&self.blocks, offset,
                          out, self.datalen as usize);
    }

    /// Copy out the value. Same as get_data but takes a slice, which
    /// has a len we can verify against.
    #[inline(always)]
    pub unsafe fn get_buf(&self, out: &mut [u8]) {
        let dlen = self.datalen as usize;
        let offset = self.offset + self.len - dlen;
        //if unlikely!(out.len() < dlen) {
            //panic!("ur buf is 2 smal");
        //}
        // TODO optimize if contiguous
        segment::copy_out(&self.blocks, offset,
                          out.as_mut_ptr(), dlen);
    }

}

/// Construct an EntryReference given a VA and a set of Blocks.
/// See also Log::copy_header
pub fn get_ref(list: &[BlockRef], idx: usize, va: usize) -> EntryReference {
    let mut header: EntryHeader;
    let href: &EntryHeader;
    let offset = va & BLOCK_OFF_MASK;
    let blk_tail = BLOCK_SIZE - offset;
    let len = size_of::<EntryHeader>();

    // only copy out the header if it is split across blocks,
    // else, set the href to point directly into the log
    if blk_tail >= len {
        let head_addr = va as *const usize as *const EntryHeader;
        href = unsafe { &*head_addr };
    } else { unsafe {
        header = EntryHeader::empty();
        copy_out(&list[idx..], offset, header.as_mut_ptr(), len);
        href = &header;
    }}

    debug_assert!(href.getdatalen() > 0);
    // https://github.com/rust-lang/rust/issues/22644
    debug_assert!( (href.getdatalen() as usize) < SEGMENT_SIZE);

    // determine which blocks belong
    let mut nblks = 1;
    let entry_len = href.len_with_header();
    if entry_len > blk_tail {
        nblks += ((entry_len - blk_tail - 1) / BLOCK_SIZE) + 1;
    }
    debug_assert!( (idx + nblks - 1) < list.len() );

    EntryReference {
        offset: offset,
        len: entry_len,
        datalen: href.getdatalen(),
        blocks: &list[idx..(idx + nblks)],
    }
}

//==----------------------------------------------------==//
//      Unit tests
//==----------------------------------------------------==//

#[cfg(IGNORE)]
mod tests {
    use super::*;

    use std::ptr;
    use std::sync::{Arc,Mutex};

    use segment::*;
    use common::*;

    use super::super::logger;

    #[test]
    fn log_alloc_until_full() {
        logger::enable();
        let memlen = 1<<27;
        let manager = segmgr_ref!(SEGMENT_SIZE, memlen);
        let log = Log::new(manager);
        let key = String::from("keykeykeykey");
        let mut val = String::from("valuevaluevalue");
        for _ in 0..200 {
            val.push_str("valuevaluevaluevaluevalue");
        }
        let obj = ObjDesc::new2(&key, &val);
        loop {
            if let Err(code) = log.append(&obj) {
                match code {
                    ErrorCode::OutOfMemory => break,
                    _ => panic!("filling log returned {:?}", code),
                }
            }
        } // loop
    }

    // TODO fill log 50%, delete random items, then manually force
    // cleaning to test it

    // FIXME rewrite these unit tests

    #[test]
    fn entry_header_readwrite() {
        logger::enable();
        // get some raw memory
        let mem: Box<[u8;32]> = Box::new([0 as u8; 32]);
        let ptr = Box::into_raw(mem);

        // put a header into it with known values
        let mut header = EntryHeader::empty();
        header.set_key_len(5);
        header.set_data_len(7);
        assert_eq!(header.getkeylen(), 5);
        assert_eq!(header.getdatalen(), 7);

        unsafe {
            ptr::write(ptr as *mut EntryHeader, header);
        }

        // reset our copy, and re-read from raw memory
        unsafe {
            header = ptr::read(ptr as *const EntryHeader);
        }
        assert_eq!(header.getkeylen(), 5);
        assert_eq!(header.getdatalen(), 7);

        // free the original memory again
        unsafe { Box::from_raw(ptr); }
    }
}
