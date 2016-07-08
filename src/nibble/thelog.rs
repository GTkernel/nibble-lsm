use common::*;
use segment::*;
use epoch::*;

use std::cmp;
use std::mem::{size_of,transmute};
use std::ptr;
use std::sync::{Arc,Mutex};

use rand::{self,Rng};

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
//      Constants
//==----------------------------------------------------==//

pub const NUM_LOG_HEADS: u32 = 1;

//==----------------------------------------------------==//
//      Entry header
//==----------------------------------------------------==//

/// Describe entry in the log. Format is:
///     | EntryHeader | Key bytes | Data bytes |
/// This struct MUST NOT contain any pointers.
#[derive(Debug)]
#[repr(packed)]
pub struct EntryHeader {
    keylen: u32,
    datalen: u32,
}

// TODO can I get rid of most of this?
// e.g. use std::ptr::read / write instead?
impl EntryHeader {

    pub fn new(desc: &ObjDesc) -> Self {
        assert!(desc.keylen() <= usize::max_value());
        assert!(desc.getvalue() != None);
        EntryHeader {
            keylen: desc.keylen() as u32,
            datalen: desc.valuelen(),
        }
    }

    pub fn empty() -> Self {
        EntryHeader {
            keylen: 0 as u32,
            datalen: 0 as u32,
        }
    }

    pub fn getdatalen(&self) -> u32 { self.datalen }
    pub fn getkeylen(&self) -> u32 { self.keylen }
    pub fn object_length(&self) -> u32 { self.datalen + self.keylen }
    pub fn len_with_header(&self) -> usize {
        (self.object_length() as usize) + size_of::<EntryHeader>()
    }

    /// Size of this (entire) entry in the log.
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

    /// Give the starting address of the object in the log, provided
    /// the address of this EntryHeader within the log.
    pub fn data_address(&self, entry: usize) -> *const u8 {
        (entry + size_of::<EntryHeader>() + self.keylen as usize)
            as *mut u8
    }

    #[cfg(test)]
    pub fn set_key_len(&mut self, l: u32) { self.keylen = l; }

    #[cfg(test)]
    pub fn set_data_len(&mut self, l: u32) { self.datalen = l; }
}


//==----------------------------------------------------==//
//      Log head
//==----------------------------------------------------==//

pub type LogHeadRef = Arc<Mutex<LogHead>>;

macro_rules! loghead_ref {
    ( $manager:expr ) => {
        Arc::new( Mutex::new(
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
        assert!(buf.len_with_header() <
                (SEGMENT_SIZE-size_of::<SegmentHeader>()),
                "object {} larger than segment {}",
                buf.len_with_header(), SEGMENT_SIZE);

        let roll: bool;

        // check if head exists
        if let None = self.segment {
            debug!("head doesn't exist");
            roll = true;
        }
        // check if the object can fit in remaining space
        else {
            let segref = self.segment.clone().unwrap();
            roll = {
                let seg = segref.read().unwrap();
                !seg.can_hold(buf)
            };
            if roll {
                debug!("rolling: head cannot hold new object");
            }
        }
        if roll {
            let socket = {
                let guard = self.manager.lock().unwrap();
                guard.socket()
            };
            debug!("rolling head, socket {:?}", socket);
            if let Err(code) = self.roll() {
                return Err(code);
            }
        }

        let segref = self.segment.clone().unwrap();
        let mut seg = segref.write().unwrap();
        match seg.append(buf) {
            Err(s) => panic!("has space but append failed: {:?}",s),
            va @ Ok(_) => va,
        }
    }

    //
    // --- Private methods ---
    //

    /// Replace the head segment.
    fn replace(&mut self) -> Status {
        match self.manager.lock() {
            Ok(mut manager) => {
                self.segment = manager.alloc();
            },
            Err(_) => panic!("lock poison"),
        }
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
            match self.manager.lock() {
                Ok(mut manager) => {
                    manager.add_closed(&segref);
                },
                Err(_) => panic!("lock poison"),
            }
        }
    }

    /// Roll head. Close current and allocate new.
    fn roll(&mut self) -> Status {
        let segref = self.segment.clone();
        if let Some(seg) = segref {
            seg.write().unwrap().close();
            self.add_closed();
        }
        self.replace()
    }

}

//==----------------------------------------------------==//
//      The log
//==----------------------------------------------------==//

pub struct Log {
    heads: Vec<LogHeadRef>,
    manager: SegmentManagerRef,
    seginfo: SegmentInfoTableRef,
    // TODO track current capacity?
}

impl Log {

    pub fn new(manager: SegmentManagerRef) -> Self {
        let seginfo = match manager.lock() {
            Err(_) => panic!("lock poison"),
            Ok(guard) => guard.seginfo(),
        };
        let mut heads: Vec<LogHeadRef>;
        heads = Vec::with_capacity(NUM_LOG_HEADS as usize);
        for _ in 0..NUM_LOG_HEADS {
            heads.push(loghead_ref!(manager.clone()));
        }
        Log {
            heads: heads,
            manager: manager.clone(),
            seginfo: seginfo,
        }
    }

    /// Append an object to the log. If successful, returns the
    /// virtual address within the log inside Ok().
    /// FIXME check key is valid UTF-8
    pub fn append(&self, buf: &ObjDesc) -> Status {
        // 1. pick a log head
        let x = unsafe { rdrand() } % NUM_LOG_HEADS;
        let head = &self.heads[x as usize];
        // 2. call append on the log head
        let va: usize = {
            let mut guard = head.lock().unwrap();
            match guard.append(buf) {
                e @ Err(_) => return e,
                Ok(va) => va,
            }
        };
        // 3. update segment info table
        // FIXME shouldn't have to lock for this
        let idx = {
            let guard = self.manager.lock().unwrap();
            guard.segment_of(va)
        };
        debug_assert_eq!(idx.is_some(), true);
        let len = buf.len_with_header();
        debug_assert!(len < SEGMENT_SIZE);
        self.seginfo.incr_live(idx.unwrap(), len);
        // 4. return virtual address of new object
        Ok(va)
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
pub struct EntryReference<'a> {
    pub offset: usize, // into first block
    pub len: usize, /// header + key + data
    pub keylen: u32,
    pub datalen: u32,
    pub blocks: &'a [BlockRef],
}

// TODO optimize for cases where the blocks are contiguous
// copying directly, or avoid copying (provide reference to it)
impl<'a> EntryReference<'a> {

    pub fn get_loc(&self) -> usize {
        self.offset + self.blocks[0].addr
    }

    /// Copy out the key
    pub unsafe fn get_key(&self) -> u64 {
        let mut offset = self.offset + size_of::<EntryHeader>();
        let mut va = self.blocks[0].addr + offset;
        ptr::read(va as *const u64)
    }

    /// Copy out the value
    pub unsafe fn get_data(&self, out: *mut u8) {
        let mut offset = self.offset + self.len
                            - self.datalen as usize;
        let block  = offset / BLOCK_SIZE;
        offset = offset % BLOCK_SIZE;
        self.copy_out(out, block, offset, self.datalen as usize)
    }

    //
    // --- Private methods ---
    //

    /// TODO clean up code
    /// nearly verbatim overlap with Segment::append_entry
    unsafe fn copy_out(&self, out: *mut u8,
                       block: usize, offset: usize,
                       remaining: usize) {
        // reassign as mutable
        let mut block = block;
        let mut remaining = remaining;

        let mut src: *const u8;
        let mut dst: *mut u8;

        // Logical offset into new buffer
        let mut poffset: usize = 0;

        let mut va = self.blocks[block].addr + offset;
        let mut amt = cmp::min(BLOCK_SIZE - offset, remaining);
        src = va as *const u8;
        dst = ((out as usize) + poffset) as *mut u8;
        ptr::copy_nonoverlapping(src, dst, amt);
        remaining -= amt;
        poffset += amt;
        block += 1;

        while remaining > 0 {
            va = self.blocks[block].addr;
            amt = cmp::min(BLOCK_SIZE, remaining);
            src = va as *const u8;
            dst = ((out as usize) + poffset) as *mut u8;
            ptr::copy_nonoverlapping(src, dst, amt);
            remaining -= amt;
            poffset += amt;
            block += 1;
        }
    }
}

//==----------------------------------------------------==//
//      Unit tests
//==----------------------------------------------------==//

#[cfg(test)]
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
