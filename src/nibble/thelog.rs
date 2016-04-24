use common::*;
use segment::*;

use std::mem::transmute;
use std::mem::size_of;
use std::ptr;
use std::ptr::copy;
use std::ptr::copy_nonoverlapping;
use std::sync::Arc;
use std::cell::RefCell;

/// Describe entry in the log. Format is:
///     | EntryHeader | Key bytes | Data bytes |
/// An invalid object may exist if it was deleted or a new version was
/// created in the log. This struct MUST NOT contain any pointers.  As
/// segments are only appended, then we can assume the last entry for
/// which the signature does not match is the last entry in a segment.
/// The signature is used to identify an EntryHeader which was
/// actually created. It is computed from an instance of EntryHeader
/// with the signature field set to zero.
#[derive(Debug)]
#[repr(packed)]
pub struct EntryHeader {
    /// Verify this entry was actually created. "Valid" just means the
    /// index points to this entry; "invalid" means this entry did
    /// exist, but was deleted. Any other value for this field means
    /// it should be ignored (and likely no other entries exist in the
    /// remainder of the segment).
    __signature: u16,
    keylen: u16,
    datalen: u32,
}

// Arbitrary non-obvious values to identify header. Both mean we
// created the entry at some point; any other value means the entry
// isn't an entry -- bogus data.
const ENTRY_HEADER_SIG_LIVE:    u16 = 0x4FDA;
const ENTRY_HEADER_SIG_DEFUNCT: u16 = 0x37B4;

// TODO put a segment summary at the front, holding the count of items?
// that way, we know when to stop searching the segment... and we don't need this entry header
// status thing

// TODO get rid of this
#[derive(Debug)]
pub enum EntryHeaderStatus {
    Live, // Created and index points to it
    Defunct, // Created, but index no longer points to it
    Invalid, // Not an entry
}

impl EntryHeader {

    pub fn new(desc: &ObjDesc) -> Self {
        assert!(desc.keylen() <= usize::max_value());
        assert!(desc.getvalue() != None);
        EntryHeader {
            __signature: ENTRY_HEADER_SIG_LIVE,
            keylen: desc.keylen() as u16,
            datalen: desc.valuelen(),
        }
    }

    pub fn empty() -> Self {
        EntryHeader {
            __signature: 0 as u16, // not a real header, yet
            keylen: 0 as u16,
            datalen: 0 as u32,
        }
    }

    pub fn getdatalen(&self) -> u32 { self.datalen }
    pub fn getkeylen(&self) -> u16 { self.keylen }

    pub fn status(&self) -> EntryHeaderStatus {
        match self.__signature {
            ENTRY_HEADER_SIG_LIVE => EntryHeaderStatus::Live,
            ENTRY_HEADER_SIG_DEFUNCT => EntryHeaderStatus::Defunct,
            _ => EntryHeaderStatus::Invalid,
        }
    }

    /// Overwrite ourself with an entry somewhere in memory.
    /// TODO use std::ptr::read?
    pub unsafe fn read(&mut self, va: usize) {
        assert!(va > 0);
        let len = size_of::<EntryHeader>();
        let src: *const u8 = transmute(va);
        let dst: *mut u8 = transmute(self);
        copy(src, dst, len);
    }

    /// Store ourself to memory.
    /// TODO use std::ptr::write?
    pub unsafe fn write(&self, va: usize) {
        assert!(va > 0);
        let len = size_of::<EntryHeader>();
        let src: *const u8 = transmute(self);
        let dst: *mut u8 = transmute(va);
        copy(src, dst, len);
    }

    /// Mark an EntryHeader invalid in memory.
    /// FIXME this should lock the containing segment before updating
    /// the state, as a compaction thread might be working on it. Or
    /// we lock the object (e.g. with an object lock table).
    pub unsafe fn invalidate(va: usize) {
        assert!(va > 0);
        let mut header = EntryHeader::empty();
        header.read(va);
        match header.status() {
            EntryHeaderStatus::Live => {},
            s => panic!("header expected to be live: {:?}", s),
        }
        header.__signature = ENTRY_HEADER_SIG_DEFUNCT;
        header.write(va);
        // TODO need memory fence?
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

    //
    // --- Internal methods used for testing only ---
    //

    #[cfg(test)]
    pub fn set_valid(&mut self) {
        self.__signature = ENTRY_HEADER_SIG_LIVE;
    }
}


// -------------------------------------------------------------------
// The log
// -------------------------------------------------------------------

pub type LogHeadRef = Arc<RefCell<LogHead>>;

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
                // TODO this line might change to just move the
                // segment to a small closed segment pool
                //self.manager.borrow_mut().newly_closed(&segref);
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
        // FIXME acquire read/writer lock
        // FIXME stale the segment, m
        unsafe { EntryHeader::invalidate(va); }
        Ok(1)
    }

    pub fn enable_cleaning(&mut self) {
        unimplemented!();
    }

    pub fn disable_cleaning(&mut self) {
        unimplemented!();
    }
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

    use test::Bencher;
    use segment::*;
    use common::*;

    #[test]
    fn log_alloc_until_full() {
        let memlen = 1<<23;
        let numseg = memlen / SEGMENT_SIZE;
        let manager = segmgr_ref!(0, SEGMENT_SIZE, memlen);
        let mut log = Log::new(manager);
        let key: &'static str = "keykeykeykey";
        let val: &'static str = "valuevaluevalue";
        let obj = ObjDesc::new(key, Some(val.as_ptr()), val.len() as u32);
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
    fn entry_header_readwrite_raw() {
        // get some raw memory
        let mem: Box<[u8;32]> = Box::new([0 as u8; 32]);
        let ptr = Box::into_raw(mem);

        // put a header into it with known values
        let mut header = EntryHeader::empty();
        match header.status() {
            EntryHeaderStatus::Invalid => {}, // ok
            _ => panic!("header must be invalid"),
        }
        header.set_valid();
        match header.status() {
            EntryHeaderStatus::Live => {}, // ok
            _ => panic!("header must be live"),
        }

        let len = size_of::<EntryHeader>();
        unsafe {
            let src: *const u8 = transmute(&header);
            let dst: *mut u8 = transmute(ptr);
            copy(src, dst, len);
        }

        // reset our copy, and re-read from raw memory
        header = EntryHeader::empty();
        unsafe {
            let src: *const u8 = transmute(ptr);
            let dst: *mut u8 = transmute(&header);
            copy(src, dst, len);
        }

        // verify what we did worked
        match header.status() {
            EntryHeaderStatus::Live => {}, // ok
            _ => panic!("header must be live"),
        }

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
        header.set_valid();
        header.keylen = 47 as u16;
        header.datalen = 1025 as u32;
        unsafe { header.write(ptr as usize); }

        // reset our copy and verify
        header = EntryHeader::empty();
        unsafe { header.read(ptr as usize); }
        match header.status() {
            EntryHeaderStatus::Live => {}, // ok
            _ => panic!("header should be live"),
        }
        assert_eq!(header.keylen, 47 as u16);
        assert_eq!(header.datalen, 1025 as u32);

        // invalidate, reset, verify
        unsafe { EntryHeader::invalidate(ptr as usize); }
        header = EntryHeader::empty();
        unsafe { header.read(ptr as usize); }
        match header.status() {
            EntryHeaderStatus::Defunct => {}, // ok
            _ => panic!("header should be defunct"),
        }
        assert_eq!(header.keylen, 47 as u16);
        assert_eq!(header.datalen, 1025 as u32);
    }


    // TODO fill log 50%, delete random items, then manually force
    // cleaning to test it
}
