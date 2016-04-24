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

    /// Overwrite ourself with an entry somewhere in memory.
    pub unsafe fn read(&mut self, va: usize) {
        assert!(va > 0);
        let len = size_of::<EntryHeader>();
        let src: *const u8 = transmute(va);
        let dst: *mut u8 = transmute(self);
        copy(src, dst, len);
    }

    /// Store ourself to memory.
    pub unsafe fn write(&self, va: usize) {
        assert!(va > 0);
        let len = size_of::<EntryHeader>();
        let src: *const u8 = transmute(self);
        let dst: *mut u8 = transmute(va);
        copy(src, dst, len);
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


    // FIXME rewrite these unit tests

//    #[test]
//    fn entry_header_readwrite_raw() {
//        // get some raw memory
//        let mem: Box<[u8;32]> = Box::new([0 as u8; 32]);
//        let ptr = Box::into_raw(mem);
//
//        // put a header into it with known values
//        let mut header = EntryHeader::empty();
//        match header.status() {
//            EntryHeaderStatus::Invalid => {}, // ok
//            _ => panic!("header must be invalid"),
//        }
//        header.set_valid();
//        match header.status() {
//            EntryHeaderStatus::Live => {}, // ok
//            _ => panic!("header must be live"),
//        }
//
//        let len = size_of::<EntryHeader>();
//        unsafe {
//            let src: *const u8 = transmute(&header);
//            let dst: *mut u8 = transmute(ptr);
//            copy(src, dst, len);
//        }
//
//        // reset our copy, and re-read from raw memory
//        header = EntryHeader::empty();
//        unsafe {
//            let src: *const u8 = transmute(ptr);
//            let dst: *mut u8 = transmute(&header);
//            copy(src, dst, len);
//        }
//
//        // verify what we did worked
//        match header.status() {
//            EntryHeaderStatus::Live => {}, // ok
//            _ => panic!("header must be live"),
//        }
//
//        // free the original memory again
//        let mem = unsafe { Box::from_raw(ptr) };
//    }
//
//    #[test]
//    fn entry_header_readwrite() {
//        // get some raw memory
//        let mem: Box<[u8;32]> = Box::new([0 as u8; 32]);
//        let ptr = Box::into_raw(mem);
//
//        // put a header into it with known values
//        let mut header = EntryHeader::empty();
//        header.set_valid();
//        header.keylen = 47 as u32;
//        header.datalen = 1025 as u32;
//        unsafe { header.write(ptr as usize); }
//
//        // reset our copy and verify
//        header = EntryHeader::empty();
//        unsafe { header.read(ptr as usize); }
//        match header.status() {
//            EntryHeaderStatus::Live => {}, // ok
//            _ => panic!("header should be live"),
//        }
//        assert_eq!(header.keylen, 47 as u32);
//        assert_eq!(header.datalen, 1025 as u32);
//
//        // invalidate, reset, verify
//        unsafe { EntryHeader::invalidate(ptr as usize); }
//        header = EntryHeader::empty();
//        unsafe { header.read(ptr as usize); }
//        match header.status() {
//            EntryHeaderStatus::Defunct => {}, // ok
//            _ => panic!("header should be defunct"),
//        }
//        assert_eq!(header.keylen, 47 as u32);
//        assert_eq!(header.datalen, 1025 as u32);
//    }
}
