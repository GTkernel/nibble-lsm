//==----------------------------------------------------==//
//      General types
//==----------------------------------------------------==//

pub type Pointer = Option<*const u8>;
pub type PointerMut = Option<*mut u8>;

//==----------------------------------------------------==//
//      Error handling
//==----------------------------------------------------==//

#[derive(Debug)]
pub enum ErrorCode {

    SegmentFull,
    SegmentClosed,

    OutOfMemory,

    KeyNotExist,

    EmptyObject,
}

pub fn err2str(code: ErrorCode) -> &'static str {
    match code {
        ErrorCode::SegmentFull   => { "Segment is full" },
        ErrorCode::SegmentClosed => { "Segment is closed" },
        ErrorCode::OutOfMemory   => { "Out of memory" },
        ErrorCode::KeyNotExist   => { "Key does not exist" },
        ErrorCode::EmptyObject   => { "Object is empty" },
    }
}

pub type Status = Result<(usize), ErrorCode>;

