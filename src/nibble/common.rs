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

/// Update T in an Option<T> where T is an int type
macro_rules! incr {
    ( $obj:expr, $by:expr ) => {
        if let Some(val) = $obj {
            $obj = Some(val + $by);
        }
    }
}

pub type Pointer = Option<*const u8>;
pub type PointerMut = Option<*mut u8>;

// -------------------------------------------------------------------
// Error handling
// -------------------------------------------------------------------

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

