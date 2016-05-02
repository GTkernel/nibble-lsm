// Macros go in separate module to satisfy circular dependencies

//==----------------------------------------------------==//
//      Common macros
//==----------------------------------------------------==//

/// Extract reference &T from Option<T>
#[macro_export]
macro_rules! r {
    ( $obj:expr ) => { $obj.as_ref().unwrap() }
}

/// Borrow on a reference &T from a RefCell<Option<T>>
#[macro_export]
macro_rules! rb {
    ( $obj:expr ) => { r!($obj).borrow() }
}

/// Same as rb! but mutable
#[macro_export]
macro_rules! rbm {
    ( $obj:expr ) => { r!($obj).borrow_mut() }
}

/// Update T in an Option<T> where T is an int type
#[macro_export]
macro_rules! incr {
    ( $obj:expr, $by:expr ) => {
        if let Some(val) = $obj {
            $obj = Some(val + $by);
        }
    }
}

//==----------------------------------------------------==//
//      Compaction macros
//==----------------------------------------------------==//

/// Make new CompactorRef
#[macro_export]
macro_rules! compref {
    ( $segmgr:expr ) => {
        Arc::new( Mutex::new( RefCell::new(
                Compactor::new($segmgr)
                )))
    }
}

//==----------------------------------------------------==//
//      Segment macros
//==----------------------------------------------------==//

/// Instantiate new Segment as a SegmentRef
#[macro_export]
macro_rules! seg_ref {
    ( $id:expr, $slot:expr, $blocks:expr ) => {
        Arc::new( RefCell::new(
                Segment::new($id, $slot, $blocks)
                ))
    }
}

/// Instantiate new Segment with zero blocks
#[macro_export]
macro_rules! seg_ref_empty {
    ( $id:expr ) => {
        Arc::new( RefCell::new(
                Segment::empty($id)
                ))
    }
}

/// Make a new segment manager and package into a shareable reference
#[macro_export]
macro_rules! segmgr_ref {
    ( $id:expr, $segsz:expr, $bytes:expr ) => {
        Arc::new( Mutex::new( RefCell::new(
                SegmentManager::new( $id, $segsz, $bytes)
                )))
    }
}

//==----------------------------------------------------==//
//      Index macros
//==----------------------------------------------------==//

/// Make a new index and package into a shareable reference
#[macro_export]
macro_rules! index_ref {
    ( ) => {
        Arc::new( Mutex::new( RefCell::new(
                Index::new()
                )))
    }
}

