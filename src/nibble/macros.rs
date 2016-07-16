use parking_lot as pl;

// Macros go in separate module to satisfy circular dependencies

//==----------------------------------------------------==//
//      Common macros
//==----------------------------------------------------==//

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
//      Segment macros
//==----------------------------------------------------==//

/// Instantiate new Segment as a SegmentRef
#[macro_export]
macro_rules! seg_ref {
    ( $id:expr, $sock:expr, $slot:expr, $blocks:expr ) => {
        Arc::new( pl::RwLock::new(
                Segment::new($id, $sock, $slot, $blocks)
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
    ( $segsz:expr, $bytes:expr ) => {
        Arc::new( Mutex::new(
                SegmentManager::new( $segsz, $bytes)
                ))
    }
}

//==----------------------------------------------------==//
//      Index macros
//==----------------------------------------------------==//

/// Make a new index and package into a shareable reference
#[macro_export]
macro_rules! index_ref {
    ( ) => {
        Arc::new( Index::new() )
    }
}

//==----------------------------------------------------==//
//      Compactor macros
//==----------------------------------------------------==//

/// Make a new compactor instance.
#[macro_export]
macro_rules! comp_ref {
    ( $manager:expr, $index:expr ) => {
        Arc::new( Mutex::new(
                Compactor::new( $manager, $index )
                ))
    }
}

