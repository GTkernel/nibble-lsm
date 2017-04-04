// Macros go in separate module to satisfy circular dependencies

//==----------------------------------------------------==//
//      Compiler hints
//==----------------------------------------------------==//

/// Wrapper for intrinsics::likely
#[macro_export]
macro_rules! likely {
    ( $b:expr ) => { unsafe { intrinsics::likely($b) } }
}

/// Wrapper for intrinsics::unlikely
#[macro_export]
macro_rules! unlikely {
    ( $b:expr ) => { unsafe { intrinsics::unlikely($b) } }
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
        Arc::new( pl::Mutex::new(
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
        Arc::new( pl::Mutex::new(
                Compactor::new( $manager, $index )
                ))
    }
}

