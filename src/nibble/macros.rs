// Macros go in separate module to satisfy circular dependencies

//==----------------------------------------------------==//
//      Common macros
//==----------------------------------------------------==//

#[allow(unused_mut)]
pub unsafe fn rdtsc() -> u64 {
    let mut low: u32;
    let mut high: u32;
    asm!("rdtsc" : "={eax}" (low), "={edx}" (high));
    ((high as u64) << 32) | (low as u64)
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
//      Segment macros
//==----------------------------------------------------==//

/// Instantiate new Segment as a SegmentRef
#[macro_export]
macro_rules! seg_ref {
    ( $id:expr, $slot:expr, $blocks:expr ) => {
        Arc::new( RwLock::new(
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
        Arc::new( Mutex::new(
                SegmentManager::new( $id, $segsz, $bytes)
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
        Arc::new( Mutex::new(
                Index::new()
                ))
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

