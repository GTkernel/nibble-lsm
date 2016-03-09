use block;

/// Identifying information for a segment
pub struct Header {
    id: usize,
}

/// Logical representation of a segment, composed of multiple smaller
/// blocks.
pub struct Segment {
    // Note: headers are written to the buffer, not necessarily kept in
    // the metadata structure here
    header: Header,
}

// TODO Segment state as enum

/// Segment manager for a set of physical memory
pub struct SegmentManager<'a> {
    manager_id: usize,
    segment_size: usize,
    allocator: block::BlockAllocator,
    segments: Vec<&'a mut Segment>,
}

impl Segment {
    pub fn new() -> Self {
        Segment { header: Header { id: 0 }, }
    }
}

// TODO use some config structure
impl<'a> SegmentManager<'a> {
    // TODO socket to initialize on
    pub fn new(id: usize, segsz: usize, len: usize) -> Self {
        let b = block::BlockAllocator::new(1<<16, len);
        SegmentManager {
            manager_id: id,
            segment_size: segsz,
            allocator: b,
            segments: Vec::new(),
        }
    }
    pub fn alloc(&self) -> Option<Segment> {
        unimplemented!();
    }
    pub fn free(&self) {
        unimplemented!();
    }
}
