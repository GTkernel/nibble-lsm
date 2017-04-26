use std::fs::File;
use std::io::{self, BufReader};
use std::io::prelude::*;
use std::collections::VecDeque;
use std::intrinsics;
use std::str::FromStr;
use kvs::memory;

#[derive(Debug)]
pub enum Op {
    Get, Set, Del
}

#[derive(Debug)]
pub struct Entry {
    pub key: u64,
    pub size: u32,
    pub op: Op,
}

impl Entry {

    pub fn new(key: u64, op: Op, size: u32) -> Self {
        Entry { key: key, op: op, size: size }
    }
}

/// Used for reading in the actual trace file, in binary format.
/// This must represent the layout from
/// https://github.gatech.edu/kernel/kvs-mutilate.git
/// TestGenerator.cc struct Entry
#[repr(packed)]
struct TraceFileEntry {
    key: u64,
    op: u8, // 0 GET 1 SET 2 DEL
    size: u32,
}

/// Object used to read in a PUT/GET/DEL trace from a file.
pub struct Trace {
    /// number of entries stored in the trace
    pub n: u32,
    /// if operation is Get or Set, value size is irrelevant
    pub rec: Vec<Entry>
}

impl Trace {

    pub fn new(path: &str) -> Self {
        let mut t = Trace { n: 0_u32, rec: Vec::with_capacity(1usize<<36) };
        if let Err(msg) = t.read_trace(path) {
            panic!("Error reading or parsing trace file {}: {}",
                path, msg);
        }
        t.n = t.rec.len() as u32;
        t
    }

    pub fn new2(path: &String) -> Self {
        Trace::new(path.as_str())
    }

    pub fn print(&self) {
        for e in &self.rec {
            println!("{:?}", e);
        }
    }

    /// Trace shold be { u64, u8, u32 } for { key, op, len }
    /// according to struct TraceFileEntry
    fn read_trace(&mut self, path: &str) -> Result<(),&str> {
        info!("Loading trace file...");

        let mut mapped = memory::MemMapFile::new(path);
        let entries = unsafe { mapped.as_slice::<TraceFileEntry>() };
        for entry in entries {
            // LSM does not allow key 0
            if entry.key == 0 { continue; }
            // FIXME LSM still cannot handle zero-sized objects
            let size = if entry.size == 0 { 1 } else { entry.size };
            // convert TraceFileEntry to Entry
            let op = match entry.op {
                0 => Op::Get,
                1 => Op::Set,
                2 => Op::Del,
                e @ _ => panic!("Unexpected op code: {}", e),
            };
            self.rec.push( Entry::new(entry.key,op,size) );
            if 0 == (self.rec.len() % 500_000_000) {
                info!("Loaded {} mil. entries",
                      self.rec.len() / 1_000_000_usize);
            }
            // XXX remove me
            if self.rec.len() > 10_000_000_000_usize {
                println!("LIMITING TRACE TO {}", self.rec.len());
                break;
            }
        }

        Ok( () )
    }
}
