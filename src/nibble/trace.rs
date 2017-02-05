use std::fs::File;
use std::io::{self, BufReader};
use std::io::prelude::*;
use std::collections::VecDeque;
use std::intrinsics;

#[derive(Debug)]
pub enum Op {
    Get, Set, Del
}

#[derive(Debug)]
pub struct Entry {
    key: u32,
    size: u32,
    op: Op,
}

impl Entry {

    pub fn new(key: u32, op: Op, size: u32) -> Self {
        Entry { key: key, op: op, size: size }
    }
}

/// Object used to read in a PUT/GET/DEL trace from a file.
pub struct Trace {
    /// number of entries stored in the trace
    n: u32,
    /// if operation is Get or Set, value size is irrelevant
    rec: VecDeque<Entry>
}

impl Trace {

    pub fn new(path: &str) -> Self {
        let mut t = Trace { n: 0u32, rec: VecDeque::with_capacity(1usize<<20) };
        if t.read_trace(path).is_err() {
            println!("Error reading or parsing trace file {}", path);
            unsafe { intrinsics::abort(); }
        }
        t.n = t.rec.len() as u32;
        t
    }

    pub fn print(&self) {
        for e in &self.rec {
            println!("{:?}", e);
        }
    }

    /// Read trace from given path.
    /// We assume trace has three space-separated columns:
    ///         key op value_size
    /// where
    ///      key:    [0-9]+          unsigned long
    ///       op:    (get|del|set)   string
    /// val size:    (na|[0-9]+)     'na' or unsigned long
    ///
    /// Lines starting exactly with '#' will be skipped.
    fn read_trace(&mut self, path: &str) -> Result<(),()> {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) => return Err( () ),
        };
        let file = BufReader::new(file);
        for line in file.lines() {
            if line.is_err() { break; }
            let line = line.unwrap();
            if line.starts_with("#") { continue; }
            let mut iter = line.split_whitespace();
            let key = match iter.next() {
                None => return Err( () ),
                Some(k) => match u32::from_str_radix(k, 10) {
                    Ok(v) => v,
                    Err(e) => return Err( () ),
                },
            };
            let op = match iter.next() {
                None => return Err( () ),
                Some(o) => match o {
                    "get" => Op::Get,
                    "set" => Op::Set,
                    "del" => Op::Del,
                    _ => return Err( () ),
                },
            };
            let size = match iter.next() {
                None => return Err( () ),
                Some(s) => match s {
                    "na" => 0u32,
                    _ => match u32::from_str_radix(s, 10) {
                        Ok(v) => v,
                        Err(e) => return Err( () ),
                    },
                },
            };
            self.rec.push_back( Entry::new(key,op,size) );
        }
        Ok( () )
    }
}
