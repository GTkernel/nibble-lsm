use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::fmt;

/// Special type to represent a NUMA socket.
#[derive(Copy,Clone)]
pub struct NodeId(pub usize);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

lazy_static! {
    pub static ref NODE_MAP: NodeMap = { NodeMap::new() };
}

#[derive(Copy,Clone)]
pub struct CpuRange {
    pub start: usize,
    pub end: usize,
}

/// Info about the socket
#[allow(dead_code)]
pub struct SocketInfo {
    ncpus: usize,
    cpus: CpuRange,
    //max_mem: usize,
}

impl SocketInfo {

    pub fn new(node: usize) -> Self {
        let range = read_node_cpus(node);
        SocketInfo {
            ncpus: range.end - range.start + 1,
            cpus: range,
        }
    }
}

/// Info about the system
#[allow(dead_code)]
pub struct NodeMap {
    sockets: Vec<SocketInfo>,
    cpu2sock: HashMap<usize, usize>,
}

impl NodeMap {

    pub fn new() -> Self {
        let nnodes = nodes();
        let mut sockets: Vec<SocketInfo> = Vec::with_capacity(nnodes);
        let mut map: HashMap<usize,usize>;
        map = HashMap::new();
        for node in 0..nnodes {
            let sock = SocketInfo::new(node);
            for cpu in sock.cpus.start..(sock.cpus.end+1) {
                map.insert(cpu, node);
            }
            sockets.push(sock);
        }
        NodeMap {
            sockets: sockets,
            cpu2sock: map,
        }
    }

    pub fn cpus_of(&self, sock: NodeId) -> CpuRange {
        assert!(sock.0<self.sockets.len(),"sock is too big: {}",sock);
        self.sockets[sock.0].cpus
    }

    pub fn sockets(&self) -> usize {
        self.sockets.len()
    }
}

/// Read file and interpret as an integer range, e.g.
/// X-Y returns CpuRange { start: X, end: Y }
/// FIXME we don't handle all corner cases
fn file_as_range(fname: &str) -> CpuRange {
    debug!("reading {}", fname);
    let mut file = match File::open(fname) {
        Err(e) => panic!("{}: {}", fname, e),
        Ok(f) => f,
    };
    let mut line = String::new();
    if let Err(e) = file.read_to_string(&mut line) {
        panic!("file {}: {}", fname, e);
    }
    // a-b,x-y  hyperthreading? take first range
    if let Some(idx) = line.find(',') {
        line = {
            let (ab,_) = line.split_at(idx);
            ab.to_owned()
        };
    }
    if let None = line.find('-') {
        panic!("no range in file");
    }
    let values: Vec<&str>;
    values = line.trim_right().split('-').collect();
    assert_eq!(values.len(), 2);
    CpuRange {
        start: usize::from_str_radix(values[0], 10).unwrap(),
        end:   usize::from_str_radix(values[1], 10).unwrap(),
    }
}

fn read_node_cpus(node: usize) -> CpuRange {
    let fname = format!(
        "/sys/devices/system/node/node{}/cpulist",
        node);
    file_as_range(fname.as_str())
}

/// Read from /proc/self/numa_maps and report how many pages are
/// allocated to each socket (index is socket ID)
///
/// FIXME differentiate page sizes, e.g.:
/// anon=2048 dirty=2048 N1=2048 kernelpagesize_kB=2048
///                              ^^^^^^^^^^^^^^^^^^^^^^
/// mapped=329 mapmax=39 N0=6 N1=323 kernelpagesize_kB=4
///                                  ^^^^^^^^^^^^^^^^^^^
pub fn numa_allocated() -> Vec<usize> {
    let nnodes = nodes();
    let mut counts: Vec<usize> = Vec::with_capacity(nnodes);
    for _ in 0..nnodes {
        counts.push(0);
    }
    let fname = "/proc/self/numa_maps";
    let mut file = match File::open(fname) {
        Err(e) => panic!("{}: {}", fname, e),
        Ok(f) => f,
    };
    let mut content = String::new();
    if let Err(e) = file.read_to_string(&mut content) {
        panic!("file {}: {}", fname, e);
    };
    for line in content.lines() {
        for field in line.split(' ') {
            if field.starts_with("N") {
                let (_,f) = field.split_at(1);
                // n is 'X' where X is [0-9]+
                // p is '=X'
                let (n,p) = f.split_at(field.find('=').unwrap()-1);
                let pp = &p[1..]; // remove =
                let node: usize = n.parse::<usize>().unwrap();
                let pages: usize = pp.parse::<usize>().unwrap();
                counts[node] += pages;
            }
        }
    }
    counts
}

/// Number of nodes in the system.
/// FIXME we assume all nodes are online
pub fn nodes() -> usize {
    let fname = "/sys/devices/system/node/online";
    file_as_range(fname).end + 1
}

/// Number of cpus in the system.
/// FIXME we assume all cpus are online
pub fn ncpus() -> usize {
    let fname = "/sys/devices/system/cpu/online";
    file_as_range(fname).end + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::logger;

    #[test]
    fn scan() {
        logger::enable();
        let map = NodeMap::new();
        let mut cpus = 0;
        for socket in map.sockets {
            cpus += socket.ncpus;
        }
        let online = ncpus();
        // FIXME account for hyperthreading... either equal or 2x
        if cpus != online {
            assert!(2*cpus == online,"cpus {} ncpus {}",cpus,online);
        }
        assert_eq!(map.cpu2sock.get(&2), Some(&0));
        assert_eq!(map.cpu2sock.get(&2048), None);
    }

    #[test]
    #[should_panic(expected = "no range in file")]
    fn read_range_empty() {
        logger::enable();
        super::file_as_range("/dev/null");
    }

    #[test]
    #[should_panic(expected = "No such file")]
    fn read_range_notexist() {
        logger::enable();
        super::file_as_range("/not/exist/anywhere");
    }

    #[test]
    fn cpulist() {
        logger::enable();
        let range = super::read_node_cpus(0);
        // can only do simple checking,
        // since this varies with each machine
        assert!(range.start < range.end);
        // TODO update if there are larger machines
        assert!(range.start < 512);
        assert!(range.end < 512);
    }

    #[test]
    fn nnodes() {
        logger::enable();
        let n = super::nodes();
        assert!(n > 0);
        // TODO update if there are larger machines
        assert!(n < 32);
    }

    #[test]
    fn check_numa_pages() {
        let v: Vec<usize> = super::numa_allocated();
        assert_eq!(v.len(), nodes());

        let mut total: usize = 0;
        for x in v {
            total += x;
        }
        assert!(total > 0);
    }
}
