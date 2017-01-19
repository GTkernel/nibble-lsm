use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::fmt;

// Linux definitions for mbind system call.
// FIXME put somewhere portable
pub const MPOL_BIND: usize       = 2;
pub const MPOL_INTERLEAVE: usize = 3;
pub const MPOL_MF_STRICT: usize  = (1usize<<0);

/// Special type to represent a NUMA socket.
#[derive(Copy,Clone)]
pub struct NodeId(pub usize);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

lazy_static! {
    pub static ref NODE_MAP: NodeMap = { NodeMap::new() };
}

/// Used to parse the CPU ID files. Some file formats
/// have varied syntax: X-Y or X,Y,Z or X-Y,Z
#[derive(Clone)]
pub struct CpuSet {
    /// A sorted set of unique core IDs.
    set: Vec<usize>,
}

impl CpuSet {

    /// Create a new CpuSet from a vector of values.
    /// We sort then remove duplicates.
    pub fn new(ids: &Vec<usize>) -> CpuSet {
        let mut v = ids.clone();
        v.sort();
        v.dedup();
        CpuSet { set: v }
    }

    pub fn len(&self) -> usize {
        self.set.len()
    }

    pub fn is_member(&self, id: usize) -> bool {
        self.set.binary_search(&id).is_ok()
    }

    pub fn lowest(&self) -> usize {
        self.set[0]
    }

    pub fn get(&self) -> Vec<usize> {
        self.set.clone()
    }
}

/// Info about the socket
#[allow(dead_code)]
pub struct SocketInfo {
    ncpus: usize,
    cpus: CpuSet,
    //max_mem: usize,
}

impl SocketInfo {

    pub fn new(node: usize) -> Self {
        let set = read_node_cpus(node);
        SocketInfo {
            ncpus: set.len(),
            cpus: set,
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
            for cpu in sock.cpus.get() {
                map.insert(cpu, node);
            }
            sockets.push(sock);
        }
        NodeMap {
            sockets: sockets,
            cpu2sock: map,
        }
    }

    pub fn cpus_of(&self, sock: NodeId) -> CpuSet {
        assert!(sock.0<self.sockets.len(),"sock is too big: {}",sock);
        self.sockets[sock.0].cpus.clone()
    }

    pub fn cpus_in(&self, sock: NodeId) -> usize {
        assert!(sock.0<self.sockets.len(),"sock is too big: {}",sock);
        self.sockets[sock.0].ncpus
    }

    pub fn sock_of(&self, cpu: usize) -> NodeId {
        match self.cpu2sock.get(&cpu) {
            None => panic!("bad key"),
            Some(id) => NodeId(*id),
        }
    }

    pub fn sockets(&self) -> usize {
        self.sockets.len()
    }

    pub fn ncpus(&self) -> usize {
        self.sockets.len() * self.sockets[0].ncpus
    }
}

/// Read first line of given file and interpret as CPU IDs.
fn read_cpu_ids(fname: &str) -> CpuSet {
    debug!("reading {}", fname);
    let mut file = match File::open(fname) {
        Err(e) => panic!("{}: {}", fname, e),
        Ok(f) => f,
    };
    let mut line = String::new();
    if let Err(e) = file.read_to_string(&mut line) {
        panic!("file {}: {}", fname, e);
    }
    // take X-Y,A-B,C,D and split on comma,
    // then process each as either a range or singular value
    let mut ids: Vec<usize> = Vec::with_capacity(32);
    for s in line.split(',') {
        let s = s.trim();
        debug!("parsing '{}'", s);
        if s.contains('-') {
            let (l,h) = s.split_at(s.find('-').unwrap());
            let h = h.trim_matches('-');
            debug!("l {} h {}", l, h);
            let low = usize::from_str_radix(l,10).unwrap();
            let high = usize::from_str_radix(h,10).unwrap();
            debug!("low {} high {}", low, high);
            for i in low..(high+1) {
                ids.push(i);
            }
        } else {
            ids.push(usize::from_str_radix(s,10).unwrap());
        }
    };
    assert!(!ids.is_empty());
    CpuSet::new(&ids)
}

fn read_node_cpus(node: usize) -> CpuSet {
    let fname = format!(
        "/sys/devices/system/node/node{}/cpulist",
        node);
    read_cpu_ids(fname.as_str())
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
    read_cpu_ids(fname).len()
}

/// Number of cpus in the system.
/// FIXME we assume all cpus are online
pub fn ncpus() -> usize {
    let fname = "/sys/devices/system/cpu/online";
    read_cpu_ids(fname).len()
}

#[cfg(IGNORE)]
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
        super::read_cpu_ids("/dev/null");
    }

    #[test]
    #[should_panic(expected = "No such file")]
    fn read_range_notexist() {
        logger::enable();
        super::read_cpu_ids("/not/exist/anywhere");
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
