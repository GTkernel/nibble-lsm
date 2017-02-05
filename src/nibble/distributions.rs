/// This code ported from the YCSB tool in the file
///    core/src/main/java/com/yahoo/ycsb/generator/ZipfianGenerator.java
/// found at https://github.com/brianfrankcooper/YCSB

use common::{rdrand, rdrandq};

use std::cmp;

pub trait DistGenerator {
    fn next(&mut self) -> u32;
    fn reset(&mut self);
}

pub struct Zipfian {
    items: i64,
    base: i64,
    countforzeta: i64,
    constant: f64,
    theta: f64,
    zeta2theta: f64,
    alpha: f64,
    zetan: f64,
    eta: f64,
}

impl Zipfian {

    pub fn new(n: u32, s: f64) -> Self {
        let mut z = Zipfian {
            items: n as i64,
            base: 0,
            constant: s,
            theta: s,
            zeta2theta: 0f64,
            alpha: 1f64/(1f64-s),
            zetan: 0f64,
            countforzeta: n as i64,
            eta: 0f64
        };
        let t = z.theta; // FIXME Rust needs non-lexical lifetimes
        z.zeta2theta = z.zeta1(2, t);
        z.zetan = Zipfian::zetastatic1(n as i64, s);
        z.eta = (1f64-(1f64/n as f64).powf(1f64-z.theta)) /
                    (1f64-z.zeta2theta/z.zetan);
        z.next();
        z
    }

    fn zeta1(&mut self, n: i64, theta: f64) -> f64 {
        self.countforzeta = n;
        Zipfian::zetastatic1(n, theta)
    }

    fn zeta2(&mut self, st: i64, n: i64, theta: f64, initial_sum: f64) -> f64 {
        self.countforzeta = n;
        Zipfian::zetastatic2(st, n, theta, initial_sum)
    }

    fn zetastatic1(n: i64, theta: f64) -> f64 {
        Zipfian::zetastatic2(0i64,n,theta,0f64)
    }

    fn zetastatic2(st: i64, n: i64, theta: f64, initial_sum: f64) -> f64 {
        let mut sum: f64 = initial_sum;
        for i in st..n {
            sum += 1f64 / (i as f64 + 1f64).powf(theta);
        }
        sum
    }

    // this code can be cleaned up once we have non-lexical lifetimes
    fn nextLong(&mut self, n: i64) -> i64 {
        if n != self.countforzeta {
            if n > self.countforzeta {
                let c = self.countforzeta;
                let t = self.theta;
                let z = self.zetan;
                self.zetan = self.zeta2(c, n, t, z);
                let t = self.theta;
                let z2 = self.zeta2theta;
                let z = self.zetan;
                self.eta = (1f64 - (2f64/n as f64).powf(1f64-t)) /
                            (1f64 - z2 / z);
            }
            else { error!("problem with zipfian"); }
        }
        let u: f64 = unsafe { rdrandq() as f64 / u64::max_value() as f64 };
        let uz: f64 = u * self.zetan;
        if uz < 1f64 {
            self.base
        } else if uz < (1f64 + 0.5f64.powf(self.theta)) {
            self.base+1
        } else {
            (n as f64 *(self.eta *u -self.eta +1f64).powf(self.alpha)
                + self.base as f64) as i64
        }
    }
}

impl DistGenerator for Zipfian {

    fn next(&mut self) -> u32 {
        let i = self.items; // need NLL
        self.nextLong(i) as u32
    }

    fn reset(&mut self) { }
}

pub struct ZipfianArray {
    n: u32,
    /// Given we execute for short periods in our experiments, we
    /// won't need to generate all data points. 'n' is the total
    /// quantity of items we would access given infinite time. 'upto'
    /// is how many operations we'll realistically perform given the
    /// duration of the experiment.
    upto: Option<u32>,
    arr: Vec<u32>,
    next: u32
}

impl ZipfianArray {

    pub fn new(n: u32, s: f64) -> Self {
        //let many = (n*4) as usize;
        // limit how many we use
        let many = cmp::min( (n*4) as usize, 1usize << 29 ) as u32;
        let mut v: Vec<u32> = Vec::with_capacity(many as usize);
        let mut zip = Zipfian::new(n, s);
        for _ in 0..many {
            v.push(zip.next());
        }
        // 1-pass fisher yates shuffle
        for i in 0..many {
            let r = unsafe { rdrand() };
            let o = (r % (many-i)) + i;
            v.swap(i as usize, o as usize);
        }
        ZipfianArray { n: many, upto: None, arr: v, next: 0 }
    }
}

impl DistGenerator for ZipfianArray {

    #[inline(always)]
    fn next(&mut self) -> u32 {
        self.next = (self.next + 1) % self.n;
        if self.upto.is_some() {
            assert!(self.next < self.upto.unwrap(),
                    "upto exceeded. increase, or shorten expmt duration");
        }
        self.arr[self.next as usize] as u32
    }

    fn reset(&mut self) {
        self.next = 0;
    }
}

pub struct Uniform {
    n: u32,
    arr: Vec<u32>,
    next: u32,
}

impl Uniform {

    /// don't pre-compute values; use rdrand
    pub fn new(n: u32) -> Self {
        let mut v = vec![];
        Uniform { n: n, arr: v, next: 0 }
    }
}

impl DistGenerator for Uniform {

    /// don't pre-compute values; use rdrand
    #[inline(always)]
    fn next(&mut self) -> u32 {
        (unsafe { rdrand() } % (self.n+1)) as u32
    }

    fn reset(&mut self) {
        self.next = 0;
    }
}