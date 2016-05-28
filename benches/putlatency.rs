#![feature(test)]

extern crate test;
extern crate nibble;

use test::Bencher;

use nibble::nib::Nibble;
use nibble::segment::ObjDesc;
use nibble::sched::pin_cpu;

// TODO test objects larger than block, and segment
// TODO put_object which must traverse chunks
// TODO a get_object which must traverse chunks
// TODO test we can determine live vs dead entries in segment
// TODO test specific cases where header cross block boundaries

// -------------------------------------------------------------------
// Raw insertion benchmarks (one object)
// -------------------------------------------------------------------

const KEY: &'static str = "1234";
const KEY_LEN: u32 = 4; // KEY.len() as u32; doesn't work yet

#[bench]
fn insert_8(b: &mut Bencher) {
    unsafe { pin_cpu(0); }
    let mut nib = Nibble::new( 1<<28 );
    const LEN: u32 = 8-KEY_LEN;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(KEY, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_16(b: &mut Bencher) {
    unsafe { pin_cpu(0); }
    let mut nib = Nibble::new( 1<<28 );
    const LEN: u32 = 16-KEY_LEN;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(KEY, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_32(b: &mut Bencher) {
    unsafe { pin_cpu(0); }
    let mut nib = Nibble::new( 1<<28 );
    const LEN: u32 = 32-KEY_LEN;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(KEY, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_64(b: &mut Bencher) {
    unsafe { pin_cpu(0); }
    let mut nib = Nibble::new( 1<<28 );
    const LEN: u32 = 64-KEY_LEN;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(KEY, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_128(b: &mut Bencher) {
    unsafe { pin_cpu(0); }
    let mut nib = Nibble::new( 1<<28 );
    const LEN: u32 = 128-KEY_LEN;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(KEY, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_256(b: &mut Bencher) {
    unsafe { pin_cpu(0); }
    let mut nib = Nibble::new( 1<<28 );
    const LEN: u32 = 256-KEY_LEN;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(KEY, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_512(b: &mut Bencher) {
    unsafe { pin_cpu(0); }
    let mut nib = Nibble::new( 1<<28 );
    const LEN: u32 = 512-KEY_LEN;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(KEY, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_1024(b: &mut Bencher) {
    unsafe { pin_cpu(0); }
    let mut nib = Nibble::new( 1<<28 );
    const LEN: u32 = 1024-KEY_LEN;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(KEY, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_2048(b: &mut Bencher) {
    unsafe { pin_cpu(0); }
    let mut nib = Nibble::new( 1<<28 );
    const LEN: u32 = 1024-KEY_LEN;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(KEY, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_4096(b: &mut Bencher) {
    unsafe { pin_cpu(0); }
    let mut nib = Nibble::new( 1<<28 );
    const LEN: u32 = 1024-KEY_LEN;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(KEY, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_8192(b: &mut Bencher) {
    unsafe { pin_cpu(0); }
    let mut nib = Nibble::new( 1<<28 );
    const LEN: u32 = 1024-KEY_LEN;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(KEY, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_16384(b: &mut Bencher) {
    unsafe { pin_cpu(0); }
    let mut nib = Nibble::new( 1<<28 );
    const LEN: u32 = 1024-KEY_LEN;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(KEY, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_32768(b: &mut Bencher) {
    unsafe { pin_cpu(0); }
    let mut nib = Nibble::new( 1<<28 );
    const LEN: u32 = 1024-KEY_LEN;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(KEY, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_65536(b: &mut Bencher) {
    unsafe { pin_cpu(0); }
    let mut nib = Nibble::new( 1<<28 );
    const LEN: u32 = 65536-KEY_LEN;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(KEY, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_131072(b: &mut Bencher) {
    unsafe { pin_cpu(0); }
    let mut nib = Nibble::new( 1<<28 );
    const LEN: u32 = 131072-KEY_LEN;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(KEY, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_262144(b: &mut Bencher) {
    unsafe { pin_cpu(0); }
    let mut nib = Nibble::new( 1<<28 );
    const LEN: u32 = 262144-KEY_LEN;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(KEY, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_524288(b: &mut Bencher) {
    unsafe { pin_cpu(0); }
    let mut nib = Nibble::new( 1<<28 );
    const LEN: u32 = 524288-KEY_LEN;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(KEY, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_1048576(b: &mut Bencher) {
    unsafe { pin_cpu(0); }
    let mut nib = Nibble::new( 1<<28 );
    const LEN: u32 = 1048576-KEY_LEN;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(KEY, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_2097152(b: &mut Bencher) {
    unsafe { pin_cpu(0); }
    let mut nib = Nibble::new( 1<<28 );
    const LEN: u32 = 2097152-KEY_LEN;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(KEY, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_4194304(b: &mut Bencher) {
    unsafe { pin_cpu(0); }
    let mut nib = Nibble::new( 1<<28 );
    const LEN: u32 = 4194304-KEY_LEN;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(KEY, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_8388608(b: &mut Bencher) {
    unsafe { pin_cpu(0); }
    let mut nib = Nibble::new( 1<<28 );
    const LEN: u32 = 8388608-KEY_LEN;
    let val: Box<[u8;LEN as usize]> = Box::new([42 as u8; LEN as usize]);
    let obj = ObjDesc::new(KEY, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
