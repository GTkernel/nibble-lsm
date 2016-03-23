#![feature(test)]

extern crate test;
extern crate nibble;

use test::Bencher;

use nibble::store::Nibble;
use nibble::segment::ObjDesc;

// TODO test objects larger than block, and segment
// TODO put_object which must traverse chunks
// TODO a get_object which must traverse chunks
// TODO test we can determine live vs dead entries in segment
// TODO test specific cases where header cross block boundaries

// -------------------------------------------------------------------
// Raw insertion benchmarks (one object)
// -------------------------------------------------------------------

#[bench]
fn insert_64(b: &mut Bencher) {
    let mut nib = Nibble::new( 1<<26 );
    let key: &'static str = "abcdefghij123456";
    const LEN: u32 = 64;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(key, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_128(b: &mut Bencher) {
    let mut nib = Nibble::new( 1<<26 );
    let key: &'static str = "abcdefghij123456";
    const LEN: u32 = 128;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(key, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_256(b: &mut Bencher) {
    let mut nib = Nibble::new( 1<<26 );
    let key: &'static str = "abcdefghij123456";
    const LEN: u32 = 256;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(key, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_384(b: &mut Bencher) {
    let mut nib = Nibble::new( 1<<26 );
    let key: &'static str = "abcdefghij123456";
    const LEN: u32 = 384;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(key, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_512(b: &mut Bencher) {
    let mut nib = Nibble::new( 1<<26 );
    let key: &'static str = "abcdefghij123456";
    const LEN: u32 = 512;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(key, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_768(b: &mut Bencher) {
    let mut nib = Nibble::new( 1<<26 );
    let key: &'static str = "abcdefghij123456";
    const LEN: u32 = 768;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(key, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_1024(b: &mut Bencher) {
    let mut nib = Nibble::new( 1<<26 );
    let key: &'static str = "abcdefghij123456";
    const LEN: u32 = 1024;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(key, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_1536(b: &mut Bencher) {
    let mut nib = Nibble::new( 1<<26 );
    let key: &'static str = "abcdefghij123456";
    const LEN: u32 = 1536;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(key, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_2048(b: &mut Bencher) {
    let mut nib = Nibble::new( 1<<26 );
    let key: &'static str = "abcdefghij123456";
    const LEN: u32 = 1024;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(key, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_2560(b: &mut Bencher) {
    let mut nib = Nibble::new( 1<<26 );
    let key: &'static str = "abcdefghij123456";
    const LEN: u32 = 2560;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(key, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_3072(b: &mut Bencher) {
    let mut nib = Nibble::new( 1<<26 );
    let key: &'static str = "abcdefghij123456";
    const LEN: u32 = 3072;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(key, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}
#[bench]
fn insert_4096(b: &mut Bencher) {
    let mut nib = Nibble::new( 1<<26 );
    let key: &'static str = "abcdefghij123456";
    const LEN: u32 = 1024;
    let val = [42 as u8; LEN as usize];
    let obj = ObjDesc::new(key, Some(val.as_ptr() as *const u8), LEN);
    b.iter( || { nib.put_object(&obj) });
}

// #[bench]
// fn insert_50(b: &mut Bencher) {
//     let mut nib = Nibble::new( 1<<26 );
//     let key: &'static str = "abcdefghij123456";
//     const LEN: u32 = 50;
//     let val = [42 as u8; LEN as usize];
//     let obj = ObjDesc::new(key, Some(val.as_ptr() as *const u8), LEN);
//     b.iter( || { nib.put_object(&obj) });
// }
// #[bench]
// fn insert_100(b: &mut Bencher) {
//     let mut nib = Nibble::new( 1<<26 );
//     let key: &'static str = "abcdefghij123456";
//     const LEN: u32 = 100;
//     let val = [42 as u8; LEN as usize];
//     let obj = ObjDesc::new(key, Some(val.as_ptr() as *const u8), LEN);
//     b.iter( || { nib.put_object(&obj) });
// }
// #[bench]
// fn insert_200(b: &mut Bencher) {
//     let mut nib = Nibble::new( 1<<26 );
//     let key: &'static str = "abcdefghij123456";
//     const LEN: u32 = 200;
//     let val = [42 as u8; LEN as usize];
//     let obj = ObjDesc::new(key, Some(val.as_ptr() as *const u8), LEN);
//     b.iter( || { nib.put_object(&obj) });
// }
// #[bench]
// fn insert_400(b: &mut Bencher) {
//     let mut nib = Nibble::new( 1<<26 );
//     let key: &'static str = "abcdefghij123456";
//     const LEN: u32 = 400;
//     let val = [42 as u8; LEN as usize];
//     let obj = ObjDesc::new(key, Some(val.as_ptr() as *const u8), LEN);
//     b.iter( || { nib.put_object(&obj) });
// }
// #[bench]
// fn insert_800(b: &mut Bencher) {
//     let mut nib = Nibble::new( 1<<26 );
//     let key: &'static str = "abcdefghij123456";
//     const LEN: u32 = 800;
//     let val = [42 as u8; LEN as usize];
//     let obj = ObjDesc::new(key, Some(val.as_ptr() as *const u8), LEN);
//     b.iter( || { nib.put_object(&obj) });
// }
// #[bench]
// fn insert_1000(b: &mut Bencher) {
//     let mut nib = Nibble::new( 1<<26 );
//     let key: &'static str = "abcdefghij123456";
//     const LEN: u32 = 1000;
//     let val = [42 as u8; LEN as usize];
//     let obj = ObjDesc::new(key, Some(val.as_ptr() as *const u8), LEN);
//     b.iter( || { nib.put_object(&obj) });
// }
// #[bench]
// fn insert_2000(b: &mut Bencher) {
//     let mut nib = Nibble::new( 1<<26 );
//     let key: &'static str = "abcdefghij123456";
//     const LEN: u32 = 1000;
//     let val = [42 as u8; LEN as usize];
//     let obj = ObjDesc::new(key, Some(val.as_ptr() as *const u8), LEN);
//     b.iter( || { nib.put_object(&obj) });
// }
// #[bench]
// fn insert_4000(b: &mut Bencher) {
//     let mut nib = Nibble::new( 1<<26 );
//     let key: &'static str = "abcdefghij123456";
//     const LEN: u32 = 1000;
//     let val = [42 as u8; LEN as usize];
//     let obj = ObjDesc::new(key, Some(val.as_ptr() as *const u8), LEN);
//     b.iter( || { nib.put_object(&obj) });
// }
