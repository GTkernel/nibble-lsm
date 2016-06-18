// http://doc.crates.io/build-script.html

// We assume you install cityhash and cuckoohash_map.hh
// to $HOME/local/ provided in the thirdparty/ directory

use std::env;
use std::path::Path;
use std::process::Command;

// Works, but src/bin/ executables cannot link to nibble if nibble has
// non-fPIC code in it.
#[allow(dead_code)]
fn make_static() {
    let outdir = env::var("OUT_DIR").unwrap();
    let mut libdir = String::from(env::var("HOME").unwrap());
    libdir.push_str("/local/lib");

    Command::new("g++")
        .args(&["./src/cuckoo.cc", "-c", "-o"])
        .arg(&format!("{}/cuckoo.o", outdir))
        .args(&["-std=c++11", "-O0", "-Ithirdparty/libcuckoo/src"])
        .status().unwrap();

    Command::new("ar").args(&["crUs", "libcuckoo.a", "cuckoo.o"])
        .current_dir(&Path::new(&outdir))
        .status().unwrap();

    println!("cargo:rustc-link-search=native={}", outdir);
    println!("cargo:rustc-link-search=native={}", libdir);
    println!("cargo:rustc-link-lib=static=cuckoo");
    println!("cargo:rustc-link-lib=static=cityhash");
}

#[allow(dead_code)]
fn make_dynamic() {
    let outdir = env::var("OUT_DIR").unwrap();
    let mut libdir = String::from(env::var("HOME").unwrap());
    libdir.push_str("/local/lib");

    Command::new("g++")
        .args(&["./src/cuckoo.cc", "-c", "-fPIC", "-o"])
        .arg(&format!("{}/cuckoo.o", outdir))
        .args(&["-std=c++11", "-O3", "-Ithirdparty/libcuckoo/src"])
        .args(&["-msse4.2", "-mtune=native", "-march=native", "-malign-double"])
        .status().unwrap();

    Command::new("g++")
        .arg(&format!("{}/cuckoo.o", outdir))
        .args(&["-shared", "-Wl,-soname,libcuckoo.so", "-flto", "-o"])
        .arg(&format!("{}/libcuckoo.so", outdir))
        .status().unwrap();

    println!("cargo:rustc-link-search={}", outdir);
    println!("cargo:rustc-link-search={}", libdir);
    println!("cargo:rustc-link-lib=cityhash");
    println!("cargo:rustc-link-lib=numa");
}

/// Commands here must be tested manually, first. See src/cuckoo.cc for
/// the commands to use and how to test it first. Then, merge updates
/// here.
fn main() {
    make_dynamic();
}
