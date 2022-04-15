use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};
fn main() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    println!("cargo:rerun-if-changed={}", "CMakeLists.txt");
    let out_include: PathBuf = [out_dir.to_str().unwrap(), "include"].iter().collect();
    let out_lib_path = Path::new(&out_dir).join("lib");
    let glog_url = "https://github.com/google/glog/archive/refs/tags/v0.5.0.tar.gz";
    let glog_hash = "2368e3e0a95cce8b5b35a133271b480f";
    cmake::Config::new(".")
        .define("GLOG_URL", glog_url)
        .define("GLOG_HASH", glog_hash)
        .generator("Ninja")
        .build();
    println!(
        "cargo:rustc-link-search=native={}",
        &out_lib_path.to_str().unwrap()
    );
    println!("cargo:rustc-link-lib=static=glogd");
    // println!("cargo:rustc-link-lib=static=stdc++");
}
