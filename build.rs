// build.rs

use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

use glob::glob;
use jobserver::Client;

fn main() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let client = match unsafe { Client::from_env() } {
        Some(client) => client,
        None => panic!("client not configured"),
    };
    let token = client.acquire().unwrap(); // blocks until it is available
                                           // generate cfg, because cfg doesn't have include mechanism so must compile all protobuf before
    for entry in glob("oneflow/core/**/*.proto").expect("Failed to read glob pattern") {
        match entry {
            Ok(path) => {
                println!("cargo:rerun-if-changed={}", path.to_str().unwrap());
                if cfg_proto_paths.contains(path.to_str().unwrap()) {
                    let mut cmd = Command::new("python3");
                    cmd.args(&[
                        "tools/cfg/template_convert.py",
                        "--project_build_dir",
                        out_dir.to_str().unwrap(),
                        "--of_cfg_proto_python_dir",
                        out_dir.to_str().unwrap(),
                        "--generate_file_type=cfg.cpp",
                        "--proto_file_path",
                        path.to_str().unwrap(),
                    ]);
                    client.configure(&mut cmd);
                    assert!(cmd.status().expect("failed to execute process").success());
                }
            }
            Err(e) => println!("{:?}", e),
        }
    }
    drop(token); // releases the token when the work is done
                 // generate c++ and python from proto
                 // build oneflow common
    let mut oneflow_common = cc::Build::new();
    for entry in glob("oneflow/core/common/**/*.cpp").expect("Failed to read glob pattern") {
        match entry {
            Ok(path) => {
                if !path.to_str().unwrap().ends_with("test.cpp") {
                    oneflow_common.file(path);
                }
            }
            Err(e) => println!("{:?}", e),
        }
    }
    oneflow_common
        .include(".")
        .include(out_include)
        .include(&out_dir)
        .include("./tools/cfg/include")
        .cpp_link_stdlib("stdc++")
        .flag("-w")
        .compile("oneflow_common");
    println!(
        "cargo:rustc-link-search=native={}",
        &out_lib_path.to_str().unwrap()
    );
    println!("cargo:rustc-link-lib=static=glogd");
    println!("cargo:rustc-link-lib=static=stdc++");
}
