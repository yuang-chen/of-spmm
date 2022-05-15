use std::collections::HashSet;
fn main() {
    let protoc_path = Path::new(&out_dir).join("bin").join("protoc");
    let proto_include_path = Path::new(&out_dir).join("include");
    for entry in glob("oneflow/core/**/*.proto").expect("Failed to read glob pattern") {
        match entry {
            Ok(path) => {
                println!("cargo:rerun-if-changed={}", path.to_str().unwrap());
                // TODO: create __init__.py for pb
                let mut cmd = Command::new(protoc_path.to_str().unwrap());
                cmd.args(&[
                    "-I",
                    proto_include_path.to_str().unwrap(),
                    "-I",
                    "./",
                    "--cpp_out",
                    out_dir.to_str().unwrap(),
                    "--python_out",
                    "python",
                    "--python_out",
                    out_dir.to_str().unwrap(), // this is for cfg to use in reflection
                    path.to_str().unwrap(),
                ]);
                client.configure(&mut cmd);
                // assert!(cmd.status().expect("failed to execute process");.success());
            }
            Err(e) => println!("{:?}", e),
        }
    }
}
