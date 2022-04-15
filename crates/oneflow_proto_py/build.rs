use std::collections::HashSet;
fn main() {
    let cfg_proto_paths = HashSet::from([
        "oneflow/core/common/error.proto",
        "oneflow/core/vm/instruction.proto",
        "oneflow/core/job/job_conf.proto",
        "oneflow/core/job/placement.proto",
        "oneflow/core/operator/op_conf.proto",
        "oneflow/core/operator/interface_blob_conf.proto",
        "oneflow/core/common/shape.proto",
        "oneflow/core/record/record.proto",
        "oneflow/core/job/resource.proto",
        "oneflow/core/register/logical_blob_id.proto",
        "oneflow/core/register/tensor_slice_view.proto",
        "oneflow/core/common/range.proto",
        "oneflow/core/framework/user_op_conf.proto",
        "oneflow/core/framework/user_op_attr.proto",
        "oneflow/core/job/sbp_parallel.proto",
        "oneflow/core/graph/boxing/collective_boxing.proto",
        "oneflow/core/register/blob_desc.proto",
        "oneflow/core/job/scope.proto",
        "oneflow/core/job/mirrored_parallel.proto",
        "oneflow/core/operator/op_attribute.proto",
        "oneflow/core/operator/arg_modifier_signature.proto",
        "oneflow/core/job/blob_lifetime_signature.proto",
        "oneflow/core/job/parallel_signature.proto",
        "oneflow/core/job/parallel_conf_signature.proto",
        "oneflow/core/job/cluster_instruction.proto",
        "oneflow/core/job/initializer_conf.proto",
        "oneflow/core/job/regularizer_conf.proto",
        "oneflow/core/job/learning_rate_schedule_conf.proto",
        "oneflow/core/common/cfg_reflection_test.proto",
        "oneflow/core/common/data_type.proto",
        "oneflow/core/common/device_type.proto",
        "oneflow/core/serving/saved_model.proto",
    ]);

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
