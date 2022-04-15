# OneFlow Protobuf

- This crate generate protobuf python apis in
- This crate is for building cfg. Because cfg doesn't have including mechanism,
  so before building it we need to do a full proto python compilation.
  While `.pb.cpp` and `.pb.h` should be component specific.
- You must make this crate a dependency if you need to compile cfg
