# jax2tf
- [jax2tf_tfcompile.py](jax2tf_tfcompile.py) is an example of how to convert JAX to Tensorflow, and then get the input files for tfcompile ([graph.pb](graph.pb) and [graph.config.pbtext](graph.config.pbtext)).
- [tf2xla_pb2.py](tf2xla_pb2.py) is generated and obtained using the following commands from the tensorflow directory:
    ```bash
    protoc tensorflow/compiler/tf2xla/tf2xla.proto --python_out=. && cp tensorflow/compiler/tf2xla/tf2xla_pb2.py path/to/
    ```
# tfcompile with WASM
- using `git apply wasm-patch.diff`, we apply the necessary changes to tfcompile for WASM target support
- tfcompile can then be configured and built using the following commands from the tensorflow directory:
  ```bash
  ./configure && bazel build --config=opt //tensorflow/compiler/aot:tfcompile
  ```
- once built, tfcompile can be used to AOT compile your graph using:
  ```bash
  bazel-bin/tensorflow/compiler/aot/tfcompile \
  --target_triple="wasm32-unknown-emscripten" \
  --target_cpu="generic" \
  --xla_cpu_multi_thread_eigen=false \
  --graph=path/to/graph.pb \
  --config=path/to/graph.config.pbtxt \
  --out_function_object=out_model.o \
  --out_header=out_header.h \
  --out_metadata_object=out_helper.o \
  --cpp_class=MyClass
  ```
- you should now have two WebAssembly object files, `out_model.o` and `out_helper.o`, and a C++ header file, `out_header.h`
