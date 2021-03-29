- [add.h](add.h) is a tfcompile-generated header file
- [add.cc](add.cc) is used to invoke the tfcompile-generated code using the header file
- The [objects](objects/) directory contains all necessary compiled dependencies that can be linked to generate a WASM binary file. [add_helper.o](objects/add_helper.o) and [add_model.o](objects/add_model.o) were generated using a modified `tfcompile` with WASM support, and the rest of the files were compiled using a command similar to what Bazel uses to build the x86 binary if one were using the `cc_binary` build macro. These commands can be seen from Bazel using the `--subcommands` option, and a sample command is given below. You may need to remove some flags that Bazel uses in its subcommands if they cause errors when trying to manually compile dependencies with emscripten.
  ```bash
  (cd /usr/local/google/home/paksha/.cache/bazel/_bazel_paksha/e3f75b9aceed7caf2cac7619be37da6a/execroot/org_tensorflow && \
  exec env - \
    PATH=/usr/local/google/home/paksha/.local/bin:/usr/lib/google-golang/bin:/usr/local/buildtools/java/jdk/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PWD=/proc/self/cwd \
    TF2_BEHAVIOR=1 \
    /usr/local/google/home/paksha/emsdk/upstream/emscripten/emcc -U_FORTIFY_SOURCE -fstack-protector -Wall -Wunused-but-set-parameter -Wno-free-nonheap-object -fno-omit-frame-pointer -g0 -O2 '-D_FORTIFY_SOURCE=1' -DNDEBUG -ffunction-sections -fdata-sections '-std=c++0x' -MD -MF bazel-out/k8-opt/bin/tensorflow/compiler/tf2xla/_objs/xla_compiled_cpu_function/xla_compiled_cpu_function.d -DEIGEN_MPL2_ONLY '-DEIGEN_MAX_ALIGN_BYTES=64' -iquote . -iquote bazel-out/k8-opt/bin -iquote external/eigen_archive -iquote bazel-out/k8-opt/bin/external/eigen_archive -iquote external/com_google_absl -iquote bazel-out/k8-opt/bin/external/com_google_absl -isystem third_party/eigen3/mkl_include -isystem bazel-out/k8-opt/bin/third_party/eigen3/mkl_include -isystem external/eigen_archive -isystem bazel-out/k8-opt/bin/external/eigen_archive -w -DAUTOLOAD_DYNAMIC_KERNELS '-std=c++14' -Wno-builtin-macro-redefined -c tensorflow/compiler/tf2xla/xla_compiled_cpu_function.cc -o bazel-out/k8-opt/bin/tensorflow/compiler/tf2xla/_objs/xla_compiled_cpu_function/xla_compiled_cpu_function.o)
  ```
- The [browser](browser/) directory contains all necessary files to run the WASM code for the addition computation in the browser. To do so, run a simple local HTTP server using Python, go to the server, and then click on the HTML file. For information, see [how to set up a local testing server](https://developer.mozilla.org/en-US/docs/Learn/Common_questions/set_up_a_local_testing_server). The files in this directory were generated using the following command, which uses emscripten to link the object files from [objects](objects/) and generate HTML & JavaScript to execute the WASM:
  ```bash
  emcc -s WASM=1 *.o -o addition.html
  ```
