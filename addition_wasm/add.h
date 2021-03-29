// Generated by tfcompile, the TensorFlow graph compiler.  DO NOT EDIT!
//
// This header was generated via ahead-of-time compilation of a TensorFlow
// graph.  An object file corresponding to this header was also generated.
// This header gives access to the functionality in that object file.
//
// clang-format off

#ifndef TFCOMPILE_GENERATED_entry_H_  // NOLINT(build/header_guard)
#define TFCOMPILE_GENERATED_entry_H_  // NOLINT(build/header_guard)



#include "tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h"
#include "tensorflow/core/platform/types.h"

namespace Eigen { struct ThreadPoolDevice; }
namespace xla { class ExecutableRunOptions; }

// (Implementation detail) Entry point to the function in the object file.
extern "C" void entry(
    void* result, const ::xla::ExecutableRunOptions* run_options,
    const void** args, void** temps, tensorflow::int64* profile_counters);





// Add represents a computation previously specified in a
// TensorFlow graph, now compiled into executable code. This extends the generic
// XlaCompiledCpuFunction class with statically type-safe arg and result
// methods. Usage example:
//
//   Add computation;
//   // ...set args using computation.argN methods
//   CHECK(computation.Run());
//   // ...inspect results using computation.resultN methods
//
// The Run method invokes the actual computation, with inputs read from arg
// buffers, and outputs written to result buffers. Each Run call may also use
// a set of temporary buffers for the computation.
//
// By default each instance of this class manages its own arg, result and temp
// buffers. The AllocMode constructor parameter may be used to modify the
// buffer allocation strategy.
//
// Under the default allocation strategy, this class is thread-compatible:
// o Calls to non-const methods require exclusive access to the object.
// o Concurrent calls to const methods are OK, if those calls are made while it
//   is guaranteed that no thread may call a non-const method.
//
// The logical function signature is:
//   (arg0: f32[1,1], arg1: f32[1,1]) -> (f32[])
//
// Memory stats:
//   arg bytes total:    8
//   arg bytes aligned:  128
//   temp bytes total:   12
//   temp bytes aligned: 128
class Add final : public tensorflow::XlaCompiledCpuFunction {
 public:
  // Number of input arguments for the compiled computation.
  static constexpr size_t kNumArgs = 2;

  // Number of variables for the compiled computation.
  static constexpr size_t kNumVariables = 0;

  // Byte size of each argument buffer. There are kNumArgs entries.
  static const ::tensorflow::int64 ArgSize(::tensorflow::int32 index) {
    return BufferInfos()[ArgIndexToBufferIndex()[index]].size();
  }

  // Returns static data used to create an XlaCompiledCpuFunction.
  static const tensorflow::XlaCompiledCpuFunction::StaticData& StaticData() {
    static XlaCompiledCpuFunction::StaticData* kStaticData = [](){
      XlaCompiledCpuFunction::StaticData* data =
        new XlaCompiledCpuFunction::StaticData;
      set_static_data_raw_function(data, entry);
      set_static_data_buffer_infos(data, BufferInfos());
      set_static_data_num_buffers(data, kNumBuffers);
      set_static_data_arg_index_table(data, ArgIndexToBufferIndex());
      set_static_data_num_args(data, kNumArgs);
      set_static_data_num_variables(data, kNumVariables);
      set_static_data_result_index(data, kResultIndex);
      set_static_data_arg_names(data, StaticArgNames());
      set_static_data_variable_names(data, StaticVariableNames());
      set_static_data_result_names(data, StaticResultNames());
      set_static_data_program_shape(data, StaticProgramShape());
      set_static_data_hlo_profile_printer_data(
          data, StaticHloProfilePrinterData());

      return data;
    }();
    return *kStaticData;
  }

  Add(AllocMode alloc_mode =
            AllocMode::ARGS_VARIABLES_RESULTS_PROFILES_AND_TEMPS)
      : XlaCompiledCpuFunction(StaticData(), alloc_mode) {}

  Add(const Add&) = delete;
  Add& operator=(const Add&) = delete;

  // Arg methods for managing input buffers. Buffers are in row-major order.
  // There is a set of methods for each positional argument, with the following
  // general form:
  //
  // void set_argN_data(void* data)
  //   Sets the buffer of type T for positional argument N. May be called in
  //   any AllocMode. Must be called before Run to have an affect. Must be
  //   called in AllocMode::RESULTS_PROFILES_AND_TEMPS_ONLY for each positional
  //   argument, to set the argument buffers.
  //
  // T* argN_data()
  //   Returns the buffer of type T for positional argument N.
  //
  // T& argN(...dim indices...)
  //   Returns a reference to the value of type T for positional argument N,
  //   with dim indices specifying which value. No bounds checking is performed
  //   on dim indices.

  void set_arg0_data(const void* data) {
    set_arg_data(0, data);
  }
  float* arg0_data() {
    return static_cast<float*>(arg_data(0));
  }
  float& arg0(size_t dim0, size_t dim1) {
    return (*static_cast<float(*)[1][1]>(
        arg_data(0)))[dim0][dim1];
  }
  const float* arg0_data() const {
    return static_cast<const float*>(arg_data(0));
  }
  const float& arg0(size_t dim0, size_t dim1) const {
    return (*static_cast<const float(*)[1][1]>(
        arg_data(0)))[dim0][dim1];
  }
  int arg0_size() const {
    return 1 * sizeof(float);
  }
  int arg0_count() const {
    return 1;
  }

  void set_arg1_data(const void* data) {
    set_arg_data(1, data);
  }
  float* arg1_data() {
    return static_cast<float*>(arg_data(1));
  }
  float& arg1(size_t dim0, size_t dim1) {
    return (*static_cast<float(*)[1][1]>(
        arg_data(1)))[dim0][dim1];
  }
  const float* arg1_data() const {
    return static_cast<const float*>(arg_data(1));
  }
  const float& arg1(size_t dim0, size_t dim1) const {
    return (*static_cast<const float(*)[1][1]>(
        arg_data(1)))[dim0][dim1];
  }
  int arg1_size() const {
    return 1 * sizeof(float);
  }
  int arg1_count() const {
    return 1;
  }

  // Result methods for managing output buffers. Buffers are in row-major order.
  // Must only be called after a successful Run call. There is a set of methods
  // for each positional result, with the following general form:
  //
  // T* resultN_data()
  //   Returns the buffer of type T for positional result N.
  //
  // T& resultN(...dim indices...)
  //   Returns a reference to the value of type T for positional result N,
  //   with dim indices specifying which value. No bounds checking is performed
  //   on dim indices.
  //
  // Unlike the arg methods, there is no set_resultN_data method. The result
  // buffers are managed internally, and may change after each call to Run.

  float* result0_data() {
    return static_cast<float*>(result_data(0));
  }
  float& result0() {
    return (*static_cast<float(*)[1]>(
        result_data(0)))[0];
  }
  const float* result0_data() const {
    return static_cast<const float*>(result_data(0));
  }
  const float& result0() const {
    return (*static_cast<const float(*)[1]>(
        result_data(0)))[0];
  }
  int result0_size() const {
    return 1 * sizeof(float);
  }
  int result0_count() const {
    return 1;
  }

  // Methods for managing variable buffers. Buffers are in row-major order.
  //
  // For read-write variables we generate the following methods:
  //
  // void set_var_X_data(T* data)
  //   Sets the buffer for variable X.  Must be called before Run if the
  //   allocation mode is RESULTS_PROFILES_AND_TEMPS_ONLY.
  //
  // T* var_X_data()
  //   Returns the buffer of type T for variable X.  If the allocation mode is
  //   RESULTS_PROFILES_AND_TEMPS_ONLY then this buffer is the same as the
  //   buffer passed to set_var_X_data.
  //
  // T& var_X(...dim indices...)
  //   Returns a reference to the value of type T for variable X,
  //   with dim indices specifying which value. No bounds checking is performed
  //   on dim indices.
  //
  // For readonly variables we generate the same set of methods, except that we
  // use `const T` instead of `T`.  We use `const T` to avoid erasing the
  // constness of the buffer passed to `set_var_X_data` but the underlying
  // buffer is not const (and thus the const can be safely const-cast'ed away)
  // unless `set_var_X_data` is called with a pointer to constant storage.

 private:
  // Number of buffers for the compiled computation.
  static constexpr size_t kNumBuffers = 4;

  static const ::xla::cpu_function_runtime::BufferInfo* BufferInfos() {
    static const ::xla::cpu_function_runtime::BufferInfo
      kBufferInfos[kNumBuffers] = {
::xla::cpu_function_runtime::BufferInfo({33ULL, ~0ULL}),
::xla::cpu_function_runtime::BufferInfo({17ULL, ~0ULL}),
::xla::cpu_function_runtime::BufferInfo({18ULL, 0ULL}),
::xla::cpu_function_runtime::BufferInfo({18ULL, 1ULL})
      };
    return kBufferInfos;
  }

  static const ::tensorflow::int32* ArgIndexToBufferIndex() {
    static constexpr ::tensorflow::int32 kArgIndexToBufferIndex[kNumArgs] = {
2, 3
    };
    return kArgIndexToBufferIndex;
  }

  // The 0-based index of the result tuple in the temporary buffers.
  static constexpr size_t kResultIndex = 0;

  // Array of names of each positional argument, terminated by nullptr.
  static const char** StaticArgNames() {
    return nullptr;
  }

  // Array of names of each positional variable, terminated by nullptr.
  static const char** StaticVariableNames() {
    return nullptr;
  }

  // Array of names of each positional result, terminated by nullptr.
  static const char** StaticResultNames() {
    return nullptr;
  }

  // Shape of the args and results.
  static const ::xla::ProgramShapeProto* StaticProgramShape() {
    static const ::xla::ProgramShapeProto* kShape = nullptr;
    return kShape;
  }

  // Metadata that can be used to pretty-print profile counters.
  static const ::xla::HloProfilePrinterData* StaticHloProfilePrinterData() {
    static const ::xla::HloProfilePrinterData* kHloProfilePrinterData =
      nullptr;
    return kHloProfilePrinterData;
  }
};


#endif  // TFCOMPILE_GENERATED_entry_H_

// clang-format on