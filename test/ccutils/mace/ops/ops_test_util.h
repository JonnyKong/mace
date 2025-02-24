// Copyright 2018 The MACE Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MACE_OPS_OPS_TEST_UTIL_H_
#define MACE_OPS_OPS_TEST_UTIL_H_

#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "mace/core/types.h"
#include "mace/core/net/serial_net.h"
#include "mace/runtimes/opencl/core/opencl_context.h"
#include "mace/core/memory/rpcmem/rpcmem.h"
#include "mace/core/tensor.h"
#include "mace/core/workspace.h"
#include "mace/core/registry/ops_registry.h"
#include "mace/core/registry/op_delegator_registry.h"
#include "mace/core/runtime/runtime_registry.h"
#include "mace/ops/registry/registry.h"
#include "mace/public/mace.h"
#include "mace/utils/memory.h"
#include "mace/utils/math.h"
#include "mace/core/quantize.h"
#include "mace/ops/testing/test_utils.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/runtimes/opencl/core/opencl_util.h"
#endif

namespace mace {
namespace ops {
namespace test {

class OpDefBuilder {
 public:
  OpDefBuilder(const char *type, const std::string &name);

  OpDefBuilder &Input(const std::string &input_name);

  OpDefBuilder &Output(const std::string &output_name);

  OpDefBuilder &OutputType(const std::vector<DataType> &output_type);

  OpDefBuilder &OutputShape(const std::vector<index_t> &output_shape);

  OpDefBuilder AddIntArg(const std::string &name, const int value);

  OpDefBuilder AddFloatArg(const std::string &name, const float value);

  OpDefBuilder AddStringArg(const std::string &name, const char *value);

  OpDefBuilder AddIntsArg(const std::string &name,
                          const std::vector<int> &values);

  OpDefBuilder AddFloatsArg(const std::string &name,
                            const std::vector<float> &values);

  void Finalize(OperatorDef *op_def) const;

  OperatorDef op_def_;
};

class OpTestContext {
 public:
  OpTestContext(int num_threads, CPUAffinityPolicy cpu_affinity_policy,
                bool use_cache = true);
  MACE_DISABLE_COPY_AND_ASSIGN(OpTestContext);

  static OpTestContext *Get(
      int num_threads = -1,
      CPUAffinityPolicy cpu_affinity_policy = AFFINITY_BIG_ONLY);
  static std::unique_ptr<OpTestContext> New(
      int num_threads = -1,
      CPUAffinityPolicy cpu_affinity_policy = AFFINITY_BIG_ONLY);
  std::unique_ptr<Runtime> NewAndInitRuntime(
      RuntimeType runtime_type, MemoryType mem_type,
      MaceEngineCfgImpl *engine_config,
      RuntimeContext *runtime_context = nullptr);
  Runtime *GetRuntime(RuntimeType runtime_type);
#ifdef MACE_ENABLE_OPENCL
  std::shared_ptr<OpenclContext> gpu_context() const;
  std::vector<MemoryType> opencl_mem_types();
  void SetOCLBufferTestFlag();
  void SetOCLImageTestFlag();
  void SetOCLImageAndBufferTestFlag();
#endif
  utils::ThreadPool *thread_pool() {
    return thread_pool_.get();
  }

 private:
#ifdef MACE_ENABLE_OPENCL
  std::shared_ptr<OpenclContext> gpu_context_;
  std::vector<MemoryType> opencl_mem_types_;
#endif
  std::unique_ptr<utils::ThreadPool> thread_pool_;
  std::shared_ptr<Rpcmem> rpcmem_;
  std::unique_ptr<RuntimeContext> runtime_context_;
  std::unique_ptr<RuntimeRegistry> runtime_registry_;
  std::map<RuntimeType, std::unique_ptr<Runtime>> runtime_map_;
};

class OpsTestNet {
 public:
  OpsTestNet() :
      op_registry_(make_unique<OpRegistry>()),
      op_delegator_registry_(make_unique<OpDelegatorRegistry>()),
      ws_(op_delegator_registry_.get(), nullptr) {
    ops::RegisterAllOps(op_registry_.get());
    ops::RegisterAllOpDelegators(op_delegator_registry_.get());
    {
      std::lock_guard<std::mutex> lock(ref_mutex_);
      ++ref_count_;
    }
  }

  ~OpsTestNet();

  template<RuntimeType D, typename T>
  void AddInputFromArray(const std::string &name,
                         const std::vector<index_t> &shape,
                         const std::vector<T> &data,
                         bool is_weight = false,
                         const float scale = 0.0,
                         const int32_t zero_point = 0) {
    auto *runtime = OpTestContext::Get()->GetRuntime(D);
    Tensor *input = ws_.CreateTensor(name, runtime, DataTypeToEnum<T>::v(),
                                     is_weight, runtime->GetBaseMemoryType());
    input->Reshape(shape);
    runtime->AllocateBufferForTensor(input, RENT_PRIVATE);
    Tensor::MappingGuard input_mapper(input);
    T *input_data = input->mutable_data<T>();
    MACE_CHECK(static_cast<size_t>(input->size()) == data.size(),
               input->size(), " VS ", data.size());
    memcpy(input_data, data.data(), data.size() * sizeof(T));
    input->SetScale(scale);
    input->SetZeroPoint(zero_point);
  }

  template<RuntimeType D, typename T>
  void AddRepeatedInput(const std::string &name,
                        const std::vector<index_t> &shape,
                        const T data,
                        bool is_weight = false) {
    auto *runtime = OpTestContext::Get()->GetRuntime(D);
    Tensor *input = ws_.CreateTensor(name, runtime, DataTypeToEnum<T>::v(),
                                     is_weight, runtime->GetBaseMemoryType());
    input->Reshape(shape);
    runtime->AllocateBufferForTensor(input, RENT_PRIVATE);
    Tensor::MappingGuard input_mapper(input);
    T *input_data = input->mutable_data<T>();
    std::fill(input_data, input_data + input->size(), data);
  }

  template<RuntimeType D, typename T>
  void AddRandomInput(const std::string &name,
                      const std::vector<index_t> &shape,
                      bool is_weight = false,
                      bool positive = true,
                      bool truncate = false,
                      const float truncate_min = 0.001f,
                      const float truncate_max = 100.f) {
    auto *runtime = OpTestContext::Get()->GetRuntime(D);
    Tensor *input = ws_.CreateTensor(name, runtime, DataTypeToEnum<T>::v(),
                                     is_weight, runtime->GetBaseMemoryType());
    input->Reshape(shape);
    runtime->AllocateBufferForTensor(input, RENT_PRIVATE);
    Tensor::MappingGuard input_mapper(input);
    T *input_data = input->mutable_data<T>();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> nd(0, 1);
    if (DataTypeToEnum<T>::value == DT_HALF ||
        DataTypeToEnum<T>::value == DT_FLOAT16) {
      std::generate(
          input_data, input_data + input->size(),
          [&gen, &nd, positive, truncate, truncate_min, truncate_max] {
            float d = nd(gen);
            if (truncate) {
              if (std::abs(d) > truncate_max) d = truncate_max;
              if (std::abs(d) < truncate_min) d = truncate_min;
            }
            return half_float::half_cast<half>(positive ? std::abs(d) : d);
          });
    } else if (DataTypeToEnum<T>::value == DT_UINT8) {
      std::generate(input_data, input_data + input->size(),
                    [&gen, &nd] {
                      return Saturate<uint8_t>(roundf((nd(gen) + 1) * 128));
                    });
    } else {
      std::generate(input_data, input_data + input->size(),
                    [&gen, &nd, positive, truncate,
                        truncate_min, truncate_max] {
                      float d = nd(gen);
                      if (truncate) {
                        if (std::abs(d) > truncate_max) d = truncate_max;
                        if (std::abs(d) < truncate_min) d = truncate_min;
                      }
                      return (positive ? std::abs(d) : d);
                    });
    }
  }

  template<RuntimeType D, typename T>
  void CopyData(const std::string &src_name,
                const std::string &dst_name) {
    auto *runtime = OpTestContext::Get()->GetRuntime(D);
    Tensor *input = ws_.GetTensor(src_name);
    Tensor *output = ws_.CreateTensor(dst_name, runtime, DataTypeToEnum<T>::v(),
                                      input->is_weight(), input->memory_type());
    const std::vector<index_t> input_shape = input->shape();
    output->Reshape(input_shape);
    runtime->AllocateBufferForTensor(output, RENT_PRIVATE);

    Tensor::MappingGuard input_guard(input);
    output->CopyBytes(input->raw_data(), input->size() * input->SizeOfType());
  }

  template<RuntimeType D, typename T>
  void TransformDataFormat(const std::string &src_name,
                           const DataFormat src_format,
                           const std::string &dst_name,
                           const DataFormat dst_format) {
    Tensor *input = ws_.GetTensor(src_name);
    auto *runtime = input->GetCurRuntime();
    Tensor *output = ws_.CreateTensor(dst_name, runtime, DataTypeToEnum<T>::v(),
                                      input->is_weight(), input->memory_type());
    const std::vector<index_t> input_shape = input->shape();
    MACE_CHECK(input_shape.size() == 4, "input shape != 4");

    if (src_format == DataFormat::NHWC && dst_format == DataFormat::NCHW) {
      index_t batch = input_shape[0];
      index_t height = input_shape[1];
      index_t width = input_shape[2];
      index_t channels = input_shape[3];
      output->Reshape({batch, channels, height, width});
      runtime->AllocateBufferForTensor(output, RENT_PRIVATE);
      Tensor::MappingGuard input_guard(input);
      Tensor::MappingGuard output_guard(output);
      const T *input_data = input->data<T>();
      T *output_data = output->mutable_data<T>();
      for (index_t b = 0; b < batch; ++b) {
        for (index_t c = 0; c < channels; ++c) {
          for (index_t h = 0; h < height; ++h) {
            for (index_t w = 0; w < width; ++w) {
              output_data[((b * channels + c) * height + h) * width + w] =
                  input_data[((b * height + h) * width + w) * channels + c];
            }
          }
        }
      }
      output->set_data_format(DataFormat::NCHW);
    } else if (src_format == DataFormat::NCHW &&
        dst_format == DataFormat::NHWC) {
      index_t batch = input_shape[0];
      index_t channels = input_shape[1];
      index_t height = input_shape[2];
      index_t width = input_shape[3];
      output->Reshape({batch, height, width, channels});
      runtime->AllocateBufferForTensor(output, RENT_PRIVATE);
      Tensor::MappingGuard input_guard(input);
      Tensor::MappingGuard output_guard(output);
      const T *input_data = input->data<T>();
      T *output_data = output->mutable_data<T>();
      for (index_t b = 0; b < batch; ++b) {
        for (index_t h = 0; h < height; ++h) {
          for (index_t w = 0; w < width; ++w) {
            for (index_t c = 0; c < channels; ++c) {
              output_data[((b * height + h) * width + w) * channels + c] =
                  input_data[((b * channels + c) * height + h) * width + w];
            }
          }
        }
      }
      output->set_data_format(DataFormat::NHWC);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }

  template<RuntimeType D, typename T>
  void TransformFilterDataFormat(const std::string &src_name,
                                 const DataFormat src_format,
                                 const std::string &dst_name,
                                 const DataFormat dst_format) {
    Tensor *input = ws_.GetTensor(src_name);
    auto *runtime = input->GetCurRuntime();
    Tensor *output = ws_.CreateTensor(dst_name, runtime, DataTypeToEnum<T>::v(),
                                      input->is_weight(), input->memory_type());
    const std::vector<index_t> input_shape = input->shape();
    MACE_CHECK(input_shape.size() == 4, "input shape != 4");
    if (src_format == DataFormat::HWOI && dst_format == DataFormat::OIHW) {
      index_t height = input_shape[0];
      index_t width = input_shape[1];
      index_t out_channels = input_shape[2];
      index_t in_channels = input_shape[3];
      index_t hw = height * width;
      index_t oi = out_channels * in_channels;
      output->Reshape({out_channels, in_channels, height, width});
      runtime->AllocateBufferForTensor(output, RENT_PRIVATE);
      Tensor::MappingGuard input_guard(input);
      Tensor::MappingGuard output_guard(output);
      const T *input_data = input->data<T>();
      T *output_data = output->mutable_data<T>();
      for (index_t i = 0; i < oi; ++i) {
        for (index_t j = 0; j < hw; ++j) {
          output_data[i * height * width + j] =
              input_data[j * out_channels * in_channels + i];
        }
      }
    } else if (src_format == DataFormat::OIHW &&
        dst_format == DataFormat::HWOI) {
      index_t out_channels = input_shape[0];
      index_t in_channels = input_shape[1];
      index_t height = input_shape[2];
      index_t width = input_shape[3];
      index_t hw = height * width;
      index_t oi = out_channels * in_channels;
      output->Reshape({height, width, out_channels, in_channels});
      runtime->AllocateBufferForTensor(output, RENT_PRIVATE);
      Tensor::MappingGuard input_guard(input);
      Tensor::MappingGuard output_guard(output);
      const T *input_data = input->data<T>();
      T *output_data = output->mutable_data<T>();
      for (index_t i = 0; i < hw; ++i) {
        for (index_t j = 0; j < oi; ++j) {
          output_data[i * out_channels * in_channels + j] =
              input_data[j * height * width + i];
        }
      }
    } else if (src_format == DataFormat::HWIO &&
        dst_format == DataFormat::OIHW) {
      index_t height = input_shape[0];
      index_t width = input_shape[1];
      index_t in_channels = input_shape[2];
      index_t out_channels = input_shape[3];
      index_t hw = height * width;
      output->Reshape({out_channels, in_channels, height, width});
      runtime->AllocateBufferForTensor(output, RENT_PRIVATE);
      Tensor::MappingGuard input_guard(input);
      Tensor::MappingGuard output_guard(output);
      const T *input_data = input->data<T>();
      T *output_data = output->mutable_data<T>();
      for (index_t m = 0; m < out_channels; ++m) {
        for (index_t c = 0; c < in_channels; ++c) {
          for (index_t k = 0; k < hw; ++k) {
            output_data[((m * in_channels) + c) * height * width + k] =
                input_data[k * out_channels * in_channels + c * out_channels +
                    m];
          }
        }
      }
    } else if (src_format == DataFormat::OHWI &&
        dst_format == DataFormat::OIHW) {
      index_t out_channels = input_shape[0];
      index_t height = input_shape[1];
      index_t width = input_shape[2];
      index_t in_channels = input_shape[3];
      output->Reshape({out_channels, in_channels, height, width});
      runtime->AllocateBufferForTensor(output, RENT_PRIVATE);
      Tensor::MappingGuard input_guard(input);
      Tensor::MappingGuard output_guard(output);
      const T *input_data = input->data<T>();
      T *output_data = output->mutable_data<T>();
      for (index_t b = 0; b < out_channels; ++b) {
        for (index_t c = 0; c < in_channels; ++c) {
          for (index_t h = 0; h < height; ++h) {
            for (index_t w = 0; w < width; ++w) {
              output_data[((b * in_channels + c) * height + h) * width + w] =
                  input_data[((b * height + h) * width + w) * in_channels + c];
            }
          }
        }
      }
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }

  template<RuntimeType D, typename Src, typename Dst>
  void Cast(const std::string &src_name, const std::string &dst_name) {
    Tensor *input = ws_.GetTensor(src_name);
    auto *runtime = input->GetCurRuntime();
    Tensor *output = ws_.CreateTensor(
        dst_name, runtime, DataTypeToEnum<Dst>::v(),
        input->is_weight(), input->memory_type());
    output->Resize(input->shape());
    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    auto input_data = input->data<Src>();
    auto output_data = output->mutable_data<Dst>();
    for (index_t i = 0; i < input->size(); ++i) {
      output_data[i] = input_data[i];
    }
  }

  // Create standalone tensor on runtime D with T type.
  template<typename T, RuntimeType D = RuntimeType::RT_CPU>
  std::unique_ptr<Tensor> CreateTensor(
      const std::vector<index_t> &shape = {},
      const std::vector<T> &data = {}) {
    auto *runtime = OpTestContext::Get()->GetRuntime(D);
    std::unique_ptr<Tensor> res = make_unique<Tensor>(
        runtime, DataTypeToEnum<T>::v(), shape);
    if (!data.empty()) {
      runtime->AllocateBufferForTensor(res.get(), RENT_PRIVATE);
      Tensor::MappingGuard res_guard(res.get());
      T *input_data = res->mutable_data<T>();
      memcpy(input_data, data.data(), data.size() * sizeof(T));
    }
    return res;
  }

  OperatorDef *NewOperatorDef() {
    op_defs_.clear();
    op_defs_.emplace_back(OperatorDef());
    return &op_defs_[op_defs_.size() - 1];
  }

  OperatorDef *AddNewOperatorDef() {
    op_defs_.emplace_back(OperatorDef());
    return &op_defs_[op_defs_.size() - 1];
  }

  inline Workspace *ws() { return &ws_; }

  bool Setup(RuntimeType runtime);

  MaceStatus Run(RunMetadata *run_metadata=nullptr);

  // DEPRECATED(liyin):
  // Test and benchmark should setup model once and run multiple times.
  // Setup time should not be counted during benchmark.
  MaceStatus RunOp(RuntimeType runtime);

  // DEPRECATED(liyin):
  // Test and benchmark should setup model once and run multiple times.
  // Setup time should not be counted during benchmark.
  MaceStatus RunOp();

  MaceStatus RunNet(const NetDef &net_def, const RuntimeType runtime);

  inline Tensor *GetOutput(const char *output_name) {
    return ws_.GetTensor(output_name);
  }

  inline Tensor *GetTensor(const char *tensor_name) {
    return ws_.GetTensor(tensor_name);
  }

  void Sync();

 private:
  std::unique_ptr<OpRegistry> op_registry_;
  std::unique_ptr<OpDelegatorRegistry> op_delegator_registry_;
  Workspace ws_;
  std::vector<OperatorDef> op_defs_;
  std::unique_ptr<BaseNet> net_;
  RuntimeType runtime_type_;

  static int ref_count_;
  static std::mutex ref_mutex_;
};

class OpsTestBase : public ::testing::Test {
 protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
#ifdef MACE_ENABLE_OPENCL
    OpTestContext::Get()->SetOCLImageTestFlag();
#endif
  }
};

}  // namespace test
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPS_TEST_UTIL_H_
