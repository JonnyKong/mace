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

#include <string>

#include "mace/utils/statistics.h"
#include "mace/benchmark_utils/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <RuntimeType D, typename T>
void FCBenchmark(
    int iters, int batch, int height, int width, int channel, int out_channel) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  if (D == RuntimeType::RT_OPENCL) {
    net.AddRandomInput<D, float>("Input", {batch, height, width, channel});
  } else {
    net.AddRandomInput<D, float>("Input", {batch, channel, height, width});
  }

  net.AddRandomInput<D, float>("Weight",
                               {out_channel, channel, height, width}, true);
  net.AddRandomInput<D, float>("Bias", {out_channel}, true);

  OpDefBuilder("FullyConnected", "FullyConnectedTest")
      .Input("Input")
      .Input("Weight")
      .Input("Bias")
      .Output("Output")
#ifdef MACE_ENABLE_OPENCL
      .AddIntArg("weight_type",
                 static_cast<int>(BufferContentType::WEIGHT_WIDTH))
#endif
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  // Warm-up
  net.Setup(D);
  for (int i = 0; i < 5; ++i) {
    net.Run();
  }
  net.Sync();

  RunMetadata run_metadata;

  mace::testing::StartTiming();
  while (iters--) {
    net.Run(&run_metadata);
  }
  net.Sync();

  std::cout << "stat length: " << run_metadata.op_stats.size() << std::endl;
  for (OperatorStats s : run_metadata.op_stats) {
    std::cout << " Queue time: " << s.stats.queue_micros;
    std::cout << " submit time: " << s.stats.submit_micros;
    std::cout << " start time: " << s.stats.start_micros;
    std::cout << " end time: " << s.stats.end_micros;
    std::cout << " Runtime: " << s.stats.end_micros - s.stats.start_micros << std::endl;
  }
}

#ifdef MACE_ENABLE_QUANTIZE
template <>
void FCBenchmark<RT_CPU, uint8_t>(
    int iters, int batch, int height, int width, int channel, int out_channel) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<RuntimeType::RT_CPU, uint8_t>(
      "Input", {batch, height, width, channel});
  net.GetTensor("Input")->SetScale(0.1);
  net.AddRandomInput<RuntimeType::RT_CPU, uint8_t>
      ("Weight", {out_channel, height, width, channel});
  net.GetTensor("Weight")->SetScale(0.1);
  net.AddRandomInput<RuntimeType::RT_CPU, int32_t>("Bias", {out_channel});

  OpDefBuilder("FullyConnected", "FullyConnectedTest")
      .Input("Input")
      .Input("Weight")
      .Input("Bias")
      .Output("Output")
      .AddIntArg("T", DT_UINT8)
      .Finalize(net.NewOperatorDef());

  net.Setup(RuntimeType::RT_CPU);
  net.GetTensor("Output")->SetScale(0.1);

  // Warm-up
  for (int i = 0; i < 2; ++i) {
    net.Run();
  }

  mace::testing::StartTiming();
  while (iters--) {
    net.Run();
  }
}
#endif  // MACE_ENABLE_QUANTIZE

}  // namespace

#define MACE_BM_FC_MACRO(N, H, W, C, OC, TYPE, DEVICE)                     \
  static void MACE_BM_FC_##N##_##H##_##W##_##C##_##OC##_##TYPE##_##DEVICE( \
      int iters) {                                                         \
    const int64_t macs =                                                   \
        static_cast<int64_t>(iters) * mace::benchmark::StatMACs(           \
            "FullyConnected", {OC, H, W, C}, {N, 1, 1, OC});               \
    const int64_t tot =                                                    \
        static_cast<int64_t>(iters) * (N + OC) * C * H * W + OC;           \
    mace::testing::MacsProcessed(macs);                                    \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                    \
    FCBenchmark<DEVICE, TYPE>(iters, N, H, W, C, OC);                      \
  }                                                                        \
  MACE_BENCHMARK(MACE_BM_FC_##N##_##H##_##W##_##C##_##OC##_##TYPE##_##DEVICE)

#if defined(MACE_ENABLE_OPENCL) && defined(MACE_ENABLE_QUANTIZE)
#define MACE_BM_FC(N, H, W, C, OC)                 \
  MACE_BM_FC_MACRO(N, H, W, C, OC, float, RT_CPU);    \
  MACE_BM_FC_MACRO(N, H, W, C, OC, float, RT_OPENCL);    \
  MACE_BM_FC_MACRO(N, H, W, C, OC, half, RT_OPENCL);     \
  MACE_BM_FC_MACRO(N, H, W, C, OC, uint8_t, RT_CPU)
#elif defined(MACE_ENABLE_OPENCL)
#define MACE_BM_FC(N, H, W, C, OC)                 \
  MACE_BM_FC_MACRO(N, H, W, C, OC, float, RT_CPU);    \
  MACE_BM_FC_MACRO(N, H, W, C, OC, float, RT_OPENCL)
#elif defined(MACE_ENABLE_QUANTIZE)
#define MACE_BM_FC(N, H, W, C, OC)                 \
  MACE_BM_FC_MACRO(N, H, W, C, OC, float, RT_CPU);    \
  MACE_BM_FC_MACRO(N, H, W, C, OC, uint8_t, RT_CPU)
#else
#define MACE_BM_FC(N, H, W, C, OC)                 \
  MACE_BM_FC_MACRO(N, H, W, C, OC, float, RT_CPU)
#endif

MACE_BM_FC(16, 1, 2048, 1, 2048);

/* MACE_BM_FC(1, 16, 16, 32, 32); */
/* MACE_BM_FC(1, 8, 8, 32, 1000); */
/* MACE_BM_FC(1, 2, 2, 512, 2); */
/* MACE_BM_FC(1, 7, 7, 512, 2048); */

}  // namespace test
}  // namespace ops
}  // namespace mace
