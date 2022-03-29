#include "mace/benchmark_utils/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {
namespace {

template <RuntimeType D, typename T>
void CustomAddBenchmark(
    int iters, int n, int h, int w, int c, int repeat_times) {
  mace::testing::StopTiming();

  OpsTestNet net;
  net.AddRandomInput<D, float>("Input0", {n, h, w, c});
  net.AddRandomInput<D, float>("Input1", {n, h, w, c});

  OpDefBuilder("CustomAdd", "CustomAddTest")
      .Input("Input0")
      .Input("Input1")
      .AddIntArg("repeat_times", repeat_times)
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  net.Setup(D);

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.Run();
    net.Sync();
    // auto expected = net.CreateTensor<float>();
    // expected->Copy(*net.GetOutput("Output"));
  }

   RunMetadata run_metadata;

  mace::testing::StartTiming();
  while (iters--) {
    net.Run(&run_metadata);
    net.Sync();
    // auto expected = net.CreateTensor<float>();
    // expected->Copy(*net.GetOutput("Output"));
  }
  std::cout << "stat length: " << run_metadata.op_stats.size() << std::endl;
  for (OperatorStats s : run_metadata.op_stats) {

    std::cout << " Queue time: " << s.stats.queue_micros;
    std::cout << " submit time: " << s.stats.submit_micros;
    std::cout << " start time: " << s.stats.start_micros;
    std::cout << " end time: " << s.stats.end_micros;

    std::cout << " Runtime: " << s.stats.end_micros - s.stats.start_micros << std::endl;
  }



}
}  // namespace

#define MACE_BM_CUSTOM_ADD_MACRO(N, H, W, C, REPEAT_TIMES, TYPE, DEVICE)             \
  static void                                                                        \
      MACE_BM_CUSTOM_ADD_##N##_##H##_##W##_##C##_##REPEAT_TIMES##_##TYPE##_##DEVICE( \
          int iters) {                                                               \
    const int64_t tot =                                                              \
        static_cast<int64_t>(iters) * N * H * W * C * REPEAT_TIMES;                  \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                              \
    CustomAddBenchmark<DEVICE, TYPE>(iters, N, H, W, C, REPEAT_TIMES);               \
  }                                                                                  \
  MACE_BENCHMARK(                                                                    \
      MACE_BM_CUSTOM_ADD_##N##_##H##_##W##_##C##_##REPEAT_TIMES##_##TYPE##_##DEVICE)

#ifdef MACE_ENABLE_OPENCL
#define MACE_BM_CUSTOM_ADD(N, H, W, C, REPEAT_TIMES)                 \
  MACE_BM_CUSTOM_ADD_MACRO(N, H, W, C, REPEAT_TIMES, float, RT_CPU); \
  MACE_BM_CUSTOM_ADD_MACRO(N, H, W, C, REPEAT_TIMES, float, RT_OPENCL);
#else
#define MACE_BM_CUSTOM_ADD(N, H, W, C, REPEAT_TIMES) \
  MACE_BM_CUSTOM_ADD_MACRO(N, H, W, C, REPEAT_TIMES, float, RT_CPU);
#endif

//MACE_BM_CUSTOM_ADD(1, 256, 256, 4, 8);
//MACE_BM_CUSTOM_ADD(1, 256, 256, 4, 64);
//MACE_BM_CUSTOM_ADD(1, 256, 256, 4, 256);
//MACE_BM_CUSTOM_ADD(1, 256, 256, 4, 1024);
//MACE_BM_CUSTOM_ADD(1, 256, 256, 4, 4096);
MACE_BM_CUSTOM_ADD(1, 512, 512, 4, 16384);

}  // namespace test
}  // namespace ops
}  // namespace mace
