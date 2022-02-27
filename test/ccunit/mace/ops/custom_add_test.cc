#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class CustomAddOpTest : public OpsTestBase {};

namespace {
template <RuntimeType D>
void SimpleCustomAdd(int repeat_times) {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("CustomAdd", "CustomAddTest")
      .Input("Input1")
      .Input("Input2")
      .AddIntArg("repeat_times", repeat_times)
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  std::vector<float> input1 = {6, 5, 4, 3, 2, 1};
  std::vector<float> input2 = {1, 2, 3, 4, 5, 6};
  std::vector<float> expected_vec(input1.size());
  for (int i = 0; i < input1.size(); i++) {
    expected_vec[i] = input1[i] + input2[i] * repeat_times;
  }

  // Add input data
  net.AddInputFromArray<D, float>("Input1", {1, 2, 3, 1}, input1);
  net.AddInputFromArray<D, float>("Input2", {1, 2, 3, 1}, input2);

  // Run
  net.RunOp(D);

  auto expected =
      net.CreateTensor<float>({1, 2, 3, 1}, expected_vec);

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(CustomAddOpTest, CPUSimpleCustomAdd) {
  SimpleCustomAdd<RuntimeType::RT_CPU>(64);
}
TEST_F(CustomAddOpTest, GPUSimpleCustomAdd) {
  SimpleCustomAdd<RuntimeType::RT_OPENCL>(64);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
