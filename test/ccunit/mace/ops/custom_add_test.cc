#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class CustomAddOpTest : public OpsTestBase {};

namespace {
template <RuntimeType D>
void SimpleCustomAdd() {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("CustomAdd", "CustomAddTest")
      .Input("Input1")
      .Input("Input2")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<D, float>("Input1", {1, 2, 3, 1}, {1, 2, 3, 4, 5, 6});
  net.AddInputFromArray<D, float>("Input2", {1, 2, 3, 1}, {1, 2, 3, 4, 5, 6});

  // Run
  net.RunOp(D);

  auto expected = net.CreateTensor<float>({1, 2, 3, 1}, {65, 130, 195, 260, 325, 390});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(CustomAddOpTest, SimpleCustomAdd) { SimpleCustomAdd<RuntimeType::RT_CPU>(); }

}  // namespace test
}  // namespace ops
}  // namespace mace
