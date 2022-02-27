#ifndef MACE_OPS_OPENCL_IMAGE_CUSTOM_ADD_H_
#define MACE_OPS_OPENCL_IMAGE_CUSTOM_ADD_H_

#include <vector>

#include "mace/core/ops/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/opencl/custom_add.h"
#include "mace/runtimes/opencl/core/opencl_helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

class CustomAddKernel : public OpenCLCustomAddKernel {
 public:
  MaceStatus Compute(OpContext *context,
                     const std::vector<const Tensor *> &input_tensors,
                     int repeat_times,
                     Tensor *output_tensor) override;

 private:
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
};

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_CUSTOM_ADD_H_
