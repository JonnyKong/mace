#ifndef MACE_OPS_OPENCL_CUSTOM_ADD_H_
#define MACE_OPS_OPENCL_CUSTOM_ADD_H_

#include "mace/public/mace.h"
#include "mace/utils/math.h"

namespace mace {

class OpContext;
class Tensor;

namespace ops {

class OpenCLCustomAddKernel {
 public:
  virtual MaceStatus Compute(
      OpContext *context,
      const std::vector<const Tensor *> &input_tensors,
      int repeat_times,
      Tensor *output_tensor) = 0;
  MACE_EMPTY_VIRTUAL_DESTRUCTOR(OpenCLCustomAddKernel);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_CUSTOM_ADD_H_
