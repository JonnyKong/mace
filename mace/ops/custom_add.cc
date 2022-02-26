// Add input2 to input1 multiple times

#include "mace/core/ops/operator.h"
#include "mace/core/registry/ops_registry.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/image/addn.h"
#endif  // MACE_ENABLE_OPENCL
#include "mace/utils/memory.h"


namespace mace {
namespace ops {

template <RuntimeType D, class T>
class CustomAddOp;

template <class T>
class CustomAddOp<RuntimeType::RT_CPU, T> : public Operation {
 public:
  explicit CustomAddOp(OpConstructContext *context) : Operation(context) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    Tensor *output = this->Output(0);
    MACE_RETURN_IF_ERROR(output->ResizeLike(inputs_[0]));
    const index_t size = output->size();

    auto output_data = output->mutable_data<T>();
    memset(static_cast<void *>(output_data), 0, size * sizeof(T));

    auto input_data_0 = inputs_[0]->template data<T>();
    auto input_data_1 = inputs_[1]->template data<T>();

    for (index_t j = 0; j < size; ++j) {
      output_data[j] = input_data_0[j];

      for (int k = 0; k < 64; k++)
        output_data[j] += input_data_1[j];
    }

    return MaceStatus::MACE_SUCCESS;
  }
};


void RegisterCustomAdd(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "CustomAdd", CustomAddOp, RuntimeType::RT_CPU, float);
  // MACE_REGISTER_GPU_OP(op_registry, "CustomAdd", CustomAddOp);
}

}  // namespace ops
}  // namespace mace
