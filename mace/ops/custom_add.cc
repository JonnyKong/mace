// Add input2 to input1 multiple times

#include "mace/core/ops/operator.h"
#include "mace/core/registry/ops_registry.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/image/custom_add.h"
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

#ifdef MACE_ENABLE_OPENCL
template<>
class CustomAddOp<RuntimeType::RT_OPENCL, float> : public Operation {
 public:
  explicit CustomAddOp(OpConstructContext *context)
      : Operation(context) {
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::CustomAddKernel>();
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }
  MaceStatus Run(OpContext *context) override {
    Tensor *output_tensor = this->Output(0);
    size_t n = this->inputs_.size();
    for (size_t i = 1; i < n; ++i) {
      MACE_CHECK(inputs_[0]->dim_size() == inputs_[i]->dim_size());
      MACE_CHECK(inputs_[0]->size() == inputs_[i]->size())
        << "Input 0: " << MakeString(inputs_[0]->shape())
        << ", size: " << inputs_[0]->size() << ". Input " << i << ": "
        << MakeString(inputs_[i]->shape()) << ", size: " << inputs_[i]->size();
    }

    return kernel_->Compute(context, inputs_, output_tensor);
  }

 private:
  std::unique_ptr<OpenCLCustomAddKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

void RegisterCustomAdd(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "CustomAdd", CustomAddOp, RuntimeType::RT_CPU, float);
  MACE_REGISTER_GPU_OP(op_registry, "CustomAdd", CustomAddOp);
}

}  // namespace ops
}  // namespace mace
