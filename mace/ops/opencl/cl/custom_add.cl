#include <common.h>

__kernel void custom_add(OUT_OF_RANGE_PARAMS
                         GLOBAL_WORK_GROUP_SIZE_DIM2
                         __read_only image2d_t input0,
                         __read_only image2d_t input1,
                         __private const int repeat_times,
                         __write_only image2d_t output) {
  const int w = get_global_id(0);
  const int hb = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (w >= global_size_dim0 || hb >= global_size_dim1) return;
#endif

  DATA_TYPE4 in0 = READ_IMAGET(input0, SAMPLER, (int2)(w, hb));
  DATA_TYPE4 in1 = READ_IMAGET(input1, SAMPLER, (int2)(w, hb));
  DATA_TYPE4 out = in0;

  for (int i = 0; i < repeat_times; i++) {
    out = out + in1;
  }

  WRITE_IMAGET(output, (int2)(w, hb), out);
}
