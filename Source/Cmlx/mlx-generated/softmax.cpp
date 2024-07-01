namespace mlx::core::metal {

const char* softmax() {
  return R"preamble(
template <typename T>
inline T softmax_exp(T x) {
  return fast::exp(x);
}
template <typename T, typename AccT = T, int N_READS = SOFTMAX_N_READS>
[[kernel]] void softmax_single_row(
    const device T* in,
    device T* out,
    constant int& axis_size,
    uint gid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  int lid = _lid;
  constexpr int SIMD_SIZE = 32;
  threadgroup AccT local_max[SIMD_SIZE];
  threadgroup AccT local_normalizer[SIMD_SIZE];
  AccT ld[N_READS];
  in += gid * axis_size + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      ld[i] = AccT(in[i]);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      ld[i] = ((lid * N_READS + i) < axis_size) ? AccT(in[i])
                                                : Limits<AccT>::finite_min;
    }
  }
  if (simd_group_id == 0) {
    local_max[simd_lane_id] = Limits<AccT>::finite_min;
    local_normalizer[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  AccT maxval = Limits<AccT>::finite_min;
  for (int i = 0; i < N_READS; i++) {
    maxval = (maxval < ld[i]) ? ld[i] : maxval;
  }
  maxval = simd_max(maxval);
  if (simd_lane_id == 0) {
    local_max[simd_group_id] = maxval;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    maxval = simd_max(local_max[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_max[0] = maxval;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maxval = local_max[0];
  AccT normalizer = 0;
  for (int i = 0; i < N_READS; i++) {
    AccT exp_x = softmax_exp(ld[i] - maxval);
    ld[i] = exp_x;
    normalizer += exp_x;
  }
  normalizer = simd_sum(normalizer);
  if (simd_lane_id == 0) {
    local_normalizer[simd_group_id] = normalizer;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    normalizer = simd_sum(local_normalizer[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_normalizer[0] = normalizer;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer = 1 / local_normalizer[0];
  out += gid * axis_size + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      out[i] = T(ld[i] * normalizer);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        out[i] = T(ld[i] * normalizer);
      }
    }
  }
}
template <typename T, typename AccT = T, int N_READS = SOFTMAX_N_READS>
[[kernel]] void softmax_looped(
    const device T* in,
    device T* out,
    constant int& axis_size,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  in += gid * axis_size;
  constexpr int SIMD_SIZE = 32;
  threadgroup AccT local_max[SIMD_SIZE];
  threadgroup AccT local_normalizer[SIMD_SIZE];
  AccT prevmax;
  AccT maxval = Limits<AccT>::finite_min;
  AccT normalizer = 0;
  for (int r = 0; r < static_cast<int>(ceildiv(axis_size, N_READS * lsize));
       r++) {
    int offset = r * lsize * N_READS + lid * N_READS;
    AccT vals[N_READS];
    if (offset + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        vals[i] = AccT(in[offset + i]);
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        vals[i] = (offset + i < axis_size) ? AccT(in[offset + i])
                                           : Limits<AccT>::finite_min;
      }
    }
    prevmax = maxval;
    for (int i = 0; i < N_READS; i++) {
      maxval = (maxval < vals[i]) ? vals[i] : maxval;
    }
    normalizer *= softmax_exp(prevmax - maxval);
    for (int i = 0; i < N_READS; i++) {
      normalizer += softmax_exp(vals[i] - maxval);
    }
  }
  prevmax = maxval;
  maxval = simd_max(maxval);
  normalizer *= softmax_exp(prevmax - maxval);
  normalizer = simd_sum(normalizer);
  prevmax = maxval;
  if (simd_lane_id == 0) {
    local_max[simd_group_id] = maxval;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maxval = simd_max(local_max[simd_lane_id]);
  normalizer *= softmax_exp(prevmax - maxval);
  if (simd_lane_id == 0) {
    local_normalizer[simd_group_id] = normalizer;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer = simd_sum(local_normalizer[simd_lane_id]);
  normalizer = 1 / normalizer;
  out += gid * axis_size;
  for (int r = 0; r < static_cast<int>(ceildiv(axis_size, N_READS * lsize));
       r++) {
    int offset = r * lsize * N_READS + lid * N_READS;
    if (offset + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        out[offset + i] = T(softmax_exp(in[offset + i] - maxval) * normalizer);
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if (offset + i < axis_size) {
          out[offset + i] =
              T(softmax_exp(in[offset + i] - maxval) * normalizer);
        }
      }
    }
  }
}
)preamble";
}

} // namespace mlx::core::metal
