import torch
from torch import nn
from torch import autograd
import cupy as cp
import math


_GROUPED_BW_TIMING_ENABLED = False

_GROUPED_BW_TIMING = {
    "prepare": [],
    "sort": [],
    "group_meta": [],
    "grouped_kernel": [],
}

_FM_GROUPED_BW_TIMING = {
    "prepare": [],
    "sort": [],
    "group_meta": [],
    "grouped_kernel": [],
}


def set_grouped_bw_timing(enabled):
    global _GROUPED_BW_TIMING_ENABLED
    _GROUPED_BW_TIMING_ENABLED = enabled


def get_grouped_bw_timing():
    return _GROUPED_BW_TIMING


def get_fm_grouped_bw_timing():
    return _FM_GROUPED_BW_TIMING


def clear_grouped_bw_timing():
    for values in _GROUPED_BW_TIMING.values():
        values.clear()

    for values in _FM_GROUPED_BW_TIMING.values():
        values.clear()


def _find_nearest_divisor(value, target):
    divisors = []
    for i in range(1, value+1):
        if value % i == 0:
            divisors.append((i, abs(target-i)))
    divisors.sort(key=lambda x: x[1])
    return divisors[0][0]


_num_threads_forward_cache = dict()


def _get_num_threads_for_forward(output_size):
    optimal_num_threads = 512
    if output_size not in _num_threads_forward_cache:
        _num_threads_forward_cache[output_size] = _find_nearest_divisor(output_size, optimal_num_threads)

    return _num_threads_forward_cache[output_size]


_num_threads_backward_cache = dict()


def _get_num_threads_for_backward(output_size):
    optimal_num_threads = 512
    if output_size not in _num_threads_backward_cache:
        _num_threads_backward_cache[output_size] = _find_nearest_divisor(output_size, optimal_num_threads)

    return _num_threads_backward_cache[output_size]


def _kernel_with_threads(kernel, threads):
    def f(grid, args):
        kernel(grid=grid, block=threads, args=args)
    return f


_feature_transformer_slice_forward_kernel_cache = dict()


def make_feature_transformer_slice_forward_kernel(max_active_features, output_size):
    '''
        @param: max_active_features
            The maximum number of features that are active
            (non-zero) for a single position. This value determines
            the shape of the inputs.
            This value is of type uint32_t.

        @param: output_size
            The number of outputs. Must match the shape of weights
            and biases.
            This value is of type uint32.
    '''
    num_threads = _get_num_threads_for_forward(output_size)
    output_thread_slice_size = output_size // num_threads
    key = (max_active_features, output_size, num_threads)
    if key not in _feature_transformer_slice_forward_kernel_cache:
        kernel = cp.RawKernel(r'''

typedef unsigned int uint32_t;
typedef int int32_t;

extern "C" __global__

/*
    @assumptions:
        The blocks must have dimensionality (BATCH_SIZE,)
        The threads must have dimensionality (N,), where
        N * output_thread_slice_size == output_size.

    @param: feature_indices
        A matrix of shape (BATCH_SIZE, max_active_features)
        containing indices of active features for each position
        in a batch. Feature index of -1 means that the slot is empty
        and the weights will not be accumulated for it. Moreover
        no further indices from this block will be considered.
        The indices form an implicit matrix of shape
        (BATCH_SIZE, NUM_INPUTS), where the first dimension index is
        inferred from the memory location (BATCH_SIZE), and the
        second dimension index is stored in the feature_indices matrix.
        The type for feature indices is int32_t.

    @param: feature_values
        A matrix of shape (BATCH_SIZE, max_active_features)
        containing the values (arity) of the corresponding
        feature index in feature_indices.
        The type for the feature value (arity) is float32.

    @param: weight
        The weight matrix of shape (NUM_INPUTS, output_size).
        Weights must be of type float32.

    @param: bias
        The bias vector of shape (output_size,).
        Bias values must be of type float32.

    @param: output
        An output matrix of shape (BATCH_SIZE, output_size).
        It may not be initialized, bias is always copied
        to the output first.
        Output values must have type float32.
*/
void feature_transformer_slice_forward(
    const int32_t* const feature_indices,
    const float*   const feature_values,
    const float*   const weight,
    const float*   const bias,
          float*   const output
) {{
    __shared__
          float          shared_output[{output_size}];

    const uint32_t       block_idx           = blockIdx.x;
    const uint32_t       slice_offset        = threadIdx.x * {output_thread_slice_size};

          float*   const output_slice        = output + block_idx * {output_size} + slice_offset;
    const float*   const bias_slice          = bias                               + slice_offset;
          float*         shared_output_slice = shared_output                      + slice_offset;

    const int32_t* const feature_index_row   = feature_indices + block_idx * {max_active_features};
    const float*   const feature_value_row   = feature_values  + block_idx * {max_active_features};

    #pragma unroll
    for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
    {{
        shared_output_slice[s] = bias_slice[s];
    }}

    for (uint32_t k = 0; k < {max_active_features}; ++k)
    {{
        const int32_t feature_index = feature_index_row[k];
        const float   feature_value = feature_value_row[k];
        if (feature_index != -1)
        {{
            const float* const weight_slice = weight + feature_index * {output_size} + slice_offset;
            #pragma unroll
            for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
            {{
                shared_output_slice[s] += weight_slice[s] * feature_value;
            }}
        }} else break;
    }}

    #pragma unroll
    for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
    {{
        output_slice[s] = shared_output_slice[s];
    }}
}}

'''.format(
            max_active_features=max_active_features,
            output_thread_slice_size=output_thread_slice_size,
            output_size=output_size),
            'feature_transformer_slice_forward')
        kernel.compile()
        _feature_transformer_slice_forward_kernel_cache[key] = _kernel_with_threads(kernel, (num_threads,))
    return _feature_transformer_slice_forward_kernel_cache[key]


_feature_transformer_slice_backward_kernel_cache = dict()


_double_feature_transformer_slice_backward_grouped_kernel_cache = {}


def make_double_feature_transformer_slice_backward_kernel(
    max_active_features,
    output_size,
):
    """
    Version 2A:
        sort + grouped CUDA reduction

    1 block = 1 unique feature

    block内:
        output方向を128 threadsで分担

    feature occurrence方向:
        group内を各threadがserial reduction

    weight_gradへのatomicAdd:
        0回
    """

    num_threads = 128

    key = (
        max_active_features,
        output_size,
        num_threads,
    )

    if key not in _double_feature_transformer_slice_backward_grouped_kernel_cache:

        kernel_code = r'''
typedef unsigned int uint32_t;
typedef int int32_t;

extern "C" __global__
void double_feature_transformer_slice_backward_grouped(
    const int32_t* __restrict__ sorted_occurrences,

    const int32_t* __restrict__ unique_features,
    const int32_t* __restrict__ group_starts,
    const int32_t* __restrict__ group_counts,

    const float* __restrict__ feature_values_0,
    const float* __restrict__ feature_values_1,

    const float* __restrict__ output_grad_0,
    const float* __restrict__ output_grad_1,

    float* __restrict__ weight_grad,

    const uint32_t batch_size
)
{{
    const uint32_t group_id = blockIdx.x;
    const uint32_t tid      = threadIdx.x;

    const int32_t feature_index =
        unique_features[group_id];

    const int32_t group_start =
        group_starts[group_id];

    const int32_t group_count =
        group_counts[group_id];

    const int32_t occurrences_per_set =
        (int32_t)(batch_size * {max_active_features});


    // --------------------------------------------------------
    // Output dimension tiling
    //
    // output_size = 1280
    // threads     = 128
    //
    // each thread:
    //   10 output dimensions
    // --------------------------------------------------------

    for (
        uint32_t out = tid;
        out < {output_size};
        out += {num_threads}
    )
    {{
        float acc = 0.0f;


        // ----------------------------------------------------
        // Reduce all occurrences of this feature
        // ----------------------------------------------------

        for (
            int32_t j = 0;
            j < group_count;
            ++j
        )
        {{
            const int32_t sorted_pos =
                group_start + j;

            const int32_t occurrence =
                sorted_occurrences[sorted_pos];


            // ------------------------------------------------
            // Feature set 0
            // ------------------------------------------------

            if (occurrence < occurrences_per_set)
            {{
                const int32_t batch =
                    occurrence / {max_active_features};

                const int32_t slot =
                    occurrence -
                    batch * {max_active_features};

                const float fv =
                    feature_values_0[
                        batch * {max_active_features} + slot
                    ];

                const float og =
                    output_grad_0[
                        batch * {output_size} + out
                    ];

                acc += og * fv;
            }}

            // ------------------------------------------------
            // Feature set 1
            // ------------------------------------------------

            else
            {{
                const int32_t local =
                    occurrence - occurrences_per_set;

                const int32_t batch =
                    local / {max_active_features};

                const int32_t slot =
                    local -
                    batch * {max_active_features};

                const float fv =
                    feature_values_1[
                        batch * {max_active_features} + slot
                    ];

                const float og =
                    output_grad_1[
                        batch * {output_size} + out
                    ];

                acc += og * fv;
            }}
        }}


        // ----------------------------------------------------
        // IMPORTANT:
        //
        // NO atomicAdd
        //
        // One block exclusively owns one feature.
        // One thread exclusively owns one output element.
        // ----------------------------------------------------

        weight_grad[
            feature_index * {output_size} + out
        ] = acc;
    }}
}}
'''

        kernel_code = kernel_code.format(
            max_active_features=max_active_features,
            output_size=output_size,
            num_threads=num_threads,
        )

        kernel = cp.RawKernel(
            kernel_code,
            'double_feature_transformer_slice_backward_grouped'
        )

        kernel.compile()

        _double_feature_transformer_slice_backward_grouped_kernel_cache[key] = \
            _kernel_with_threads(
                kernel,
                (num_threads,)
        )

    return _double_feature_transformer_slice_backward_grouped_kernel_cache[key]


_fm_embedding_grouped_backward_kernel_cache = {}


def make_fm_embedding_grouped_backward_kernel(
    max_active_features,
    factor_dim,
):
    num_threads = 32
    key = (
        max_active_features,
        factor_dim,
        num_threads,
    )

    if key not in _fm_embedding_grouped_backward_kernel_cache:
        kernel_code = r'''
typedef unsigned int uint32_t;
typedef int int32_t;

extern "C" __global__
void fm_embedding_grouped_backward(
    const int32_t* __restrict__ sorted_occurrences,
    const int32_t* __restrict__ unique_features,
    const int32_t* __restrict__ group_starts,
    const int32_t* __restrict__ group_counts,
    const float* __restrict__ grad_output_0,
    const float* __restrict__ grad_output_1,
    float* __restrict__ v_grad,
    const uint32_t batch_size
)
{{
    const uint32_t group_id = blockIdx.x;
    const uint32_t tid = threadIdx.x;

    const int32_t feature_index = unique_features[group_id];
    const int32_t group_start = group_starts[group_id];
    const int32_t group_count = group_counts[group_id];
    const int32_t occurrences_per_set =
        (int32_t)(batch_size * {max_active_features});

    for (
        uint32_t dim = tid;
        dim < {factor_dim};
        dim += {num_threads}
    )
    {{
        float acc = 0.0f;

        for (int32_t j = 0; j < group_count; ++j)
        {{
            const int32_t occurrence =
                sorted_occurrences[group_start + j];

            if (occurrence < occurrences_per_set)
            {{
                acc += grad_output_0[
                    occurrence * {factor_dim} + dim
                ];
            }}
            else
            {{
                const int32_t local =
                    occurrence - occurrences_per_set;

                acc += grad_output_1[
                    local * {factor_dim} + dim
                ];
            }}
        }}

        v_grad[
            feature_index * {factor_dim} + dim
        ] = acc;
    }}
}}
'''

        kernel_code = kernel_code.format(
            max_active_features=max_active_features,
            factor_dim=factor_dim,
            num_threads=num_threads,
        )

        kernel = cp.RawKernel(
            kernel_code,
            'fm_embedding_grouped_backward',
        )
        kernel.compile()

        _fm_embedding_grouped_backward_kernel_cache[key] = \
            _kernel_with_threads(
                kernel,
                (num_threads,),
        )

    return _fm_embedding_grouped_backward_kernel_cache[key]


class FMEmbeddingGroupedFunction(autograd.Function):
    @staticmethod
    def forward(ctx, feature_indices_0, feature_indices_1, v):
        assert len(feature_indices_0.shape) == 2
        assert len(feature_indices_1.shape) == 2
        assert feature_indices_0.shape == feature_indices_1.shape
        assert feature_indices_0.dtype == torch.int32
        assert feature_indices_1.dtype == torch.int32
        assert len(v.shape) == 2
        assert v.dtype == torch.float32
        assert feature_indices_0.is_cuda
        assert feature_indices_1.is_cuda
        assert v.is_cuda
        assert feature_indices_0.device == feature_indices_1.device
        assert v.device == feature_indices_0.device
        assert feature_indices_0.is_contiguous()
        assert feature_indices_1.is_contiguous()
        assert v.is_contiguous()

        num_inputs = v.shape[0]

        idx0 = torch.clamp(
            feature_indices_0,
            0,
            num_inputs - 1,
        )
        idx1 = torch.clamp(
            feature_indices_1,
            0,
            num_inputs - 1,
        )

        valid0 = (feature_indices_0 >= 0).unsqueeze(-1)
        valid1 = (feature_indices_1 >= 0).unsqueeze(-1)

        with torch.profiler.record_function(
            "NNUE/fm_embedding_forward_idx0"
        ):
            v_feat0_indexed = v[idx0]
        v_feat0 = v_feat0_indexed * valid0

        with torch.profiler.record_function(
            "NNUE/fm_embedding_forward_idx1"
        ):
            v_feat1_indexed = v[idx1]
        v_feat1 = v_feat1_indexed * valid1

        ctx.save_for_backward(
            feature_indices_0,
            feature_indices_1,
        )
        ctx.num_inputs = num_inputs
        ctx.factor_dim = v.shape[1]

        return v_feat0, v_feat1

    @staticmethod
    def backward(ctx, grad_output_0, grad_output_1):
        global _GROUPED_BW_TIMING_ENABLED
        global _FM_GROUPED_BW_TIMING

        timing = _GROUPED_BW_TIMING_ENABLED

        assert not ctx.needs_input_grad[0]
        assert not ctx.needs_input_grad[1]

        with torch.profiler.record_function(
            "NNUE/fm_grouped_prepare"
        ):
            if timing:
                ev_prepare_start = torch.cuda.Event(enable_timing=True)
                ev_prepare_end = torch.cuda.Event(enable_timing=True)
                ev_prepare_start.record()

            grad_output_0 = grad_output_0.contiguous()
            grad_output_1 = grad_output_1.contiguous()

            (
                feature_indices_0,
                feature_indices_1,
            ) = ctx.saved_tensors

            device = feature_indices_0.device
            batch_size = feature_indices_0.shape[0]
            max_active_features = feature_indices_0.shape[1]
            total_per_set = batch_size * max_active_features

            v_grad = torch.zeros(
                ctx.num_inputs,
                ctx.factor_dim,
                dtype=torch.float32,
                device=device,
            )

            feature_ids_0 = torch.clamp(
                feature_indices_0.reshape(-1),
                0,
                ctx.num_inputs - 1,
            )
            feature_ids_1 = torch.clamp(
                feature_indices_1.reshape(-1),
                0,
                ctx.num_inputs - 1,
            )

            all_feature_ids = torch.cat(
                (
                    feature_ids_0,
                    feature_ids_1,
                ),
                dim=0,
            )

            all_original_ids = torch.cat(
                (
                    feature_indices_0.reshape(-1),
                    feature_indices_1.reshape(-1),
                ),
                dim=0,
            )
            valid = all_original_ids >= 0
            valid_feature_ids = all_feature_ids[valid]

            all_occurrences = torch.arange(
                2 * total_per_set,
                dtype=torch.int32,
                device=device,
            )
            valid_occurrences = all_occurrences[valid]

            if timing:
                ev_prepare_end.record()
                _FM_GROUPED_BW_TIMING["prepare"].append(
                    (ev_prepare_start, ev_prepare_end)
                )

        with torch.profiler.record_function(
            "NNUE/fm_grouped_sort"
        ):
            if timing:
                ev_sort_start = torch.cuda.Event(enable_timing=True)
                ev_sort_end = torch.cuda.Event(enable_timing=True)
                ev_sort_start.record()

            sorted_feature_ids, sort_order = torch.sort(
                valid_feature_ids
            )
            sorted_occurrences = valid_occurrences[sort_order]

            if timing:
                ev_sort_end.record()
                _FM_GROUPED_BW_TIMING["sort"].append(
                    (ev_sort_start, ev_sort_end)
                )

        with torch.profiler.record_function(
            "NNUE/fm_grouped_group_meta"
        ):
            if timing:
                ev_group_start = torch.cuda.Event(enable_timing=True)
                ev_group_end = torch.cuda.Event(enable_timing=True)
                ev_group_start.record()

            unique_features, group_counts = torch.unique_consecutive(
                sorted_feature_ids,
                return_counts=True,
            )
            group_counts = group_counts.to(dtype=torch.int32)
            group_starts = (
                torch.cumsum(
                    group_counts,
                    dim=0,
                    dtype=torch.int32,
                )
                - group_counts
            )
            num_groups = unique_features.numel()

            if timing:
                ev_group_end.record()
                _FM_GROUPED_BW_TIMING["group_meta"].append(
                    (ev_group_start, ev_group_end)
                )

        if num_groups > 0:
            kernel = make_fm_embedding_grouped_backward_kernel(
                max_active_features,
                ctx.factor_dim,
            )

            with torch.profiler.record_function(
                "NNUE/fm_grouped_kernel"
            ):
                if timing:
                    ev_kernel_start = torch.cuda.Event(enable_timing=True)
                    ev_kernel_end = torch.cuda.Event(enable_timing=True)
                    ev_kernel_start.record()

                kernel(
                    grid=(int(num_groups),),
                    args=(
                        sorted_occurrences.data_ptr(),
                        unique_features.data_ptr(),
                        group_starts.data_ptr(),
                        group_counts.data_ptr(),
                        grad_output_0.data_ptr(),
                        grad_output_1.data_ptr(),
                        v_grad.data_ptr(),
                        batch_size,
                    ),
                )

                if timing:
                    ev_kernel_end.record()
                    _FM_GROUPED_BW_TIMING["grouped_kernel"].append(
                        (ev_kernel_start, ev_kernel_end)
                    )

        return None, None, v_grad


def debug_compare_fm_embedding_grouped(
    feature_indices_0,
    feature_indices_1,
    v,
    grad_output_0,
    grad_output_1,
):
    reference_v = v.detach().clone().requires_grad_(True)
    grouped_v = v.detach().clone().requires_grad_(True)

    num_inputs = reference_v.shape[0]
    idx0 = torch.clamp(
        feature_indices_0,
        0,
        num_inputs - 1,
    )
    idx1 = torch.clamp(
        feature_indices_1,
        0,
        num_inputs - 1,
    )
    valid0 = (feature_indices_0 >= 0).unsqueeze(-1)
    valid1 = (feature_indices_1 >= 0).unsqueeze(-1)

    reference_output_0 = reference_v[idx0] * valid0
    reference_output_1 = reference_v[idx1] * valid1

    grouped_output_0, grouped_output_1 = \
        FMEmbeddingGroupedFunction.apply(
            feature_indices_0,
            feature_indices_1,
            grouped_v,
        )

    reference_grad, = torch.autograd.grad(
        outputs=(reference_output_0, reference_output_1),
        inputs=reference_v,
        grad_outputs=(grad_output_0, grad_output_1),
    )
    grouped_grad, = torch.autograd.grad(
        outputs=(grouped_output_0, grouped_output_1),
        inputs=grouped_v,
        grad_outputs=(grad_output_0, grad_output_1),
    )

    def make_metrics(actual, reference):
        abs_diff = torch.abs(actual - reference)
        reference_norm = torch.linalg.vector_norm(reference)
        relative_diff = (
            torch.linalg.vector_norm(abs_diff)
            / torch.clamp(
                reference_norm,
                min=torch.finfo(reference.dtype).eps,
            )
        )
        return {
            "max_abs_diff": abs_diff.max().item(),
            "mean_abs_diff": abs_diff.mean().item(),
            "relative_diff": relative_diff.item(),
        }

    results = {
        "v_feat0": make_metrics(
            grouped_output_0,
            reference_output_0,
        ),
        "v_feat1": make_metrics(
            grouped_output_1,
            reference_output_1,
        ),
        "v_grad": make_metrics(
            grouped_grad,
            reference_grad,
        ),
    }

    print("[FM grouped correctness]")
    for name, metrics in results.items():
        print(f"  {name}")
        print(
            f"    max abs diff : "
            f"{metrics['max_abs_diff']:.9e}"
        )
        print(
            f"    mean abs diff: "
            f"{metrics['mean_abs_diff']:.9e}"
        )
        print(
            f"    relative diff: "
            f"{metrics['relative_diff']:.9e}"
        )

    return results


class DoubleFeatureTransformerSliceFunction(autograd.Function):

    @staticmethod
    def forward(ctx, feature_indices_0, feature_values_0, feature_indices_1, feature_values_1, weight, bias):
        ctx.save_for_backward(feature_indices_0, feature_values_0, feature_indices_1, feature_values_1, weight, bias)

        assert len(feature_indices_0.shape) == 2
        assert len(feature_values_0.shape) == 2
        assert feature_indices_0.shape[0] == feature_values_0.shape[0]
        assert feature_indices_0.shape[1] == feature_values_0.shape[1]
        assert feature_indices_0.dtype == torch.int32
        assert feature_values_0.dtype == torch.float32

        assert len(feature_indices_1.shape) == 2
        assert len(feature_values_1.shape) == 2
        assert feature_indices_1.shape[0] == feature_values_1.shape[0]
        assert feature_indices_1.shape[1] == feature_values_1.shape[1]
        assert feature_indices_1.dtype == torch.int32
        assert feature_values_1.dtype == torch.float32

        assert len(weight.shape) == 2
        assert weight.dtype == torch.float32

        assert len(bias.shape) == 1
        assert bias.dtype == torch.float32

        assert feature_indices_0.is_cuda
        assert feature_values_0.is_cuda
        assert feature_indices_1.is_cuda
        assert feature_values_1.is_cuda
        assert weight.is_cuda
        assert bias.is_cuda

        assert feature_values_0.device == feature_indices_0.device
        assert feature_values_1.device == feature_indices_1.device
        assert feature_indices_0.device == feature_indices_1.device
        assert weight.device == feature_indices_0.device
        assert bias.device == feature_indices_0.device

        assert feature_indices_0.is_contiguous()
        assert feature_values_0.is_contiguous()
        assert feature_indices_1.is_contiguous()
        assert feature_values_1.is_contiguous()
        assert weight.is_contiguous()
        assert bias.is_contiguous()

        device = feature_indices_0.device
        batch_size = feature_indices_0.shape[0]
        max_active_features = feature_indices_0.shape[1]
        output_size = weight.shape[1]

        output0 = torch.empty(batch_size, output_size, dtype=torch.float32, device=device, requires_grad=True)
        output1 = torch.empty(batch_size, output_size, dtype=torch.float32, device=device, requires_grad=True)

        kernel = make_feature_transformer_slice_forward_kernel(max_active_features, output_size)
        kernel(
            grid=(batch_size,),
            args=(
                feature_indices_0.data_ptr(),
                feature_values_0.data_ptr(),
                weight.data_ptr(),
                bias.data_ptr(),
                output0.data_ptr()
            )
        )

        kernel(
            grid=(batch_size,),
            args=(
                feature_indices_1.data_ptr(),
                feature_values_1.data_ptr(),
                weight.data_ptr(),
                bias.data_ptr(),
                output1.data_ptr()
            )
        )

        return output0, output1

    @staticmethod
    def backward(ctx, grad_output_0, grad_output_1):
        global _GROUPED_BW_TIMING_ENABLED
        global _GROUPED_BW_TIMING

        timing = _GROUPED_BW_TIMING_ENABLED

        # ==========================================================
        # 1. prepare
        #
        # ・contiguous
        # ・saved tensor展開
        # ・weight_grad / bias_grad確保
        # ・bias gradient
        # ・flatten
        # ・padding除去
        # ・occurrence生成
        # ==========================================================

        if timing:
            ev_prepare_start = torch.cuda.Event(enable_timing=True)
            ev_prepare_end = torch.cuda.Event(enable_timing=True)
            ev_prepare_start.record()

        assert not ctx.needs_input_grad[0]
        assert not ctx.needs_input_grad[1]
        assert not ctx.needs_input_grad[2]
        assert not ctx.needs_input_grad[3]

        grad_output_0 = grad_output_0.contiguous()
        grad_output_1 = grad_output_1.contiguous()

        (
            feature_indices_0,
            feature_values_0,
            feature_indices_1,
            feature_values_1,
            weight,
            bias,
        ) = ctx.saved_tensors

        device = feature_indices_0.device

        batch_size = feature_indices_0.shape[0]
        max_active_features = feature_indices_0.shape[1]
        output_size = weight.shape[1]

        assert feature_indices_1.shape[0] == batch_size
        assert feature_indices_1.shape[1] == max_active_features

        # ====================================================
        # weight_grad / bias_grad
        # ====================================================

        weight_grad = torch.zeros(
            weight.shape[0],
            weight.shape[1],
            dtype=torch.float32,
            device=device,
        )

        bias_grad = torch.zeros(
            output_size,
            dtype=torch.float32,
            device=device,
        )

        # ====================================================
        # bias gradient
        # ====================================================

        bias_grad.copy_(
            grad_output_0.sum(dim=0)
            + grad_output_1.sum(dim=0)
        )

        # ====================================================
        # feature indices flatten
        # ====================================================

        total_per_set = batch_size * max_active_features

        feature_ids_0 = feature_indices_0.reshape(-1)
        feature_ids_1 = feature_indices_1.reshape(-1)

        all_feature_ids = torch.cat(
            (
                feature_ids_0,
                feature_ids_1,
            ),
            dim=0,
        )

        # ====================================================
        # padding(-1)除外
        # ====================================================

        valid = all_feature_ids >= 0

        valid_feature_ids = all_feature_ids[valid]

        all_occurrences = torch.arange(
            2 * total_per_set,
            dtype=torch.int32,
            device=device,
        )

        valid_occurrences = all_occurrences[valid]

        if timing:
            ev_prepare_end.record()
            _GROUPED_BW_TIMING["prepare"].append(
                (ev_prepare_start, ev_prepare_end)
            )

        # ==========================================================
        # 2. sort
        #
        # feature ID sort
        # occurrenceの並び替えまで含める
        # ==========================================================

        if timing:
            ev_sort_start = torch.cuda.Event(enable_timing=True)
            ev_sort_end = torch.cuda.Event(enable_timing=True)
            ev_sort_start.record()

        sorted_feature_ids, sort_order = torch.sort(
            valid_feature_ids
        )

        sorted_occurrences = valid_occurrences[
            sort_order
        ]

        if timing:
            ev_sort_end.record()
            _GROUPED_BW_TIMING["sort"].append(
                (ev_sort_start, ev_sort_end)
            )

        # ==========================================================
        # 3. group_meta
        #
        # unique feature検出
        # group_counts
        # group_starts
        # ==========================================================

        if timing:
            ev_group_start = torch.cuda.Event(enable_timing=True)
            ev_group_end = torch.cuda.Event(enable_timing=True)
            ev_group_start.record()

        unique_features, group_counts = torch.unique_consecutive(
            sorted_feature_ids,
            return_counts=True,
        )

        group_counts = group_counts.to(
            dtype=torch.int32
        )

        group_starts = (
            torch.cumsum(
                group_counts,
                dim=0,
                dtype=torch.int32,
            )
            - group_counts
        )

        num_groups = unique_features.numel()

        if timing:
            ev_group_end.record()
            _GROUPED_BW_TIMING["group_meta"].append(
                (ev_group_start, ev_group_end)
            )

        # ==========================================================
        # 4. grouped CUDA kernel
        #
        # この中に
        #   ・同一featureのgradient reduction
        #   ・weight_gradへの最終書き込み
        # が両方含まれる
        # ==========================================================

        if num_groups > 0:

            kernel = make_double_feature_transformer_slice_backward_kernel(
                max_active_features,
                output_size,
            )

            if timing:
                ev_kernel_start = torch.cuda.Event(enable_timing=True)
                ev_kernel_end = torch.cuda.Event(enable_timing=True)
                ev_kernel_start.record()

            kernel(
                grid=(int(num_groups),),
                args=(
                    sorted_occurrences.data_ptr(),

                    unique_features.data_ptr(),
                    group_starts.data_ptr(),
                    group_counts.data_ptr(),

                    feature_values_0.data_ptr(),
                    feature_values_1.data_ptr(),

                    grad_output_0.data_ptr(),
                    grad_output_1.data_ptr(),

                    weight_grad.data_ptr(),

                    batch_size,
                ),
            )

            if timing:
                ev_kernel_end.record()
                _GROUPED_BW_TIMING["grouped_kernel"].append(
                    (ev_kernel_start, ev_kernel_end)
                )

        # ==========================================================
        # debug
        # ==========================================================

        if torch.rand(()) < 0.001:
            print(
                f"[Grouped BW] "
                f"occurrences={valid_feature_ids.numel():,} "
                f"unique_features={num_groups:,} "
                f"ratio={num_groups / max(valid_feature_ids.numel(), 1):.4f}"
            )

        return (
            None,
            None,
            None,
            None,
            weight_grad,
            bias_grad,
        )


class DoubleFeatureTransformerSlice(nn.Module):
    # factor_dimを追加
    def __init__(self, num_inputs, num_outputs, factor_dim):
        super(DoubleFeatureTransformerSlice, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.factor_dim = factor_dim

        sigma = math.sqrt(1/num_inputs)
        self.weight = nn.Parameter(torch.rand(num_inputs, num_outputs, dtype=torch.float32) * (2 * sigma) - sigma)
        self.bias = nn.Parameter(torch.rand(num_outputs, dtype=torch.float32) * (2 * sigma) - sigma)

        # self.vを追加
        self.v = nn.Parameter((torch.rand(num_inputs, factor_dim, dtype=torch.float32) * (2 * sigma) - sigma) * 0.1)

    def forward(self, feature_indices_0, feature_values_0, feature_indices_1, feature_values_1):
        # 1. 既存の高速な線形パス
        t_self, t_opp = DoubleFeatureTransformerSliceFunction.apply(
            feature_indices_0, feature_values_0,
            feature_indices_1, feature_values_1,
            self.weight, self.bias
        )

        # 2. FM要素の抽出
        v_feat0, v_feat1 = FMEmbeddingGroupedFunction.apply(
            feature_indices_0,
            feature_indices_1,
            self.v,
        )

        return t_self, t_opp, v_feat0, v_feat1


if __name__ == '__main__':
    import time
    import sys
    import os

    def FeatureTransformerSliceFunctionEmulate(feature_indices, feature_values, weight, bias):
        batch_size = feature_indices.shape[0]
        num_inputs = weight.shape[0]
        max_active_features = feature_indices.shape[1]
        inputs = torch.zeros(batch_size, num_inputs, dtype=torch.float32, device=weight.device)
        for i in range(batch_size):
            for j in range(max_active_features):
                feature = feature_indices[i, j]
                value = feature_values[i, j]
                inputs[i, feature] += value

        return torch.mm(inputs, weight) + bias

    def test():
        BATCH_SIZE = 16
        INPUT_SIZE = 10
        MAX_ACTIVE_FEATURES = 32
        STRIDE = 128
        MAX_ERROR = 1e-4

        torch.manual_seed(0)
        weight0 = torch.rand(INPUT_SIZE, STRIDE, dtype=torch.float32, requires_grad=True)
        bias0 = torch.rand(STRIDE, dtype=torch.float32, requires_grad=True)
        torch.manual_seed(0)
        weight1 = torch.rand(INPUT_SIZE, STRIDE, dtype=torch.float32, requires_grad=True)
        bias1 = torch.rand(STRIDE, dtype=torch.float32, requires_grad=True)
        indices0 = (torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES) * INPUT_SIZE).to(dtype=torch.int32)
        indices1 = (torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES) * INPUT_SIZE).to(dtype=torch.int32)
        values0 = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32)
        values1 = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32)

        output00 = FeatureTransformerSliceFunctionEmulate(indices0.clone(), values0.clone(), weight0, bias0)
        output01 = FeatureTransformerSliceFunctionEmulate(indices1.clone(), values1.clone(), weight0, bias0)
        # output10 = FeatureTransformerSliceFunction.apply(indices0.clone().cuda(), values0.clone().cuda(), weight1.cuda(), bias1.cuda())
        # output11 = FeatureTransformerSliceFunction.apply(indices1.clone().cuda(), values1.clone().cuda(), weight1.cuda(), bias1.cuda())
        output10, output11 = DoubleFeatureTransformerSliceFunction.apply(indices0.clone().cuda(), values0.clone().cuda(), indices1.clone().cuda(), values1.clone().cuda(), weight1.cuda(), bias1.cuda())

        assert torch.max(output00.cpu() - output10.cpu()) < MAX_ERROR
        assert torch.max(output01.cpu() - output11.cpu()) < MAX_ERROR
        (output00 - output01).sum().backward()
        (output10 - output11).sum().backward()
        assert torch.max(weight0.grad.cpu() - weight1.grad.cpu()) < MAX_ERROR
        assert torch.max(bias0.grad.cpu() - bias1.grad.cpu()) < MAX_ERROR
        print('Tests passed.')

    def bench():
        INPUT_SIZE = 40960
        BATCH_SIZE = 8192
        ITERS = 64
        STRIDE = 264
        MAX_ACTIVE_FEATURES = 64

        layer = DoubleFeatureTransformerSlice(INPUT_SIZE, STRIDE).cuda()
        indices0 = torch.cat([torch.sort((torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES * 3 // 4) * INPUT_SIZE), dim=1)[0].to(dtype=torch.int32), torch.full((BATCH_SIZE, MAX_ACTIVE_FEATURES // 4), -1, dtype=torch.int32)], dim=1).cuda()
        values0 = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32).cuda()
        indices1 = torch.cat([torch.sort((torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES * 3 // 4)) * INPUT_SIZE, dim=1)[0].to(dtype=torch.int32), torch.full((BATCH_SIZE, MAX_ACTIVE_FEATURES // 4), -1, dtype=torch.int32)], dim=1).cuda()
        values1 = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32).cuda()

        output0, output1 = layer(indices0, values0, indices1, values1)

        device = indices0.device

        start = time.time()

        for i in range(ITERS):
            output0, output1 = layer(indices0, values0, indices1, values1)
            output0 = torch.clamp(output0, 0.0, 1.0)
            output1 = torch.clamp(output1, 0.0, 1.0)

            g = ((output0 - output1)**2).mean()
            g.backward()

            torch.cuda.synchronize()

        end = time.time()

        # for param in layer.parameters():
        #    print(param.grad)

        print('{} pos/s'.format((ITERS * BATCH_SIZE) / (end - start)))

    test()
    bench()
