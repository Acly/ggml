#include "ggml-impl.h"
#include "ggml-blas.h"
#include "ggml-backend-impl.h"

#include <future>
#include <vector>
#include <cstring>

#if defined(GGML_BLAS_USE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#elif defined(GGML_BLAS_USE_MKL)
#   include <mkl.h>
#elif defined(GGML_BLAS_USE_BLIS)
#   include <blis.h>
#elif defined(GGML_BLAS_USE_NVPL)
#   include <nvpl_blas.h>
#else
#   include <cblas.h>
#endif

struct ggml_backend_blas_context {
    int n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<char[]> work_data;
    size_t work_size = 0;
#ifndef GGML_USE_OPENMP
    std::vector<std::future<void>> tasks;
#endif
};

static void ggml_backend_blas_mul_mat(ggml_backend_blas_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    const int64_t ne_plane      = ne01*ne00;
    const size_t  desired_wsize = type == GGML_TYPE_F32 ? 0 : ne03*ne02*ne_plane*sizeof(float);

    if (ctx->work_size < desired_wsize) {
        ctx->work_data.reset(new char[desired_wsize]);
        ctx->work_size = desired_wsize;
    }
    void * wdata = ctx->work_data.get();

    // convert src0 to float
    if (type != GGML_TYPE_F32) {
        const auto * type_traits = ggml_get_type_traits(type);
        ggml_to_float_t const to_float = type_traits->to_float;

        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                const void  *       x      = (char *)  src0->data + i02*nb02          + i03*nb03;
                      float * const wplane = (float *) wdata      + i02*ne_plane      + i03*ne02*ne_plane;

                const int min_cols_per_thread = 4096;
                const int min_rows_per_thread = std::max((int)(min_cols_per_thread/ne00), 1);
                const int n_threads = std::max(std::min(ctx->n_threads, (int)(ne01/min_rows_per_thread)), 1);

#ifdef GGML_USE_OPENMP
                #pragma omp parallel for num_threads(n_threads)
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    to_float((const char *) x + i01*nb01, wplane + i01*ne00, ne00);
                }
#else
                for (int i = 1; i < n_threads; i++) {
                    const int64_t start =       i*ne01/n_threads;
                    const int64_t end   = (i + 1)*ne01/n_threads;
                    if (start < end) {
                        ctx->tasks.push_back(std::async(std::launch::async, [=]() {
                            for (int64_t i01 = start; i01 < end; i01++) {
                                to_float((const char *) x + i01*nb01, wplane + i01*ne00, ne00);
                            }
                        }));
                    }
                }
                {
                    // reuse the current thread for the first task
                    const int64_t start = 0;
                    const int64_t end   = ne01/n_threads;
                    for (int64_t i01 = start; i01 < end; i01++) {
                        to_float((const char *) x + i01*nb01, wplane + i01*ne00, ne00);
                    }
                }
#endif
            }
        }

#ifndef GGML_USE_OPENMP
        // wait for all tasks to finish
        for (auto & task : ctx->tasks) {
            task.get();
        }
        ctx->tasks.clear();
#endif
    }

#if defined(OPENBLAS_VERSION)
    openblas_set_num_threads(ctx->n_threads);
#endif

#if defined(GGML_BLAS_USE_BLIS)
    bli_thread_set_num_threads(ctx->n_threads);
#endif

#if defined(GGML_BLAS_USE_NVPL)
    nvpl_blas_set_num_threads(ctx->n_threads);
#endif

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const int64_t i03 = i13/r3;
            const int64_t i02 = i12/r2;

            const float * x = (float *) ((char *) src0->data + i02*nb02 + i03*nb03);
            const float * y = (float *) ((char *) src1->data + i12*nb12 + i13*nb13);
                  float * d = (float *) ((char *)  dst->data + i12*nb2  + i13*nb3);

            if (type != GGML_TYPE_F32) {
                x = (float *) wdata + i02*ne_plane + i03*ne02*ne_plane;
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        ne1, ne01, ne10,
                        1.0f,   y, ne10,
                                x, ne00,
                        0.0f,   d, ne01);
        }
    }
}

static void ggml_backend_blas_out_prod(ggml_backend_blas_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(ne0  == ne00);
    GGML_ASSERT(ne1  == ne10);
    GGML_ASSERT(ne2  == ne02);
    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne3  == ne13);
    GGML_ASSERT(ne03 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == sizeof(float));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    // Arguments to ggml_compute_forward_out_prod (expressed as major,minor)
    // src0: (k,n)
    // src1: (k,m)
    // dst:  (m,n)
    //
    // Arguments to sgemm (see https://github.com/Reference-LAPACK/lapack/blob/master/BLAS/SRC/sgemm.f)
    // Also expressed as (major,minor)
    // a: (m,k): so src1 transposed
    // b: (k,n): so src0
    // c: (m,n)
    //
    // However, if ggml_is_transposed(src1) is true, then
    // src1->data already contains a transposed version, so sgemm mustn't
    // transpose it further.

    int n = src0->ne[0];
    int k = src0->ne[1];
    int m = src1->ne[0];

    CBLAS_TRANSPOSE transposeA;
    int lda;

    if (!ggml_is_transposed(src1)) {
        transposeA = CblasTrans;
        lda = m;
    } else {
        transposeA = CblasNoTrans;
        lda = k;
    }

    float * a = (float *) ((char *) src1->data);
    float * b = (float *) ((char *) src0->data);
    float * c = (float *) ((char *) dst->data);

    cblas_sgemm(CblasRowMajor, transposeA, CblasNoTrans, m, n, k, 1.0, a, lda, b, n, 0.0, c, n);

    GGML_UNUSED(ctx);
}

namespace {

int64_t div_up(int64_t x, int64_t multiple) {
    return (x + multiple - 1) / multiple;
}

float * work_data_f32(ggml_backend_blas_context * ctx, int64_t num_elements) {
    int64_t size = num_elements * sizeof(float);
    if (size > ctx->work_size) {
        ctx->work_data.reset(new char[size]);
        ctx->work_size = size;
    }
    return (float *) ctx->work_data.get();
}

} // namespace

static void ggml_backend_blas_conv_2d(ggml_backend_blas_context * ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[1];
    const ggml_tensor * kernel = dst->src[0];

    const int32_t stride_x = dst->op_params[0];
    const int32_t stride_y = dst->op_params[1];
    const int32_t pad_x = dst->op_params[2];
    const int32_t pad_y = dst->op_params[3];

    const int64_t c = src->ne[0]; // number of input channels
    GGML_ASSERT(c == kernel->ne[1]);
    const int64_t src_w = src->ne[1];
    const int64_t src_h = src->ne[2];
    const float * src_data = (const float *)src->data;
    const int64_t knl_count = kernel->ne[0]; // number of kernels / output channels
    const int64_t knl_w = kernel->ne[2];
    const int64_t knl_h = kernel->ne[3];
    const float * knl_data = (const float *)kernel->data;
    const int64_t dst_w = dst->ne[1];
    const int64_t dst_h = dst->ne[2];
    float       * dst_data = (float *)dst->data;

    // Handle 1x1 convolutions with stride 1 -> use gemm directly
    if (knl_w == 1 && knl_h == 1 && stride_x == 1 && stride_y == 1) {
        int m = dst->ne[3] * dst_w * dst_h;
        int n = knl_count;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, c, 1.0f, src_data, c, knl_data, n, 0.0f, dst_data, n);
        return;
    }

    // Rewrite the convolution as a matrix multiplication (im2col)
    // uses some parts from tensorflow's implementation
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/conv_ops_using_gemm.cc
    GGML_ASSERT(knl_w > 1 || knl_h > 1 || stride_x > 1);
    GGML_ASSERT(pad_x == 0 || pad_x == knl_w / 2);
    GGML_ASSERT(pad_y == 0 || pad_y == knl_h / 2);

    int knl_left_offset = (dst_w - 1) * stride_x + knl_w - src_w;
    int knl_top_offset = (dst_h - 1) * stride_y + knl_h - src_h;
    knl_left_offset = pad_x > 0 ? (knl_left_offset + 1) / 2 : knl_left_offset / 2;
    knl_top_offset = pad_y > 0 ? (knl_top_offset + 1) / 2 : knl_top_offset / 2;

    const int64_t max_chunk_size = 16 * 1024 * 1024;
    const int64_t knl_n = c * knl_w * knl_h;
    const int64_t patch_n = dst->ne[3] * dst_w * dst_h;
    const int64_t patches_per_chunk = max_chunk_size / (knl_n * sizeof(float));
    const int64_t values_per_chunk = div_up(max_chunk_size, sizeof(float));
    const int64_t chunk_n = div_up(patch_n, patches_per_chunk);
    float * tmp = work_data_f32(ctx, values_per_chunk); // for im2col result

    for (int64_t chunk_i = 0; chunk_i < chunk_n; ++chunk_i) {
        const int64_t patch_start = chunk_i * patches_per_chunk;
        const int64_t patch_end = std::min(patch_start + patches_per_chunk, patch_n);

        for (int64_t patch_i = patch_start; patch_i < patch_end; ++patch_i) {
            const int64_t batch = patch_i / (dst_w * dst_h);
            const int64_t dst_y = (patch_i / dst_w) % dst_h;
            const int64_t dst_x = patch_i % dst_w;
            const float * src_patch = (const float*)src->data + (batch * c * src_w * src_h);
            const int64_t src_y_origin = dst_y * stride_y - knl_top_offset;
            const int64_t src_x_origin = dst_x * stride_x - knl_left_offset;
            float * tmp_patch = tmp + ((patch_i % patches_per_chunk) * knl_n);

            for (int64_t knl_y = 0; knl_y < knl_h; ++knl_y) {
                const int64_t src_y = src_y_origin + knl_y;
                float * tmp_row = tmp_patch + (knl_y * knl_w * c);

                if (src_y < 0 || src_y >= src_h) {
                    memset(tmp_row, 0, knl_w * c * sizeof(float));
                } else {
                    const int64_t pad_left = std::max(0i64, -src_x_origin);
                    const int64_t pad_right = std::max(0i64, src_x_origin + knl_w - src_w);
                    const int64_t center = knl_w - pad_left - pad_right;
                    if (pad_left > 0) {
                        memset(tmp_row, 0, pad_left * c * sizeof(float));
                    }
                    if (center > 0) {
                        const int64_t src_x = std::max(0i64, src_x_origin);
                        const float * src_start = src_patch + (src_y * src_w + src_x) * c;
                        memcpy(tmp_row + pad_left * c, src_start, center * c * sizeof(float));
                    }
                    if (pad_right > 0) {
                        memset(tmp_row + (pad_left + center) * c, 0, pad_right * c * sizeof(float));
                    }
                }
            }
        } // for each patch

        int m = patch_end - patch_start;
        int n = knl_count;
        int k = knl_n;
        float * dst_chunk_data = dst_data + patch_start * knl_count;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, 1.0f, tmp, k, knl_data, n, 0.0f, dst_chunk_data, n);
    } // for each chunk    
}

// backend interface

static const char * ggml_backend_blas_get_name(ggml_backend_t backend) {
    return "BLAS";

    GGML_UNUSED(backend);
}

static void ggml_backend_blas_free(ggml_backend_t backend) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_blas_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_blas_mul_mat(ctx, node);
                break;

            case GGML_OP_OUT_PROD:
                ggml_backend_blas_out_prod(ctx, node);
                break;

            case GGML_OP_CONV_2D:
                ggml_backend_blas_conv_2d(ctx, node);
                break;

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

static struct ggml_backend_i blas_backend_i = {
    /* .get_name                = */ ggml_backend_blas_get_name,
    /* .free                    = */ ggml_backend_blas_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_blas_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

static ggml_guid_t ggml_backend_blas_guid(void) {
    static ggml_guid guid = { 0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    return &guid;
}

ggml_backend_t ggml_backend_blas_init(void) {
    ggml_backend_blas_context * ctx = new ggml_backend_blas_context;

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_blas_guid(),
        /* .interface = */ blas_backend_i,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_blas_reg(), 0),
        /* .context   = */ ctx,
    };

#if defined(OPENBLAS_VERSION) && defined(GGML_USE_OPENMP)
    if (openblas_get_parallel() != OPENBLAS_OPENMP) {
        GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but OpenBLAS was compiled without OpenMP support\n", __func__);
    }
#endif

#if defined(BLIS_ENABLE_CBLAS) && defined(GGML_USE_OPENMP) && !defined(BLIS_ENABLE_OPENMP)
    GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but BLIS was compiled without OpenMP support\n", __func__);
#endif

    return backend;
}

bool ggml_backend_is_blas(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_blas_guid());
}

void ggml_backend_blas_set_n_threads(ggml_backend_t backend_blas, int n_threads) {
    GGML_ASSERT(ggml_backend_is_blas(backend_blas));

    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend_blas->context;
    ctx->n_threads = n_threads;
}

// device interface

static const char * ggml_backend_blas_device_get_name(ggml_backend_dev_t dev) {
    return "BLAS";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_blas_device_get_description(ggml_backend_dev_t dev) {
    #if defined(GGML_BLAS_USE_ACCELERATE)
        return "Accelerate";
    #elif defined(GGML_BLAS_USE_MKL)
        return "MKL";
    #elif defined(GGML_BLAS_USE_BLIS)
        return "BLIS";
    #elif defined(GGML_BLAS_USE_NVPL)
        return "NVPL";
    #elif defined(OPENBLAS_VERSION)
        return "OpenBLAS";
    #else
        return "BLAS";
    #endif

    GGML_UNUSED(dev);
}

static void ggml_backend_blas_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // TODO
    *free = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_blas_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_blas_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_blas_device_get_name(dev);
    props->description = ggml_backend_blas_device_get_description(dev);
    props->type        = ggml_backend_blas_device_get_type(dev);
    ggml_backend_blas_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_blas_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_blas_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_blas_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_blas_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_blas_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
        {
            // BLAS usually is only faster for large matrices
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];

            const int64_t ne10 = src1->ne[0];

            const int64_t ne0 = op->ne[0];
            const int64_t ne1 = op->ne[1];

            // TODO: find the optimal value
            const int64_t min_batch = 32;

            return ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   ne0*ne1 >= min_batch*min_batch &&
                //    (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != NULL);
        }

        case GGML_OP_OUT_PROD:
            return op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_matrix(src0) &&
                   ggml_is_matrix(src1) &&
                   ggml_is_contiguous(src0) &&
                   (ggml_is_contiguous(src1) || ggml_is_transposed(src1)) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != NULL);

        case GGML_OP_CONV_2D:
            return true;


        default:
            return false;

    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_blas_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_blas_device_i = {
    /* .get_name             = */ ggml_backend_blas_device_get_name,
    /* .get_description      = */ ggml_backend_blas_device_get_description,
    /* .get_memory           = */ ggml_backend_blas_device_get_memory,
    /* .get_type             = */ ggml_backend_blas_device_get_type,
    /* .get_props            = */ ggml_backend_blas_device_get_props,
    /* .init_backend         = */ ggml_backend_blas_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_blas_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_blas_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_blas_device_supports_op,
    /* .supports_buft        = */ ggml_backend_blas_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

static const char * ggml_backend_blas_reg_get_name(ggml_backend_reg_t reg) {
    return "BLAS";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_blas_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_blas_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_blas_device = {
        /* .iface   = */ ggml_backend_blas_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_blas_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static void * ggml_backend_blas_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *)ggml_backend_blas_set_n_threads;
    }
    return NULL;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static const struct ggml_backend_reg_i ggml_backend_blas_reg_i = {
    /* .get_name         = */ ggml_backend_blas_reg_get_name,
    /* .get_device_count = */ ggml_backend_blas_reg_get_device_count,
    /* .get_device       = */ ggml_backend_blas_reg_get_device,
    /* .get_proc_address = */ ggml_backend_blas_get_proc_address,
};

ggml_backend_reg_t ggml_backend_blas_reg(void) {
    static struct ggml_backend_reg ggml_backend_blas_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_blas_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_blas_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_blas_reg)
