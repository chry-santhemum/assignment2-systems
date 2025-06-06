import triton
import triton.language as tl
import torch
import math

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    Q_tile = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # running max and denom
    m_running = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    denom_running = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    out_running = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    for key_tile_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_tile = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        V_tile = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        V_dtype = V_tile.dtype

        S_tile = tl.dot(Q_tile, K_tile.T) * scale
        if is_causal:
            # Calculate actual row and column indices
            row_start = query_tile_index * Q_TILE_SIZE
            col_start = key_tile_index * K_TILE_SIZE
            
            # Create index grids
            row_offsets = row_start + tl.arange(0, Q_TILE_SIZE)[:, None]
            col_offsets = col_start + tl.arange(0, K_TILE_SIZE)[None, :]
            
            # Apply causal mask
            causal_mask = row_offsets >= col_offsets
            S_tile = tl.where(causal_mask, S_tile, float('-inf'))

        m_new = tl.maximum(m_running, tl.max(S_tile, axis=1).to(tl.float32))  # new maximum
        P_tile = tl.exp(S_tile - m_new[:, None])
        denom_running = tl.exp(m_running - m_new) * denom_running + tl.sum(P_tile, axis=1).to(tl.float32)
        out_running = tl.dot(P_tile.to(V_dtype), V_tile).to(tl.float32) + tl.exp(m_running - m_new)[:, None] * out_running
        m_running = m_new

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_dtype = O_block_ptr.type.element_ty
    O_tile = (out_running / denom_running[:, None]).to(O_dtype)
    tl.store(O_block_ptr, O_tile, boundary_check=(0,))

    L_dtype = L_block_ptr.type.element_ty
    L_tile = tl.log(denom_running) + m_running
    tl.store(L_block_ptr, L_tile, boundary_check=(0,))


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Expected CUDA tensors"
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), "Expected contiguous tensors"

        N_QUERIES, N_KEYS = Q.shape[1], K.shape[1]
        D = Q.shape[-1]
        batch_size = Q.shape[0]
        Q_TILE_SIZE, K_TILE_SIZE = 16, 16
        num_Q_tiles = math.ceil(N_QUERIES / Q_TILE_SIZE)
        scale = 1 / math.sqrt(D)

        ctx.Q_TILE_SIZE = Q_TILE_SIZE
        ctx.K_TILE_SIZE = K_TILE_SIZE
        ctx.scale = scale

        O = torch.empty_like(Q)
        L = torch.empty((batch_size, N_QUERIES,), dtype=torch.float32, device=Q.device)

        flash_fwd_kernel[(num_Q_tiles, batch_size)](
            Q, K, V, 
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS,
            scale,
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )

        ctx.save_for_backward(Q, K, V, L)

        return O

