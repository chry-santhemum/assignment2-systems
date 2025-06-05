import torch
import math


class FlashAttentionPyTorch(torch.autograd.Function):
    """
    PyTorch implementation of Flash Attention 2 foward pass.
    Incredibly slow but demonstrates the idea.
    """
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # Set these values for now
        # Assume they divide the seq_len, so no out of bounds errors
        Q_TILE_SIZE = 16
        KV_TILE_SIZE = 16

        # Shapes: [B, N, D]
        batch_size = Q.shape[0]
        N_Q, N_KV = Q.shape[1], K.shape[1]
        assert N_KV == V.shape[1]
        # move tensors to cuda if not already
        Q, K, V = Q.cuda(), K.cuda(), V.cuda()
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Expected CUDA tensors"
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), "Expected contiguous tensors"

        out = torch.zeros_like(Q)
        L = torch.zeros(batch_size, N_Q, device=Q.device)

        for i in range(math.ceil(N_Q / Q_TILE_SIZE)):
            Q_tile = Q[:, i*Q_TILE_SIZE : (i+1)*Q_TILE_SIZE, :]

            m_running = torch.full((batch_size, Q_TILE_SIZE, 1), float("-inf"), device=Q.device)
            denom_running = torch.zeros(batch_size, Q_TILE_SIZE, 1, device=Q.device)
            out_running = torch.zeros_like(Q_tile)

            for j in range(math.ceil(N_KV / KV_TILE_SIZE)):
                K_tile = K[:, j*KV_TILE_SIZE : (j+1)*KV_TILE_SIZE, :]
                V_tile = V[:, j*KV_TILE_SIZE : (j+1)*KV_TILE_SIZE, :]

                S_tile = torch.einsum("bnd, bmd->bnm", Q_tile, K_tile) / math.sqrt(Q_tile.shape[-1])
                if is_causal:
                    # determine which elements to mask
                    Q_range = torch.arange(i*Q_TILE_SIZE, (i+1)*Q_TILE_SIZE, device=Q.device)
                    K_range = torch.arange(j*KV_TILE_SIZE, (j+1)*KV_TILE_SIZE, device=Q.device)
                    mask_tile = (Q_range[:, None] < K_range[None, :]).unsqueeze(0)
                    S_tile = S_tile.masked_fill(mask_tile, 1e-6)

                m_new = torch.max(m_running, torch.max(S_tile, dim=-1, keepdim=True)[0])

                P_tile = torch.exp(S_tile - m_new)
                denom_running = torch.exp(m_running - m_new) * denom_running + P_tile.sum(dim=-1, keepdim=True)
                out_running = P_tile @ V_tile + torch.einsum("bnm, bmd -> bnd", torch.diag_embed(torch.exp(m_running - m_new).squeeze(-1)), out_running)
                m_running = m_new
            
            out_tile = out_running / denom_running
            out[:, i*Q_TILE_SIZE : (i+1)*Q_TILE_SIZE, :] = out_tile
            L[:, i*Q_TILE_SIZE : (i+1)*Q_TILE_SIZE] = torch.log(denom_running.squeeze(-1)) + m_running.squeeze(-1)
            ctx.save_for_backward(Q.cpu(), K.cpu(), V.cpu(), L.cpu())

        return out.cpu()

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
    