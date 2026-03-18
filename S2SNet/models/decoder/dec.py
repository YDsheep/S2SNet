import torch
import torch.nn as nn
import torch.nn.functional as F

class SSU(nn.Module):
    def __init__(self, dim_in):
        """
        Shifted Sequence Upsampler (SSU)
        """
        super().__init__()
        self.dim = dim_in

        self.pe = nn.Parameter(torch.randn(1, 1, 1, 4, self.dim) * 0.02)

        self.proj_seq = nn.Parameter(torch.empty(4, 5))
        nn.init.xavier_uniform_(self.proj_seq)

        self.proj_feat = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
            nn.GELU()
        )

        self.alpha = nn.Parameter(torch.zeros(1))

    def _project_grid(self, F_H, F_L):

        B, C, H, W = F_H.shape
        F_L_seq = F_L.contiguous().view(B, C, H, 2, W, 2).permute(0, 2, 4, 3, 5, 1).contiguous().view(B, H, W, 4, C)
        F_L_seq = F_L_seq + self.pe

        # (B, C, H, W) -> (B, H, W, 1, C)
        F_H_seq = F_H.permute(0, 2, 3, 1).unsqueeze(3)
        S = torch.cat([F_H_seq, F_L_seq], dim=3)

        # S: (B, H, W, 5, C), W: (4, 5) -> Y: (B, H, W, 4, C)
        Y = torch.einsum('b h w s c, k s -> b h w k c', S, self.proj_seq)
        Y = self.proj_feat(Y)

        # (B, H, W, 4, C) -> (B, C, 2H, 2W)
        Out = Y.view(B, H, W, 2, 2, C).permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, 2 * H, 2 * W)
        return Out

    def forward(self, F_H, F_L):
        """
        F_H:(B, C, H, W)
        F_L:(B, C, 2H, 2W)
        """
        out_A = self._project_grid(F_H, F_L)

        F_L_shifted = torch.roll(F_L, shifts=(-1, -1), dims=(2, 3))
        F_H_pad = F.pad(F_H, (0, 1, 0, 1), mode='replicate')
        F_H_shifted = F.avg_pool2d(F_H_pad, kernel_size=2, stride=1)
        out_B_shifted = self._project_grid(F_H_shifted, F_L_shifted)
        out_B = torch.roll(out_B_shifted, shifts=(1, 1), dims=(2, 3))

        out_final = out_A + self.alpha * out_B

        return out_final


if __name__ == "__main__":
    B, C, H, W = 2, 64, 16, 16
    F_H = torch.randn(B, C, H, W)
    F_L = torch.randn(B, C, H * 2, W * 2)

    ssp_up = SSPUp(dim_in=C)

    out = ssp_up(F_H, F_L)

    print(f"F_H shape: {F_H.shape}")
    print(f"F_L shape: {F_L.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output requires grad: {out.requires_grad}")