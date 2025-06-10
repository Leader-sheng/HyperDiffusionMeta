import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicalConstraintModule(nn.Module):
    def __init__(self, grid_size=24, time_steps=4):
        super().__init__()
        # 
        self.interaction_matrix = nn.Parameter(
            torch.nn.init.orthogonal_(torch.empty(grid_size, grid_size)) * 0.5)

        self.temporal_weights = nn.Parameter(torch.ones(time_steps))

        # 
        self.adaptor = nn.Sequential(
            nn.Conv3d(1, 16, (2, 3, 3), padding=(0, 1, 1)),
            nn.GELU(),
            nn.Conv3d(16, 32, (3, 3, 3), padding=(1, 1, 1)),
            nn.GELU(),
            nn.Conv3d(32, 1, (1, 3, 3), padding=(0, 1, 1))
        )
        nn.init.constant_(self.temporal_weights, 1.0)
        self.grid_size = grid_size

    def forward(self, features):
        features = features.to(torch.float32)
        disp_region = features[..., 24:, 24:]  # [B,T,C,H,W]
        stress_region = features[..., :24, 24:]
        disp_region = disp_region.permute(0, 2, 1, 3, 4)  # [B,1,T,24,24]

        disp_processed = self.adaptor(disp_region)  # =4
        disp_processed = disp_processed * self.temporal_weights.view(1, 1, -1, 1, 1)

        predicted_stress = torch.einsum('ij,bcthw->btcij',
                                        self.interaction_matrix,
                                        disp_processed)

        #
        t_steps = predicted_stress.size(1)
        target_stress = stress_region[:, :t_steps]  # [B,4,1,24,24]

        # 
        mid_feat = self.adaptor[0:2](disp_region)
        mid_loss = 0.2 * torch.norm(mid_feat, p=2)

        # 
        base_loss = F.mse_loss(predicted_stress, target_stress)
        smooth_loss = 0.01 * torch.norm(self.interaction_matrix.diff(dim=1), p=2)
        temporal_loss = 0.01 * torch.norm(disp_processed.diff(dim=2), p=1)

        return base_loss + smooth_loss + temporal_loss + mid_loss



