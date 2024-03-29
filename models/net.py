from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import *
from .patchmatch import PatchMatch
from utils import *


class FeatureNet(nn.Module):
    """Feature Extraction Network: to extract features of original images from each view"""

    def __init__(self):
        """Initialize different layers in the network"""

        super(FeatureNet, self).__init__()

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        # [B,8,H,W]
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)
        # [B,16,H/2,W/2]
        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)
        # [B,32,H/4,W/4]
        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.conv7 = ConvBnReLU(32, 32, 3, 1, 1)
        # [B,64,H/8,W/8]
        self.conv8 = ConvBnReLU(32, 64, 5, 2, 2)
        self.conv9 = ConvBnReLU(64, 64, 3, 1, 1)
        self.conv10 = ConvBnReLU(64, 64, 3, 1, 1)

        self.output1 = nn.Conv2d(64, 64, 1, bias=False)
        self.inner1 = nn.Conv2d(32, 64, 1, bias=True)
        self.inner2 = nn.Conv2d(16, 64, 1, bias=True)
        self.output2 = nn.Conv2d(64, 32, 1, bias=False)
        self.output3 = nn.Conv2d(64, 16, 1, bias=False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward method

        Args:
            x: images from a single view, in the shape of [B, C, H, W]. Generally, C=3

        Returns:
            output_feature: a python dictionary contains extracted features from stage_1 to stage_3
                keys are "stage_1", "stage_2", and "stage_3"
        """
        output_feature = {}

        conv1 = self.conv1(self.conv0(x))
        conv4 = self.conv4(self.conv3(self.conv2(conv1)))

        conv7 = self.conv7(self.conv6(self.conv5(conv4)))
        conv10 = self.conv10(self.conv9(self.conv8(conv7)))

        output_feature["stage_3"] = self.output1(conv10)

        intra_feat = F.interpolate(conv10, scale_factor=2, mode="bilinear") + self.inner1(conv7)
        del conv7, conv10
        output_feature["stage_2"] = self.output2(intra_feat)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear") + self.inner2(conv4)
        del conv4
        output_feature["stage_1"] = self.output3(intra_feat)

        del intra_feat
        return output_feature


class Refinement(nn.Module):
    """Depth map refinement network"""

    def __init__(self):
        """Initialize"""

        super(Refinement, self).__init__()

        # img: [B,3,H,W]
        self.conv0 = ConvBnReLU(in_channels=3, out_channels=8)


        # depth map:[B,1,H/2,W/2]
        self.conv1 = ConvBnReLU(in_channels=1, out_channels=8)
        self.conv2 = ConvBnReLU(in_channels=8, out_channels=8)
        self.deconv = nn.ConvTranspose2d(
            in_channels=8, out_channels=8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
        )

        self.bn = nn.BatchNorm2d(8)
        self.bn16 = nn.BatchNorm2d(16)
        self.bn32 = nn.BatchNorm2d(32)
        self.bn64 = nn.BatchNorm2d(64)
        self.conv3 = ConvBnReLU(in_channels=16, out_channels=8)
        self.res = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1, bias=False)

        self.mlpconv0 = ConvBnReLU(16, 16, 3, 1, 1)
        self.mlpconv1 = ConvBnReLU(16, 32, 5, 2, 2) #W/2
        self.mlpconv2 = ConvBnReLU(32, 32, 3, 1, 1)
        self.mlpconv3 = ConvBnReLU(32, 64, 5, 2, 2) #W/4
        self.mlpconv4 = ConvBnReLU(64, 64, 3, 1, 1)
        self.mlpconv5 = ConvBnReLU(64, 128, 5, 2, 2) #W/8
        self.mlpconv6 = ConvBnReLU(128, 128, 3, 1, 1)

        self.mlpdeconv6 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
        )
        self.mlpdeconv4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
        )
        self.mlpdeconv3 = nn.ConvTranspose2d(
            in_channels=64, out_channels=16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
        )

        self.mlpconvOut = nn.Conv2d(32, 6, 1)
        self.activationTanh = nn.Tanh()
        self.activationSigmoid = nn.Sigmoid()

    def forward(
            self, img: torch.Tensor, depth_0: torch.Tensor, depth_min: torch.Tensor, depth_max: torch.Tensor
    ) -> torch.Tensor:
        """Forward method

        Args:
            img: input reference images (B, 3, H, W)
            depth_0: current depth map (B, 1, H//2, W//2)
            depth_min: pre-defined minimum depth (B, )
            depth_max: pre-defined maximum depth (B, )

        Returns:
            depth: refined depth map (B, 1, H, W)
        """

        batch_size = depth_min.size()[0]
        # pre-scale the depth map into [0,1]
        depth = (depth_0 - depth_min.view(batch_size, 1, 1, 1)) / (
                depth_max.view(batch_size, 1, 1, 1) - depth_min.view(batch_size, 1, 1, 1)
        )

        conv0 = self.conv0(img)
        deconv = F.relu(self.bn(self.deconv(self.conv2(self.conv1(depth)))), inplace=True)

        cat = torch.cat((deconv, conv0), dim=1)

        feature_conv0 = self.mlpconv0(cat)
        feature_conv1 = self.mlpconv2(self.mlpconv1(feature_conv0))
        feature_conv2 = self.mlpconv4(self.mlpconv3(feature_conv1))
        feature_conv3 = self.mlpconv6(self.mlpconv5(feature_conv2))

        feature_deconv1 = F.relu(self.bn64(self.mlpdeconv6(feature_conv3)), inplace=True)
        feature_cat1 = torch.cat((feature_conv2, feature_deconv1), dim=1)
        feature_deconv2 = F.relu(self.bn32(self.mlpdeconv4(feature_cat1)), inplace=True)

        feature_cat2 = torch.cat((feature_conv1, feature_deconv2), dim=1)
        feature_deconv3 = F.relu(self.bn16(self.mlpdeconv3(feature_cat2)), inplace=True)

        feature_cat3 = torch.cat((feature_conv0, feature_deconv3), dim=1)
        feature = self.mlpconvOut(feature_cat3)

        del deconv, conv0, feature_cat2, feature_cat1, feature_deconv2, feature_deconv1
        del feature_conv0, feature_conv1, feature_conv2, cat

        eps = 1e-2  # 1e-3 in case of gaussian distribution
        self.mu0 = self.activationTanh(torch.unsqueeze(feature[:, 0, :], 1))
        self.mu1 = self.activationTanh(torch.unsqueeze(feature[:, 1, :], 1))

        self.sigma0 = torch.clamp(self.activationSigmoid(torch.unsqueeze(feature[:, 2, :], 1)), eps, 1.0)*(depth_max.view(batch_size, 1, 1, 1) - depth_min.view(batch_size, 1, 1, 1))
        self.sigma1 = torch.clamp(self.activationSigmoid(torch.unsqueeze(feature[:, 3, :], 1)), eps, 1.0)*(depth_max.view(batch_size, 1, 1, 1) - depth_min.view(batch_size, 1, 1, 1))

        self.pi0 = self.activationSigmoid(torch.unsqueeze(feature[:, 4, :], 1))
        final_edge = self.activationSigmoid(torch.unsqueeze(feature[:, 5, :], 1))

        self.pi1 = 1. - self.pi0


        # Mode with the highest density value as final prediction
        u_mask = (self.pi0 / self.sigma0 > self.pi1 / self.sigma1).float()

        res = self.mu0 * u_mask + self.mu1 * (1. - u_mask)
        depth_interpolated = F.interpolate(depth, scale_factor=2, mode="nearest")
        depth = depth_interpolated + res
        depth = depth * (
                depth_max.view(batch_size, 1, 1, 1) - depth_min.view(batch_size, 1, 1, 1)) + depth_min.view(
            batch_size, 1, 1, 1)

        self.mu0 = depth_interpolated + self.mu0
        self.mu0 = self.mu0 * (
                depth_max.view(batch_size, 1, 1, 1) - depth_min.view(batch_size, 1, 1, 1)) + depth_min.view(
            batch_size, 1, 1, 1)
        self.mu1 = depth_interpolated + self.mu1
        self.mu1 = self.mu1 * (
                depth_max.view(batch_size, 1, 1, 1) - depth_min.view(batch_size, 1, 1, 1)) + depth_min.view(
            batch_size, 1, 1, 1)

        return depth,final_edge,u_mask,self.pi0,self.mu0,self.mu1,self.sigma0,self.sigma1



class DDLMVS(nn.Module):
    """ Implementation of complete structure of DDLMVS"""

    def __init__(
            self,
            patchmatch_interval_scale: List[float] = [0.005, 0.0125, 0.025],
            propagation_range: List[int] = [6, 4, 2],
            patchmatch_iteration: List[int] = [1, 2, 2],
            patchmatch_num_sample: List[int] = [8, 8, 16],
            propagate_neighbors: List[int] = [0, 8, 16],
            evaluate_neighbors: List[int] = [9, 9, 9],
    ) -> None:
        """Initialize modules in DDLMVS

        Args:
            patchmatch_interval_scale: depth interval scale in patchmatch module
            propagation_range: propagation range
            patchmatch_iteration: patchmatch interation number
            patchmatch_num_sample: patchmatch number of samples
            propagate_neighbors: number of propagation neigbors
            evaluate_neighbors: number of propagation neigbors for evaluation
        """
        super(DDLMVS, self).__init__()

        self.stages = 4
        self.feature = FeatureNet()
        self.patchmatch_num_sample = patchmatch_num_sample

        num_features = [8, 16, 32, 64]

        self.propagate_neighbors = propagate_neighbors
        self.evaluate_neighbors = evaluate_neighbors
        # number of groups for group-wise correlation
        self.G = [4, 8, 8]

        for i in range(self.stages - 1):

            if i == 2:
                patchmatch = PatchMatch(
                    random_initialization=True,
                    propagation_out_range=propagation_range[i],
                    patchmatch_iteration=patchmatch_iteration[i],
                    patchmatch_num_sample=patchmatch_num_sample[i],
                    patchmatch_interval_scale=patchmatch_interval_scale[i],
                    num_feature=num_features[i + 1],
                    G=self.G[i],
                    propagate_neighbors=self.propagate_neighbors[i],
                    stage=i + 1,
                    evaluate_neighbors=evaluate_neighbors[i],
                )
            else:
                patchmatch = PatchMatch(
                    random_initialization=False,
                    propagation_out_range=propagation_range[i],
                    patchmatch_iteration=patchmatch_iteration[i],
                    patchmatch_num_sample=patchmatch_num_sample[i],
                    patchmatch_interval_scale=patchmatch_interval_scale[i],
                    num_feature=num_features[i + 1],
                    G=self.G[i],
                    propagate_neighbors=self.propagate_neighbors[i],
                    stage=i + 1,
                    evaluate_neighbors=evaluate_neighbors[i],
                )
            setattr(self, f"patchmatch_{i + 1}", patchmatch)
        self.upsample_net = Refinement()

    def forward(
            self,
            imgs: Dict[str, torch.Tensor],
            # edges: Dict[str, torch.Tensor],
            proj_matrices: Dict[str, torch.Tensor],
            depth_min: torch.Tensor,
            depth_max: torch.Tensor
    ) -> Dict[str, Any]:
        """Forward method for DDLMVS

        Args:
            imgs: different stages of images (B, 3, H, W) stored in the dictionary
            proj_matrics: different stages of camera projection matrices (B, 4, 4) stored in the dictionary
            depth_min: minimum virtual depth (B, )
            depth_max: maximum virtual depth (B, )

        Returns:
            output dictionary of DDLMVS, containing refined depthmap, depth patchmatch
                and photometric_confidence.
        """
        imgs_0 = torch.unbind(imgs["stage_0"], 1)
        imgs_1 = torch.unbind(imgs["stage_1"], 1)
        imgs_2 = torch.unbind(imgs["stage_2"], 1)
        imgs_3 = torch.unbind(imgs["stage_3"], 1)
        del imgs

        self.imgs_0_ref = imgs_0[0]
        self.imgs_1_ref = imgs_1[0]
        self.imgs_2_ref = imgs_2[0]
        self.imgs_3_ref = imgs_3[0]
        del imgs_1, imgs_2, imgs_3

        self.proj_matrices_0 = torch.unbind(proj_matrices["stage_0"].float(), 1)
        self.proj_matrices_1 = torch.unbind(proj_matrices["stage_1"].float(), 1)
        self.proj_matrices_2 = torch.unbind(proj_matrices["stage_2"].float(), 1)
        self.proj_matrices_3 = torch.unbind(proj_matrices["stage_3"].float(), 1)
        del proj_matrices

        assert len(imgs_0) == len(self.proj_matrices_0), "Different number of images and projection matrices"

        # step 1. Multi-scale feature extraction
        features = []
        for img in imgs_0:
            output_feature = self.feature(img)
            features.append(output_feature)
        del imgs_0
        ref_feature, src_features = features[0], features[1:]

        depth_min = depth_min.float()
        depth_max = depth_max.float()

        # step 2. Learning-based patchmatch
        depth = None
        view_weights = None
        depth_patchmatch = {}
        refined_depth = {}
        refined_edge = {}

        for l in reversed(range(1, self.stages)):
            src_features_l = [src_fea[f"stage_{l}"] for src_fea in src_features]
            projs_l = getattr(self, f"proj_matrices_{l}")
            ref_proj, src_projs = projs_l[0], projs_l[1:]

            if l > 1:
                depth, _, view_weights = getattr(self, f"patchmatch_{l}")(
                    ref_feature=ref_feature[f"stage_{l}"],
                    src_features=src_features_l,
                    ref_proj=ref_proj,
                    src_projs=src_projs,
                    depth_min=depth_min,
                    depth_max=depth_max,
                    depth=depth,
                    img=getattr(self, f"imgs_{l}_ref"),
                    view_weights=view_weights,
                )
            else:
                depth, score, _ = getattr(self, f"patchmatch_{l}")(
                    ref_feature=ref_feature[f"stage_{l}"],
                    src_features=src_features_l,
                    ref_proj=ref_proj,
                    src_projs=src_projs,
                    depth_min=depth_min,
                    depth_max=depth_max,
                    depth=depth,
                    img=getattr(self, f"imgs_{l}_ref"),
                    view_weights=view_weights,
                )

            del src_features_l, ref_proj, src_projs, projs_l

            depth_patchmatch[f"stage_{l}"] = depth

            depth = depth[-1].detach()
            if l > 1:
                # upsampling the depth map and pixel-wise view weight for next stage
                depth = F.interpolate(depth, scale_factor=2, mode="nearest")
                view_weights = F.interpolate(view_weights, scale_factor=2, mode="nearest")

        # step 3. Refinement
        depth,final_edge,pseudo_pi,pi,mu0,mu1,sigma0,sigma1 = self.upsample_net(self.imgs_0_ref, depth, depth_min, depth_max)
        refined_depth['stage_0'] = depth
        refined_edge['stage_0'] = final_edge

        del depth, ref_feature, src_features

        if self.training:
            return {
                "refined_depth": refined_depth,
                "depth_patchmatch": depth_patchmatch,
                "refined_edge": refined_edge,
                "pseudo_pi": pseudo_pi,
                "pi": pi,
                "mu0": mu0,
                "mu1": mu1,
                "sigma0" : sigma0,
                "sigma1" : sigma1
            }

        else:
            num_depth = self.patchmatch_num_sample[0]
            score_sum4 = 4 * F.avg_pool3d(
                F.pad(score.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0
            ).squeeze(1)
            # [B, 1, H, W]
            depth_index = depth_regression(
                score, depth_values=torch.arange(num_depth, device=score.device, dtype=torch.float)
            ).long()
            depth_index = torch.clamp(depth_index, 0, num_depth - 1)
            photometric_confidence = torch.gather(score_sum4, 1, depth_index)
            photometric_confidence = F.interpolate(photometric_confidence, scale_factor=2, mode="nearest")
            photometric_confidence = photometric_confidence.squeeze(1)

            return {
                "refined_depth": refined_depth,
                "depth_patchmatch": depth_patchmatch,
                "photometric_confidence": photometric_confidence,
                "refined_edge": refined_edge,
                "pseudo_pi": pseudo_pi,
                "pi": pi,
                "mu0": mu0,
                "mu1": mu1,
                "sigma0" : sigma0,
                "sigma1" : sigma1
            }


def DDLMVS_loss(
        depth_patchmatch: Dict[str, torch.Tensor],
        refined_depth: Dict[str, torch.Tensor],
        edges_est: Dict[str, torch.Tensor],
        depth_gt: Dict[str, torch.Tensor],
        mask: Dict[str, torch.Tensor],
        pi: Dict[str, torch.Tensor],
        mu0: Dict[str, torch.Tensor],
        mu1: Dict[str, torch.Tensor],
        sigma0: Dict[str, torch.Tensor],
        sigma1: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """DDLMVS loss function

    Args:
        depth_patchmatch: depth map predicted by ddlmvs
        refined_depth: refined depth map predicted by ddlmvs
        edges_est: edge map predicted by ddlmvs
        depth_gt: ground truth depth map
        mask: mask for filter valid points
        pi: pi map predicted by ddlmvs
        mu0: mu0 map predicted by ddlmvs
        mu1: mu1 map predicted by ddlmvs
        sigma0: sigma0 map predicted by ddlmvs
        sigma1: sigma1 map predicted by ddlmvs

    Returns:
        loss: result loss value
    """
    stage = 4
    # depth loss
    loss = 0
    for l in range(1, stage):
        depth_gt_l = depth_gt[f"stage_{l}"]
        mask_l = mask[f"stage_{l}"] > 0.5
        depth2 = depth_gt_l[mask_l]

        depth_patchmatch_l = depth_patchmatch[f"stage_{l}"]
        for i in range(len(depth_patchmatch_l)):
            depth1 = depth_patchmatch_l[i][mask_l]
            loss = loss + F.smooth_l1_loss(depth1, depth2, reduction="mean")

    l = 0
    depth_refined_l = refined_depth[f"stage_{l}"]
    depth_gt_l = depth_gt[f"stage_{l}"]
    mask_l = mask[f"stage_{l}"] > 0.5
    depth1 = depth_refined_l[mask_l]
    depth2 = depth_gt_l[mask_l]
    loss = loss + F.smooth_l1_loss(depth1, depth2, reduction="mean")
    main_loss = loss

    # edge smoothness loss
    laplacian_depthy = torch.abs(2*refined_depth[f'stage_{l}'][:,:,1:-1,:] - refined_depth[f'stage_{l}'][:,:,:-2,:] - refined_depth[f'stage_{l}'][:,:,2:,:])
    laplacian_depthx = torch.abs(2*refined_depth[f'stage_{l}'][:,:,:,1:-1] - refined_depth[f'stage_{l}'][:,:,:,:-2] - refined_depth[f'stage_{l}'][:,:,:,2:])

    depths_X = F.pad(torch.abs(2*depth_gt_l[ :,:,:, 1:-1] - depth_gt_l[ :,:,:, :-2]-depth_gt_l[ :,:,:, 2:]) > 5, (1, 1, 0, 0, 0, 0, 0, 0))
    depths_Y = F.pad(torch.abs(2*depth_gt_l[ :,:,1:-1, : ] - depth_gt_l[ :,:,:-2, :]-depth_gt_l[:,:,2:, :]) > 5, (0, 0, 1, 1, 0, 0, 0, 0))

    depths_GRAD=torch.logical_or(depths_Y,  depths_X).float()

    BETA = -20.
    tv_h = (mask_l[:,:,1:-1,:]*torch.abs(laplacian_depthy)*torch.exp(BETA*edges_est[f'stage_{l}'][:,:,1:-1,:])).sum()
    tv_w = (mask_l[:,:,:,1:-1]*torch.abs(laplacian_depthx)*torch.exp(BETA*edges_est[f'stage_{l}'][:,:,:,1:-1])).sum()

    EDGEL2SIM = 8* F.smooth_l1_loss(depths_GRAD, edges_est[f'stage_{l}'], reduction="mean") #5 12 oct
    TV2LOSS = 2.5*(tv_h+tv_w)/len(depth1) # 2500


    #Bimodal loss
    eps = 1e-5
    Laplacian0 = 0.5 * pi[mask_l]     * torch.exp(-(torch.abs(mu0[mask_l]-depth2)/sigma0[mask_l]))/sigma0[mask_l]
    Laplacian1 = 0.5 * (1-pi[mask_l]) * torch.exp(-(torch.abs(mu1[mask_l]-depth2)/sigma1[mask_l]))/sigma1[mask_l]
    ####Bimodal loss negative log likelihood of the weighted laplacians

    BIMODAL_LOSS = -1.* torch.log(Laplacian0 + Laplacian1+eps)
    BIMODAL_LOSS = BIMODAL_LOSS.sum()/len(depth1)

    # total loss
    loss = 2*loss  +TV2LOSS +EDGEL2SIM + BIMODAL_LOSS

    print("TOTAL LOSS ",loss," MAIN LOSS ",main_loss," EDGEL2SIM ",EDGEL2SIM, " TV2LOSS ",TV2LOSS," BIMODAL_LOSS ",BIMODAL_LOSS)

    return loss
