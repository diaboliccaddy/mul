import torch
from fe_method.features import Features
from utils_loc.mvtec3d_util import *
import numpy as np
import math
import os
import random
from utils_loc.cpu_knn import fill_missing_values
import torch.nn as nn
import torch.nn.functional as F


class QFormerGate(nn.Module):
    def __init__(self, num_modalities=3, hidden_dim=32, num_queries=4):
        super().__init__()

        self.num_modalities = num_modalities
        self.num_queries = num_queries

        # learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(num_queries, hidden_dim)
        )

        # project modality scores
        self.modality_proj = nn.Linear(1, hidden_dim)

        # cross attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )

        # weight prediction
        self.weight_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modalities)
        )

    def forward(self, scores):
        """
        scores shape: [3] -> (xyz, rgb, infra)
        """

        scores = scores.view(1, -1, 1)

        modality_tokens = self.modality_proj(scores)

        queries = self.query_tokens.unsqueeze(0)

        attn_out, _ = self.attn(
            queries,
            modality_tokens,
            modality_tokens
        )

        pooled = attn_out.mean(dim=1)

        score_signal = scores.squeeze(-1)
        weights = self.weight_head(pooled)
        weights = weights + score_signal.detach()

        weights = F.softmax(weights / 0.3, dim=-1)

        final_score = torch.sum(weights * scores.squeeze(-1), dim=1)

        return final_score, weights



class CrossModalAttention(nn.Module):

    def __init__(self, dim=3):
        super().__init__()

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.scale = dim ** -0.5

    def forward(self, x):

        # x: [3]

        x = x.unsqueeze(0)  # [1,3]

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn = torch.softmax((q @ k.transpose(-1, -2)) * self.scale, dim=-1)

        out = attn @ v

        return x.squeeze(0) + out.squeeze(0)

##Single modality
class RGBFeatures(Features):
    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb = sample[0]
        infra = sample[1]
        pc = sample[2]
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,_ = self(sample[0], sample[1], sample[2])
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        self.patch_rgb_lib.append(rgb_patch)
        
    def predict(self, sample, label, pixel_mask):

        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
            
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,_ = self(rgb, infra, pc)
        
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        

        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s,pixel_mask)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,_ = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T


        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
 

        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()

        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)



   
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
  

        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
      
        

        
        s = torch.tensor([s_rgb])
        


        self.s_lib.append(s)

    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label,pixel_mask):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

    
        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
 
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
  

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))

        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')


        s = torch.tensor([s_rgb])

        #--------------------------------------------------------------
        # object-level preds  or labels
        self.image_preds.append(s.numpy())
        self.image_labels.append(label)

        # pixel-level preds  or labels
        ## RGB
        self.rgb_pixel_preds.extend(s_map_rgb.flatten().numpy())
        self.rgb_pixel_labels.extend(pixel_mask[0].flatten().numpy())
        
        self.rgb_predictions.append(s_map_rgb.detach().cpu().squeeze().numpy())
        self.rgb_gts.append(pixel_mask[0].detach().cpu().squeeze().numpy())

        ## Infra

        ## PC
     
        #-------------------------------------------------------------

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):
     
        
        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0) 

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0) 
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0) 
            w_dist = torch.cdist(m_star, self.patch_infra_lib)  

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)   

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_infra_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D)))) 
        s = w * s_star
        
        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear',align_corners=True)
        s_map = self.blur(s_map)

        return s, s_map

    def run_coreset(self):
      
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
      

        self.rgb_mean = torch.mean(self.patch_rgb_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)


        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        

        if self.f_coreset < 1:

            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_rgb_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]

            
class InfraFeatures(Features):
    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb = sample[0]
        infra = sample[1]
        pc = sample[2]

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,_ = self(sample[0], sample[1], sample[2])

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch = xyz_patch.squeeze(0).T
        
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T

        # self.patch_xyz_lib.append(xyz_patch)
        # self.patch_rgb_lib.append(rgb_patch)
        self.patch_infra_lib.append(infra_patch)
        
    def predict(self, sample, label, pixel_mask):

        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
            
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,_ = self(rgb, infra, pc)
        
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s, pixel_mask)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,_ = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T


        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        

        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()

        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)


        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))

        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')
        

        
        s = torch.tensor([s_infra])
        

        self.s_lib.append(s)


    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label,pixel_mask):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()

        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)

        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))

        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')

        s = torch.tensor([s_infra])
 
        #--------------------------------------------------------------
        # object-level preds  or labels
        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
         # pixel-level preds  or labels
        ## RGB


        ## Infra
        self.infra_pixel_preds.extend(s_map_infra.flatten().numpy())
        self.infra_pixel_labels.extend(pixel_mask[1].flatten().numpy())
        
        self.infra_predictions.append(s_map_infra.detach().cpu().squeeze().numpy())
        self.infra_gts.append(pixel_mask[1].detach().cpu().squeeze().numpy())
        ## PC

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)   

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0) 
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)   
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0)   
            w_dist = torch.cdist(m_star, self.patch_infra_lib)   

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)   

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_infra_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        
        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear',align_corners=True)
        s_map = self.blur(s_map)

        return s, s_map

    def run_coreset(self):

        self.patch_infra_lib = torch.cat(self.patch_infra_lib, 0)

        self.infra_mean = torch.mean(self.patch_infra_lib)
        self.infra_std = torch.std(self.patch_infra_lib)

        
        self.patch_infra_lib = (self.patch_infra_lib - self.infra_mean)/self.infra_std

        if self.f_coreset < 1:

            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_infra_lib,
                                                            n=int(self.f_coreset * self.patch_infra_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_infra_lib = self.patch_infra_lib[self.coreset_idx]  

class PCFeatures(Features):
    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb = sample[0]
        infra = sample[1]
        pc = sample[2]

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,_ = self(sample[0], sample[1], sample[2])

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch = xyz_patch.squeeze(0).T  
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.patch_xyz_lib.append(xyz_patch)

        
    def predict(self, sample, label, pixel_mask):

        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
            
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,pts = self(rgb, infra, pc)
        
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s,pixel_mask,center_idx,pts)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx,pts = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T


        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        # 2D-true dist 
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)

        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        
    
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size,center_idx,pts, modal='xyz')

             
        s = torch.tensor([s_xyz])
        
        self.s_lib.append(s)

    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label,pixel_mask,center_idx,pts):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist 
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)

        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size,center_idx,pts, modal='xyz')


        s = torch.tensor([s_xyz])

        #--------------------------------------------------------------
        # object-level preds  or labels
        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        # pixel-level preds  or labels
        ## RGB


        ## Infra

        ## PC
        self.pc_pixel_preds.extend(s_map_xyz.flatten())
        self.pc_pixel_labels.extend(pixel_mask[2].flatten().numpy())
        
        self.pc_pts.append(pts[0,:])
        self.pc_predictions.append(s_map_xyz.squeeze())
        self.pc_gts.append(pixel_mask[2].detach().cpu().squeeze().numpy())
        #-------------------------------------------------------------

    def compute_single_s_s_map(self, patch, dist, feature_map_dims,center_idx,pts, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)   

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  
        elif modal=='rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0)  
            w_dist = torch.cdist(m_star, self.patch_infra_lib)  

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)   

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)/1000
        elif modal=='rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)/1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_infra_lib[nn_idx[0, 1:]], dim=1)/1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star
        if modal=='xyz':
            s_xyz_map = min_val
            if not center_idx.dtype == torch.long:
                center_idx = center_idx.long()

            sample_data = pts[0,center_idx]
            s_xyz_map = s_xyz_map.cpu().numpy()
            full_s_xyz_map = fill_missing_values(sample_data,s_xyz_map,pts, k=1)
            
            return s, full_s_xyz_map


        

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)      
        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_xyz_lib)
        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std


        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
 



class PCRGBFeatures(Features):

    def __init__(self, args):
        super().__init__(args)
        self.qformer_gate = QFormerGate(num_modalities=2).to(self.device)
        self.cross_modal = CrossModalAttention(dim=2).to(self.device)

    # ================= MEMORY =================
    def add_sample_to_mem_bank(self, sample, class_name=None):

        rgb, infra, pc = sample

        rgb_f, infra_f, xyz_f, center, neighbor_idx, center_idx, _ = self(rgb, infra, pc)

        # ---- XYZ ----
        xyz_patch = torch.cat(xyz_f, 1).squeeze(0).T

        # ---- RGB (FIXED SHAPE) ----
        rgb_patch = torch.cat(rgb_f, 1)
        rgb_patch = rgb_patch.squeeze(0)
        rgb_patch = rgb_patch.permute(1, 2, 0)
        rgb_patch = rgb_patch.reshape(-1, rgb_patch.shape[-1])

        self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch)

    # ================= PREDICT =================
    def predict(self, sample, label, pixel_mask):

        rgb, infra, pc = sample

        label_s = int(any(label)) if isinstance(label, (list, tuple)) else int(label)

        rgb_f, infra_f, xyz_f, center, neighbor_idx, center_idx, pts = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_f, 1).squeeze(0).T

        rgb_patch = torch.cat(rgb_f, 1)
        rgb_patch = rgb_patch.squeeze(0)
        rgb_patch = rgb_patch.permute(1, 2, 0)
        rgb_patch = rgb_patch.reshape(-1, rgb_patch.shape[-1])

        self.compute_s_s_map(xyz_patch, rgb_patch, label_s, pixel_mask, center_idx, pts)

    # ================= CORE =================
    def compute_s_s_map(self, xyz_patch, rgb_patch, label, pixel_mask, center_idx, pts):

        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)

        s_xyz, s_map_xyz = self.compute_single_s_s_map(
            xyz_patch, dist_xyz, center_idx=center_idx, pts=pts, modal='xyz'
        )

        s_rgb, s_map_rgb = self.compute_single_s_s_map(
            rgb_patch, dist_rgb, modal='rgb'
        )

        # ===== FUSION =====
        modal_scores = torch.stack([s_xyz, s_rgb]).to(self.device)
        modal_scores = torch.log1p(modal_scores)

        conf = torch.tensor([
            torch.std(dist_xyz),
            torch.std(dist_rgb)
        ], device=self.device)

        conf = conf / (conf.sum() + 1e-6)
        scores = modal_scores + 0.3 * conf

        _, weights = self.qformer_gate(scores)
        weights = torch.softmax(weights.squeeze(), dim=0)

        s = torch.sum(weights * modal_scores)
        s = torch.exp(s) - 1

        s = float(s.detach().cpu())

        # ===== STORE =====
        self.image_preds.append(s)
        self.image_labels.append(int(label))

        self.rgb_pixel_preds.extend(s_map_rgb.flatten().cpu().numpy())
        self.rgb_pixel_labels.extend(pixel_mask[0].flatten().cpu().numpy())
        # -------- RGB MAP STORAGE (MISSING) --------
        self.rgb_predictions.append(s_map_rgb.detach().cpu().squeeze().numpy())
        self.rgb_gts.append(pixel_mask[0].detach().cpu().squeeze().numpy())

        # ---- PC SAFE ALIGN ----
        if isinstance(s_map_xyz, torch.Tensor):
            pc_vals = s_map_xyz.flatten().cpu().numpy()
        else:
            pc_vals = s_map_xyz.flatten()

        pc_gt = pixel_mask[2].detach().cpu().numpy().flatten()

        min_len = min(len(pc_vals), len(pc_gt))

        self.pc_pixel_preds.extend(pc_vals[:min_len])
        self.pc_pixel_labels.extend(pc_gt[:min_len])
        # -------- PC MAP STORAGE (MISSING) --------
        self.pc_predictions.append(s_map_xyz if isinstance(s_map_xyz, np.ndarray)
                                else s_map_xyz.detach().cpu().numpy())

        self.pc_gts.append(pixel_mask[2].detach().cpu().squeeze().numpy())

    # ================= SINGLE MODAL =================
    def compute_single_s_s_map(self, patch, dist, center_idx=None, pts=None, modal='xyz'):

        min_val_full, min_idx_full = torch.min(dist, dim=1)

        if modal != 'xyz':
            k = max(1, int(0.1 * min_val_full.shape[0]))
            topk_vals, topk_idx = torch.topk(min_val_full, k=k)

            min_val = topk_vals
            min_idx = min_idx_full[topk_idx]
            patch = patch[topk_idx]
        else:
            min_val = min_val_full
            min_idx = min_idx_full

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val) / 1000

        m_test = patch[s_idx].unsqueeze(0)

        if modal == 'xyz':
            mem_lib = self.patch_xyz_lib
        else:
            mem_lib = self.patch_rgb_lib

        m_star = mem_lib[min_idx[s_idx]].unsqueeze(0)

        w_dist = torch.cdist(m_star, mem_lib)
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)

        mem_subset = mem_lib[nn_idx[0, 1:]]

        # ---- SAFE DIM ALIGN ----
        if m_test.shape[1] != mem_subset.shape[1]:
            min_dim = min(m_test.shape[1], mem_subset.shape[1])
            m_test = m_test[:, :min_dim]
            mem_subset = mem_subset[:, :min_dim]

        m_star_knn = torch.linalg.norm(m_test - mem_subset, dim=1)

        D = torch.sqrt(torch.tensor(patch.shape[1], dtype=torch.float32))
        w = 1 - (torch.exp(s_star / D) / torch.sum(torch.exp(m_star_knn / D)))

        s = w * s_star

        # ===== MAP =====
        if modal == "xyz":

            s_map = min_val_full.cpu().numpy()

            if center_idx is not None:
                center_idx = center_idx.long()
                sample_data = pts[0, center_idx]

                s_map = fill_missing_values(sample_data, s_map, pts, k=1)

            return s, s_map

        else:
            base_map = min_val_full

            numel = base_map.shape[0]
            side = int(math.sqrt(numel))

            if side * side != numel:
                side = int(np.sqrt(numel))
                base_map = base_map[:side * side]

            s_map = base_map.view(1, 1, side, side)
            s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear', align_corners=True)
            s_map = self.blur(s_map)

            return s, s_map

    # ================= CORESET =================
    def run_coreset(self):

        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_xyz_lib)

        self.rgb_mean = torch.mean(self.patch_rgb_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std
        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std

class PCInfraFeatures(Features):

    def __init__(self, args):
        super().__init__(args)

        self.qformer_gate = QFormerGate(num_modalities=2).to(self.device)
        #self.cross_modal = CrossModalAttention(dim=2).to(self.device)

    # ---------------- MEMORY ----------------
    def add_sample_to_mem_bank(self, sample, class_name=None):
        rgb, infra, pc = sample

        # ---------- FORWARD PASS ----------
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, _ = self(rgb, infra, pc)

        # ---------- XYZ ----------
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch = xyz_patch.squeeze(0).T   # (N, 1152)

        # ---------- INFRA ----------
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.squeeze(0)              # (C, H, W)
        infra_patch = infra_patch.permute(1, 2, 0)        # (H, W, C)
        infra_patch = infra_patch.reshape(-1, infra_patch.shape[-1])  # (N, 768)

        # ---------- STORE ----------
        self.patch_xyz_lib.append(xyz_patch)
        self.patch_infra_lib.append(infra_patch)

    # ---------------- PREDICT ----------------
    def predict(self, sample, label, pixel_mask):

        rgb, infra, pc = sample

        # ---------- FORWARD PASS ----------
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, pts = self(rgb, infra, pc)

        # ---------- XYZ ----------
        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch = xyz_patch.squeeze(0).T

        # ---------- INFRA ----------
        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.squeeze(0)
        infra_patch = infra_patch.permute(1, 2, 0)
        infra_patch = infra_patch.reshape(-1, infra_patch.shape[-1])

        # ---------- CALL MAIN LOGIC ----------
        self.compute_s_s_map(
            xyz_patch,
            infra_patch,
            label,
            pixel_mask,
            center_idx,
            pts
        )

    # ---------------- CORE FUSION ----------------
    def compute_s_s_map(self, xyz_patch, infra_patch, label, pixel_mask, center_idx, pts):

        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)

        xyz_size = (int(math.sqrt(xyz_patch.shape[0])),)*2
        infra_size = (int(math.sqrt(infra_patch.shape[0])),)*2

        s_xyz, s_map_xyz = self.compute_single_s_s_map(
            xyz_patch,
            dist_xyz,
            xyz_size,
            center_idx,
            pts,
            'xyz'
        )
        s_infra, s_map_infra = self.compute_single_s_s_map(
            infra_patch,
            dist_infra,
            infra_size,
            modal='infra'
        )

        # -------- MODAL SCORES --------
        modal_scores = torch.stack([
            s_xyz.detach(),
            s_infra.detach()
        ]).to(self.device)

        # -------- LOG SCALE --------
        modal_scores = torch.log1p(modal_scores)

        # -------- RELIABILITY ESTIMATION --------
        min_xyz, _ = torch.min(dist_xyz, dim=1)
        min_infra, _ = torch.min(dist_infra, dim=1)

        conf_xyz = torch.var(min_xyz)
        conf_infra = torch.var(min_infra)

        conf = torch.tensor(
            [conf_xyz, conf_infra],
            device=self.device,
            dtype=torch.float32
        )

        conf = conf / (conf.sum() + 1e-6)

        # moderate sharpening
        conf = torch.softmax(conf * 2.0, dim=0)

        modal_scores = modal_scores * (
            0.7 + conf
        )



        # -------- QFORMER --------
        _, weights = self.qformer_gate(modal_scores)
        weights = torch.softmax(weights.squeeze(), dim=0)

        # -------- HYBRID FUSION --------
        weighted_scores = weights * modal_scores

        temp = 0.7

        s = (
            0.7 * torch.max(weighted_scores)
            +
            0.3 * torch.mean(weighted_scores)
        ) / temp

        # numerical stability
        s = torch.nan_to_num(
            s,
            nan=0.0,
            posinf=1e3,
            neginf=-1e3
        )

        s = s.detach().cpu()

        # ========= STORE =========
        self.image_preds.append(float(s))
        if isinstance(label, (list, tuple)):
            label_val = int(any(label))
        else:
            label_val = int(label)

        self.image_labels.append(label_val)

        # Infra maps
        self.infra_pixel_preds.extend(s_map_infra.flatten().numpy())
        self.infra_pixel_labels.extend(pixel_mask[1].flatten().numpy())
        self.infra_predictions.append(s_map_infra.detach().cpu().squeeze().numpy())
        self.infra_gts.append(pixel_mask[1].detach().cpu().squeeze().numpy())

                # PC maps
        # Resize GT mask to match prediction
        # ---------- PC MAP ----------
        if isinstance(s_map_xyz, torch.Tensor):
            pc_vals = s_map_xyz.flatten().cpu().numpy()
        else:
            pc_vals = s_map_xyz.flatten()

        # ---------- MATCH GT SIZE TO PRED ----------
        pc_gt = pixel_mask[2].detach().cpu().numpy().flatten()

        # 🔥 CRITICAL: ALIGN LENGTHS
        min_len = min(len(pc_vals), len(pc_gt))

        pc_vals = pc_vals[:min_len]
        pc_gt = pc_gt[:min_len]
        # --------------------------------------------------------------
        # Point cloud predictions

        if isinstance(s_map_xyz, torch.Tensor):
            pc_vals = s_map_xyz.flatten().cpu().numpy()
        else:
            pc_vals = s_map_xyz.flatten()

        self.pc_pixel_preds.extend(pc_vals)

        self.pc_pixel_labels.extend(
            pixel_mask[2].flatten().numpy()
        )

        self.pc_pts.append(pts[0, :])

        self.pc_predictions.append(
            s_map_xyz.squeeze()
        )

        self.pc_gts.append(
            pixel_mask[2].detach().cpu().squeeze().numpy()
        )
        # ---------- STORE ----------
        self.pc_pixel_preds.extend(pc_vals)
        self.pc_pixel_labels.extend(pc_gt)



    def compute_single_s_s_map(self, patch, dist, feature_map_dims, center_idx=None, pts=None, modal='xyz'):

        min_val_full, min_idx_full = torch.min(dist, dim=1)

        # -------- TOP-K FOR SCORING ONLY --------
        if modal != 'xyz':
            k = max(1, int(0.1 * min_val_full.shape[0]))
            topk_vals, topk_idx = torch.topk(min_val_full, k=k)

            patch_reduced = patch[topk_idx]
            min_val_reduced = topk_vals
            min_idx_reduced = min_idx_full[topk_idx]
        else:
            patch_reduced = patch
            min_val_reduced = min_val_full
            min_idx_reduced = min_idx_full

        # -------- SCORE --------
        s_idx = torch.argmax(min_val_reduced)
        s_star = torch.max(min_val_reduced) / 1000

        m_test = patch_reduced[s_idx].unsqueeze(0)

        # ---------- SELECT CORRECT MEMORY ----------
        if modal == 'xyz':
            mem_lib = self.patch_xyz_lib
        elif modal == 'infra':
            mem_lib = self.patch_infra_lib
        else:
            raise ValueError(f"Unknown modal: {modal}")

        # ---------- MATCH ----------
        m_star = mem_lib[min_idx_reduced[s_idx]].unsqueeze(0)

        w_dist = torch.cdist(m_star, mem_lib)

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)

        # ---------- REWEIGHT ----------
        mem_subset = mem_lib[nn_idx[0, 1:]]

        # -------- HARD ALIGN FEATURE DIM --------
        if m_test.shape[1] != mem_subset.shape[1]:
            min_dim = min(m_test.shape[1], mem_subset.shape[1])
            m_test = m_test[:, :min_dim]
            mem_subset = mem_subset[:, :min_dim]

        m_star_knn = torch.linalg.norm(
            m_test - mem_subset,
            dim=1
        )

        # ---------- WEIGHT ----------
        D = torch.sqrt(torch.tensor(patch.shape[1], device=patch.device, dtype=torch.float32))

        w = 1 - (
            torch.exp(s_star / D) /
            (torch.sum(torch.exp(m_star_knn / D)) + 1e-6)
        )

        s = w * s_star

        # ---------- MAP ----------
        if modal == "xyz":
            s_xyz_map = min_val_full

            if not center_idx.dtype == torch.long:
                center_idx = center_idx.long()

            sample_data = pts[0, center_idx]

            s_xyz_map = s_xyz_map.cpu().numpy()

            full_s_xyz_map = fill_missing_values(
                sample_data,
                s_xyz_map,
                pts,
                k=1
            )

            return s, full_s_xyz_map

        else:
            s_map = min_val_full.view(1, 1, *feature_map_dims)

            s_map = torch.nn.functional.interpolate(
                s_map,
                size=(224, 224),
                mode='bilinear',
                align_corners=True
            )

            s_map = self.blur(s_map)

            return s, s_map

    # ---------------- CORESET ----------------
    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_infra_lib = torch.cat(self.patch_infra_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_xyz_lib) + 1e-6

        self.infra_mean = torch.mean(self.patch_infra_lib)
        self.infra_std = torch.std(self.patch_infra_lib) + 1e-6

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std
        self.patch_infra_lib = (self.patch_infra_lib - self.infra_mean)/self.infra_std



class RGBInfraFeatures(Features):
    def __init__(self, args):
        super().__init__(args)

        self.qformer_gate = QFormerGate(num_modalities=2).to(self.device)
        #self.cross_modal = CrossModalAttention(dim=2).to(self.device)

    # ---------------- MEMORY ----------------
    def add_sample_to_mem_bank(self, sample, class_name=None):

        rgb, infra, pc = sample

        rgb_f, infra_f, xyz_f, center, neighbor_idx, center_idx, _ = self(rgb, infra, pc)

        rgb_patch = torch.cat(rgb_f, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_f, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T

        self.patch_rgb_lib.append(rgb_patch)
        self.patch_infra_lib.append(infra_patch)

    # ---------------- PREDICT ----------------
    def predict(self, sample, label, pixel_mask):

        label_s = 1 if (label[0]==1 or label[1]==1 or label[2]==1) else 0

        rgb = sample[0].to(self.device)
        infra = sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_f, infra_f, xyz_f, center, neighbor_idx, center_idx, pts = self(rgb, infra, pc)

        rgb_patch = torch.cat(rgb_f, 1).reshape(rgb_f[0].shape[1], -1).T
        infra_patch = torch.cat(infra_f, 1).reshape(infra_f[0].shape[1], -1).T

        self.compute_s_s_map(rgb_patch, infra_patch, label_s, pixel_mask)

    # ---------------- CORE FUSION ----------------
    def compute_s_s_map(self, rgb_patch, infra_patch, label, pixel_mask):

        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()

        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)

        rgb_size = (int(math.sqrt(rgb_patch.shape[0])),)*2
        infra_size = (int(math.sqrt(infra_patch.shape[0])),)*2

        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_size, modal='rgb')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_size, modal='infra')

        # ========= STRONG QFORMER FUSION =========
        modal_scores = torch.stack([
            s_rgb.detach(),
            s_infra.detach()
        ]).to(self.device)


        # -------- LOG SCALE --------
        modal_scores = torch.log1p(modal_scores)

        # -------- RELIABILITY ESTIMATION --------
        min_rgb, _ = torch.min(dist_rgb, dim=1)
        min_infra, _ = torch.min(dist_infra, dim=1)

        conf_rgb = torch.var(min_rgb)
        conf_infra = torch.var(min_infra)

        conf = torch.tensor(
            [conf_rgb, conf_infra],
            device=self.device,
            dtype=torch.float32
        )

        # normalize
        conf = conf / (conf.sum() + 1e-6)

        # 🔥 sharpen confidence (VERY IMPORTANT)
        conf = torch.softmax(conf * 2.0, dim=0)

        # apply confidence BEFORE fusion
        modal_scores = modal_scores * conf

        # -------- QFORMER --------
        _, weights = self.qformer_gate(modal_scores)
        weights = torch.softmax(weights.squeeze(), dim=0)

        # -------- HYBRID FUSION --------
        weighted_scores = weights * modal_scores

        temp = 0.7

        s = (
            0.7 * torch.max(weighted_scores)
            +
            0.3 * torch.mean(weighted_scores)
        ) / temp

        # numerical stability
        s = torch.nan_to_num(
            s,
            nan=0.0,
            posinf=1e3,
            neginf=-1e3
        )

        s = s.detach().cpu()

        # -------- STORE --------
        self.image_preds.append(s.numpy())
        self.image_labels.append(label)

        # RGB
        self.rgb_pixel_preds.extend(s_map_rgb.flatten().cpu().numpy())
        self.rgb_pixel_labels.extend(pixel_mask[0].flatten().numpy())
        self.rgb_predictions.append(s_map_rgb.detach().cpu().squeeze().numpy())
        self.rgb_gts.append(pixel_mask[0].detach().cpu().squeeze().numpy())

        # Infra
        self.infra_pixel_preds.extend(s_map_infra.flatten().cpu().numpy())
        self.infra_pixel_labels.extend(pixel_mask[1].flatten().numpy())
        self.infra_predictions.append(s_map_infra.detach().cpu().squeeze().numpy())
        self.infra_gts.append(pixel_mask[1].detach().cpu().squeeze().numpy())

    # ---------------- SHARED ----------------
    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='rgb'):

        min_val_full, min_idx_full = torch.min(dist, dim=1)

        min_val = min_val_full
        min_idx = min_idx_full

        if modal != 'xyz':
            k = max(1, int(0.1 * min_val.shape[0]))
            topk_vals, topk_idx = torch.topk(min_val, k=k)

            patch = patch[topk_idx]
            min_val = topk_vals
            min_idx = min_idx[topk_idx]

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        m_test = patch[s_idx].unsqueeze(0)

        if modal == 'rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)
        else:
            m_star = self.patch_infra_lib[min_idx[s_idx]].unsqueeze(0)
            w_dist = torch.cdist(m_star, self.patch_infra_lib)

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)

        if modal == 'rgb':
            mem_subset = self.patch_rgb_lib[nn_idx[0, 1:]]

            if m_test.shape[1] != mem_subset.shape[1]:
                min_dim = min(m_test.shape[1], mem_subset.shape[1])

                m_test = m_test[:, :min_dim]
                mem_subset = mem_subset[:, :min_dim]

            m_star_knn = torch.linalg.norm(
                m_test - mem_subset,
                dim=1
            ) / 1000
        else:
            mem_subset = self.patch_infra_lib[nn_idx[0, 1:]]

            if m_test.shape[1] != mem_subset.shape[1]:
                min_dim = min(m_test.shape[1], mem_subset.shape[1])

                m_test = m_test[:, :min_dim]
                mem_subset = mem_subset[:, :min_dim]

            m_star_knn = torch.linalg.norm(
                m_test - mem_subset,
                dim=1
            ) / 1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / torch.sum(torch.exp(m_star_knn / D)))
        s = w * s_star

        s_map = min_val_full.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear', align_corners=True)
        s_map = self.blur(s_map)

        return s, s_map

    # ---------------- CORESET ----------------
    def run_coreset(self):

        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_infra_lib = torch.cat(self.patch_infra_lib, 0)

        # ✅ FIXED NORMALIZATION
        self.rgb_mean = torch.mean(self.patch_rgb_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib) + 1e-6

        self.infra_mean = torch.mean(self.patch_infra_lib)
        self.infra_std = torch.std(self.patch_infra_lib) + 1e-6

        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std
        self.patch_infra_lib = (self.patch_infra_lib - self.infra_mean)/self.infra_std

        if self.f_coreset < 1:
            idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                               n=int(self.f_coreset * self.patch_rgb_lib.shape[0]),
                                               eps=self.coreset_eps)
            self.patch_rgb_lib = self.patch_rgb_lib[idx]

            idx = self.get_coreset_idx_randomp(self.patch_infra_lib,
                                               n=int(self.f_coreset * self.patch_infra_lib.shape[0]),
                                               eps=self.coreset_eps)
            self.patch_infra_lib = self.patch_infra_lib[idx]


class TripleRGBInfraPointFeatures(Features):



    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.qformer_gate = QFormerGate().to(self.device)
        #self.cross_modal = CrossModalAttention().to(self.device)
        # store modality weights for analysis
        self.modal_weights = []




    def add_sample_to_mem_bank(self, sample, class_name=None):

        rgb = sample[0]
        infra = sample[1]
        pc = sample[2]

        outputs = self(sample[0], sample[1], sample[2])

        if outputs is None:
            print("❌ MODEL RETURNED NONE")
            return

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, _ = outputs

        if len(xyz_feature_maps) == 0:
            print("⚠️ EMPTY XYZ FEATURES → USING FALLBACK")

            # create dummy patch so pipeline doesn't break
            xyz_patch = torch.zeros((10, 128)).to(self.device)
            rgb_patch = torch.zeros((10, 768)).to(self.device)
            infra_patch = torch.zeros((10, 768)).to(self.device)

            self.patch_xyz_lib.append(xyz_patch.cpu())
            self.patch_rgb_lib.append(rgb_patch.cpu())
            self.patch_infra_lib.append(infra_patch.cpu())
            return

        xyz_patch = torch.cat(xyz_feature_maps, 1)

        xyz_patch = xyz_patch.squeeze(0).T
        
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T


        infra_patch = torch.cat(infra_feature_maps, 1)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T


        self.patch_xyz_lib.append(xyz_patch)
        self.patch_rgb_lib.append(rgb_patch)
        self.patch_infra_lib.append(infra_patch)
   

        
    def predict(self, sample, label,pixel_mask):

        if label[0]==1 or label[1]==1 or label[2]==1:
            label_s = 1
        else:
            label_s = 0
        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)
        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, pts = self(rgb, infra, pc)
        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T
        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        self.compute_s_s_map(xyz_patch, rgb_patch, infra_patch, label_s, pixel_mask, center_idx, pts)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        rgb = sample[0].to(self.device)
        infra =sample[1].to(self.device)
        pc = sample[2].to(self.device)

        rgb_feature_maps, infra_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, pts = self(rgb, infra, pc)

        xyz_patch = torch.cat(xyz_feature_maps, 1).to(self.device)
        xyz_patch = xyz_patch.squeeze(0).T


        rgb_patch = torch.cat(rgb_feature_maps, 1).to(self.device)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        infra_patch = torch.cat(infra_feature_maps, 1).to(self.device)
        infra_patch = infra_patch.reshape(infra_patch.shape[1], -1).T
        
        xyz_patch = ((xyz_patch - self.xyz_mean)/self.xyz_std).cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean)/self.rgb_std).cpu()
        infra_patch = ((infra_patch - self.infra_mean)/self.infra_std).cpu()
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)


        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size,center_idx, pts, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_infra, s_map_infra = self.compute_single_s_s_map(infra_patch, dist_infra, infra_feat_size, modal='infra')

        s = torch.tensor([[s_xyz, s_rgb, s_infra]])


        self.s_lib.append(s)

    def compute_s_s_map(self, xyz_patch, rgb_patch, infra_patch, label, pixel_mask, center_idx, pts):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # ---------- normalize patches ----------
        xyz_patch = ((xyz_patch - self.xyz_mean) / self.xyz_std).cpu()
        rgb_patch = ((rgb_patch - self.rgb_mean) / self.rgb_std).cpu()
        infra_patch = ((infra_patch - self.infra_mean) / self.infra_std).cpu()

        # ---------- compute distances ----------
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_infra = torch.cdist(infra_patch, self.patch_infra_lib)

        # ---------- feature map sizes ----------
        infra_feat_size = (int(math.sqrt(infra_patch.shape[0])), int(math.sqrt(infra_patch.shape[0])))
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

        # ---------- PatchCore anomaly scores ----------
        s_xyz, s_map_xyz = self.compute_single_s_s_map(
            xyz_patch, dist_xyz, xyz_feat_size, center_idx, pts, modal='xyz'
        )

        s_rgb, s_map_rgb = self.compute_single_s_s_map(
            rgb_patch, dist_rgb, rgb_feat_size, modal='rgb'
        )

        s_infra, s_map_infra = self.compute_single_s_s_map(
            infra_patch, dist_infra, infra_feat_size, modal='infra'
        )

        # ---------- build modality score vector ----------
        scores = torch.tensor(
            [s_xyz, s_rgb, s_infra],
            device=self.device,
            dtype=torch.float32
        )

        # normalize scores
        scores = (scores - scores.mean()) / (scores.std() + 1e-6)
        _, weights = self.qformer_gate(scores)
        weights = torch.softmax(weights.squeeze(), dim=0)
        # ---------- cross-modal interaction ----------
        #scores_attn = self.cross_modal(scores)



        # -------- BASE MODAL SCORES --------
        modal_scores = torch.tensor(
            [s_xyz, s_rgb, s_infra],
            device=self.device,
            dtype=torch.float32
        )

        # -------- DIFFICULTY ESTIMATION --------
        # -------------------------------------------------
        # RELIABILITY / UNCERTAINTY ESTIMATION
        # aligns with Eq. (8)
        # -------------------------------------------------

        min_xyz, _ = torch.min(dist_xyz, dim=1)
        min_rgb, _ = torch.min(dist_rgb, dim=1)
        min_infra, _ = torch.min(dist_infra, dim=1)

        conf_xyz = torch.var(min_xyz)
        conf_rgb = torch.var(min_rgb)
        conf_infra = torch.var(min_infra)



        difficulty = (
            conf_xyz +
            conf_rgb +
            conf_infra
        )
        # normalize difficulty
        difficulty = difficulty / (difficulty + 1e-6)

        # smooth scaling (0 → easy, 1 → hard)
        difficulty_factor = torch.sigmoid(difficulty)

        # -------- ADAPTIVE MODULATION --------
        adaptive_scores = modal_scores * (
            1 + 0.5 * difficulty_factor
        )

        # -------------------------------------------------
        # SCORE NORMALIZATION
        # aligns with Eq. (10)
        # -------------------------------------------------

        # -------- QFORMER WEIGHTS --------
        _, weights = self.qformer_gate(adaptive_scores)

        weights = torch.softmax(weights.squeeze(), dim=0)

        # -------- FINAL FUSION --------
        weighted_scores = weights * adaptive_scores

        # -------------------------------------------------
        # ENTROPY-GUIDED DECISION STRATEGY
        # aligns with Eq. (14)-(16)
        # -------------------------------------------------
# -------- FINAL FUSION --------
        weighted_scores = weights * adaptive_scores

        # -------- ROBUST AGGREGATION --------
        sorted_scores, _ = torch.sort(
            weighted_scores,
            descending=True
        )

        # stability logic
        if torch.std(sorted_scores) < 0.05:

            s = torch.mean(sorted_scores)

        elif sorted_scores[0] > 2.0 * sorted_scores[1]:

            s = torch.mean(sorted_scores[1:])

        else:

            s = torch.mean(sorted_scores[:2])

        # -------------------------------------------------
        # NUMERICAL STABILITY
        # -------------------------------------------------

        s = torch.nan_to_num(
            s,
            nan=0.0,
            posinf=1e3,
            neginf=-1e3
        )
        # -------------------------------------------------

        # --------------------------------------------------------------
        # object-level preds
        self.image_preds.append(s.detach().cpu().numpy())
        self.image_labels.append(label)

        # --------------------------------------------------------------
        # RGB pixel predictions
        self.rgb_pixel_preds.extend(s_map_rgb.flatten().numpy())
        self.rgb_pixel_labels.extend(pixel_mask[0].flatten().numpy())

        self.rgb_predictions.append(s_map_rgb.detach().cpu().squeeze().numpy())
        self.rgb_gts.append(pixel_mask[0].detach().cpu().squeeze().numpy())

        # --------------------------------------------------------------
        # Infra pixel predictions
        self.infra_pixel_preds.extend(s_map_infra.flatten().numpy())
        self.infra_pixel_labels.extend(pixel_mask[1].flatten().numpy())

        self.infra_predictions.append(s_map_infra.detach().cpu().squeeze().numpy())
        self.infra_gts.append(pixel_mask[1].detach().cpu().squeeze().numpy())

        # --------------------------------------------------------------
        # Point cloud predictions
        self.pc_pixel_preds.extend(s_map_xyz.flatten())
        self.pc_pixel_labels.extend(pixel_mask[2].flatten().numpy())

        self.pc_pts.append(pts[0, :])
        self.pc_predictions.append(s_map_xyz.squeeze())
        self.pc_gts.append(pixel_mask[2].detach().cpu().squeeze().numpy())


        #-------------------------------------------------------------

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, center_idx=None, pts=None, modal='xyz'):

        # -------- CORRECT DIST --------
        min_val_full, min_idx_full = torch.min(dist, dim=1)

        min_val = min_val_full
        min_idx = min_idx_full

        # -------- TOP-K FILTER (ONLY RGB/INFRA) --------
        if modal != 'xyz':
            k = max(1, int(0.1 * min_val.shape[0]))
            topk_vals, topk_idx = torch.topk(min_val, k=k)

            patch = patch[topk_idx]
            min_val = topk_vals
            min_idx = min_idx[topk_idx]

        # -------------------------------------------------
        # TOP-K ROBUST ANOMALY SCORING
        # aligns with Eq. (4) in methodology
        # -------------------------------------------------

        s_idx = torch.argmax(min_val)

        s_star = torch.max(min_val) / 1000

        # -------------------------------------------------
        # PATCHCORE REWEIGHTING (PRESERVED)
        # -------------------------------------------------

        s_idx = torch.argmax(min_val)

        m_test = patch[s_idx].unsqueeze(0)

        if modal == 'xyz':
            mem_lib = self.patch_xyz_lib
        elif modal == 'rgb':
            mem_lib = self.patch_rgb_lib
        else:
            mem_lib = self.patch_infra_lib

        m_star = mem_lib[min_idx[s_idx]].unsqueeze(0)

        w_dist = torch.cdist(m_star, mem_lib)

        _, nn_idx = torch.topk(
            w_dist,
            k=self.n_reweight,
            largest=False
        )

        mem_subset = mem_lib[nn_idx[0, 1:]]

        # SAFE FEATURE ALIGN
        if m_test.shape[1] != mem_subset.shape[1]:
            min_dim = min(m_test.shape[1], mem_subset.shape[1])

            m_test = m_test[:, :min_dim]
            mem_subset = mem_subset[:, :min_dim]

        m_star_knn = torch.linalg.norm(
            m_test - mem_subset,
            dim=1
        )

        D = torch.sqrt(
            torch.tensor(
                patch.shape[1],
                device=patch.device,
                dtype=torch.float32
            )
        )

        w = 1 - (
            torch.exp(s_star / D) /
            (torch.sum(torch.exp(m_star_knn / D)) + 1e-6)
        )

        # final PatchCore score
        s = w * s_star

        # -------- MAP GENERATION --------
        if modal == "xyz":
            s_xyz_map = min_val_full   # ✅ IMPORTANT (NOT min_val)

            if not center_idx.dtype == torch.long:
                center_idx = center_idx.long()

            sample_data = pts[0, center_idx]

            s_xyz_map = s_xyz_map.cpu().numpy()

            full_s_xyz_map = fill_missing_values(
                sample_data,
                s_xyz_map,
                pts,
                k=1
            )

            return s, full_s_xyz_map

        else:
            s_map = min_val_full.view(1, 1, *feature_map_dims)  # ✅ IMPORTANT
            s_map = torch.nn.functional.interpolate(
                s_map, size=(224, 224),
                mode='bilinear',
                align_corners=True
            )
            s_map = self.blur(s_map)

            return s, s_map

    def run_coreset(self):

        # -------- CONCAT --------
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_infra_lib = torch.cat(self.patch_infra_lib, 0)

        # -------- CORRECT STATS --------
        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_xyz_lib) + 1e-6

        self.rgb_mean = torch.mean(self.patch_rgb_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib) + 1e-6

        self.infra_mean = torch.mean(self.patch_infra_lib)
        self.infra_std = torch.std(self.patch_infra_lib) + 1e-6

        # -------- NORMALIZE --------
        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean) / self.xyz_std
        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean) / self.rgb_std
        self.patch_infra_lib = (self.patch_infra_lib - self.infra_mean) / self.infra_std

        # -------- CORESET --------
        if self.f_coreset < 1:

            xyz_idx = self.get_coreset_idx_randomp(
                self.patch_xyz_lib,
                n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                eps=self.coreset_eps
            )

            rgb_idx = self.get_coreset_idx_randomp(
                self.patch_rgb_lib,
                n=int(self.f_coreset * self.patch_rgb_lib.shape[0]),
                eps=self.coreset_eps
            )

            infra_idx = self.get_coreset_idx_randomp(
                self.patch_infra_lib,
                n=int(self.f_coreset * self.patch_infra_lib.shape[0]),
                eps=self.coreset_eps
            )

            self.patch_xyz_lib = self.patch_xyz_lib[xyz_idx]
            self.patch_rgb_lib = self.patch_rgb_lib[rgb_idx]
            self.patch_infra_lib = self.patch_infra_lib[infra_idx]