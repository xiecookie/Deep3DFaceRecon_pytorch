"""This script is the differentiable renderer for Deep3DFaceRecon_pytorch
    Attention, antialiasing step is missing in current version.
"""
import pytorch3d.ops
import torch
import torch.nn.functional as F
import kornia
from kornia.geometry.camera import pixel2cam
import numpy as np
from typing import List
from scipy.io import loadmat
from torch import nn

from pytorch3d.structures import Meshes

"""
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    DirectionalLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
)
"""

class MeshRenderer(nn.Module):
    def __init__(self,
                 rasterize_fov,
                 znear=0.1,
                 zfar=10,
                 rasterize_size=224):
        super(MeshRenderer, self).__init__()

        self.rasterize_size = rasterize_size
        self.fov = rasterize_fov
        self.znear = znear
        self.zfar = zfar

    def forward(self, rasterizer, vertex, tri, feat=None):
        """
        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            depth              -- torch.tensor, size (B, 1, H, W)
            features(optional) -- torch.tensor, size (B, C, H, W) if feat is not None

        Parameters:
            vertex          -- torch.tensor, size (B, N, 3)
            tri             -- torch.tensor, size (B, M, 3) or (M, 3), triangles
            feat(optional)  -- torch.tensor, size (B, N ,C), features
        """
        device = vertex.device
        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 0] = -vertex[..., 0]
        print("vertex : ")
        print(np.min(vertex.cpu().numpy()[0, :, :2]))
        print(np.max(vertex.cpu().numpy()[0, :, :2]))
        print(vertex.shape)
        print(vertex[0, :20])
            
        tri = tri.type(torch.int32).contiguous()

        # rasterize
        batch_size = vertex.size()[0]
        tri = tri.unsqueeze(0)
        tri = tri.expand(batch_size, tri.size()[1], tri.size()[2])

        mesh = Meshes(vertex.contiguous()[..., :3], tri)
        fragments = rasterizer(mesh)
        rast_out = fragments.pix_to_face.squeeze(-1)
        depth = fragments.zbuf

        # render depth
        depth = depth.permute(0, 3, 1, 2)
        mask = (rast_out > 0).float().unsqueeze(1)
        depth = mask * depth

        image = None
        if feat is not None:
            attributes = feat.reshape(-1, 3)[mesh.faces_packed()]
            image = pytorch3d.ops.interpolate_face_attributes(fragments.pix_to_face,
                                                              fragments.bary_coords,
                                                              attributes)
            image = image.squeeze(-2).permute(0, 3, 1, 2)
            image = mask * image

        return mask, depth, image

    def rasterize(self, rasterizer, vertex, tri, feat=None):
        device = vertex.device
        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 0] = -vertex[..., 0]

        vertex = vertex.contiguous()[..., :3]
        tri = tri.type(torch.int32).contiguous()[None]

#         print("in rasterize")
#         print(vertex.shape)
#         print(tri.shape)
        
        # batch_size = vertex.size()[0]
        # tri = tri.unsqueeze(0)
        # tri = tri.expand(batch_size, tri.size()[1], tri.size()[2])

        mesh = Meshes(vertex, tri)
        fragments = rasterizer(mesh)
        return fragments

    def transform(self, rasterizer, vertex, tri, feat=None):
        device = vertex.device
        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 0] = -vertex[..., 0]

        vertex = vertex.contiguous()[..., :3]
        tri = tri.type(torch.int32).contiguous()[None]

#         print("in rasterize")
#         print(vertex.shape)
#         print(tri.shape)
        
        # batch_size = vertex.size()[0]
        # tri = tri.unsqueeze(0)
        # tri = tri.expand(batch_size, tri.size()[1], tri.size()[2])

        mesh = Meshes(vertex, tri)
        mesh_new = rasterizer.transform(mesh)
        return mesh_new