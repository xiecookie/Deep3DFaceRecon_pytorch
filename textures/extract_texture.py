"""This script is the test script for Deep3DFaceRecon_pytorch
"""

import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import MyVisualizer
from util.preprocess import align_img
from PIL import Image
import numpy as np
from util.load_mats import load_lm3d
import torch
from data.flist_dataset import default_flist_reader
from scipy.io import loadmat, savemat
from textures.uv_creator import UVCreator
import cv2


def get_data_path(root='examples'):
    im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]
    lm_path = [i.replace('png', 'txt').replace('jpg', 'txt') for i in im_path]
    lm_path = [os.path.join(i.replace(i.split(os.path.sep)[-1], ''), 'detections', i.split(os.path.sep)[-1]) for i in
               lm_path]

    return im_path, lm_path


def read_data(im_path, lm_path, lm3d_std, to_tensor=True):
    # to RGB
    im = Image.open(im_path).convert('RGB')
    W, H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, _ = align_img(im, lm, lm3d_std)
    if to_tensor:
        im = torch.tensor(np.array(im) / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
    return im, lm

def extract_texture(model, i):
    print("in extract texture : ")
    device_name = 'cuda'
    model.facemodel.to(device_name)
    model.mesh_rasterizer.to(device_name)
    print("model.pred_vertex: "+str(model.pred_vertex.shape))
    print("model.facemodel.face_buf: "+str(model.facemodel.face_buf.shape))
    fragment = model.renderer.rasterize(model.mesh_rasterizer, model.pred_vertex, model.facemodel.face_buf)
    visible_face = torch.unique(fragment.pix_to_face)[1:]  # exclude face id -1
    visible_vert = model.facemodel.face_buf[visible_face]
    visible_vert = torch.unique(visible_vert)
    
#     new_mesh = model.renderer.transform(model.mesh_rasterizer, model.pred_vertex, model.facemodel.face_buf)
#     shift_vert = new_mesh._verts_padded[0]
#     print(shift_vert.shape)
#     print(shift_vert[0][:10])
    shift_vert = model.pred_vertex[0]
    
    vert_alpha = torch.zeros([shift_vert.shape[0], 1], device=device_name)
    vert_alpha[visible_vert] = 1
    print("shift_vert: "+str(shift_vert.shape))
    print("vert_alpha: "+str(vert_alpha.shape))
    nsh_shift_vert_alpha = torch.cat([shift_vert, vert_alpha], axis=-1)
    uv_creator = UVCreator('230', device=device_name)
    uv_size = 1024
    uv_creator.save_mesh(model, model.input_img)
    uvmap = uv_creator.create_bfm_uv_torch(
        nsh_shift_vert_alpha, model.input_img, uv_size)
    uvmap = uvmap[..., :3].cpu().numpy()[:, :, (2,1,0)] * 255.0
    cv2.imwrite(str(i)+'_uv_test.jpg',uvmap)

def center_crop(image, img_size):
    # set img_size to None will not resize image
    _, height, width = image.shape
    if width > img_size:
        w_s = (width - img_size) // 2
        image = image[:, :, w_s:w_s + img_size]
    if height > img_size:
        h_s = (height - img_size) // 2
        image = image[:, h_s:h_s + img_size, :]

    return image