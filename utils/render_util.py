#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from urllib.parse import uses_params
from matplotlib.pyplot import axis
import numpy as np
import random
from regex import D
import torch
import os

from torch.functional import norm
from DeepLearningFrameWork.utils import img2tensor

from torch._C import device
lightDistance = 2.14
viewDistance = 2.75 # 39.98 degrees FOV

class svBRDF():
    def __init__(self, opt):
        self.opt = opt
        self.size = opt['size']
        self.nbRendering = opt['nbRendering']
        self.useAugmentation = opt.get('useAug',False)
        # self.lampIntensity = opt["lampIntensity"]
        if opt.get('lampIntensity', False):
            self.lampIntensity = opt['lampIntensity']

    def split_img(self, imgs):
        """Split input image to $split_num images.

        Args:
            imgs (list[ndarray] | ndarray): Images with shape (h, w, c).
            split_num (int): number of input images containing
            split_axis (int): which axis the split images is concated.

        Returns:
            list[ndarray]: Split images.
        """
        split_num = self.opt['split_num']
        split_axis = self.opt['split_axis']
        concat = self.opt['concat']
        svbrdf_norm = self.opt['svbrdf_norm']
        order = self.opt['order']
        gamma_correct = self.opt.get('gamma_correct', '')

        def _norm(img, gamma_correct='pdrs', order='pndrs'):
            n,d,r,s = None, None, None, None
            if order == 'pndrs':
                p,n,d,r,s = img
            elif order == 'dnrs':
                d,n,r,s = img
            elif order == 'ndrs':
                n,d,r,s = img
            r = np.mean(r,axis=-1,keepdims=True)
            n = numpy_norm(n*2-1,-1)*0.5+0.5
            if 'd' in gamma_correct:
                d = d**2.2
            if 'r' in gamma_correct:
                r = r**2.2
            if 's' in gamma_correct:
                s = s**2.2
            result = []
            if order == 'pndrs' or order=='ndrs':
                result = [n,d,r,s]
            elif order == 'dnrs':
                result = [d,n,r,s]
            result = [preprocess(x) for x in result]
            return result

        if split_num == 1:
            return imgs
        else:
            if isinstance(imgs, list):
                imglist = [np.split(v,split_num,axis=split_axis) for v in imgs]
                if svbrdf_norm:
                    imglist = [_norm(v,gamma_correct,order) for v in imglist]
                if concat:
                    return [np.concatenate(v,axis=-1) for v in imglist]
            else:
                imglist = np.split(imgs,split_num,axis=split_axis)
                if svbrdf_norm:
                    imglist = _norm(imglist, gamma_correct, order)
                if concat:
                    return np.concatenate(imglist,axis=-1)

    def brdf2uint8(self, svbrdf, n_xy=True, gamma_correct=None):
        if gamma_correct is None:
            gamma_correct = self.opt.get('gamma_correct', '')
        if not n_xy:
            svbrdf = svbrdf/2+0.5
            n = svbrdf[:,0:3]
            d = svbrdf[:,3:6]
            r = svbrdf[:,6:7]
            s = svbrdf[:,7:10]
        else:
            n = svbrdf[:, 0:2]
            n = self.unsqueeze_normal(n, device='cpu')/2+0.5
            d = svbrdf[:,2:5]/2+0.5
            r = svbrdf[:,5:6]/2+0.5
            s = svbrdf[:,6:9]/2+0.5
        
        if 'd' in gamma_correct:
            d = d**0.4545
        if 'r' in gamma_correct:
            r = r**0.4545
        if 's' in gamma_correct:
            s = s**0.4545
        r = torch.cat([r]*3, dim=1)
        result = torch.cat([n,d,r,s], dim=3)
        return result
    def fullRandomLightsSurface(self):
        if self.nbRendering == 1:
            centerPos = torch.zeros([1,2])
            currentLightPos = torch.cat([centerPos, torch.ones([self.nbRendering, 1])* lightDistance], axis = -1)
        else:
            nbRenderings = self.nbRendering - 1
            currentLightPos = torch.rand(nbRenderings, 2) * 2.4 - 1.2
            centerPos = torch.zeros([1,2])
            currentLightPos = torch.cat([centerPos,currentLightPos], axis = 0)
            currentLightPos = torch.cat([currentLightPos, torch.ones([self.nbRendering, 1])* lightDistance], axis = -1)

        # [n, 3]
        return currentLightPos.float()
    def fullRandomCosine(self, n_diff = 3, n_spec = 6,batch_size=0,surface_size=256):
        # direct lighting
        currentViewPos_diff = self.random_dir_cosine(n = n_diff,batch=batch_size)
        currentLightPos_diff = self.random_dir_cosine(n = n_diff,batch=batch_size)
        wo_d = torch.tile(currentViewPos_diff.unsqueeze(-1).unsqueeze(-1),(1,1,1,surface_size,surface_size))
        wi_d = torch.tile(currentLightPos_diff.unsqueeze(-1).unsqueeze(-1),(1,1,1,surface_size,surface_size))
        
        #specular 
        currentViewPos_spec = self.random_dir_cosine(n = n_spec,batch=batch_size)
        currentLightPos_spec = currentViewPos_spec * torch.from_numpy(np.array([[[-1,-1,1]]], dtype=np.float32)).cuda()
        currentShift = torch.cat([torch.empty([batch_size,n_spec,2], dtype=torch.float32).uniform_(-1.0, 1.0),torch.zeros([batch_size,n_spec,1])+0.0001],axis=-1).cuda()
        
        gaussian_view = torch.exp(torch.empty([batch_size,n_spec,1], dtype=torch.float32).normal_(0.5,0.75)).cuda()
        gaussian_light = torch.exp(torch.empty([batch_size,n_spec,1], dtype=torch.float32).normal_(0.5,0.75)).cuda()
        
        currentViewPos_spec = currentViewPos_spec*gaussian_view + currentShift
        currentLightPos_spec = currentLightPos_spec*gaussian_light + currentShift
        
        surface = self.surface(surface_size).unsqueeze(0).cuda()
        wo_s = torch_norm(torch.tile(currentViewPos_spec.unsqueeze(-1).unsqueeze(-1),(1,1,1,surface_size,surface_size)) - surface,dim=2) 
        wi_s = torch_norm(torch.tile(currentLightPos_spec.unsqueeze(-1).unsqueeze(-1),(1,1,1,surface_size,surface_size)) - surface,dim=2) 
        
        wo = torch.cat([wo_d,wo_s],dim=1)
        wi = torch.cat([wi_d,wi_s],dim=1)
        return wi, wo
    def fixedLightsSurface(self):
        nbRenderings = self.nbRendering
        currentLightPos = torch.from_numpy(np.array([0.0, 0.0, lightDistance])).view(1,3)
        currentLightPos = torch.cat([currentLightPos]*nbRenderings, dim=0)
        # [n, 3]
        return currentLightPos.float()

    def fixedView(self):
        currentViewPos = torch.from_numpy(np.array([0.0, 0.0, viewDistance])).view(1,3)
        # [1,1,3]
        return currentViewPos.float()

    def shuffle_sample(self, max_num, num, n, ob, l, v, p, d, mask):
        idx = np.random.permutation(np.arange(n.shape[-1]*n.shape[-2]-torch.sum(mask).numpy()))
        arr = [n.unsqueeze(0),ob.unsqueeze(0),l,v,p[:,:2],d]
        c = 0
        samples = []
        for entity in arr:
            c +=entity.shape[1]
            samples.append(entity[0])
        tmp = torch.cat(samples, dim=0)
        tmp = tmp.masked_select(~mask).view(tmp.shape[0], -1)[:,idx[:num]].numpy().T
        if max_num is not None:
            trace = np.zeros((max_num, c))
            trace[:num, :] = tmp
            idx = np.random.permutation(np.arange(max_num))
            trace = trace[idx, :]
            mask = idx < num
        else:
            trace = tmp
            mask = idx[:num] > -1
        return trace.astype(np.float32), mask
        
    def surface(self, size,range=1):
        x_range = torch.linspace(-range,range,size)
        y_range = torch.linspace(-range,range,size)
        y_mat, x_mat = torch.meshgrid(x_range, y_range)
        pos = torch.stack([x_mat, -y_mat, torch.zeros(x_mat.shape)],axis=0)
        pos = torch.unsqueeze(pos,0)
        return pos

    def torch_generate(self, camera_pos_world, light_pos_world,surface=None, pos = None):
        # permute = self.opt['permute_channel']
        size = self.opt['size']
        nl,_ = light_pos_world.shape
        nv,_ = camera_pos_world.shape
        if pos is None and surface is None:
            pos = self.surface(size)
        elif surface is not None:
            pos = surface

        
        light_pos_world = light_pos_world.reshape(nl,1,1,3)
        camera_pos_world = camera_pos_world.reshape(nv,1,1,3)

        light_pos_world = light_pos_world.permute(0,3,1,2).contiguous()
        camera_pos_world = camera_pos_world.permute(0,3,1,2).contiguous()
        view_dir_world = torch_norm(camera_pos_world - pos, dim=1)

        # pos = torch.tile(pos,[n,1,1,1])
        light_dis_square = torch.sum(torch.square(light_pos_world - pos),1,keepdims=True)

        light_dir_world = torch_norm(light_pos_world - pos, dim=1)

        return light_dir_world, view_dir_world, light_dis_square, pos
    
    def plane_normal(self, batch = 0):
        size = self.opt['size']
        pos = torch.zeros((2,size,size),dtype=torch.float32)
        z_mat = torch.ones((1,size,size),dtype=torch.float32)
        pos = torch_norm(torch.cat([pos, z_mat],axis=0),dim=0)
        if batch > 0:
            pos = pos.unsqueeze(0).repeat(batch,1,1,1)
        return pos

    def sphere_normal(self, batch = 0, padding=False):
        size = self.opt['size']
        r = 1
        x_range = torch.linspace(-1,1,size)
        y_range = torch.linspace(-1,1,size)
        y_mat, x_mat = torch.meshgrid(x_range, y_range)
        pos = torch.stack([x_mat, -y_mat],axis=0)
        z_mat = torch.maximum(1-pos[0:1]**2-pos[1:]**2, torch.zeros_like(pos[1:]))
        normal = torch.cat([pos, z_mat],axis=0)
        mask = ~(torch.sum(normal**2, axis=0, keepdim=True) > 1)
        normal = normal * mask
        if padding:
            normal = normal + torch.cat([torch.zeros((2,size,size)), (~mask).float()],dim=0)
        normal = torch_norm(normal,dim=0)
        if batch > 0:
            normal = normal.unsqueeze_(0).repeat(batch,1,1,1)
            mask.unsqueeze_(0)
        return normal, mask
   
    def random_dir(self, batch = 0, n = 1, theta_range=[0,70], phi_range=[0,360]):
        if batch == 0:
            shape = (1, n)
        else:
            shape = (batch, 1, n)
        theta = torch.empty(shape, dtype=torch.float32).uniform_(theta_range[0]/180*np.pi,theta_range[1]/180*np.pi)
        phi = torch.empty(shape, dtype=torch.float32).uniform_(phi_range[0]/180*np.pi,phi_range[1]/180*np.pi)

        x = torch.sin(theta)*torch.cos(phi)
        y = torch.sin(theta)*torch.sin(phi)
        z = torch.cos(theta)
        return torch.cat([x,y,z],dim=-2)

    def random_pos(self, batch = 0, n = 1, r=[-1,1]):
        if batch == 0:
            shape = (2, n)
        else:
            shape = (batch, 2, n)
        xy = torch.empty(shape, dtype=torch.float32).uniform_(r[0],r[1])
        if batch == 0:
            shape = (1, n)
        else:
            shape = (batch, 1, n)
        z = torch.zeros(shape, dtype=torch.float32)
        return torch.cat([xy,z], dim=-2)
    
    def random_dir_cosine(self, batch = 0, n = 1, lowEps = 0.001, highEps =0.05):
        if batch == 0:
            shape = (n,1)
        else:
            shape = (batch,n,1)
        r1 = torch.empty(shape, dtype=torch.float32).uniform_(0.0 + lowEps, 1.0 - highEps).cuda()
        r2 = torch.empty(shape, dtype=torch.float32).uniform_(0.0, 1.0).cuda()
        r = torch.sqrt(r1)
        phi = 2 * math.pi * r2
        
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        z = torch.sqrt(1.0 - torch.square(r))
        finalVec = torch.cat([x, y, z], dim=-1) #Dimension here is [batchSize, n,3]
        return finalVec
    def _render(self,inputs,l,v,dis, n_xy=False, r_single=True, use_spec=True, isAmbient=False, per_point=False):
        INV_PI = 1.0 / math.pi
        EPS = 1e-12
        eps = torch.ones_like(dis)*1e-12
        def GGX(NoH, roughness):
            alpha = roughness  * roughness
            tmp = alpha / torch.max(eps,  (NoH * NoH * (alpha * alpha - 1.0) + 1.0  ) )
            return tmp * tmp * INV_PI

        def SmithG(NoV, NoL, roughness):
            def _G1(NoM, k):
                return 1 / (NoM * (1.0 - k ) + k)

            k = torch.max(eps, roughness * roughness * 0.5)
            return _G1(NoL,k) * _G1(NoV, k)

        def Fresnel(F0, VoH):
            coeff = VoH * (-5.55473 * VoH - 6.98316)
            return F0 + (1.0 - F0) *(2**coeff)
        
        n,d,r,s = self._seperate_brdf(inputs, n_xy, r_single)

        if self.opt["svbrdf_norm"]:
            r = r*0.5+0.5
            d = d*0.5+0.5
            s = s*0.5+0.5
        else:
            n = n*2-1
        if per_point:
            dim = -2
        else:
            dim = -3
        n = torch_norm(n, dim=dim)
        h = torch_norm((l+v) * 0.5 , dim=dim)

        r = torch.max(r, torch.ones_like(r)*0.001)
        NoH = torch_dot(n,h, dim=dim)
        NoV = torch_dot(n,v, dim=dim)
        NoL = torch_dot(n,l, dim=dim)
        VoH = torch_dot(v,h, dim=dim)


        NoH = torch.max(NoH,eps )
        NoV = torch.max(NoV, eps)
        NoL = torch.max(NoL,eps)
        VoH = torch.max(VoH, eps)
        
        f_d = d * (1-s) * INV_PI

        D = GGX(NoH,r)
        G = SmithG(NoV, NoL, r)
        F = Fresnel(s, VoH)
        f_s = D * G * F * 0.25

        if use_spec:
            res =  (f_d + f_s) * NoL * torch.pi
        else:
            res = (f_d) * NoL * torch.pi
        lampIntensity = self.get_intensity(isAmbient,self.useAugmentation, self.nbRendering)

        res *= lampIntensity
        return res
    def _render_back(self,inputs,l,v,dis, n_xy=False, r_single=True, use_spec=True, isAmbient=False, per_point=False):
        INV_PI = 1.0 / math.pi
        EPS = 1e-12
        eps = torch.ones_like(dis)*1e-12
        def GGX(NoH, roughness):
            alpha = roughness  * roughness
            tmp = alpha / torch.max(eps,  (NoH * NoH * (alpha * alpha - 1.0) + 1.0  ) )
            return tmp * tmp * INV_PI

        def SmithG(NoV, NoL, roughness):
            def _G1(NoM, k):
                return NoM / (NoM * (1.0 - k ) + k)

            k = torch.max(eps, roughness * roughness * 0.5)
            return _G1(NoL,k) * _G1(NoV, k)

        def Fresnel(F0, VoH):
            coeff = VoH * (-5.55473 * VoH - 6.98316)
            return F0 + (1.0 - F0) *(2**coeff)
        
        n,d,r,s = self._seperate_brdf(inputs, n_xy, r_single)

        if self.opt["svbrdf_norm"]:
            r = r*0.5+0.5
            d = d*0.5+0.5
            s = s*0.5+0.5
        else:
            n = n*2-1
        if per_point:
            dim = -2
        else:
            dim = -3
        n = torch_norm(n, dim=dim)
        h = torch_norm((l+v) * 0.5 , dim=dim)

        NoH = torch_dot(n,h, dim=dim)
        NoV = torch_dot(n,v, dim=dim)
        NoL = torch_dot(n,l, dim=dim)
        VoH = torch_dot(v,h, dim=dim)


        NoH = torch.max(NoH, eps)
        NoV = torch.max(NoV, eps)
        NoL = torch.max(NoL, eps)
        VoH = torch.max(VoH, eps)

        f_d = d * INV_PI

        D = GGX(NoH,r)
        G = SmithG(NoV, NoL, r)
        F = Fresnel(s, VoH)
        f_s = D * G * F / (4.0 * NoL * NoV + EPS)

        if use_spec:
            res =  (f_d + f_s) * NoL * math.pi/dis
        else:
            res = (f_d) * NoL * math.pi /dis
        lampIntensity = self.get_intensity(isAmbient,self.useAugmentation, self.nbRendering)
        if isinstance(lampIntensity,torch.Tensor):
            shape = [res.shape[s] if s == 0 else 1 for s in range(len(res.shape))]
            res = res * lampIntensity.view(*shape)
        else: 
            res *= lampIntensity
        # res = torch.clip(res, 0, 1)
        return res
    
    def get_intensity(self, isAmbient=False, useAugmentation=False, ln=1):
        if not isAmbient:
            if useAugmentation:
                #The augmentations will allow different light power and exposures                 
                stdDevWholeBatch = torch.exp(torch.randn(()).normal_(mean = -2.0, std = 0.5))
                #add a normal distribution to the stddev so that sometimes in a minibatch all the images are consistant and sometimes crazy.
                lampIntensity = torch.abs(torch.randn((ln)).normal_(mean = 1.5, std = stdDevWholeBatch)) # Creates a different lighting condition for each shot of the nbRenderings Check for over exposure in renderings
                #autoExposure
                autoExposure = torch.exp(torch.randn(()).normal_(mean = np.log(1), std = 0.4))
                lampIntensity = lampIntensity * autoExposure
            elif hasattr(self, 'lampIntensity'):
                lampIntensity = self.lampIntensity
            else:
                lampIntensity = 3.0 #Look at the renderings when not using augmentations
        else:
            #If this uses ambient lighting we use much small light values to not burn everything.
            if useAugmentation:
                lampIntensity = torch.exp(torch.randn(()).normal_(mean = torch.log(0.15), std = 0.5)) #No need to make it change for each rendering.
            else:
                lampIntensity = 0.15
        return lampIntensity
    
    def get_svbrdfs(self, imgs):
        return img2tensor(self.split_img(imgs),bgr2rgb=False, float32=True, normalization=False)
    
    def homo2sv(self, brdf):
        if len(brdf.shape) >= 2:
            n, c = brdf.shape
            svbrdf = torch.ones((n,c,self.size, self.size),dtype=torch.float32)*(brdf.unsqueeze(-1).unsqueeze(-1))
        else:
            c = brdf.shape[0]
            svbrdf = torch.ones((c, self.size, self.size), dtype=torch.float32)*(brdf.unsqueeze(1).unsqueeze(1))
        return svbrdf

    def squeeze_normal(self, n):
        '''
            n: normal, shape: (3, h, w)
            return: svbrdf, shape:( 2,h,w)
        '''
        n = torch_norm(n, 1)[:,:2]
        return n
    def unsqueeze_normal(self, n, device='cpu'):
        b,c,h,w = n.shape
        # tmp = torch.ones((b, 1, h, w),dtype=torch.float32,device=device) - torch.sum(n**2, dim=1)
        # n = torch.cat([n,torch.sqrt(tmp)], dim=1)
        n = torch.cat([n,torch.ones((b, 1, h, w),dtype=torch.float32,device=device)], dim=1)
        n = torch_norm(n, 1)
        return n
    
    def _seperate_brdf(self, svbrdf, n_xy=False, r_single=True, device='cpu'):
        if svbrdf.dim() == 4:
            b,c,h,w = svbrdf.shape
        else:
            b = 0
            svbrdf = svbrdf.unsqueeze(0)
        
        if self.opt['order'] == 'pndrs' or self.opt['order'] == 'ndrs':
            if not n_xy:
                n = svbrdf[:,0:3]
                d = svbrdf[:,3:6]
                if r_single:
                    r = svbrdf[:,6:7]
                    s = svbrdf[:,7:10]
                else:
                    r = svbrdf[:, 6:7]
                    s = svbrdf[:, 9:12]
            else:
                n = svbrdf[:, 0:2]
                n = self.unsqueeze_normal(n, device=device)
                d = svbrdf[:,2:5]
                if r_single:
                    r = svbrdf[:,5:6]
                    s = svbrdf[:,6:9]
                else:
                    r = svbrdf[:,5:6]
                    s = svbrdf[:,8:11]
        elif self.opt['order'] == 'dnrs':
            d = svbrdf[:,0:3]
            n = svbrdf[:,3:6]
            r = svbrdf[:,6:7]
            s = svbrdf[:,7:10]
        if b != 0:
            n = n.unsqueeze(1)
            d = d.unsqueeze(1)
            r = r.unsqueeze(1)
            s = s.unsqueeze(1)
        return n,d,r,s
    def get_svbrdf_torch(self,svbrdfimg):
        # [n,h,w,c]
        normal,diffuse,roughness,specular = torch.split(svbrdfimg,256,2)
        roughness = torch.mean(roughness,-1,keepdim=True)
        svbrdf = torch.cat([normal,diffuse,roughness,specular],-1)
        # [n,c,h,w]
        svbrdf = svbrdf.permute(0,3,1,2)
        return svbrdf*2-1 #[-1,1]
        
    def squeeze_brdf(self, svbrdf, device='cpu'):
        n,d,r,s = self._seperate_brdf(svbrdf, device=device)
        n = self.squeeze_normal(n)
        n = n.squeeze(0)
        d = d.squeeze(0)
        r = r.squeeze(0)
        s = s.squeeze(0)
        return torch.cat([n,d,r,s],dim=0)
    def unsqueeze_brdf(self, svbrdf, device='cpu'):
        n,d,r,s = self._seperate_brdf(svbrdf,n_xy=True, device=device)
        n = n.squeeze(0)
        d = d.squeeze(0)
        r = r.squeeze(0)
        s = s.squeeze(0)
        return torch.cat([n,d,r,s],dim=1)
    def generate_pos(self,view_pos,light_pos= None, r_single=True, toLDR=None, device='cpu'):
        if toLDR is None:
            toLDR = self.opt['toLDR']
        view_pos= torch.Tensor(view_pos,device="cpu").unsqueeze(0)
        if light_pos is None:
            light_pos = view_pos
        else:
            light_pos= torch.Tensor(light_pos,device=device).unsqueeze(0)
            
        light_dir, view_dir, light_dis, surface = self.torch_generate(view_pos, light_pos)
        light_dis = torch.max(light_dir[:,2:3],torch.mean(torch.ones_like(light_dir),2,keepdim=True)*0.001)
        light_dir = light_dir.to(device)
        view_dir = view_dir.to(device)
        light_dis = light_dis.to(device)

        return light_dir,view_dir,light_dis

    def render_from_pos(self, svbrdf, view_pos,light_pos= None, r_single=True, toLDR=None, device='cpu'):
        if toLDR is None:
            toLDR = self.opt['toLDR']
        view_pos= torch.Tensor(view_pos,device=device).unsqueeze(0)
        if light_pos is None:
            light_pos = view_pos
        else:
            light_pos= torch.Tensor(light_pos,device=device).unsqueeze(0)
            
        light_dir, view_dir, light_dis, surface = self.torch_generate(view_pos, light_pos)
        light_dis = torch.max(light_dir[:,2:3],torch.mean(torch.ones_like(light_dir),2,keepdim=True)*0.001)
        svbrdf = svbrdf.to(device)
        light_dir = light_dir.to(device)
        view_dir = view_dir.to(device)
        light_dis = light_dis.to(device)
        render_result = self._render(svbrdf, light_dir, view_dir, light_dis)
        if toLDR:
            render_result = toLDR_torch(render_result)
            render_result = toHDR_torch(render_result)
        return render_result
    def render(self, svbrdf, single_pos=False, obj_pos=None, light_dir=None,\
         view_dir=None, light_dis=None, surface=None, random_light=True, colocated=True,\
              n_xy=False, r_single=True, toLDR=False, use_spec = True, per_point=False, isAmbient=False):
        if toLDR is None:
            toLDR = self.opt['toLDR']
        if light_dir is None:
            if random_light:
                light_pos = self.fullRandomLightsSurface().cuda()
            else:
                light_pos = self.fixedLightsSurface()
            if colocated:
                view_pos = light_pos
            else:
                view_pos = self.fixedView()

            if obj_pos is None and single_pos:
                obj_pos = torch.rand((self.nbRendering, 3), dtype=torch.float32)*2-1
                obj_pos[:, 2] = 0.0
            light_dir, view_dir, light_dis, surface = self.torch_generate(view_pos, light_pos, pos=obj_pos)
        if light_dis is None:
            light_dis = torch.max(light_dir[:,:,2:3],torch.mean(torch.ones_like(light_dir),2,keepdim=True)*0.001)
        render_result = self._render(svbrdf, light_dir, view_dir, light_dis, n_xy=n_xy, r_single=r_single, use_spec=use_spec, per_point=per_point, isAmbient=isAmbient)
        if toLDR:
            render_result = toLDR_torch(render_result)
            render_result = toHDR_torch(render_result)
        # [n,c,h,w]
        return render_result, light_dir, view_dir, surface, light_dis
    def set_lamp_intensity(self, lamp_intensity):
        self.lampIntensity = lamp_intensity
    
    def reset_lamp_intensity(self):
        if hasattr(self, 'lampIntensity'):
            self.lampIntensity = self.opt['lampIntensity']
        else:
            del self.lampIntensity

def toLDR_torch(img, gamma=True, sturated=True):
    if gamma:
        img = img**0.4545
    if sturated:
        img = torch.clip(img,0,1)
    img = (img.numpy()*255).astype(np.uint8)
    img = torch.from_numpy(img)
    return img

def toHDR_torch(img, gamma=True):
    img = img.float()/255
    if gamma:
        img = img ** 2.2
    return img

def preprocess(img):
    # [0,1] => [-1,1]
    return img * 2.0 - 1.0

def deprocess(img):
    # [-1,1] => [0,1]
    return (img + 1.0) / 2.0
def torch_norm(arr, dim=1):
    length = torch.sqrt(torch.sum(arr * arr, dim = dim, keepdims=True))
    return arr / (length + 1e-12)
    
def numpy_norm(arr, dim=-1):
    length = np.sqrt(np.sum(arr * arr, axis = dim, keepdims=True))
    return arr / (length + 1e-12)
def torch_dot(a,b, dim=-3):
    return torch.sum(a*b,dim=dim,keepdims=True)


