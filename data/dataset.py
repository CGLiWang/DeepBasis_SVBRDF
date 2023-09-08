import os
import torch
from torch.utils import data as data
from utils import svBRDF, FileClient, imfrombytes,img2tensor
import torch.nn.functional as F

class DeepBasisDataset(data.Dataset):
    def __init__(self,opt):
        super(DeepBasisDataset).__init__()
        self.opt = opt
        # self.eps = torch.ones([256,256])*-0.8
        self.file_client = None
        self.io_backend_opt = {
            'type': 'disk'
        }
        
        svbrdf_folder = opt['svbrdf_root']
        svbrdf_names = os.listdir(svbrdf_folder)
        svbrdf_names.sort()
        self.svbrdf_paths = []
        for name in svbrdf_names:
            self.svbrdf_paths.append(os.path.join(svbrdf_folder,name))
        brdf_opt = {
            'nbRendering':1,
            'size': 256,
            'split_num': 5,
            'split_axis': 1,
            'concat': True,
            'svbrdf_norm': True,
            'permute_channel': True,
            'order': 'pndrs',
            'lampIntensity': 1
        }
        self.svBRDF_utils = svBRDF(brdf_opt)
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
            
    def __getitem__(self, index):
        img_bytes = self.file_client.get(self.svbrdf_paths[index], 'brdf')
        img = imfrombytes(img_bytes, float32=True)[:,:,::-1]
        if self.opt.get("real_input",False):
            inputs = img2tensor(img.copy(),bgr2rgb=False).unsqueeze(0)
            inputs = F.interpolate(inputs,size=256,mode='bilinear').squeeze(0)
            if self.opt.get("input_gamma",False):
                inputs = inputs**2.2
            pass
        else:
            inputs = img[:,:256]
            inputs = img2tensor(inputs.copy(),bgr2rgb=False)
            svbrdfs = self.svBRDF_utils.get_svbrdfs(img)
                
        if self.opt.get('log', False):
            log_inputs = log_normalization(inputs)
            inputs = torch.cat([inputs,log_inputs],0)            
            
        if self.opt.get("real_input",False):
            return {'inputs':inputs*2-1,'name':os.path.basename(self.svbrdf_paths[index])}
        else:
            return {'inputs':inputs*2-1,'svbrdfs':svbrdfs,'name':os.path.basename(self.svbrdf_paths[index])}
    def __len__(self):
        return len(self.svbrdf_paths)
def log_normalization(img, eps=1e-2):
    return (torch.log(img+eps)-torch.log(torch.ones((1,))*eps))/(torch.log(1+torch.ones((1,))*eps)-torch.log(torch.ones((1,))*eps))