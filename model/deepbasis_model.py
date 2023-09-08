import os
from collections import OrderedDict
from copy import deepcopy
from tkinter import image_names
from torch.nn.parallel import DataParallel, DistributedDataParallel
from tqdm import tqdm
import torch
import numpy as np
import os.path as osp
from torch.cuda.amp import GradScaler as GradScaler
from utils import svBRDF,imwrite,tensor2img,torch_norm
from torch.autograd import Variable

from network.deepbasis_net import GlobalNet,LocalNet

class DeepBasisModel():

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda')
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
        self.renderer = svBRDF(brdf_opt)
        self.init_rendering()
        self.init_network()
        if args.mode == 'train':
            self.schedulers = []
            self.optimizers = []
            self.init_training_settings()
        
    def init_rendering(self):
        # init lighting direction
        surface = self.renderer.surface(384,1.5).to('cpu')
        view_pos = np.array([0,0,self.args.fovZ])
        view_pos= torch.Tensor(view_pos,device="cpu").unsqueeze(0)
        light_dir, view_dir, _, _ = self.renderer.torch_generate(view_pos, view_pos,pos=surface)
        self.light_dir = light_dir.cuda()
        self.view_dir = view_dir.cuda()
        x = 192
        y = 192
        self.sample_light_dir = self.light_dir[:,:,x-128:x+128,y-128:y+128]
        self.sample_view_dir = self.view_dir[:,:,x-128:x+128,y-128:y+128]
        self.sample_light_dis = torch.max(self.sample_view_dir[:,2:3],torch.mean(torch.ones_like(self.sample_view_dir),2,keepdim=True)*0.001)
        
        # init lambertain svbrdf
        up_normal = torch.concat([torch.zeros([1,256,256]),torch.zeros([1,256,256]),torch.ones([1,256,256])],0)*0.5+0.5
        lam_diffuse = torch.ones_like(up_normal)*0.1
        lam_specular = torch.ones_like(up_normal)*0.5
        lam_roughness = torch.ones_like(up_normal)*0.15
        lam_roughness = torch.mean(lam_roughness,0,keepdim=True)
        lam_svbrdf = torch.cat([up_normal,lam_diffuse,lam_roughness,lam_specular],0).cuda()
        self.lam_svbrdf = lam_svbrdf*2-1
    def init_network(self):
        self.net_g = GlobalNet()
        self.net_g = self.model_to_device(self.net_g)

        self.net_l = LocalNet()
        self.net_l = self.model_to_device(self.net_l)
        
        if not self.args.mode == 'train':
            self.load_network(self.net_g,self.args.loadpath_network_g)
            self.load_network(self.net_l,self.args.loadpath_network_l)

    def init_training_settings(self):
        self.net_g.train()
        self.net_l.train()

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        
    def setup_optimizers(self):
        
        optim_params = [param for param in self.net_g.parameters()]
        self.optimizer_g = torch.optim.Adam(optim_params,2e-5)
        self.optimizers.append(self.optimizer_g)
        for p in self.net_g.parameters():
            p.requires_grad = True
        
        optim_params_l = [param for param in self.net_l.parameters()]
        self.optimizer_m = torch.optim.AdamW(optim_params_l,lr=2e-3,weight_decay=0.,betas=[0.9, 0.9])
        self.optimizers.append(self.optimizer_m)
        for p in self.net_l.parameters():
            p.requires_grad = True
    def setup_schedulers(self):

        for optimizer in self.optimizers:
            self.schedulers.append(
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400000, eta_min=1e-7))


    def log_normalization(self,img, eps=1e-2):
        return (torch.log(img+eps)-torch.log(torch.ones((1,),device="cuda")*eps))/(torch.log(1+torch.ones((1,),device='cuda')*eps)-torch.log(torch.ones((1,),device='cuda')*eps))     
    def feed_data(self, data, random=True):
        
        if self.args.mode == 'real':
            self.inputs = data['inputs'].cuda()
        else:
            self.svbrdf = data['svbrdfs'].cuda()
            # start = time.time()
            if random:
                x_d, y_d = np.clip(np.random.normal(0,0.33,2)*64,-64,64)
                x = int(x_d) + 192
                y = int(y_d) + 192
            else:
                x = 192
                y = 192
            self.sample_light_dir = self.light_dir[:,:,x-128:x+128,y-128:y+128]
            self.sample_view_dir = self.view_dir[:,:,x-128:x+128,y-128:y+128]
            self.sample_light_dis = torch.max(self.sample_view_dir[:,2:3],torch.mean(torch.ones_like(self.sample_view_dir),2,keepdim=True)*0.001)

            inputs = self.renderer._render(self.svbrdf,self.sample_light_dir,self.sample_view_dir,self.sample_light_dis).squeeze(1)
            inputs = torch.clip(inputs,0,1)
            log_inputs = self.log_normalization(inputs)
            lam_rendering = self.renderer._render(self.lam_svbrdf,self.sample_light_dir,self.sample_view_dir,self.sample_light_dis).squeeze(1)
            lam_rendering = torch.tile(torch.clip(lam_rendering,0,1),[inputs.shape[0],1,1,1])
            
            self.inputs = torch.cat([inputs,log_inputs,lam_rendering],1)*2-1

            # self.inputs = data['inputs'].cuda()
            # print(time.time()-start)
        self.name = data['name']
        
    def weight_compute(self,weight):
        weights = weight*0.5+0.5
        weights_sum = torch.sum(weights,1,keepdim=True)
        weights_clip = weights/(weights_sum+1e-6)
        weights_clip = weights_clip.unsqueeze(2)
        return weights_clip
    def get_svbrdf_sum(self,weights, basis):
        svbrdf = torch.sum(weights*basis,1,keepdim=False)
        return svbrdf
    def optimize_parameters(self, current_iter):
        loss_dict = OrderedDict()

        self.optimizer_g.zero_grad()
        self.optimizer_m.zero_grad()


        normal_kesi, other_kesi = torch.split(self.svbrdf,[3,7],1)
        
        theta = np.random.uniform(0,np.pi/6,self.args.bSize)
        rotate_matrix = torch.from_numpy(np.array([[np.cos(theta),-np.sin(theta),np.zeros_like(theta)],
                            [np.sin(theta), np.cos(theta),np.zeros_like(theta)],
                            [np.zeros_like(theta),np.zeros_like(theta),np.ones_like(theta)]], dtype="float32")).cuda()
        rotate_matrix = torch.permute(rotate_matrix,(2,0,1))
        rotate_matrix = rotate_matrix.view(self.args.bSize,1,1,3,3)
        normal_kesi = normal_kesi.permute(0,2,3,1).unsqueeze(-1)
        normal_kesi = torch.matmul(rotate_matrix,normal_kesi)
        normal_noise = normal_kesi.squeeze(-1).permute(0,3,1,2)
        
        kesi = torch.empty((self.args.bSize,7,1,1),dtype=torch.float32,device='cuda').uniform_(0,0.3)
        one_plus_kesi= torch.ones_like(kesi,device='cuda')+kesi
        other_noise = ((other_kesi*0.5+0.5)+kesi)/one_plus_kesi
        
        svbrdf_noise = torch.cat([normal_noise,other_noise*2-1],1)
        rendering_image = self.renderer._render(svbrdf_noise,self.sample_light_dir,self.sample_view_dir,self.sample_light_dis).squeeze(1)
        new_input = torch.clip(rendering_image,0,1)
        log_new_input = self.log_normalization(new_input)
        new_inputs = torch.cat([new_input,log_new_input,torch.tile(self.sample_light_dir,[self.args.bSize,1,1,1])],1)*2-1
        self.inputs = torch.cat([self.inputs,new_inputs],0)
        weight,basis = self.net_g(self.inputs)

        weight = self.weight_compute(weight)
        basis_map = torch.tile(basis.unsqueeze(-1).unsqueeze(-1),(1,1,1,256,256))
        temp_svbrdf =  torch.sum(weight*basis_map,1,keepdim=False) 
        render_1 = self.renderer._render(temp_svbrdf,self.sample_light_dir,self.sample_view_dir,self.sample_light_dis).squeeze(1)
        render_2 = self.inputs[:,:3]*0.5+0.5
        error = (torch.clip(render_1,0,1)-render_2)
        
        delta_basis = self.net_l(torch.cat([self.inputs,error],1),basis,weight.squeeze(2))  
        multi_basis = delta_basis + basis_map
        n,d,r,s = torch.split(multi_basis,[3,3,1,3],2)
        new_n = torch_norm(n,2)
        multi_basis = torch.concat([new_n,d,r,s],2)
        multi_basis = torch.clip(multi_basis,-1,1)
        
        fake_svbrdf = self.get_svbrdf_sum(weight,multi_basis)

        weight, new_weight = torch.split(weight,[self.args.bSize,self.args.bSize],0)
        multi_basis, new_multi_basis = torch.split(multi_basis,[self.args.bSize,self.args.bSize],0)
        fake_svbrdf, new_fake_svbrdf = torch.split(fake_svbrdf,[self.args.bSize,self.args.bSize],0)
        final_basis_map = multi_basis
    

        l_total = 0
            
        # L1 loss
        l_pix = torch._C._nn.l1_loss(self.svbrdf,fake_svbrdf,torch.nn.functional._Reduction.get_enum('mean'))
        loss_dict['l_pix'] = l_pix
        l_total += l_pix
        
        # variation-consistency loss
        rotate_matrix = rotate_matrix.view(self.args.bSize,1,1,1,3,3)
        normal_basis, other_basis  = torch.split(multi_basis,[3,7],2)
        normal_basis = normal_basis.permute(0,1,3,4,2).unsqueeze(-1)
        normal_basis = torch.matmul(rotate_matrix,normal_basis).squeeze(-1)
        normal_basis = normal_basis.permute(0,1,4,2,3)
        
        kesi = kesi.unsqueeze(1)
        one_plus_kesi = one_plus_kesi.unsqueeze(1)
        other_basis = ((other_basis*0.5+0.5)+kesi)/one_plus_kesi
        basis_noise = torch.cat([normal_basis,other_basis*2-1],2)

        l_basis = torch.mean(torch.abs(basis_noise-new_multi_basis))
        # l_w = torch.mean(torch.abs(torch.rot90(weight,1,dims=[-2,-1])-new_weight))
        l_sv = torch.mean(torch.abs(svbrdf_noise-new_fake_svbrdf))
            
        l_kesi = l_basis + l_sv
        l_total += self.args.weight_vc * l_kesi
        loss_dict["l_kesi"] = self.args.weight_vc * l_kesi    
        l_total.backward()
        
        self.optimizer_g.step()
        self.optimizer_m.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
    def reduce_loss_dict(self, loss_dict):
        with torch.no_grad():

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict

    def test(self):            

        self.net_g.eval()
        self.net_l.eval()
        if not self.args.mode == 'real':
            with torch.no_grad():
                self.pred(self.inputs,self.sample_light_dir,self.sample_view_dir,self.sample_light_dis)
        else:
            with torch.no_grad():
                x = 192
                y = 192
                sample_light_dir = self.light_dir[:,:,x-128:x+128,y-128:y+128]
                sample_view_dir = self.view_dir[:,:,x-128:x+128,y-128:y+128]
                sample_light_dis = torch.max(sample_view_dir[:,2:3],torch.mean(torch.ones_like(sample_view_dir),2,keepdim=True)*0.001)
                lam_rendering = self.renderer._render(self.lam_svbrdf,sample_light_dir,sample_view_dir,sample_light_dis).squeeze(1)
                lam_rendering = torch.clip(lam_rendering,0,1)
                inputs = torch.cat([self.inputs,lam_rendering*2-1],1)
                self.pred(inputs,sample_light_dir,sample_view_dir,sample_light_dis)
                
                lr_list = []
                vr_list = []
                dis_list = []
                start_x = max(x-10,128)
                end_x = min(x+10,256)
                start_y = max(y-10,128)
                end_y = min(y+10,256)
                x_list = []
                y_list = []
                for i in range(start_x,end_x):
                    for j in range(start_y,end_y):
                        sample_light_dir = self.light_dir[:,:,i-128:i+128,j-128:j+128]
                        sample_view_dir = self.view_dir[:,:,i-128:i+128,j-128:j+128]
                        sample_light_dis = torch.max(sample_view_dir[:,2:3],torch.mean(torch.ones_like(sample_view_dir),2,keepdim=True)*0.001)
                        lr_list.append(sample_light_dir)
                        vr_list.append(sample_view_dir)
                        dis_list.append(sample_light_dis)
                        x_list.append(i)
                        y_list.append(j)
                light_dir = torch.cat(lr_list,0)
                vr_dir = torch.cat(vr_list,0)
                dis = torch.cat(dis_list,0)   
                render_test = self.renderer._render(self.fake_svbrdf,light_dir,vr_dir,dis).squeeze(1).squeeze(0)
                cur_loss = torch.mean(torch.abs(torch.clip(render_test,0,1)-(self.inputs[:,:3]*0.5+0.5)),dim=[1,2,3])
                for i in range(0,len(x_list)):
                    if x_list[i] == x and y_list[i] == y:
                        last_error = cur_loss[i]    
                        break
                index = torch.argmin(cur_loss)
                while True:
                    new_x = x_list[index]
                    new_y = y_list[index]
                    sample_light_dir = self.light_dir[:,:,new_x-128:new_x+128,new_y-128:new_y+128]
                    sample_view_dir = self.view_dir[:,:,new_x-128:new_x+128,new_y-128:new_y+128]
                    sample_light_dis = torch.max(sample_view_dir[:,2:3],torch.mean(torch.ones_like(sample_view_dir),2,keepdim=True)*0.001)
                    lam_rendering = self.renderer._render(self.lam_svbrdf,sample_light_dir,sample_view_dir,sample_light_dis).squeeze(1)
                    lam_rendering = torch.clip(lam_rendering,0,1)
                    inputs = torch.cat([self.inputs,lam_rendering*2-1],1)
                    self.pred(inputs,sample_light_dir,sample_view_dir,sample_light_dis)
                
                    lr_list = []
                    vr_list = []
                    dis_list = []
                    start_x = max(new_x-10,128)
                    end_x = min(new_x+10,256)
                    start_y = max(new_y-10,128)
                    end_y = min(new_y+10,256)
                    x_list = []
                    y_list = []
                    for i in range(start_x,end_x):
                        for j in range(start_y,end_y):
                            sample_light_dir = self.light_dir[:,:,i-128:i+128,j-128:j+128]
                            sample_view_dir = self.view_dir[:,:,i-128:i+128,j-128:j+128]
                            sample_light_dis = torch.max(sample_view_dir[:,2:3],torch.mean(torch.ones_like(sample_view_dir),2,keepdim=True)*0.001)
                            lr_list.append(sample_light_dir)
                            vr_list.append(sample_view_dir)
                            dis_list.append(sample_light_dis)
                            x_list.append(i)
                            y_list.append(j)
                    light_dir = torch.cat(lr_list,0)
                    vr_dir = torch.cat(vr_list,0)
                    dis = torch.cat(dis_list,0)   
                    render_test = self.renderer._render(self.fake_svbrdf,light_dir,vr_dir,dis).squeeze(1).squeeze(0)
                    cur_loss = torch.mean(torch.abs(torch.clip(render_test,0,1)-(self.inputs[:,:3]*0.5+0.5)),dim=[1,2,3])
                    for i in range(0,len(x_list)):
                        if x_list[i] == new_x and y_list[i] == new_y:
                            this_error = cur_loss[i]    
                            break
                    index = torch.argmin(cur_loss)
                    if this_error.item() >= last_error.item():
                        break
                    else:
                        last_error = this_error
                        x = new_x
                        y = new_y
                        # print(x,y)
                sample_light_dir = self.light_dir[:,:,x-128:x+128,y-128:y+128]
                sample_view_dir = self.view_dir[:,:,x-128:x+128,y-128:y+128]
                sample_light_dis = torch.max(sample_view_dir[:,2:3],torch.mean(torch.ones_like(sample_view_dir),2,keepdim=True)*0.001)
                lam_rendering = self.renderer._render(self.lam_svbrdf,sample_light_dir,sample_view_dir,sample_light_dis).squeeze(1)
                lam_rendering = torch.clip(lam_rendering,0,1)
                inputs = torch.cat([self.inputs,lam_rendering*2-1],1)
                self.pred(inputs,sample_light_dir,sample_view_dir,sample_light_dis)

                light_x = -(y-192) / 384 * 3
                light_y = (x-192) / 384 * 3
                with open(os.path.join(self.args.save_root,self.name[0][:-4].replace(".png",".txt")),"w+") as f:
                    f.write("%f,%f,%.3f %f,%f,%.3f" % (light_x,light_y,self.args.fovZ,light_x,light_y,self.args.fovZ))
                self.pred_svbrdf = self.fake_svbrdf.clone().detach()
                self.pred_single_basis = self.single_basis
                self.pred_final_basis = self.final_basis_map
            self.BasisOpt(sample_light_dir,sample_view_dir,sample_light_dis)    


        self.net_g.train()
        self.net_l.train()
    def pred(self,inputs,lr,vr,dis):
        weight,basis = self.net_g(inputs) 
        self.weight = self.weight_compute(weight)
        # [B,N,C,H,W] multi-scale details
        basis_map = torch.tile(basis.unsqueeze(-1).unsqueeze(-1),(1,1,1,256,256))
        temp_svbrdf =  torch.sum(self.weight*basis_map,1,keepdim=False) 
        render_1 = self.renderer._render(temp_svbrdf,lr,vr,dis).squeeze(1)
        render_2 = self.inputs[:,:3]*0.5+0.5
        error = (torch.clip(render_1,0,1)-render_2)
        delta_basis = self.net_l(torch.cat([inputs,error],1),basis,self.weight.squeeze(2))
        multi_basis = delta_basis + basis_map
        n,d,r,s = torch.split(multi_basis,[3,3,1,3],2)
        new_n = torch_norm(n,2)
        multi_basis = torch.concat([new_n,d,r,s],2)
        multi_basis = torch.clip(multi_basis,-1,1)

        self.final_basis_map = multi_basis
        self.fake_svbrdf = self.get_svbrdf_sum(self.weight,self.final_basis_map)
        self.single_basis = basis
        self.delta_basis = delta_basis
    def BasisOpt(self,sample_light_dir,sample_view_dir,sample_light_dis):
        v_delta_basis = self.delta_basis.detach()
        # v_delta_basis = Variable(self.delta_basis.detach(),requires_grad=True)
        v_global_basis = Variable(self.single_basis.detach(),requires_grad=True)
        weights = self.weight.detach()
        optimizer3 = torch.optim.AdamW([v_global_basis],1e-3)
        for i in range(500):
            optimizer3.zero_grad()
            
            basis_map = torch.tile(v_global_basis.view(1,10,10,1,1),(1,1,1,256,256))
            multi_basis = v_delta_basis + basis_map
            n,d,r,s = torch.split(multi_basis,[3,3,1,3],2)
            new_n = torch_norm(n,2)
            multi_basis = torch.concat([new_n,d,r,s],2)
            multi_basis = torch.clip(multi_basis,-1,1)
            fake_svbrdf = self.get_svbrdf_sum(weights,multi_basis)
            img = self.renderer._render(fake_svbrdf,sample_light_dir,sample_view_dir,sample_light_dis).squeeze(1)
            img = torch.clip(img,0,1)
            loss = torch.mean(torch.abs(img-(self.inputs[:,:3]*0.5+0.5)))
            loss.backward()
            # print(loss)
            optimizer3.step()
        self.middle = fake_svbrdf
        self.fake_svbrdf = fake_svbrdf
        self.single_basis = v_global_basis
        self.final_basis_map = multi_basis

    def validation(self, dataloader, current_iter):


        pbar = tqdm(total=len(dataloader), unit='image')
        svbrdfs_vis = torch.ones([len(dataloader),3,256*2,256*4])
        self. metric_results = 0
        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data,random=False)
            self.test()
            img_name = self.name[0][:-4]
            if  self.args.mode=='real':
                self.save_real_visuals(img_name,self.fake_svbrdf)
                self.save_basis_weight(img_name)
    
            if self.args.mode == 'train' or self.args.mode == 'test':
                svbrdfs_vis[idx] = self.save_visuals()
                error = torch.abs(self.fake_svbrdf-self.svbrdf).mean()
                self.metric_results += error

            torch.cuda.empty_cache()
            pbar.update(1)
            pbar.set_description(f'Testing')

        pbar.close()
            

        if self.args.mode == 'train' or self.args.mode == 'test':
            imwrite(tensor2img(svbrdfs_vis*0.5+0.5),osp.join(self.args.save_root, "visualization",f'{current_iter}.png'))
            self.metric_results /= (idx + 1)


    def save_visuals(self):
        svbrdf_gt_pre = torch.cat([self.svbrdf,self.fake_svbrdf],-2)
        normal,diffsue, roughness,specular = torch.split(svbrdf_gt_pre,[3,3,1,3],1)
        svbrdf_normal = torch.cat([normal,diffsue, torch.tile(roughness,(1,3,1,1)),specular],-1)

        return svbrdf_normal.squeeze(0)
    def save_real_visuals(self, name, pred):
        output = pred*0.5+0.5
        normal, diffuse, roughness, specular = torch.split(output,[3,3,1,3],dim=1)
        roughness = torch.tile(roughness,[1,3,1,1])
        output = torch.cat([normal,diffuse,roughness,specular],dim=-1)
        output_img = tensor2img(output,rgb2bgr=True)
        path = osp.join(self.args.save_root,"results","BasisOpt",name+".png") 
        imwrite(output_img, path, float2int=False)
        
        output = self.pred_svbrdf*0.5+0.5
        normal, diffuse, roughness, specular = torch.split(output,[3,3,1,3],dim=1)
        roughness = torch.tile(roughness,[1,3,1,1])
        output = torch.cat([normal,diffuse,roughness,specular],dim=-1)
        output_img = tensor2img(output,rgb2bgr=True)
        path = osp.join(self.args.save_root,"results","Pred",name+".png") 
        imwrite(output_img, path, float2int=False)
    def save_basis_weight(self,name):
        weight_map = torch.tile(self.weight.view(10,1,256,256),[1,3,1,1])
        self.pred_single_basis = torch.tile(self.pred_single_basis.squeeze(0).unsqueeze(-1).unsqueeze(-1),(1,1,256,256))*0.5+0.5
        sn,sd,sr,ss = torch.split(self.pred_single_basis,[3,3,1,3],1)
        n,d,r,s = torch.split(self.pred_final_basis.squeeze(0),[3,3,1,3],1)
        vis_map = torch.cat([n*0.5+0.5,d*0.5+0.5,torch.tile(r,[1,3,1,1])*0.5+0.5,s*0.5+0.5,weight_map,sn,sd,torch.tile(sr,[1,3,1,1]),ss],-1)
        vis_list = torch.split(vis_map,1,0)
        vis_rs = torch.cat(vis_list,-2)
        save_path = os.path.join(self.args.save_root,"results","pred_basis",name+".png")
        imwrite(tensor2img(vis_rs),save_path)
        
        
        weight_map = torch.tile(self.weight.view(10,1,256,256),[1,3,1,1])
        self.single_basis = torch.tile(self.single_basis.squeeze(0).unsqueeze(-1).unsqueeze(-1),(1,1,256,256))*0.5+0.5
        sn,sd,sr,ss = torch.split(self.single_basis,[3,3,1,3],1)
        n,d,r,s = torch.split(self.final_basis_map.squeeze(0),[3,3,1,3],1)
        vis_map = torch.cat([n*0.5+0.5,d*0.5+0.5,torch.tile(r,[1,3,1,1])*0.5+0.5,s*0.5+0.5,weight_map,sn,sd,torch.tile(sr,[1,3,1,1]),ss],-1)
        vis_list = torch.split(vis_map,1,0)
        vis_rs = torch.cat(vis_list,-2)
        save_path = os.path.join(self.args.save_root,"results","afterOpt_basis",name+".png")
        imwrite(tensor2img(vis_rs),save_path)
    def save(self, current_iter):
        self.save_network(self.net_g,'net_g',current_iter)
        self.save_network(self.net_l, 'net_l', current_iter)

    def model_to_device(self, net):
        net = net.float().to(self.device)
        return net
    def update_learning_rate(self, current_iter, warmup_iter=-1):
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)
    def save_network(self, net, net_label, current_iter, param_key='params'):
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{current_iter}.pth'
        save_dir = osp.join(self.args.save_root,"model")
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        save_path = osp.join(save_dir,save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        torch.save(save_dict, save_path)
    def get_current_learning_rate(self):
        return [param_group['lr'] for param_group in self.optimizers[0].param_groups]
    def get_current_log(self):
        return self.log_dict
    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net
    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        net = self.get_bare_model(net)
        # logger.info(f'Loading {net.__class__.__name__} model from {load_path}.')
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        # remove unnecessary 'module.'

        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
            if k.startswith('step_counter'):
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """Print keys with differnet name or different size when loading models.

        1. Print keys with differnet names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        if crt_net_keys != load_net_keys:
            logger.warning('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                logger.warning(f'  {v}')
            logger.warning('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                logger.warning(f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    logger.warning(f'Size different, ignore [{k}]: crt_net: '
                                   f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)