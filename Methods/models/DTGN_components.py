import copy
import torch
from torch.nn.parameter import Parameter
import numpy as np  
import torch.nn as nn 
import torch.nn.functional as F  
from Utils.logging_utils import log_wandb
from typing import List, Union,Tuple, Dict, Optional, Iterable
from dataclasses import dataclass
import matplotlib.pyplot as plt
from collections import OrderedDict     
from Utils.utils import RunninStatsManager, cosine_rampdown, standardize, match_dims, ordered_dict_mean
from .base_modular import ModularBaseNet, conv_block_base, FixedParameter, bn_eval
from torch.nn import BatchNorm1d
from Utils.utils import create_mask

from simple_parsing import choice
device = nn_device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Inv_Block(nn.Module):  
    def __init__(self, in_dim: int, grad_multiplier: float=None, activation=None, pooling_inv=False, type='linear', deviation_statistic='z_score', use_bn=False, rs_steps=2000, z_score_per_batch=True):
        super().__init__()
        self.use_bn = use_bn  
        self.in_dim = copy.copy(in_dim)       
        self.z_score_per_batch=z_score_per_batch 
        # self.cos = torch.nn.CosineEmbeddingLoss(margin=0.5, reduction='none')
        self.deviation_statistic = deviation_statistic    
        self.rs_steps=rs_steps

        self.runing_activation_window_buffer = RunninStatsManager(rs_steps, keep_median=(deviation_statistic=='mad')) #RunninsStatsManager() #Statistics()

        self._runing_activation_window_buffer_copy = RunninStatsManager(rs_steps, keep_median=(deviation_statistic=='mad')) #RunninsStatsManager() #Statistics()

        if activation=='relu':  
            self.activation=nn.ReLU()
        elif activation=='sigmoid':
            self.activation=nn.Sigmoid()
        elif activation=='tanh':
            self.activation=nn.Tanh()
        else:
            self.activation = None
        
        if pooling_inv:
            mpkernel = 3     
            self.pooling_flatten = nn.Sequential(nn.MaxPool2d(mpkernel), nn.Flatten())
            self.in_dim[-1]=(self.in_dim[-1] - mpkernel) // mpkernel + 1
            self.in_dim[-2]=(self.in_dim[-2] - mpkernel) // mpkernel + 1
        else:
            self.pooling_flatten = nn.Identity()
        
        # self.small_func = small
        in_bn_dim = int(in_dim[0]/2)
        if type=='conv': 
            #conv
            if self.activation is None: 
                #small
                self.layers = nn.ModuleList([ nn.Conv2d(int(self.in_dim[0]/2), int(self.in_dim[0]/2), 3, padding=1),
                                              nn.Conv2d(int(self.in_dim[0]/2), int(self.in_dim[0]/2), 3, padding=1)])
            else:
                #large
                self.layers = nn.ModuleList([ nn.Sequential(nn.Conv2d(int(self.in_dim[0]/2), int(self.in_dim[0]), 3, padding=1), self.activation, nn.ConvTranspose2d(int(self.in_dim[0]),int(self.in_dim[0]/2), 3, padding=1)), 
                                              nn.Sequential(nn.Conv2d(int(self.in_dim[0]/2), int(self.in_dim[0]), 3, padding=1), self.activation, nn.ConvTranspose2d(int(self.in_dim[0]),int(self.in_dim[0]/2), 3, padding=1))])
        elif type=='linear':   
            # linear
            in_dim = np.prod(self.in_dim)
            if self.activation is None:  
                #small 
                self.layers = nn.ModuleList([ nn.Linear(int(in_dim/2), int(in_dim/2)), nn.Linear(int(in_dim/2), int(in_dim/2))])
                in_bn_dim = int(in_dim/2)
            else:
                #large
                self.layers = nn.ModuleList([ nn.Sequential(nn.Linear(int(in_dim/2), in_dim), self.activation, nn.Linear(in_dim, int(in_dim/2))), nn.Sequential(nn.Linear(int(in_dim/2), in_dim), self.activation, nn.Linear(in_dim, int(in_dim/2)))])
                in_bn_dim = int(in_dim/2)
        else:
            raise NotImplementedError
        if self.use_bn:
            self.bn1 = BatchNorm1d(in_bn_dim, momentum=1., affine=True, track_running_stats=False)
            self.bn2 = BatchNorm1d(in_bn_dim, momentum=1., affine=True, track_running_stats=False)
        else:
            self.bn1=torch.nn.Identity()
            self.bn2=torch.nn.Identity()
        #self.attach_grad_multiplier(grad_multiplier)
    
    def reset_stats(self):
        self.runing_activation_window_buffer = RunninStatsManager(self.rs_steps, keep_median=(self.deviation_statistic=='mad')) #RunninsStatsManager() #Statistics()
        self._runing_activation_window_buffer_copy = RunninStatsManager(self.rs_steps, keep_median=(self.deviation_statistic=='mad')) #RunninsStatsManager() #Statistics()
        
    def forward(self,_, x, calculate_stats = False, *args, **kwargs):
        #x = F.normalize(x)   
        x = self.pooling_flatten(x)
        n = int(x.shape[1]/2)     
        x1, x2 = x[:,:n], x[:,n:]
        f = self.bn1(self.layers[0](x2))
        y1 = torch.add(f,x1)
        g = self.bn2(self.layers[1](y1))
        y2 = torch.add(g,x2)       
        out = torch.cat([y2,y1],dim=1)
        out = out.view(out.size(0), -1)
        #out = torch.norm(torch.norm(out, p=2, dim=1).mean(0), p=2)
        #out = torch.norm(out.view(out.size(0),-1), p=2, dim=1)          
        structural_out = F.mse_loss(out, torch.zeros_like(out), reduction='none').mean(1) # [batch_size]
        #torch.norm(out, p=2, dim=1)#/out.size(1) #torch.square(torch.norm(out, p=2, dim=1))
        # structural_out = self.cos(out, torch.ones_like(out).to(out.device), torch.ones(out.size(0)).to(out.device)) + F.mse_loss(out, torch.zeros_like(out), reduction='none').mean(1)
        if calculate_stats: 
            if self.training:                                                
                delta = structural_out - self.runing_activation_window_buffer.mean() if self.runing_activation_window_buffer._count!=0. else torch.zeros_like(structural_out).to(device)#-10
            else:
                delta = structural_out - self.runing_activation_window_buffer.mean() 
            
            z_score = self.calculate_z_score(structural_out)
            
            return structural_out, z_score, delta
        else:
            return structural_out, None, None

    def dropback_stats(self):
        #dropsback the stats to the backup stats    
        if self._runing_activation_window_buffer_copy.n > 0:
            self.runing_activation_window_buffer.load_state_dict(self._runing_activation_window_buffer_copy.state_dict())
    
    def backup_stats(self):
        self._runing_activation_window_buffer_copy.load_state_dict(self.runing_activation_window_buffer.state_dict())

    def calculate_z_score(self, inv):
        #TODO: z-score with MAD   
        if not self.runing_activation_window_buffer._count.item() >2:
            return torch.zeros_like(inv).to(device)

        elif self.deviation_statistic=='z_score':        
            mean = self.runing_activation_window_buffer.mean() 
            std = self.runing_activation_window_buffer.stddev()#torch.sqrt(self.runing_var_buffer) #np.std(self.runing_activation_window_buffer)
            if self.z_score_per_batch:
                z_score = torch.ones_like(inv)*(inv.mean() - mean)/std
            else:
                z_score = (inv - mean)/std                
            return z_score #.to(device)
        elif self.deviation_statistic=='mad':
            score = (inv - self.runing_activation_window_buffer.median)/self.runing_activation_window_buffer.mad()
            return score#.to(device)
    
    def consolidate_stats_from_inner_to_outer(self):
        self.runing_activation_window_buffer.consolidate_stats_from_inner_to_outer()
    
    def consolidate_stats_from_outer_to_inner(self):
        self.runing_activation_window_buffer.consolidate_stats_from_outer_to_inner()

class Decoder(nn.Module):                                  
    def __init__(self, out_h, out_channels, in_channels, track_running_stats, use_bn, affine_bn, momentum,  kernel_size, rs_steps, padding, deviation_statistic, 
                                final_activation='tanh', activation_target='None', module_name='None', module_type=None, n_heads=1):
        # H_out ​= (H_in​−1) *stride − 2×padding + (kernel_size-1) + output_padding + 1
        super().__init__()
        self.module_name=module_name
        self.module_type=module_type
        self.in_channels=in_channels
        self.rs_steps=rs_steps
        self.criterion = nn.MSELoss(reduction='none')
        self.deviation_statistic = deviation_statistic   
        self.n_heads=n_heads 
        if final_activation=='tanh':
            self.activation=nn.Tanh()
        elif final_activation=='sigmoid':
            self.activation=nn.Sigmoid()
        elif final_activation=='relu':
            self.activation=nn.ReLU()
        else:
            raise NotImplementedError

        if activation_target=='tanh':  
            self.activation_target=nn.Tanh()
        elif activation_target=='sigmoid':
            self.activation_target=nn.Sigmoid()
        elif activation_target=='relu':
            self.activation_target=nn.ReLU()
        else:
            self.activation_target=nn.Identity()

        if module_type!='expert':
            if out_h%2!=0:
                kernel_size2=kernel_size+1
            else:
                kernel_size2=kernel_size  
            stride=2
            if module_type =='linear':
                padding=0
                stride=1    
                kernel_size=2
                kernel_size2=1 
            assert n_heads>0
            if n_heads==1:
                self.decoder=nn.Sequential(OrderedDict([
                            ('conv_t1', nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size-1, padding=padding, stride=stride)),  
                            ('norm', nn.BatchNorm2d(out_channels, momentum=momentum, affine=affine_bn,
                                track_running_stats=track_running_stats)) if use_bn else ('norm', nn.Identity()),
                            ('relu_2', nn.ReLU()),
                            ('conv_t2', nn.ConvTranspose2d(out_channels, in_channels, kernel_size=kernel_size2, padding=0, stride=1)),  
                            ('tanh', self.activation),

                        ]))
            else:
                self.decoder=nn.ModuleList()
                for h in range(n_heads):
                    decoder=nn.Sequential(OrderedDict([
                                ('conv_t1', nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size-1, padding=padding, stride=stride)),  
                                ('norm', nn.BatchNorm2d(out_channels, momentum=momentum, affine=affine_bn,
                                    track_running_stats=track_running_stats)) if use_bn else ('norm', nn.Identity()),
                                ('relu_2', nn.ReLU()),
                                ('conv_t2', nn.ConvTranspose2d(out_channels, in_channels, kernel_size=kernel_size2, padding=0, stride=1)),  
                                ('tanh', self.activation),

                            ]))
                    self.decoder.append(decoder)
            
        else:
            modules=[]
            out_size=(5,)
            # for i in range(4):
            #     if i ==3:
            #         in_channels=self.in_channels
            #     else:
            #         in_channels=out_channels 
            modules.append(nn.Sequential(OrderedDict([                           
                    ('conv_t1', nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, padding=padding, stride=2)),  
                    ('norm', nn.BatchNorm2d(out_channels, momentum=1., affine=affine_bn,
                        track_running_stats=track_running_stats)) if use_bn else ('norm', nn.Identity()),
                    ('relu_2', nn.ReLU()),
                    ('conv_t2', nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=0, stride=1)),  
                    ('tanh', nn.ReLU()),
                ])))
            modules.append(nn.Sequential(OrderedDict([
                    ('conv_t1', nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=padding, stride=2)),  
                    ('norm', nn.BatchNorm2d(out_channels, momentum=1., affine=affine_bn,
                        track_running_stats=track_running_stats)) if use_bn else ('norm', nn.Identity()),
                    ('relu_2', nn.ReLU()),
                    ('conv_t2', nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=0, stride=1)),  
                    ('tanh', nn.ReLU()),
                ])))
            modules.append(nn.Sequential(OrderedDict([
                    ('conv_t1', nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, padding=padding, stride=2)),  
                    ('norm', nn.BatchNorm2d(out_channels, momentum=1., affine=affine_bn,
                        track_running_stats=track_running_stats)) if use_bn else ('norm', nn.Identity()),
                    ('relu_2', nn.ReLU()),
                    ('conv_t2', nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=0, stride=1)),  
                    ('tanh', nn.ReLU()),
                ])))
            modules.append(nn.Sequential(OrderedDict([
                    ('conv_t1', nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, padding=padding, stride=2)),  
                    ('norm', nn.BatchNorm2d(out_channels, momentum=1., affine=affine_bn,
                        track_running_stats=track_running_stats)) if use_bn else ('norm', nn.Identity()),
                    ('relu_2', nn.ReLU()),
                    ('conv_t2', nn.ConvTranspose2d(out_channels, in_channels, kernel_size=3, padding=0, stride=1)),  
                    ('tanh', self.activation),
                ])))

            self.decoder=nn.Sequential(*modules)


        self.runing_activation_window_buffer = RunninStatsManager(rs_steps, keep_median=(deviation_statistic=='mad')) #RunninsStatsManager() #Statistics()

        self._runing_activation_window_buffer_copy = RunninStatsManager(rs_steps, keep_median=(deviation_statistic=='mad')) #RunninsStatsManager() #Statistics()

    def dropback_stats(self):
        #dropsback the stats to the backup stats    
        if self._runing_activation_window_buffer_copy.n > 0:                                                    
            self.runing_activation_window_buffer.load_state_dict(self._runing_activation_window_buffer_copy.state_dict())
    
    def backup_stats(self):
        self._runing_activation_window_buffer_copy.load_state_dict(self.runing_activation_window_buffer.state_dict())

    def reset_stats(self):
        self.runing_activation_window_buffer = RunninStatsManager(self.rs_steps, keep_median=(self.deviation_statistic=='mad')) #RunninsStatsManager() #Statistics()
        self._runing_activation_window_buffer_copy = RunninStatsManager(self.rs_steps, keep_median=(self.deviation_statistic=='mad')) #RunninsStatsManager() #Statistics()

    def forward(self, x_in, e, calculate_stats=False, visualize=False, info='', *arwgs, **kwargs):
        # loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
        n = x_in.size(0)
        structural_out=0
        if self.n_heads>1:
            d=[]                    
            for head in self.decoder:
                dd = head(e)
                d.append(dd)
                structural_out += self.criterion(dd, self.activation_target(x_in.detach())).view(n, -1).mean(-1)
            d=torch.stack(d).mean(0)
        else:
            d = self.decoder(e)
            structural_out = self.criterion(d, self.activation_target(x_in.detach())).view(n, -1).mean(-1)

        # loss = loss_fn(d, self.activation_target(x_in.detach()))
        if self.module_type=='linear':
            d = d.squeeze()
            x_in = torch.flatten(x_in, start_dim=1)    

        if visualize:
            # print(self.module_name, info)
            plt.figure(figsize=(9, 2))   
            # h = x_in[0].size(1)
            for i, item in enumerate(x_in):
                if i >= 9: break
                plt.subplot(2, 9, i+1)
                plt.imshow(item.sum(0).detach().cpu())
                
            for i, item in enumerate(d):
                if i >= 9: break
                plt.subplot(2, 9, 9+i+1)
                plt.imshow(item.sum(0).detach().cpu())
            log_wandb({f'Rec_AE/{self.module_name}{info}': plt})

        if calculate_stats:
            if self.training:                                                
                delta = structural_out - self.runing_activation_window_buffer.mean() if self.runing_activation_window_buffer._count!=0. else torch.zeros_like(structural_out).to(device) #-10
            else:
                delta = structural_out - self.runing_activation_window_buffer.mean() 
            
            z_score = self.calculate_z_score(structural_out)
            return structural_out, z_score, delta
        else:
            return structural_out, None, None
    
    def calculate_z_score(self, inv):
        #TODO: z-score with MAD                                
        if not self.runing_activation_window_buffer._count.item() >2:
            return torch.zeros_like(inv).to(device)

        elif self.deviation_statistic=='z_score':
            mean = self.runing_activation_window_buffer.mean() 
            std = self.runing_activation_window_buffer.stddev()#torch.sqrt(self.runing_var_buffer) #np.std(self.runing_activation_window_buffer)
            # z_score = (inv - mean)/std
            z_score = torch.ones_like(inv)*(inv.mean() - mean)/std
            return z_score #.to(device)
        elif self.deviation_statistic=='mad':
            score = (inv - self.runing_activation_window_buffer.median)/self.runing_activation_window_buffer.mad()
            return score#.to(device)

    def consolidate_stats_from_inner_to_outer(self): 
        self.runing_activation_window_buffer.consolidate_stats_from_inner_to_outer()
    
    def consolidate_stats_from_outer_to_inner(self):
        self.runing_activation_window_buffer.consolidate_stats_from_outer_to_inner()

class LMC_conv_block(conv_block_base):    
    @dataclass
    class Options(conv_block_base.Options):
        anneal_structural_lr: bool = 0 # -         
        deviation_statistic: str = choice('z_score', 'mad', 'cos_distance', 'L2_distance', 'L2_speed', 'L1_distance_normalized', 'L1_speed', default='z_score')# - 
        activation_structural: str = choice('sigmoid', 'relu', 'tanh', default='sigmoid')# - 
        structure_inv: str =choice('ae', 'linear_no_act','linear_act', 'pool_only_large_lin_act', 'conv_act', 'conv_no_act', 'conv_only_large_act', 'conv_only_large_no_act', 'pool_lin_no_act', 'pool_only_large_lin_no_act', default='linear_no_act')# - 
        use_abs_score: bool = 1# if 'True' use absolute value of the z-score
        bn_structural: bool = 0# if 'True', use batch norm in the structural component
        
        running_stats_steps: int = 2000# number of steps for the running statistics computation 
        total_steps_before_freeze: float = 15000.# (for continual CL) in case of automatic freeze, after this number of steps the structural component is frozen
        detach_structural: bool = True # if 'True' the structural component is detached from the functional one (default is True)
        log_reconstructions: bool = True # if 'True' modules will log the reconstructions to wandb

        ##################
        #Backup system mechanism was used for continual CL only

        use_backup_system: bool = True #-
        use_backup_system_structural: bool = True #-
        update_backup_stats: bool = 0# -  
        #####################

        z_score_per_batch: bool = True # if 'True' calculates z-scores at a batch level (i.e. first takes the mean over the structural activation over the batch)
        keep_bn_in_eval_after_freeze: bool = False # if 'True' keep batch norm in eval mode after the module is frozen

        use_structural: bool = 1 # if 'False' structural component is not used

        activation_target_decoder: str = choice('sigmoid', 'relu', 'tanh', 'None', default='None')# - 
        use_bn_decoder: int = 1 #whether to use batchnomr in the docer of ae
        momentum_bn_decoder: float = 0.1 #momentum of decoder batchnorm
        affine_bn_decoder: int = 1 #affine parameter of the batchnorm in the decoder

        ###################################
        #parameters for GatedOh (output head module)

        structure_inv_oh: str = choice('ae', 'linear_no_act', 'linear_act', default='linear_no_act') # -
        use_bn_decoder_oh: bool = 0 # -
        activate_after_str_oh: bool = 0 # -
        ###################################

        normalize_oh: bool =  0 # if True, nromalize the input to the structural component of the gated output head
        projection_layer_oh: bool = 0 # if True, adds a projection layer before classifier and structural

        n_heads_decoder: int = 1 # number of structural components (if >1 the average structural score is calculated)

    def __init__(self, in_h, in_channels, out_channels, i_size, name=None, module_type='conv', initial_inv_block_lr=0.001,  deviation_threshold=3, freeze_module_after_step_limit=False, deeper=False, options:Options=Options(), num_classes: int=0, **kwargs):
        super().__init__(in_channels, out_channels,i_size, name, module_type, 1, 1, bias=True, deeper=deeper, options=options, n_classes=num_classes, **kwargs)                  
        
        self.in_h=in_h
        self.num_classes=num_classes
        self.args: LMC_conv_block.Options = copy.copy(options)
        self.deviation_threshold = deviation_threshold
        self.initial_inv_block_lr = initial_inv_block_lr
        self.args.freeze_module_after_step_limit = freeze_module_after_step_limit
        
        ############################################################         
        self.register_buffer('annealing_factor_buffer', torch.tensor(1.))
        ############################################################
        
        ########################
        # this many update_stats calls (e.g. outer loops) the module's structural component will be learned
        # after this number of steps the module's structural component is fixed (we cound outer update steps)  
        self.register_buffer('_total_steps', torch.tensor(float(self.args.total_steps_before_freeze)))
        #counter for steps taken  
        self.register_buffer('_current_step_buffer', torch.tensor(0.))
        ########################
        # self.small_inv_nlk = small    
        ########################################################################
        # Structure of structural layers
        ########################################################################
        self.pooling_inv=False
        self.inv_type = 'linear' 
        if self.args.use_structural:
            self.init_structural()

        self.inv_block_structural_backup_buffer = None
        self.functional_backup_state_dict_buffer = None

        self.register_buffer('optimizers_update_needed_buffer', torch.tensor(0.))
        self.hooks = []    
        self.register_buffer('block_lr_buffer', torch.tensor(self.initial_inv_block_lr))
        self.register_buffer('_frozen_structural_buffer', torch.tensor(0.))
        self.register_buffer('_frozen_functional_buffer', torch.tensor(0.))
        # self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.runing_activation_window_record = []

        if num_classes>0: # and self.module_type!='expert':
            representation_dim = self.out_channels*self.out_h*self.out_h
            self.decoder=nn.Linear(representation_dim,num_classes)
            self.classifier=nn.Sequential(nn.Flatten(), self.decoder)
        else:
            self.classifier=nn.Identity()

    def init_structural(self):
        if self.args.structure_inv!='ae':
            if self.args.structure_inv=='linear_no_act':
                #only linear layers no activation function
                self.inv_blk_activation = None
                self.inv_type = 'linear'
                self.in_dim_inv = [self.out_channels*self.out_h*self.out_h]
            elif self.args.structure_inv=='pool_lin_no_act':
                #only linear layers no activation function
                self.pooling_inv=True
                self.inv_type = 'linear'
                self.inv_blk_activation = None
                self.in_dim_inv = [self.out_channels,self.out_h,self.out_h]
            elif self.args.structure_inv=='linear_act':
                #linear layers with activation inbetween (larger structural)
                self.inv_blk_activation = self.args.activation_structural
                self.inv_type = 'linear'
                self.in_dim_inv = [self.out_channels*self.out_h*self.out_h]
            elif self.args.structure_inv=='pool_only_large_lin_no_act':
                if np.prod([self.out_channels,self.out_h,self.out_h])>10000: #self.in_channels == 3:
                    self.pooling_inv=True
                    self.in_dim_inv = [self.out_channels,self.out_h,self.out_h]
                else:
                    self.in_dim_inv = [self.out_channels*self.out_h*self.out_h]
                self.inv_type = 'linear'
                self.inv_blk_activation = None
            elif self.args.structure_inv=='pool_only_large_lin_act':
                if np.prod([self.out_channels,self.out_h,self.out_h])>10000: #self.in_channels == 3:
                    self.pooling_inv=True
                    self.in_dim_inv = [self.out_channels,self.out_h,self.out_h]
                else:
                    self.in_dim_inv = [self.out_channels*self.out_h*self.out_h]
                self.inv_type = 'linear'
                self.inv_blk_activation = self.args.activation_structural
            elif self.args.structure_inv=='conv_act':
                #convolutional structural layers with activtion inbetween
                self.inv_blk_activation = self.args.activation_structural
                self.inv_type = 'conv'
                self.in_dim_inv = [self.out_channels,self.out_h,self.out_h]
            elif self.args.structure_inv=='conv_no_act':
                #convolutional structural layers with no activation
                self.inv_blk_activation = None
                self.inv_type = 'conv'
                self.in_dim_inv = [self.out_channels,self.out_h,self.out_h]
            elif self.args.structure_inv=='conv_only_large_act':
                #use convolutional layers only for large layers (e.g. input imnet), all the others use liner, use activation inbetween
                self.inv_blk_activation = self.args.activation_structural
                if self.in_channels == 3:
                    self.inv_type = 'conv'
                    self.in_dim_inv = [self.out_channels,self.out_h,self.out_h]
                else:
                    self.in_dim_inv = [self.out_channels*self.out_h*self.out_h]
            elif self.args.structure_inv=='conv_only_large_no_act':
                #use convolutional layers only for large layers (e.g. input imnet), all the others use linear, no activatioin
                self.inv_blk_activation = None
                if self.in_channels == 3:
                    self.inv_type = 'conv'
                    self.in_dim_inv = [self.out_channels,self.out_h,self.out_h]
                else:
                    self.in_dim_inv = [self.out_channels*self.out_h*self.out_h]
            else:
                raise NotImplementedError
            ########################################################################
            self.inv_block = Inv_Block(in_dim=self.in_dim_inv, activation=self.inv_blk_activation, pooling_inv=self.pooling_inv, type=self.inv_type, deviation_statistic=self.args.deviation_statistic, use_bn=self.args.bn_structural, rs_steps=self.args.running_stats_steps, z_score_per_batch=self.args.z_score_per_batch) #, grad_multiplier=self.annealing_factor_buffer)
        else:
            self.in_dim_inv = [self.out_channels,self.out_h,self.out_h]
            self.inv_block = Decoder(self.in_h,out_channels=self.out_channels, in_channels=self.in_channels, track_running_stats=self.args.track_running_stats_bn,  affine_bn=self.args.affine_bn_decoder, momentum=self.args.momentum_bn_decoder,use_bn=self.args.use_bn_decoder, kernel_size=self.args.kernel_size,
                                                    deviation_statistic=self.args.deviation_statistic, padding=self.args.padding, rs_steps=self.args.running_stats_steps, final_activation=self.args.activation_structural, activation_target=self.args.activation_target_decoder, module_name=self.name, module_type=self.module_type, n_heads=self.args.n_heads_decoder)

    @property
    def active(self):
        if not self.module_learned:
             return True
        if not self._frozen_functional_buffer:
            return True
        else:
            return False
    
    @property 
    def current_step(self):
        return self._current_step_buffer #if self._current_step_buffer <= self._total_steps else self._total_steps

    @current_step.setter
    def current_step(self, v): 
        if self.args.freeze_module_after_step_limit and v % self._total_steps==0:
            self._reset_backups()
        self._current_step_buffer = v
    
    def reset_stats(self):
        self.inv_block.reset_stats()

    def reset(self):
        super().reset()
        #Reset things for new modules                
        self.annealing_factor_buffer.data = torch.tensor(1.).to(device)          
        self.block_lr_buffer = torch.tensor(self.initial_inv_block_lr)
        self._current_step_buffer.data = torch.tensor(0.).to(device)
        self._frozen_structural_buffer = torch.tensor(0.)
        self._frozen_functional_buffer = torch.tensor(0.)
        self._clear_backup_structural()
        self._clear_backup_functional()
        # self.reset_stats()
        # self._reinit_structural()

    def update_stats(self, v: torch.Tensor, v_backup: torch.Tensor, m: torch.Tensor = torch.tensor(1.), threshold=0.5, record=False):
        '''
        Update the running stats of the module
        '''
        #######################
        threshold=0.                       
        idx = torch.where((m>threshold).int()>0)
        m = m[idx]      
        v = v[idx].detach()  
        v_backup = v_backup[idx].detach()
        if self.active: 
            self.current_step = self.current_step + 1 
        ###update uning stats###   
        if len(m)>0: 
            if not record:                        
                if self.inv_block_structural_backup_buffer is not None or not self._frozen_structural_buffer:
                    #if backup exists or not frozen
                    assert not any(v<0)         
                    self.inv_block.runing_activation_window_buffer.push(v.mean())

                if self.inv_block_structural_backup_buffer is not None and self.args.update_backup_stats:
                    assert not any(v_backup<0) 
                    self.inv_block_structural_backup_buffer.runing_activation_window_buffer.push(v_backup.mean())

                # print(self.name)     
                if not self.module_learned: 
                    ########################      
                    if self.args.anneal_structural_lr:                
                        self.annealing_factor_buffer.data = cosine_rampdown(self.current_step, self._total_steps)
                        self.block_lr_buffer.data = self.initial_inv_block_lr * self.annealing_factor_buffer

            #this is for beam search, records the stats and then pushes them later only to the modules of the selected path
            else:
                if self.inv_block_structural_backup_buffer is not None or not self._frozen_structural_buffer: #frozen:
                    #if backup exists or not frozen
                    assert not any(v<0)         
                    # assert self.runing_activation_window_record
                    assert not self.args.update_backup_stats
                    self.runing_activation_window_record.append(v.mean())

    def update_running_stats_from_record(self, idx):
        # used for beam seach             
        if len(self.runing_activation_window_record)>0:
            if not self._frozen_structural_buffer:     
                try:       
                    self.inv_block.runing_activation_window_buffer.push(self.runing_activation_window_record[idx])
                except:
                    print(idx)
                    print(len(self.runing_activation_window_record))
        self.clean_runing_stats_record()
    
    def clean_runing_stats_record(self):
        self.runing_activation_window_record = []


    def forward(self, x, info='', log_at_this_iter=False):

        x_m = self.module(x)  # <-modules functional output     

        if self.module_type=='expert':
              x_m, _ = x_m

        if self.args.detach_structural:    
            in_structural = x_m.detach()
        else:
            in_structural = x_m  

        if self.args.use_structural:  
            if self.inv_block_structural_backup_buffer is not None:             
                #we detach it for the new structrual, since it should learn to gate                                                                                                                                                        
                str_to_minimize, _, _ = self.inv_block(x, in_structural.detach().view(x_m.size(0), *self.in_dim_inv), calculate_stats=self.args.deviation_statistic=='z_score', visualize=(log_at_this_iter and self.args.log_reconstructions), info=info)
                #we dont detach here - use detach_structural flag               
                str_to_use_for_masks, z_score, delta = self.inv_block_structural_backup_buffer(x, in_structural.view(in_structural.size(0), *self.in_dim_inv), calculate_stats=self.args.deviation_statistic=='z_score')      
            else:
                #no backup
                str_to_minimize, z_score, delta = self.inv_block(x, in_structural.view(x_m.size(0), *self.in_dim_inv), calculate_stats=self.args.deviation_statistic=='z_score', visualize=(log_at_this_iter and self.args.log_reconstructions), info=info)
                str_to_use_for_masks = str_to_minimize.clone()            
                str_to_use_for_masks = str_to_use_for_masks.detach()
        else:
            str_to_minimize=torch.zeros(x_m.size(0), device=device)
            str_to_use_for_masks=str_to_minimize
            delta = torch.zeros_like(str_to_minimize, device=device)
            outlier_score = torch.zeros_like(str_to_minimize, device=device)
            outlier = torch.tensor([False]*len(outlier_score)).to(device) 
            z_score = torch.zeros_like(str_to_minimize, device=device)
        # structural outputs:
        # str_to_minimize - the structural loss that can be used to calculate gradients and backprop
        # str_to_use_for_masks - the structural loss that should be used for normalizing and calculating module activations weights

        if self.args.deviation_statistic=='z_score':
            outlier_score = torch.abs(z_score) if self.args.use_abs_score else z_score
        else:
            raise NotImplementedError

        if self.module_learned:   
            outlier = outlier_score>self.deviation_threshold
        else:
            # if module is not learned (i.e. functional component is not fixed) the module does not report outliers
            outlier = torch.tensor([False]*len(outlier_score)).to(device)  
        if self.args.dropout>0:
            x_m = self.dropout(x_m)

        x_m = self.classifier(x_m)    

        return x_m, str_to_minimize, str_to_use_for_masks, delta, outlier_score, outlier, int(self.module_learned), int(self._frozen_functional_buffer)
          
    def freeze_functional(self, inner_loop_free=True, if_eval_bn=False):
        if not self._frozen_functional_buffer:  
            self._frozen_functional_buffer = torch.tensor(1.)
            for p in self.module.parameters():
                #setting requires_grad to False would prevent updating in the inner loop as well
                p.requires_grad = inner_loop_free
                #this would prevent updating in the outer loop:
                p.__class__ = FixedParameter       
            for p in self.classifier.parameters(): 
                p.requires_grad = inner_loop_free
                #this would prevent updating in the outer loop:
                p.__class__ = FixedParameter 
            if self.args.keep_bn_in_eval_after_freeze:  
                bn_eval(self)
                self.bn_eval_buffer=torch.tensor(1.)
            print(f'freezing functional {self.name}')
            return True
        else:            
            print(f'could not freze functional as it is already frozen at {self.name}')
            return False
    
    def freeze_learned_module(self):
        if self.module_learned and self.active and not self._frozen_functional_buffer:
            self._maybe_dropback_structural(dropback_stats=True)
            self._maybe_dropback_functional()
            return self.freeze_functional()
        else:
            return False
    
    def unfreeze_learned_module(self, unfreeze_str=False):
        success = self.unfreeze_functional()
        if unfreeze_str:
            self.unfreeze_structural()
        if success:
            self.reset_step_count()
            self._maybe_backup_structural()
        return success

    def unfreeze_structural(self):       
        if self._frozen_structural_buffer and self.args.use_structural:
            for p in self.inv_block.parameters():
                    #setting requires_grad to False would prevent updating in the inner loop as well
                    p.requires_grad = True 
                    #this would prevent updating in the outer loop:
                    p.__class__ = Parameter    
            print(f'unfreezing {self.name}')    
            self._frozen_structural_buffer = torch.tensor(0.)    
            return True
        else:
            print(f'could not freze structural as it is already frozen at {self.name}')
            return False

    def freeze_structural(self):    
        if not self._frozen_structural_buffer and self.args.use_structural:
            self._frozen_structural_buffer = torch.tensor(1.)
            for p in self.inv_block.parameters():
                #setting requires_grad to False would prevent updating in the inner loop as well
                p.requires_grad = False 
                #this would prevent updating in the outer loop:
                p.__class__ = FixedParameter            
            print(f'freezing structural {self.name}')
            return True
        else:
            print(f'could not freze structural as it is already frozen at {self.name}')
            return False

    def reset_step_count(self):
        self._current_step_buffer = torch.tensor(0.).to(device)

    def _maybe_dropback_structural(self, dropback_stats:bool, unfreeze:bool=False):
        if self.inv_block_structural_backup_buffer is not None:
            # print(self.inv_block_structural_backup_buffer)
            if dropback_stats:
                self.inv_block_structural_backup_buffer.dropback_stats()           
            self.inv_block = self._clone_inv_blck(copy.deepcopy(self.inv_block_structural_backup_buffer.state_dict()))
            if unfreeze:
                for p in self.inv_block.parameters():
                    p.__class__ = nn.Parameter
                    p.requires_grad = True
            self._clear_backup_structural()
            return True
        else:
            print(f'unable to dropback to structural backup as not structural backup exists for module {self.name}')
    
    def _clear_backup_structural(self):
        self.__delattr__('inv_block_structural_backup_buffer')
        self.inv_block_structural_backup_buffer = None
    
    def _clear_backup_functional(self):                       
        self.__delattr__('functional_backup_state_dict_buffer')
        self.functional_backup_state_dict_buffer = None

    def _maybe_dropback_functional(self):          
        if self.functional_backup_state_dict_buffer is not None:
                #drop back to previous backed-up dynamiccs     
                self.module = self._clone_func_blck(copy.deepcopy(self.functional_backup_state_dict_buffer.state_dict()))
                # self.module.load_state_dict(self.functional_backup_state_dict_buffer)
                self._clear_backup_functional()
                return True
        else:
            print(f'unable to drop back functionl as there is not functional backup at module {self.name}')
            return False

    def _maybe_backup_structural(self, verbose=True):
        if self.inv_block_structural_backup_buffer is None and self.args.use_backup_system and self.args.use_backup_system_structural:                            
            self.inv_block_structural_backup_buffer = self._clone_inv_blck(copy.deepcopy(self.inv_block.state_dict()))
            ############################################################
            #creating backup of stats at this point, in case we need to return to this state
            self.inv_block_structural_backup_buffer.backup_stats()
            ############################################################
            return True
        else:
            if verbose:
                if self.args.use_backup_system and self.args.use_backup_system_structural:
                    print(f'unable to backup structural as a backup already exists for module {self.name}')
            return False
    
    def _freeze_structural_backup(self):       
        if self.inv_block_structural_backup_buffer is not None:   
            for p in self.inv_block_structural_backup_buffer.parameters():
                p.__class__ = FixedParameter
                p.requires_grad = False          
                #TODO: is it okat to have it here?
        else:
            print(f'unable to freeze structural backup as it does not exists at {self.name}')

    def _reinit_structural(self):
        if self.args.structure_inv!='ae':    
            self.inv_block = Inv_Block(in_dim=self.in_dim_inv, activation=self.inv_blk_activation, pooling_inv=self.pooling_inv,type=self.inv_type, deviation_statistic=self.args.deviation_statistic, 
                                                                                                                                use_bn=self.args.bn_structural, rs_steps=self.args.running_stats_steps, z_score_per_batch=self.args.z_score_per_batch).to(device)
        else:
            self.inv_block = Decoder(self.in_h,out_channels=self.out_channels, in_channels=self.in_channels, track_running_stats=self.args.track_running_stats_bn,  affine_bn=self.args.affine_bn_decoder, momentum=self.args.momentum_bn_decoder,use_bn=self.args.use_bn_decoder, kernel_size=self.args.kernel_size, 
                            deviation_statistic=self.args.deviation_statistic, padding=self.args.padding, rs_steps=self.args.running_stats_steps, final_activation=self.args.activation_structural, activation_target=self.args.activation_target_decoder, module_name=self.name, module_type=self.module_type, n_heads=self.args.n_heads_decoder).to(device)
 
    
    def _maybe_backup_functional(self, verbose=True):
        if self.functional_backup_state_dict_buffer is None and self.args.use_backup_system:                       
            self.functional_backup_state_dict_buffer = self._clone_func_blck(self.module.state_dict())
            return True
        else:
            if verbose:   
                if self.args.use_backup_system:
                    print(f'unable to backup functional as a backup already exists for module {self.name}')
            return False

    def _freeze_functional_backup(self):           
        if self.functional_backup_state_dict_buffer is not None:
            for p in self.functional_backup_state_dict_buffer.parameters():
                    p.__class__ = FixedParameter
                    p.requires_grad = False
        else:
            print(f'unable to freeze functional backup as it does not exists at {self.name}')

    def _clear_backups(self):
        self._clear_backup_functional()
        self._clear_backup_structural()
      
    def unfreeze_functional(self):       
        if self._frozen_functional_buffer: 
            for p in self.module.parameters():
                p.__class__ = nn.Parameter
                p.requires_grad = True                       
            self._frozen_functional_buffer = torch.tensor(0.)        
            print(f'unfreezing {self.name}')        
            self._maybe_backup_functional()
            return True
        else:
            return False
    
    def _clone_func_blck(self, state_dict=None):
        new_inv_block = copy.deepcopy(self.module) 
        new_inv_block.to(device)
        new_inv_block.load_state_dict(state_dict)
        return new_inv_block

    def _clone_inv_blck(self, state_dict=None):
        if self.args.structure_inv!='ae':
            new_inv_block = Inv_Block(in_dim=self.in_dim_inv, activation=self.inv_blk_activation, pooling_inv=self.pooling_inv,type=self.inv_type, deviation_statistic=self.args.deviation_statistic, use_bn=self.args.bn_structural, rs_steps=self.args.running_stats_steps, z_score_per_batch=self.args.z_score_per_batch).to(device)
        else:
            new_inv_block = Decoder(self.in_h,out_channels=self.out_channels, in_channels=self.in_channels, track_running_stats=self.args.track_running_stats_bn,  affine_bn=self.args.affine_bn_decoder, momentum=self.args.momentum_bn_decoder,use_bn=self.args.use_bn_decoder, kernel_size=self.args.kernel_size,
                                                                    deviation_statistic=self.args.deviation_statistic, padding=self.args.padding, rs_steps=self.args.running_stats_steps, final_activation=self.args.activation_structural, activation_target=self.args.activation_target_decoder, module_name=self.name, module_type=self.module_type, n_heads=self.args.n_heads_decoder).to(device)

        new_inv_block.load_state_dict(state_dict)
        return new_inv_block

    def _reset_backups(self):    
        '''
        Creates a new 'snapshot' of the components - in the future when it drops back, it will drop back to the current state
        '''
        assert self.active  
        self.module_learned=torch.tensor(1.)
        if self.args.use_backup_system:
            #1. remove current functional backup, as new should be created
            self._clear_backup_functional()
            print(f'resetting backups on {self.name}')
            #2. create a new functional backup
            fb_created = self._maybe_backup_functional()
            assert fb_created or not self.args.use_backup_system
            self._freeze_functional_backup()
            if not self.active:
                self.freeze_functional(inner_loop_free=True) #<- can still be updated in the inner loop
            #if no structural backup exist, create it
            if self.args.use_backup_system_structural:
                if self.inv_block_structural_backup_buffer is None:
                    #3a. create new structural backup if we dont have one
                    # self.freeze_and_backup_structure()        
                    sb_created = self._maybe_backup_structural()      
                    assert sb_created or not self.args.use_backup_system
                    self._freeze_structural_backup()
                    if sb_created:
                        self._reinit_structural()
                else:
                    #3b. otherwise just make a new stats backup        
                    self.inv_block_structural_backup_buffer.backup_stats()     
            else:
                self.freeze_structural()       
        else:
            self.freeze_structural()
        self.optimizers_update_needed_buffer = torch.tensor(1.)
    
    def on_finished_outer_loop(self, finished:bool):
        if self.args.use_structural:
            if finished:
                self.inv_block.consolidate_stats_from_inner_to_outer()
            else:
                self.inv_block.consolidate_stats_from_outer_to_inner()
            if self.inv_block_structural_backup_buffer is not None:
                if finished:
                    self.inv_block_structural_backup_buffer.consolidate_stats_from_inner_to_outer()
                else:
                    self.inv_block_structural_backup_buffer.consolidate_stats_from_outer_to_inner()

    def on_module_addition_at_my_layer(self, inner_loop_free=True):
        if self.active:
            #when adding a module to the layer of this module
            self._maybe_dropback_functional()
            self._maybe_dropback_structural(dropback_stats=True)
            self.freeze_functional(inner_loop_free=inner_loop_free)
            if not self._frozen_structural_buffer:
                self.freeze_structural()      
        if self.args.keep_bn_in_eval_after_freeze:  
                bn_eval(self)
                self.bn_eval_buffer=torch.tensor(1.)
    
    def train(self, *args, **kwargs):                                     
        if not self.args.keep_bn_in_eval_after_freeze or self.active or not int(self.bn_eval_buffer.item()):
            return super().train(*args, **kwargs)
        else:
            r = super().train(*args, **kwargs)
            bn_eval(self)
            return r  



class ComponentList(nn.ModuleList):
    def _init__(self, *args, **kwargs):
        super(ComponentList).__init__(*args,**kwargs)
    
    @property
    def all_modules_learned(self):
        return not any(not x.module_learned for x in self)

    @property
    def optimizer_update_needed(self):
        ret = 0
        for m in self:
            if hasattr(m, 'optimizers_update_needed_buffer'):
                if m.optimizers_update_needed_buffer:
                    m.optimizers_update_needed_buffer=torch.tensor(0.)
                    ret = 1
        return ret

class GatedOh(LMC_conv_block):
    @dataclass                      
    class Options(LMC_conv_block.Options):  
        pass
    def __init__(self, out_h, in_channels, hidden_size, num_classes, module_type='linear', options: Options=Options(), **kwargs):
        options = copy.deepcopy(options)
        options.use_bn=False     
        options.use_backup_system=False  
        options.structure_inv=options.structure_inv_oh 
        options.use_bn_decoder=options.use_bn_decoder_oh
        options.use_structural=True
        
        self.normalize = options.normalize_oh
        self.projection_layer_oh = options.projection_layer_oh

        super().__init__(out_h, in_channels, hidden_size, out_h, module_type=module_type, options=options, **kwargs)
        
        self.num_classes = num_classes 
        self.out_features = num_classes  
        if module_type=='linear':
            if self.args.activate_after_str_oh:   
                self.module = nn.Sequential(OrderedDict([
                                ('flatten', nn.Flatten().to(device)), 
                                ('lin', nn.Linear(in_channels, self.out_channels).to(device) if self.projection_layer_oh else nn.Identity())                                
                            ]))
            elif not self.projection_layer_oh:
                self.module =  nn.Sequential(OrderedDict([('flatten', nn.Flatten().to(device))]))
        
        representation_dim = self.out_channels*self.out_h*self.out_h
        self.classifier=nn.Linear(representation_dim, num_classes).to(device)

    def forward(self, x):
        n=x.size(0)   
        x_m = self.module(x)  # <-modules functional output     
        if self.normalize:
            in_structural = F.normalize(x_m).detach().view(x_m.size(0),*self.in_dim_inv)
        else:
            in_structural = x_m.detach().view(x_m.size(0),*self.in_dim_inv)  
        str_to_minimize, _, _ = self.inv_block(x, in_structural, calculate_stats=self.args.deviation_statistic=='z_score', visualize=False, info='')
        # str_for_running_stats
        if self.args.activate_after_str_oh:
            x_m=F.relu(x_m)    
        if self.args.dropout>0:
            x_m = self.dropout(x_m)
        x_m=x_m.reshape(n,-1)
        x_m = self.classifier(x_m)     
        return x_m, str_to_minimize


class Sub_DTGN(nn.Module):
    @dataclass
    class Options():
        anneal_structural_lr: bool = 0 # -
        deviation_statistic: str = choice('z_score', 'mad', 'cos_distance', 'L2_distance', 'L2_speed', 'L1_distance_normalized', 'L1_speed', default='z_score')# -
        activation_structural: str = choice('sigmoid', 'relu', 'tanh', default='sigmoid')# -
        structure_inv: str =choice('ae', 'linear_no_act','linear_act', 'pool_only_large_lin_act', 'conv_act', 'conv_no_act', 'conv_only_large_act', 'conv_only_large_no_act', 'pool_lin_no_act', 'pool_only_large_lin_no_act', default='linear_no_act')# -
        use_abs_score: bool = 1# if 'True' use absolute value of the z-score
        bn_structural: bool = 0# if 'True', use batch norm in the structural component

        running_stats_steps: int = 2000# number of steps for the running statistics computation
        total_steps_before_freeze: float = 15000.# (for continual CL) in case of automatic freeze, after this number of steps the structural component is frozen
        detach_structural: bool = True # if 'True' the structural component is detached from the functional one (default is True)
        log_reconstructions: bool = True # if 'True' modules will log the reconstructions to wandb

        ##################
        #Backup system mechanism was used for continual CL only

        use_backup_system: bool = True #-
        use_backup_system_structural: bool = True #-
        update_backup_stats: bool = 0# -
        #####################

        z_score_per_batch: bool = True # if 'True' calculates z-scores at a batch level (i.e. first takes the mean over the structural activation over the batch)
        keep_bn_in_eval_after_freeze: bool = False # if 'True' keep batch norm in eval mode after the module is frozen

        use_structural: bool = 1 # if 'False' structural component is not used

        activation_target_decoder: str = choice('sigmoid', 'relu', 'tanh', 'None', default='None')# -
        use_bn_decoder: int = 1 #whether to use batchnomr in the docer of ae
        momentum_bn_decoder: float = 0.1 #momentum of decoder batchnorm
        affine_bn_decoder: int = 1 #affine parameter of the batchnorm in the decoder

        ###################################
        #parameters for GatedOh (output head module)

        structure_inv_oh: str = choice('ae', 'linear_no_act', 'linear_act', default='linear_no_act') # -
        use_bn_decoder_oh: bool = 0 # -
        activate_after_str_oh: bool = 0 # -
        ###################################

        normalize_oh: bool =  0 # if True, nromalize the input to the structural component of the gated output head
        projection_layer_oh: bool = 0 # if True, adds a projection layer before classifier and structural

        n_heads_decoder: int = 1 # number of structural components (if >1 the average structural score is calculated)
        regime='normal'
        #if True the structural component is only optimized for modules which are 'free' - not yet frozen/learned
        optmize_structure_only_free_modules: bool = True
        # wether to mask the outputs of the structural components as well
        mask_str_loss:bool=0
        #the way to combine module outputs
        concat:str = choice('sum', 'beam', default='sum')
        # if 'True' uses task id (if given) to select the corresponding modules in layer forward; has no effect if task_id is not given to the forward (use train- and test-time oracle parameters of the metalearner)
        module_oracle:bool = 0
        dropout:float=0.2  #dropout probability
        kernel_size: int = 3 #conv kernel size

    def __init__( self, in_h, in_channels, out_channels, i_size, name=None, module_type='conv', initial_inv_block_lr=0.001,  deviation_threshold=3, freeze_module_after_step_limit=False, deeper=False, layer_n_modules: int = 2, options:Options=Options(), num_classes: int=0, **kwargs):
        # super().__init__(in_h, in_channels, out_channels, i_size, module_type=module_type, options=options, **kwargs)
        super().__init__()

        self.in_channels = in_channels
        self.depth = 1
        self.out_channels = out_channels
        self.module_type =module_type
        self.num_classes = num_classes
        self.name = name
        self.module_kwargs = kwargs
        self.args: Sub_DTGN.Options = copy.copy(options)
        self.MODEL_DEPTH = 2
        self.dropout = nn.Dropout(self.args.dropout)
        # **********
        self.register_buffer('_total_steps', torch.tensor(float(self.args.total_steps_before_freeze)))
        self.register_buffer('_current_step_buffer', torch.tensor(0.))
        self.register_buffer('bn_eval_buffer', torch.tensor(0.))
        self.register_buffer('annealing_factor_buffer', torch.tensor(1.))
        self.register_buffer('module_learned_buffer', torch.tensor(0.))
        self.deviation_threshold = deviation_threshold
        self.initial_inv_block_lr = initial_inv_block_lr
        self.num_managers = 1    #This parameter needs to be adjusted

        self.in_h= in_h
        out_head = in_h

        self.block_constructor=LMC_conv_block
        self.args.freeze_module_after_step_limit = freeze_module_after_step_limit
        self.i_size=i_size
        self.deeper = deeper
        LMCs = ComponentList()

        self.num_LMC = 2 #layer_n_modules          #This parameter needs to be adjusted
        self.beam_results = []
        print("\nBuilding %d lower level modules in branch" % self.num_LMC)
        ### Allow building a single MLP at depth 0
        if (self.MODEL_DEPTH == 0) and (self.num_LMC == 1):
            # Single expert as the MLP
            conv = self.block_constructor( in_h, in_channels, out_channels, i_size, name=name, module_type=module_type, initial_inv_block_lr=initial_inv_block_lr,
                                                    deviation_threshold=deviation_threshold, freeze_module_after_step_limit=freeze_module_after_step_limit, deeper=deeper,
                                                                                           options=options, num_classes=num_classes)
            out_head = conv.out_h
            self.out_h = conv.out_h
            self.out_channels = conv.out_channels
            return  conv, out_head, conv

         ### Build leaf nodes: LMCs
        for m_i in range (0,int(self.num_LMC)):

             module_type='conv'
             conv = self.block_constructor( in_h, in_channels, out_channels, i_size, name=name+f'.{m_i}', module_type=module_type, initial_inv_block_lr=initial_inv_block_lr,
                                            deviation_threshold=deviation_threshold, freeze_module_after_step_limit=freeze_module_after_step_limit, deeper=deeper,  options=options, num_classes=num_classes)

             self.out_h = conv.out_h
             LMCs.append(conv)

        self.SubDTGN_conv = LMCs # `previous_layer_models` being the models in the hierarchical layer


        # Branch Build manager & merge block
        ########################################################################
        # Structure of structural layers
        ########################################################################
        self.pooling_inv=False
        self.inv_type = 'linear'

        merge_blocks = []
        self.init_structural()
        # self.get_combined(previous_layer_models, inv_block)

        self.inv_block_structural_backup_buffer = None
        self.functional_backup_state_dict_buffer = None

        self.register_buffer('optimizers_update_needed_buffer', torch.tensor(0.))
        self.hooks = []
        self.register_buffer('block_lr_buffer', torch.tensor(self.initial_inv_block_lr))
        self.register_buffer('_frozen_structural_buffer', torch.tensor(0.))
        self.register_buffer('_frozen_functional_buffer', torch.tensor(0.))
        # self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.runing_activation_window_record = []

        # merge_blocks.append(merge_block)
        #      ### Finalize the model
        # self.final_manager = previous_layer_models[0]

        if num_classes>0: # and self.module_type!='expert':
            representation_dim = self.out_channels*self.out_h*self.out_h
            self.decoder=nn.Linear(representation_dim,num_classes)
            self.classifier=nn.Sequential(nn.Flatten(), self.decoder)
        else:
            self.classifier=nn.Identity()
        print("The branch is done!")

    def init_structural(self):
        if self.args.structure_inv!='ae':
            if self.args.structure_inv=='linear_no_act':
                #only linear layers no activation function
                self.inv_blk_activation = None
                self.inv_type = 'linear'
                self.in_dim_inv = [self.out_channels*self.out_h*self.out_h]
            elif self.args.structure_inv=='pool_lin_no_act':
                #only linear layers no activation function
                self.pooling_inv=True
                self.inv_type = 'linear'
                self.inv_blk_activation = None
                self.in_dim_inv = [self.out_channels,self.out_h,self.out_h]
            elif self.args.structure_inv=='linear_act':
                #linear layers with activation inbetween (larger structural)
                self.inv_blk_activation = self.args.activation_structural
                self.inv_type = 'linear'
                self.in_dim_inv = [self.out_channels*self.out_h*self.out_h]
            elif self.args.structure_inv=='pool_only_large_lin_no_act':
                if np.prod([self.out_channels,self.out_h,self.out_h])>10000: #self.in_channels == 3:
                    self.pooling_inv=True
                    self.in_dim_inv = [self.out_channels,self.out_h,self.out_h]
                else:
                    self.in_dim_inv = [self.out_channels*self.out_h*self.out_h]
                self.inv_type = 'linear'
                self.inv_blk_activation = None
            elif self.args.structure_inv=='pool_only_large_lin_act':
                if np.prod([self.out_channels,self.out_h,self.out_h])>10000: #self.in_channels == 3:
                    self.pooling_inv=True
                    self.in_dim_inv = [self.out_channels,self.out_h,self.out_h]
                else:
                    self.in_dim_inv = [self.out_channels*self.out_h*self.out_h]
                self.inv_type = 'linear'
                self.inv_blk_activation = self.args.activation_structural
            elif self.args.structure_inv=='conv_act':
                #convolutional structural layers with activtion inbetween
                self.inv_blk_activation = self.args.activation_structural
                self.inv_type = 'conv'
                self.in_dim_inv = [self.out_channels,self.out_h,self.out_h]
            elif self.args.structure_inv=='conv_no_act':
                #convolutional structural layers with no activation
                self.inv_blk_activation = None
                self.inv_type = 'conv'
                self.in_dim_inv = [self.out_channels,self.out_h,self.out_h]
            elif self.args.structure_inv=='conv_only_large_act':
                #use convolutional layers only for large layers (e.g. input imnet), all the others use liner, use activation inbetween
                self.inv_blk_activation = self.args.activation_structural
                if self.in_channels == 3:
                    self.inv_type = 'conv'
                    self.in_dim_inv = [self.out_channels,self.out_h,self.out_h]
                else:
                    self.in_dim_inv = [self.out_channels*self.out_h*self.out_h]
            elif self.args.structure_inv=='conv_only_large_no_act':
                #use convolutional layers only for large layers (e.g. input imnet), all the others use linear, no activatioin
                self.inv_blk_activation = None
                if self.in_channels == 3:
                    self.inv_type = 'conv'
                    self.in_dim_inv = [self.out_channels,self.out_h,self.out_h]
                else:
                    self.in_dim_inv = [self.out_channels*self.out_h*self.out_h]
            else:
                raise NotImplementedError
            ########################################################################
            self.SubDTGN_inv = Inv_Block(in_dim=self.in_dim_inv, activation=self.inv_blk_activation, pooling_inv=self.pooling_inv, type=self.inv_type, deviation_statistic=self.args.deviation_statistic, use_bn=self.args.bn_structural, rs_steps=self.args.running_stats_steps, z_score_per_batch=self.args.z_score_per_batch) #, grad_multiplier=self.annealing_factor_buffer)
        else:
            self.in_dim_inv = [self.out_channels,self.out_h,self.out_h]
            self.SubDTGN_inv = Decoder(self.in_h,out_channels=self.out_channels, in_channels=self.in_channels, track_running_stats=self.args.track_running_stats_bn,  affine_bn=self.args.affine_bn_decoder, momentum=self.args.momentum_bn_decoder,use_bn=self.args.use_bn_decoder, kernel_size=self.args.kernel_size,
                                                    deviation_statistic=self.args.deviation_statistic, padding=self.args.padding, rs_steps=self.args.running_stats_steps, final_activation=self.args.activation_structural, activation_target=self.args.activation_target_decoder, module_name=self.name, module_type=self.module_type, n_heads=self.args.n_heads_decoder)


    def get_combined(self, previous_layer_models, manager):
         self.SubDTGN_conv = previous_layer_models
         self.SubDTGN_inv = manager
         # merge_block = ComponentList()
         # components_sub = ComponentList()
         #
         # components_sub.append(left_block)
         # components_sub.append(right_block)
         #
         # merge_block.append(components_sub)
         #
         # merge_block.append(manager)
         # return merge_block

    def forward(self, x,  beam_results, projection_phase_flag, info='', log_at_this_iter=False,task_id=None, temp=None, inner_loop=False, decoder = None, record_running_stats=False, env_assignments=None, detach_head=False, str_prior_temp=None,**kwargs):
        ####################################################
        # temp = 0.01
        # str_prior_temp = 0.01
        ####################################################
        x_m, str_loss_SubDTGN = self.forward_SubDTGN(x, info=info, log_at_this_iter=log_at_this_iter,task_id=task_id, temp=temp, inner_loop=inner_loop, decoder = decoder, record_running_stats=record_running_stats, env_assignments=env_assignments, detach_head=detach_head, str_prior_temp=str_prior_temp, projection_phase_flag =projection_phase_flag, beam_results = beam_results) # <-modules functional output

        if self.module_type=='expert':
              x_m, _ = x_m

        if self.args.detach_structural:
            in_structural = x_m.detach()
        else:
            in_structural = x_m

        if self.args.use_structural:
            if self.inv_block_structural_backup_buffer is not None:
                #we detach it for the new structrual, since it should learn to gate
                str_to_minimize, _, _ = self.SubDTGN_inv(x, in_structural.detach().view(x_m.size(0), *self.in_dim_inv), calculate_stats=self.args.deviation_statistic=='z_score', visualize=(log_at_this_iter and self.args.log_reconstructions), info=info)
                #we dont detach here - use detach_structural flag
                str_to_use_for_masks, z_score, delta = self.inv_block_structural_backup_buffer(x, in_structural.view(in_structural.size(0), *self.in_dim_inv), calculate_stats=self.args.deviation_statistic=='z_score')
            else:
                #no backup
                str_to_minimize, z_score, delta = self.SubDTGN_inv(x, in_structural.view(x_m.size(0), *self.in_dim_inv), calculate_stats=self.args.deviation_statistic=='z_score', visualize=(log_at_this_iter and self.args.log_reconstructions), info=info)
                str_to_use_for_masks = str_to_minimize.clone()
                str_to_use_for_masks = str_to_use_for_masks.detach()
        else:
            str_to_minimize=torch.zeros(x_m.size(0), device=device)
            str_to_use_for_masks=str_to_minimize
            delta = torch.zeros_like(str_to_minimize, device=device)
            outlier_score = torch.zeros_like(str_to_minimize, device=device)
            outlier = torch.tensor([False]*len(outlier_score)).to(device)
            z_score = torch.zeros_like(str_to_minimize, device=device)
        # structural outputs:
        # str_to_minimize - the structural loss that can be used to calculate gradients and backprop
        # str_to_use_for_masks - the structural loss that should be used for normalizing and calculating module activations weights

        if self.args.deviation_statistic=='z_score':
            outlier_score = torch.abs(z_score) if self.args.use_abs_score else z_score
        else:
            raise NotImplementedError

        if self.module_learned:
            outlier = outlier_score>self.deviation_threshold
        else:
            # if module is not learned (i.e. functional component is not fixed) the module does not report outliers
            outlier = torch.tensor([False]*len(outlier_score)).to(device)
        if self.args.dropout>0:
            x_m = self.dropout(x_m)


        return x_m, str_to_minimize, str_to_use_for_masks, delta, outlier_score, outlier, int(self.module_learned), int(self._frozen_functional_buffer), str_loss_SubDTGN

    def forward_SubDTGN_layer(self, X, beam_results, temp, info='', log_at_this_iter=False,  **kwargs):
        #collect module outputs in collections
        X_layer = []
        str_layer = []
        str_from_backup_detached_layer = []
        str_from_backup_layer = []
        deviation_layer = []
        z_scores_layer = []
        outliers_layer = []
        module_learned_layer = []
        fixed_functional_layer = []
        # deviation_func_str_layer = []
        ########################################

        if len(beam_results)>0:
            #in case of beam search
            for X in beam_results:
                X=X[-1]
                beam_X_layer = []
                beam_str_layer = []
                beam_str_from_backup_detached_layer = []
                beam_str_from_backup_layer = []
                beam_deviation_layer = []
                beam_z_scores_layer = []
                beam_outliers_layer = []
                beam_module_learned_layer = []
                beam_fixed_functional_layer = []
                beam_deviation_func_str_layer =[]
                for _, conv in enumerate(self.SubDTGN_conv):
                    #collect outputs from each module at layer for each beam path

                    func_out, str_to_minimize, str_from_backup, deviation, outliear_score, outlier, is_learned, fixed_functional = conv(X, info=info, log_at_this_iter=log_at_this_iter)
                    beam_z_scores_layer.append(outliear_score)
                    beam_module_learned_layer.append(is_learned) #[m_i]=age
                    beam_fixed_functional_layer.append(fixed_functional) #[m_i]=fixed_functional
                    beam_outliers_layer.append(outlier)
                    beam_X_layer.append(func_out)
                    beam_str_layer.append(str_to_minimize) # torch.norm(torch.norm(inv.mean(dim=0), p=2, dim=0).mean(0), p=2)
                    beam_str_from_backup_detached_layer.append(str_from_backup.detach())
                    beam_str_from_backup_layer.append(str_from_backup)
                    beam_deviation_layer.append(deviation)


                assert len(beam_X_layer)>0
                beam_str_layer_for_masks_layer = torch.stack(beam_str_from_backup_detached_layer)


                beam_deviation_layer = torch.stack(beam_deviation_layer) if beam_deviation_layer[0] is not None else torch.zeros_like(beam_str_layer)
                beam_module_learned_layer = torch.tensor(beam_module_learned_layer).to(device)
                beam_fixed_functional_layer = torch.tensor(beam_fixed_functional_layer).to(device)

                #############################################
                beam_z_scores_layer = torch.stack(beam_z_scores_layer)

                idx_old_modules = torch.where(beam_module_learned_layer>0)
                idx_new_modules = torch.where(beam_module_learned_layer==0)

                #
                if self.catch_outliers_for_old_modules and not self.args.K_train:
                    if self.training and len(idx_new_modules[0])>0 and len(idx_old_modules[0])>0:
                        idx_outlier_old_modules = torch.where(beam_outliers_layer[idx_old_modules].sum(0)==len(idx_old_modules[0]))[0] #outliers for all old modules
                        beam_str_layer_for_masks_layer[idx_new_modules,idx_outlier_old_modules]=1e-10
                        # print(str_layer_for_masks_layer)
                        beam_deviation_layer[idx_new_modules,idx_outlier_old_modules]=1e-10

                # add the current beam path output to the collection of outputs for the lauyer
                X_layer.extend(beam_X_layer)
                str_layer.extend(beam_str_layer)
                str_from_backup_layer.extend(beam_str_from_backup_layer)
                str_from_backup_detached_layer.append(beam_str_layer_for_masks_layer)

                deviation_layer.append(beam_deviation_layer)
                z_scores_layer.append(beam_z_scores_layer)
                fixed_functional_layer.append(beam_fixed_functional_layer)
                module_learned_layer.append(beam_module_learned_layer)

                # outliers_layer.extend(beam_outliers_layer)
            #############################################
            #TODO: if no new modules adeed to a layer, then dont fix funcitonal compnent of a module
            str_layer = torch.stack(str_layer)
            str_layer_from_backup_layer = torch.stack(str_from_backup_layer)
            str_layer_for_masks_layer = torch.cat(str_from_backup_detached_layer)
            z_scores_layer = torch.stack(z_scores_layer).mean(0)
            deviation_layer = torch.zeros_like(z_scores_layer) #torch.cat(deviation_layer)
            fixed_functional_layer=torch.cat(fixed_functional_layer)
            module_learned_layer =torch.cat(module_learned_layer)
            return torch.stack(X_layer), str_layer, str_layer_from_backup_layer, str_layer_for_masks_layer, deviation_layer, z_scores_layer, fixed_functional_layer, beam_module_learned_layer
        else:
            for _, conv in enumerate(self.SubDTGN_conv):
                func_out, str_to_minimize, str_from_backup, deviation, outliear_score, outlier, is_learned, fixed_functional= conv(X, info=info, log_at_this_iter=log_at_this_iter)

                #str_to_minimize - structural loss
                #str_from_backup - structural loss from backup component (should be used for masking i.e. the weighter sum for the layer output)
                #deviation - str_to_minimize - current running stats. mean
                #outliear_score - z_score
                #outlier - 'True' if outliear_score > deviation_threshold (if module is already fixed then False)
                #is_learned - 'True' if fixed
                #fixed_functional - 'True' if functional component is fixed
                z_scores_layer.append(outliear_score)
                module_learned_layer.append(is_learned)
                fixed_functional_layer.append(fixed_functional)
                outliers_layer.append(outlier)
                X_layer.append(func_out)
                str_layer.append(str_to_minimize)
                str_from_backup_detached_layer.append(str_from_backup.detach())
                str_from_backup_layer.append(str_from_backup)
                deviation_layer.append(deviation)

            assert len(X_layer)>0
            # end1 = time.time()
            str_layer = torch.stack(str_layer) # structural layer from the currently used structural component
            str_layer_from_backup_layer = torch.stack(str_from_backup_layer) # structural loss from backup (ignore if the backup mechanism is not being used)
            str_layer_for_masks_layer = torch.stack(str_from_backup_detached_layer) # detached tructural loss for masking

            deviation_layer = torch.stack(deviation_layer) if deviation_layer[0] is not None else torch.zeros_like(str_layer)
            module_learned_layer = torch.tensor(module_learned_layer).to(device)
            fixed_functional_layer = torch.tensor(fixed_functional_layer).to(device)

            #############################################
            outliers_layer = torch.stack(outliers_layer)
            z_scores_layer = torch.stack(z_scores_layer)
            temp = temp * torch.ones(outliers_layer.size(1)).to(device) #is temp changes here?

            idx_old_modules = torch.where(module_learned_layer>0)
            idx_new_modules = torch.where(module_learned_layer==0)

            # if self.catch_outliers_for_old_modules and not self.args.K_train:
            #     if self.training and len(idx_new_modules[0])>0 and len(idx_old_modules[0])>0:
            #         #only consider free modules for outlier samples
            #         idx_outlier_old_modules = torch.where(outliers_layer[idx_old_modules].sum(0)==len(idx_old_modules[0]))[0] #outliers for all old modules
            #         str_layer_for_masks_layer[idx_new_modules,idx_outlier_old_modules]=1e-10
            #         deviation_layer[idx_new_modules,idx_outlier_old_modules]=1e-10

            #############################################
            #TODO: if no new modules adeed to a layer, then dont fix funcitonal compnent of a module
            if isinstance(X_layer[0], Tuple):
                X_1 = torch.stack(list(map(lambda x: x[0], X_layer)))
                X_2 = torch.stack(list(map(lambda x: x[1], X_layer)))
                X=(X_1, X_2)
            else:
                X = torch.stack(X_layer)
            # return X, str_layer, str_layer_from_backup_layer, str_layer_for_masks_layer, deviation_layer, z_scores_layer, fixed_functional_layer, module_learned_layer
            return X, str_layer, str_layer_from_backup_layer, str_layer_for_masks_layer, deviation_layer, z_scores_layer, fixed_functional_layer, module_learned_layer, X_layer

    def forward_SubDTGN(self, X,  projection_phase_flag, beam_results, info='', log_at_this_iter=False, task_id=None, temp=None, inner_loop=False, decoder = None, record_running_stats=False, env_assignments=None, detach_head=False, str_prior_temp=None,**kwargs):
        #start = time.time()

        # if env_assignments is not None:
        #     raise NotImplementedError
        # if self.encoder is not None:
        #     X = self.encoder(X)
        # if self.args.module_type=='linear':
        #         X=X.view(X.size(0), -1)
        # temp = self.args.temp
        #
        # if str_prior_temp is not None:
        #     str_prior_temp=str_prior_temp
        #     if str_prior_temp<self.min_str_prior_temp.item(): #store minimal temp
        #         self.min_str_prior_temp = str_prior_temp.clone()#.to(device)
        # else:
        #     str_prior_temp=self.args.str_prior_temp
        #
        # if not self.training:
        #     str_prior_temp = min(self.min_str_prior_temp.item(),str_prior_temp)
        #
        # if isinstance(X, Tuple):
        #     n=X[0].size(0)
        #     c = X[0].shape[1]
        # else:
        #     n = X.shape[0]
        #     c = X.shape[1]
        #
        # beam_results = []
        # if self.args.concat=='beam' and self.args.beam_width:
        #     #will store the best beam
        #     beam_results.append(([],[],X))
        self.beam_results = beam_results
        added =False
        n = X.shape[0]
        c = X.shape[1]
        # str_loss_total = torch.zeros(self.MODEL_DEPTH).to(device)
        str_loss_total = torch.zeros(1).to(device)
        outlier_signal = []
        total_mask = []
        deviation_mask = []
        raw_mask = []
        # add_modules_at_leayer=set()
        fixed_modules_functional = []
        fixed_modules = []
        X_layer = []

        ################################
        fw_layer = self.forward_SubDTGN_layer( X, beam_results, temp, info=info,log_at_this_iter =log_at_this_iter)
        X_tmp, str_loss, str_loss_from_backup, str_loss_detached_for_mask, deviation_layer, z_scores_layer, fixed_modules_function_layer, fixed_module, X_list = fw_layer
        # print("str_loss.requires_grad is "+ str(str_loss.requires_grad))

        fixed_modules_functional.append(fixed_modules_function_layer)
        fixed_modules.append(fixed_module)

        deviation_layer = torch.abs(deviation_layer)
        if self.args.str_prior_factor>0:
        #bias module selection towards majority in the batch
            str_prior = torch.softmax(-torch.log(str_loss_detached_for_mask.mean(1))/str_prior_temp, dim=0) #torch.softmax(-torch.log(str_loss_detached_for_mask.mean(1))/self.args.str_prior_temp, dim=0)
        else:
            str_prior = torch.softmax(torch.ones_like(deviation_layer.mean(1)).to(deviation_layer.device), dim=0)

        # fzy
        if self.args.concat=='beam':
            module_indicies = np.tile(np.arange(1),len(beam_results))
            most_likel_module = module_indicies[str_prior.argmax().item()]
        else:
            most_likel_module = str_prior.argmax().item()
        # fzy
        ###################################################################
        # Manage projection phase and module addition #
        # added=False
        # if self.args.automated_module_addition and self.training:
        #     if not projection_phase_flag:
        #         # ====================================
        #         #unfreeze functional for modules that were frozen in the previous projection phase
        #         if len(self.modules_to_unfreeze_func)>0:
        #             for m in self.modules_to_unfreeze_func:
        #                 m.unfreeze_functional()
        #                 m.args.detach_structural=True
        #             self.modules_to_unfreeze_func=[]
        #             self.optimizer, _ = self.get_optimizers()
        #         #====================================
        #          if self.args.concat=='beam':
        #              module_indicies = np.tile(np.arange(len(self.components[l])),len(beam_results))
        #              self.most_likel_module = module_indicies[self.str_prior.argmax().item()]
        #          else:
        #              self.most_likel_module = self.str_prior.argmax().item()
        #         added, params_changed, new_m_idx =self.maybe_add_modules(z_scores_layer.mean(1), fixed_module, layer=l, bottom_up=True, module_i=most_likel_module, init_stats=self.args.init_stats)
        #         # if new module was added start the projection phase
        #         if added or (self.args.treat_unfreezing_as_addition and params_changed):
        #             if not self.args.no_projection_phase:
        #                 ##########################
        #                 #Start projection phase
        #                 # - let the gradients from structural losses from layers above flow
        #                 ##########################
        #                 # current layer
        #                 for ni in new_m_idx:
        #                     #the current layers structural components are detached (no gradient flows)
        #                     layer[ni].args.detach_structural=True
        #                 #layers below
        #                 if self.args.fix_layers_below_on_addition:
        #                     for other_layer in self.components[:l]:
        #                         for module_below in other_layer:
        #                             if module_below.module_learned:
        #                                 frozen=module_below.freeze_functional()
        #                                 if frozen:
        #                                     #if it was not frozen before, will be unfrozen after the projection phase
        #                                     self.modules_to_unfreeze_func.append(module_below)
        #
        #                             # module_below.args.detach_structural=True
        #                 #layers above
        #                 for other_layer in self.components[l+1:]:
        #                     for module_above in other_layer:
        #                         # other_c._reset_backups()
        #                         frozen=module_above.freeze_functional()
        #                         if frozen:
        #                             #if it was not frozen before, will be unfrozen after the projection phase
        #                             self.modules_to_unfreeze_func.append(module_above)
        #                         #let the gradient flow from structural
        #                         module_above.args.detach_structural=False
        #             #no module addition for the next self.args.projection_phase_length iterations
        #             self._steps_since_last_addition=self._steps_since_last_addition*0.
        #             projection_phase_flag=True
        #         if params_changed:
        #             self.optimizer, self.optimizer_structure = self.get_optimizers()
        ###################################################################

        ###################################################################
        # Claculate mask for weigting thew modules' outputs
        likelihood = torch.softmax(-torch.log((str_loss_detached_for_mask))/temp, dim=0)
        # mask = torch.softmax( torch.log(likelihood) + self.args.str_prior_factor * torch.log(str_prior.unsqueeze(0).repeat(n,1).T), dim=0)
        mask_unnorm = likelihood * str_prior.unsqueeze(0).repeat(n,1).T
        mask = (mask_unnorm)/mask_unnorm.sum(0)  # torch.softmax(torch.log(mask_unnorm), dim=0)
        ###################################################################

        total_mask.append(mask)
        raw_mask.append(str_loss_from_backup)

        if self.args.concat=='beam':
            assert not self.args.module_oracle
            components =  np.tile(self.SubDTGN_conv[:len(self.SubDTGN_conv)-int(added)],len(beam_results))  #list(itertools.chain.from_iterable(itertools.repeat(x, len(beam_results)) for x in self.components[l][:len(self.components[l])-int(added)]))
            assert len(components)==str_loss.size(0)==mask.size(0)==str_loss_from_backup.size(0)
            for m, v, v_backup, module, in zip(mask, str_loss.detach(), str_loss_from_backup, components):
                module.update_stats(v, v_backup, m, record=True)
        else:
            if self.training and not inner_loop and record_running_stats:
                ################################################
                for m, str_l, str_l_backup, module in zip(mask, str_loss.detach(), str_loss_from_backup, self.SubDTGN_conv):
                    module.update_stats(str_l, str_l_backup, m)
            ###################################################################
        ##################################################################
        deviation_mask.append(deviation_layer)
        outlier_signal.append(z_scores_layer)
        ##########################################################################################
        # Calculate structural losses for the layer
        ##########################################################################################
        if not inner_loop:
            if torch.sum(fixed_modules_function_layer) != len(fixed_modules_function_layer) or self.args.regime=='normal':
                if self.args.optmize_structure_only_free_modules and not projection_phase_flag:
                    idx_free = torch.lt(fixed_modules_function_layer,1)
                    str_act = (str_loss[idx_free]) if torch.sum(idx_free)>0 else torch.zeros_like(str_loss)
                    if self.args.mask_str_loss:
                        if torch.sum(idx_free)>0:
                             str_act*=F.normalize(mask[torch.lt(fixed_modules_function_layer,1)],dim=0, p=1)
                    str_act=str_act.mean(1).mean(0)
                else:
                    str_act = str_loss
                    if self.args.mask_str_loss:
                        str_act*=mask
                    str_act=str_act.mean(1).mean(0)

                if str_loss_from_backup.requires_grad and self.args.regime=='normal':
                        str_act += (str_loss_from_backup*mask).mean(1).mean(0)
                str_loss_total[0]+=str_act
                # print("str_loss_total.requires_grad"+str(str_loss_total.requires_grad))
        else:
            # in the inner loop of continual meta CL
            str_act = str_loss
            if self.args.mask_str_loss:
                str_act*=mask
            str_loss_total[0]+=str_act.mean(1).mean(0)
        ##########################################################################################

        #############################################
         # Calculate the layer output
        #############################################
        if self.args.concat=='sum':
            if torch.sum(torch.isnan(mask))>0:
                X_layer_output = X_tmp.sum(0)
                X_layer.append(X_layer_output)
            else:
                # X = torch.einsum("ijklm,ij->jklm", X_tmp, mask)
                if isinstance(X_tmp, Tuple):
                    X_layer_output_1 = (X_tmp[0] * mask.view(mask.size(0), mask.size(1), *[1]*len(X_tmp[0].shape[2:]))).sum(0)
                    X_layer_output_2 = (X_tmp[1] * mask.view(mask.size(0), mask.size(1), *[1]*len(X_tmp[1].shape[2:]))).sum(0)
                    X_layer_output = (X_layer_output_1, X_layer_output_2)
                    X_layer.append(X_layer_output)
                else:
                     X_layer_output = (X_tmp * mask.view(mask.size(0), mask.size(1), *[1]*len(X_tmp.shape[2:]))).sum(0)
                     X_layer.append(X_layer_output)

                    #fzy
                  #if most_likel_module == 0:
                    #    X_layer_output = X_tmp[0]
                   # else:
                       # X_layer_output = X_tmp[1]

        # elif self.args.concat=='beam':
        #     beam_results, str_prior = self.update_beam_results(str_prior, X_tmp, beam_results, l, added)
        else:
             raise NotImplementedError
            #############################################

        # X_layer_output = self.avgpool(X_layer_output)
        # sort_deviation_func_str = sorted(deviation_func_str, reverse= False)
        # sorted_id = sorted(range(len(deviation_func_str)), key=lambda k: deviation_func_str[k], reverse=False)
        # id = sorted_id[0]
        # X_layer_output = X_list[id]
        # X_layer.append(X_layer_output)
        str_loss_total = torch.sum(str_loss_total)
        # print("str_loss_total.requires_grad is " + str(str_loss_total.requires_grad))
        return X_layer_output, str_loss_total


    def update_stats(self, v: torch.Tensor, v_backup: torch.Tensor, m: torch.Tensor = torch.tensor(1.), threshold=0.5, record=False):
        '''
        Update the running stats of the module
        '''
        #######################
        threshold=0.
        idx = torch.where((m>threshold).int()>0)
        m = m[idx]
        v = v[idx].detach()
        v_backup = v_backup[idx].detach()
        if self.active:
            self.current_step = self.current_step + 1

        ###update uning stats###
        if len(m)>0:
            if not record:
                if self.inv_block_structural_backup_buffer is not None or not self._frozen_structural_buffer:
                    #if backup exists or not frozen
                    assert not any(v<0)
                    # self.inv_block.runing_activation_window_buffer.push(v.mean())
                    self.SubDTGN_inv.runing_activation_window_buffer.push(v.mean())
                if self.inv_block_structural_backup_buffer is not None and self.args.update_backup_stats:
                    assert not any(v_backup<0)
                    self.inv_block_structural_backup_buffer.runing_activation_window_buffer.push(v_backup.mean())

                # print(self.name)
                # module_learned = self.module_learned()
                if not self.module_learned:
                    ########################
                    if self.args.anneal_structural_lr:
                        self.annealing_factor_buffer.data = cosine_rampdown(self.current_step, self._total_steps)
                        self.block_lr_buffer.data = self.initial_inv_block_lr * self.annealing_factor_buffer

            #this is for beam search, records the stats and then pushes them later only to the modules of the selected path
            else:
                if self.inv_block_structural_backup_buffer is not None or not self._frozen_structural_buffer: #frozen:
                    #if backup exists or not frozen
                    assert not any(v<0)
                    # assert self.runing_activation_window_record
                    assert not self.args.update_backup_stats
                    self.runing_activation_window_record.append(v.mean())
    @property
    def active(self):
        # module_learned = self.module_learned()
        if not self.module_learned:
             return True
        if not self._frozen_functional_buffer:
            return True
        else:
            return False

    @property
    def current_step(self):
        return self._current_step_buffer #if self._current_step_buffer <= self._total_steps else self._total_steps

    @current_step.setter
    def current_step(self, v):
        if self.args.freeze_module_after_step_limit and v % self._total_steps==0:
            self._reset_backups()
        self._current_step_buffer = v

    def _reset_backups(self):
        '''
        Creates a new 'snapshot' of the components - in the future when it drops back, it will drop back to the current state
        '''
        assert self.active
        self.module_learned=torch.tensor(1.)
        if self.args.use_backup_system:
            #1. remove current functional backup, as new should be created
            self._clear_backup_functional()
            print(f'resetting backups on {self.name}')
            #2. create a new functional backup
            fb_created = self._maybe_backup_functional()
            assert fb_created or not self.args.use_backup_system
            self._freeze_functional_backup()
            if not self.active:
                self.freeze_functional(inner_loop_free=True) #<- can still be updated in the inner loop
            #if no structural backup exist, create it
            if self.args.use_backup_system_structural:
                if self.inv_block_structural_backup_buffer is None:
                    #3a. create new structural backup if we dont have one
                    # self.freeze_and_backup_structure()
                    sb_created = self._maybe_backup_structural()
                    assert sb_created or not self.args.use_backup_system
                    self._freeze_structural_backup()
                    if sb_created:
                        self._reinit_structural()
                else:
                    #3b. otherwise just make a new stats backup
                    self.inv_block_structural_backup_buffer.backup_stats()
            else:
                self.freeze_structural()
        else:
            self.freeze_structural()
        self.optimizers_update_needed_buffer = torch.tensor(1.)

    def on_finished_outer_loop(self, finished:bool):
        if self.args.use_structural:
            if finished:
                self.SubDTGN_inv.consolidate_stats_from_inner_to_outer()
            else:
                self.SubDTGN_inv.consolidate_stats_from_outer_to_inner()

            if self.inv_block_structural_backup_buffer is not None:
                if finished:
                    self.inv_block_structural_backup_buffer.consolidate_stats_from_inner_to_outer()
                else:
                    self.inv_block_structural_backup_buffer.consolidate_stats_from_outer_to_inner()

    def freeze_functional(self, inner_loop_free=True, if_eval_bn=False):
        if not self._frozen_functional_buffer:
            self._frozen_functional_buffer = torch.tensor(1.)
            for p in self.SubDTGN_conv.parameters():
                #setting requires_grad to False would prevent updating in the inner loop as well
                p.requires_grad = inner_loop_free
                #this would prevent updating in the outer loop:
                p.__class__ = FixedParameter
            # fzy
            for DTGN_conv in self.SubDTGN_conv:
                DTGN_conv.module_learned=torch.tensor(1.)
                DTGN_conv.freeze_functional(inner_loop_free=False)

            for p in self.classifier.parameters():
                p.requires_grad = inner_loop_free
                #this would prevent updating in the outer loop:
                p.__class__ = FixedParameter

            if self.args.keep_bn_in_eval_after_freeze:
                bn_eval(self)
                self.bn_eval_buffer=torch.tensor(1.)
            print(f'freezing functional {self.name}')
            return True
        else:
            print(f'could not freze functional as it is already frozen at {self.name}')
            return False

    def freeze_structural(self):
        if not self._frozen_structural_buffer and self.args.use_structural:
            self._frozen_structural_buffer = torch.tensor(1.)
            for p in self.SubDTGN_inv.parameters():
                #setting requires_grad to False would prevent updating in the inner loop as well
                p.requires_grad = False
                #this would prevent updating in the outer loop:
                p.__class__ = FixedParameter

            for DTGN in self.SubDTGN_conv:
                DTGN.freeze_structural()

            print(f'freezing structural {self.name}')
            return True
        else:
            print(f'could not freze structural as it is already frozen at {self.name}')
            return False

    @property
    def module_learned(self):
        return int(self.module_learned_buffer.item())

    @module_learned.setter
    def module_learned(self, v):
        self.module_learned_buffer.data = torch.tensor(float(v))
        for DTGN in self.SubDTGN_conv:
            DTGN.module_learned_buffer.data = torch.tensor(float(v))


    def on_module_addition_at_my_layer(self, inner_loop_free=True):
        if self.active:
            #when adding a module to the layer of this module
            self._maybe_dropback_functional()
            self._maybe_dropback_structural(dropback_stats=True)
            self.freeze_functional(inner_loop_free=inner_loop_free)
            if not self._frozen_structural_buffer:
                self.freeze_structural()
        if self.args.keep_bn_in_eval_after_freeze:
                bn_eval(self)
                self.bn_eval_buffer=torch.tensor(1.)

    def _maybe_dropback_functional(self):
        if self.functional_backup_state_dict_buffer is not None:
                #drop back to previous backed-up dynamiccs
                self.module = self._clone_func_blck(copy.deepcopy(self.functional_backup_state_dict_buffer.state_dict()))
                # self.module.load_state_dict(self.functional_backup_state_dict_buffer)
                self._clear_backup_functional()
                return True
        else:
            print(f'unable to drop back functionl as there is not functional backup at module {self.name}')
            return False

    def _maybe_dropback_structural(self, dropback_stats:bool, unfreeze:bool=False):
        if self.inv_block_structural_backup_buffer is not None:
            # print(self.inv_block_structural_backup_buffer)
            if dropback_stats:
                self.inv_block_structural_backup_buffer.dropback_stats()
            self.SubDTGN_inv = self._clone_inv_blck(copy.deepcopy(self.inv_block_structural_backup_buffer.state_dict()))
            if unfreeze:
                for p in self.SubDTGN_inv.parameters():
                    p.__class__ = nn.Parameter
                    p.requires_grad = True
            self._clear_backup_structural()
            return True
        else:
            print(f'unable to dropback to structural backup as not structural backup exists for module {self.name}')

    def reset(self):
        self.module_learned_buffer = torch.tensor(0.)
        #Reset things for new modules
        self.annealing_factor_buffer.data = torch.tensor(1.).to(device)
        self.block_lr_buffer = torch.tensor(self.initial_inv_block_lr)
        self._current_step_buffer.data = torch.tensor(0.).to(device)
        self._frozen_structural_buffer = torch.tensor(0.)
        self._frozen_functional_buffer = torch.tensor(0.)
        self._clear_backup_structural()
        self._clear_backup_functional()
        for DTGN in self.SubDTGN_conv:
            DTGN.reset()


    def unfreeze_functional(self):
        if self._frozen_functional_buffer:
            for p in self.SubDTGN_conv.parameters():
                p.__class__ = nn.Parameter
                p.requires_grad = True
            self._frozen_functional_buffer = torch.tensor(0.)

            for DTGN_conv in self.SubDTGN_conv:
                DTGN_conv.unfreeze_functional()

            print(f'unfreezing {self.name}')
            return True
        else:
            return False

    def _maybe_backup_functional(self, verbose=True):
        if self.functional_backup_state_dict_buffer is None and self.args.use_backup_system:
            self.functional_backup_state_dict_buffer = self._clone_func_blck(self.module.state_dict())
            return True
        else:
            if verbose:
                if self.args.use_backup_system:
                    print(f'unable to backup functional as a backup already exists for module {self.name}')
            return False

    def unfreeze_learned_module(self, unfreeze_str=False):
        success = self.unfreeze_functional()
        if unfreeze_str:
            self.unfreeze_structural()
        if success:
            self.reset_step_count()
            self._maybe_backup_structural()
        return success

    def reset_step_count(self):
        for p in self.SubDTGN_conv:
            p.reset_step_count()
        self._current_step_buffer = torch.tensor(0.).to(device)

    def _maybe_backup_structural(self, verbose=True):
        if self.inv_block_structural_backup_buffer is None and self.args.use_backup_system and self.args.use_backup_system_structural:
            self.inv_block_structural_backup_buffer = self._clone_inv_blck(copy.deepcopy(self.SubDTGN_inv.state_dict()))
            ############################################################
            #creating backup of stats at this point, in case we need to return to this state
            self.inv_block_structural_backup_buffer.backup_stats()
            ############################################################
            return True
        else:
            if verbose:
                if self.args.use_backup_system and self.args.use_backup_system_structural:
                    print(f'unable to backup structural as a backup already exists for module {self.name}')
            return False

    def _clone_inv_blck(self, state_dict=None):
        if self.args.structure_inv!='ae':
            new_inv_block = Inv_Block(in_dim=self.in_dim_inv, activation=self.inv_blk_activation, pooling_inv=self.pooling_inv,type=self.inv_type, deviation_statistic=self.args.deviation_statistic, use_bn=self.args.bn_structural, rs_steps=self.args.running_stats_steps, z_score_per_batch=self.args.z_score_per_batch).to(device)
        else:
            new_inv_block = Decoder(self.in_h,out_channels=self.out_channels, in_channels=self.in_channels, track_running_stats=self.args.track_running_stats_bn,  affine_bn=self.args.affine_bn_decoder, momentum=self.args.momentum_bn_decoder,use_bn=self.args.use_bn_decoder, kernel_size=self.args.kernel_size,
                                                                    deviation_statistic=self.args.deviation_statistic, padding=self.args.padding, rs_steps=self.args.running_stats_steps, final_activation=self.args.activation_structural, activation_target=self.args.activation_target_decoder, module_name=self.name, module_type=self.module_type, n_heads=self.args.n_heads_decoder).to(device)

        new_inv_block.load_state_dict(state_dict)
        return new_inv_block

    def unfreeze_structural(self):
        if self._frozen_structural_buffer and self.args.use_structural:
            for p in self.SubDTGN_inv.parameters():
                #setting requires_grad to False would prevent updating in the inner loop as well
                p.requires_grad = True
                #this would prevent updating in the outer loop:
                p.__class__ = Parameter
            for DTGN_conv in self.SubDTGN_conv:
                DTGN_conv.unfreeze_structural()
            print(f'unfreezing {self.name}')
            self._frozen_structural_buffer = torch.tensor(0.)
            return True
        else:
            print(f'could not freze structural as it is already frozen at {self.name}')
            return False

    def _clear_backup_structural(self):
        self.__delattr__('inv_block_structural_backup_buffer')
        self.inv_block_structural_backup_buffer = None

    def _clear_backup_functional(self):
        self.__delattr__('functional_backup_state_dict_buffer')
        self.functional_backup_state_dict_buffer = None

    def _freeze_functional_backup(self):
        if self.functional_backup_state_dict_buffer is not None:
            for p in self.functional_backup_state_dict_buffer.parameters():
                    p.__class__ = FixedParameter
                    p.requires_grad = False
        else:
            print(f'unable to freeze functional backup as it does not exists at {self.name}')
