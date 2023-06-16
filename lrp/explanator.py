'''
Created on 23 Sep 2022

@author: danhbuithi
'''

import torch  
from lrp import basic
from lrp.basic import forward_hook, embedding_relpop
from lrp.model import relMSSimilarityModel

def register_forward_hook_4_model(model):
    children = list(model.children())
    
    if isinstance(model, basic.RelProp) == True :
        model.register_forward_hook(forward_hook)
    for child in children: 
        register_forward_hook_4_model(child)

class LRPGenerator(object):
    '''
    classdocs
    '''

    def __init__(self, model: relMSSimilarityModel):
        '''
        Constructor
        '''
        self.model = model 
        self.model.eval()
        
        
    def forward(self, mz_f, mz_mask, loss_f, loss_mask):
        self.model(mz_f, mz_mask, loss_f, loss_mask)
        
    def _collect_cam_values(self, layers, N):    
        R = []
        for block in layers:
            grad = block.get_attn_gradients()
            grad = grad.view(N, -1, grad.shape[-2], grad.shape[-1]) #batch x nhead x Lt x Ls
            
            cam = block.get_attn_cam()
            grad = cam.view(N, -1, grad.shape[-2], grad.shape[-1])
            
            cam = grad * cam 
            
            cam = cam.clamp(min=0).mean(dim=1) #batch x Lt x Ls
            #print(cam[cam > 0])
            R.append(cam)
            
        joint_attention = R[0]
        for x in R[1:]:
            joint_attention += x
        joint_attention /= len(R)
        
        return joint_attention
    
    
                    
    def generate(self, mz_f, mz_mask, loss_f, loss_mask):
        output = self.model(mz_f, mz_mask, loss_f, loss_mask)
        kwargs = {'alpha' : 1}
        
        #N = output.shape[0] #batch size
        
        #R = output.detach()
        R = torch.ones(output.shape).to(output.device)
        output = torch.sum(output)
        
        self.model.zero_grad()
        output.backward(retain_graph=True)
        
        full_mz_R, _ = self.model.rel_prop(R, **kwargs) #N x L x K
        full_mz_R = embedding_relpop(self.model.mz_encoder.defect_emb, mz_f, full_mz_R, 102, **kwargs)
        full_mz_R = full_mz_R.detach()
        full_mz_R[full_mz_R < 0] = 0
         
        fragment_weights = self.model.mz_encoder.transformer_encoder.rollout(start_layer=0)
        fragment_weights = fragment_weights.detach() 
        fragment_weights[fragment_weights < 0] = 0
        
        '''
        full_loss_R = embedding_relpop(self.model.loss_encoder.defect_emb, loss_f, full_loss_R, 102, **kwargs)
        full_loss_R = full_loss_R.detach()
        full_loss_R[full_loss_R < 0] = 0
        ''' 
        v = output.detach() 
        return v, full_mz_R, fragment_weights
        
        
        
        
        
            
            
            
        
        