'''
Created on 16 Sep 2022

@author: danhbuithi
'''

import math
import copy 

import torch 
from torch import nn
 
from lrp import basic


class relScaleDotProductAttention(nn.Module):
    
    def __init__(self, dropout_p):
        super(relScaleDotProductAttention, self).__init__()
        
        self.matmul_1 = basic.relMatmul()
        self.add = basic.relAdd()
        self.softmax = basic.relSoftmax(dim=-1)
        
        self.dropout = None 
        if dropout_p > 0.0:
            self.dropout = basic.relDropout(dropout_p)
        
        self.matmul_2 = basic.relMatmul()
        
        self.attn_cam = None
        self.attn = None
        self.attn_gradients = None
        
    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn
        
    def save_attn_cam(self, cam):
        self.attn_cam = cam 
        
        
    def get_attn_cam(self):
        return self.attn_cam 
    
    
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
        
    def get_attn_gradients(self):
        return self.attn_gradients
        
    def forward(self, q, k, v, attn_mask=None):
        _, _, E = q.shape
        q = q / math.sqrt(E)
        
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn = self.matmul_1((q, k.transpose(-1, -2)))
        
        self.attn_mask = attn_mask
        if attn_mask is not None:
            attn = self.add((attn, attn_mask))
            
        attn = self.softmax(attn)
        self.save_attn(attn)
        
        #if self.training == False: 
        attn.register_hook(self.save_attn_gradients)
        
        if self.dropout is not None:
            attn = self.dropout(attn)
            
        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = self.matmul_2((attn, v))
        return output#, attn
        
    def rel_prop(self, R, **kwargs):
        attnR, vR = self.matmul_2.rel_prop(R, **kwargs) #R1: attn, R2: v
        attnR /= 2 
        vR /= 2
        
        self.save_attn_cam(attnR)
            
        if self.dropout is not None:
            attnR = self.dropout.rel_prop(attnR, **kwargs)
            
        attnR = self.softmax.rel_prop(attnR, **kwargs)
        
        if self.attn_mask is not None:
            attnR, _ =self.add.rel_prop(attnR, **kwargs) #attn, attn_mask
            
            
        qR, kR = self.matmul_1.rel_prop(attnR, **kwargs)
        qR /= 2 
        kR /= 2
        kR = kR.transpose(-1, -2)
        return qR, kR, vR
            
                
class relMultiheadAttention(nn.Module):
    
    def __init__(self, emb_dim, num_heads, dropout_p=0.0, bias=True):
        super(relMultiheadAttention, self).__init__()
        
        self.q_proj = basic.relLinear(emb_dim, emb_dim, bias)
        self.k_proj = basic.relLinear(emb_dim, emb_dim, bias)
        self.v_proj = basic.relLinear(emb_dim, emb_dim, bias)
        
        self.scaled_dot_product = relScaleDotProductAttention(dropout_p)
        self.out_proj = basic.relLinear(emb_dim, emb_dim, bias)
        
        self.num_heads = num_heads
        
    def get_attn_cam(self):
        return self.scaled_dot_product.attn_cam
    
    def get_attn_gradients(self):
        return self.scaled_dot_product.attn_gradients
    
    
    def _reshape_to_batches(self, x):
            batch_size, seq_len, in_feature = x.size()
            head_dim = in_feature // self.num_heads
            return x.reshape(batch_size, seq_len, self.num_heads, head_dim)\
                    .permute(0, 2, 1, 3)\
                    .reshape(batch_size * self.num_heads, seq_len, head_dim)
    
    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.num_heads
        out_dim = in_feature * self.num_heads
        return x.reshape(batch_size, self.num_heads, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)
            
    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):
        bsz, src_len, _ = k.shape
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len). \
            expand(-1, self.num_heads, -1, -1).reshape(bsz*self.num_heads, 1, src_len)
            
            if attn_mask is None: 
                attn_mask = key_padding_mask 
            else: 
                attn_mask = attn_mask.masked_fill(key_padding_mask, float('-inf'))
                
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float('-inf'))
            attn_mask = new_attn_mask
           
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
      
        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
    
        
        #attn_output, attn_output_weights = self.scaled_dot_product(q, k, v, attn_mask)
        #attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
        #attn_output_weights = attn_output_weights.sum(dim=1) / self.num_heads
        
        attn_output = self.scaled_dot_product(q, k, v, attn_mask) #B*nheads * Nt x head_dim
        attn_output = self._reshape_from_batches(attn_output) # B x Nt x D
        attn_output = self.out_proj(attn_output) #B*nheads, x Nt x D
        
         
        return attn_output
    
    
    def rel_prop(self, R, **kwargs):

        R = self.out_proj.rel_prop(R, **kwargs)
        R = self._reshape_to_batches(R)
        qR, kR, vR = self.scaled_dot_product.rel_prop(R, **kwargs)
        
        qR = self._reshape_from_batches(qR)
        kR = self._reshape_from_batches(kR)
        vR = self._reshape_from_batches(vR)
        
        qR = self.q_proj.rel_prop(qR, **kwargs)
        kR = self.k_proj.rel_prop(kR, **kwargs)
        vR = self.v_proj.rel_prop(vR, **kwargs)
        
        
        return qR, kR, vR
        
class relTransformerEncoderLayer(nn.Module):
    
    def __init__(self, d_model, nheads, dim_feedforward, dropout=0.1):
        super(relTransformerEncoderLayer, self).__init__()
        
        self.clone_1 = basic.relClone()
        self.self_attn = relMultiheadAttention(d_model, nheads, dropout_p=dropout)
        self.dropout1 = basic.relDropout(dropout)
        self.add1 = basic.relAdd()
        self.norm1 = basic.relLayerNorm(d_model)
        
        
        self.clone_2 = basic.relClone()
        self.linear1 = basic.relLinear(d_model, dim_feedforward)
        self.activate_fn = basic.relReLU()
        
        self.dropout = basic.relDropout(dropout)
        self.linear2 = basic.relLinear(dim_feedforward, d_model)
        self.dropout2 = basic.relDropout(dropout)
        self.add2 = basic.relAdd()
        self.norm2 = basic.relLayerNorm(d_model)
        
        
        
    def forward(self, src, attn_mask=None, src_key_padding_mask=None):
        x1, x2, x3, x4 = self.clone_1(src, 4) 
    
        h = self.self_attn(x1, x2, x3, attn_mask, src_key_padding_mask)
        h = self.dropout1(h)
        
        x = self.add1((x4, h))
        x = self.norm1(x)
        
        x1, x2 = self.clone_2(x, 2)
        h = self.linear1(x1) 
        h = self.activate_fn(h)
        h = self.dropout(h)
        h = self.linear2(h)
        h  = self.dropout2(h)
        
        x = self.add2((x2, h))
        x = self.norm2(x)
        return x
    
    
    def rel_prop(self, R, **kwargs):
    
        R = self.norm2.rel_prop(R, **kwargs)
        x2R, hR = self.add2.rel_prop(R, **kwargs)
        
        hR = self.dropout2.rel_prop(hR, **kwargs)
        hR = self.linear2.rel_prop(hR, **kwargs)
        hR = self.dropout.rel_prop(hR, **kwargs)
        hR = self.activate_fn.rel_prop(hR, **kwargs)
        x1R = self.linear1.rel_prop(hR, **kwargs)        
        xR = self.clone_2.rel_prop((x1R, x2R), **kwargs)
        
        xR = self.norm1.rel_prop(xR, **kwargs)
        x4R, hR = self.add1.rel_prop(xR, **kwargs)
        
        hR = self.dropout1.rel_prop(hR, **kwargs)
        x1R, x2R, x3R = self.self_attn.rel_prop(hR, **kwargs)
        xR = self.clone_1.rel_prop((x1R, x2R, x3R, x4R), **kwargs)
        return xR
    
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention

class relTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(relTransformerEncoder, self).__init__()
        
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        
    def forward(self, src, attn_mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, attn_mask, src_key_padding_mask)
        return output 
    
    
    def rel_prop(self, R, **kwargs):
        for mod in self.layers[::-1]:
            R = mod.rel_prop(R, **kwargs)
        
        return R 
    
    def rollout(self, start_layer):
        cams = []
        for blk in self.layers:
            grad = blk.self_attn.get_attn_gradients()
            cam = blk.self_attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cams.append(cam.unsqueeze(0))
            
        rollout = compute_rollout_attention(cams, start_layer)
        rollout[:, 0, 0] = rollout[:, 0].min()
        return rollout[:, 0]
        
    


class relTransformerDecoderLayer(nn.Module):
    
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(relTransformerDecoderLayer, self).__init__()
        
        self.tgt_clone1 = basic.relClone()
        self.mem_clone1 = basic.relClone()
        
        self.cross_attn = relMultiheadAttention(d_model, nhead, dropout)
        self.dropout2 = basic.relDropout(dropout)
        self.add1 = basic.relAdd()
        self.norm2 = basic.relLayerNorm(d_model)
        
        self.tgt_clone2 = basic.relClone()
        self.linear1 = basic.relLinear(d_model, dim_feedforward)
        self.activate_fn = basic.relReLU()
        self.dropout = basic.relDropout(dropout)
        self.linear2 = basic.relLinear(dim_feedforward, d_model)
        self.dropout3 = basic.relDropout(dropout)
        self.add2 = basic.relAdd()
        
        self.norm3 = basic.relLayerNorm(d_model)
        
    def get_attn_cam(self):
        return self.cross_attn.get_attn_cam()
    
    def get_attn_gradients(self):
        return self.cross_attn.get_attn_gradients()
    
    
    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None):
        t1, t2 = self.tgt_clone1(tgt, 2) 
        m1, m2 = self.mem_clone1(memory, 2)
        
        h = self.cross_attn(t1, m1, m2, memory_mask, memory_key_padding_mask)
        h = self.dropout2(h)
        
        x = self.add1((t2, h))
        x = self.norm2(x)
        
        x1, x2 = self.tgt_clone2(x, 2)
        h = self.linear1(x1)
        h = self.activate_fn(h)
        h = self.dropout(h)
        h = self.linear2(h)
        h = self.dropout3(h)
        
        x = self.add2((x2, h))
        x = self.norm3(x)
        return x 
        
    def rel_prop(self, R, **kwargs):
        R = self.norm3.rel_prop(R, **kwargs)
        x2R, hR = self.add2.rel_prop(R, **kwargs)

        hR = self.dropout3.rel_prop(hR, **kwargs)
    
        hR = self.linear2.rel_prop(hR, **kwargs)
        hR = self.dropout.rel_prop(hR, **kwargs)
        hR = self.activate_fn.rel_prop(hR, **kwargs)
        x1R = self.linear1.rel_prop(hR, **kwargs)
        
        xR = self.tgt_clone2.rel_prop((x1R, x2R), **kwargs)
        xR = self.norm2.rel_prop(xR, **kwargs)
        t2R, hR = self.add1.rel_prop(xR, **kwargs)
        
        hR = self.dropout2.rel_prop(hR, **kwargs)
        t1R, m1R, m2R = self.cross_attn.rel_prop(hR, **kwargs)
        
        mR = self.mem_clone1.rel_prop((m1R, m2R), **kwargs)
        tR = self.tgt_clone1.rel_prop((t1R, t2R), **kwargs)
        return tR, mR
        
class relTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super(relTransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.mem_clone = basic.relClone()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        
    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None):
        memory_clones = self.mem_clone(memory, self.num_layers)
        for i, mod in enumerate(self.layers):
            tgt = mod(tgt, memory_clones[i], memory_mask, memory_key_padding_mask)
        return tgt 
    
    
    def rel_prop(self, R, **kwargs):
        memory_clones = []
        for mod in reversed(self.layers):
            R, mem = mod.rel_prop(R, **kwargs)
            memory_clones.append(mem)
        mR = self.mem_clone.rel_prop(reversed(memory_clones))
        return R, mR
    
class relMultiheadLinear(basic.RelProp):
   
    def __init__(self, in_dim, out_dim, nheads, mid_dim=256, dropout=0.1):
        super(relMultiheadLinear, self).__init__()
        self.dropout_layer = basic.relDropout(dropout)
        
        self.nheads = nheads
        dim = in_dim // nheads

        self.fc = basic.relSequential(basic.relLinear(dim, mid_dim),
                                         basic.relReLU(),
                                         basic.relLinear(mid_dim, out_dim))
        
    def forward(self, x): #N x L x D
        batch_size, L, _ = x.shape 
        x = x.view(batch_size, L, self.nheads, -1) #N x L x nhead x d
        x = self.dropout_layer(x)
            
        x = self.fc(x) #N x L x nheads x d
        x = torch.flatten(x, start_dim=2) #N x L x nheads*d
        return x 
    
    
    def rel_prop(self, R, **kwargs):
        batch_size, L, _ = R.shape
        R = R.view(batch_size, L, self.nheads, -1) #N x L x nheads x d 
        R = self.fc.rel_prop(R, **kwargs)
        R = self.dropout_layer.rel_prop(R, **kwargs)
        R = R.view(batch_size, L, -1)
        return R 
        
        
class relMSDiffModel(nn.Module):
    def __init__(self, nnominals, ndefects, hidden_dim=128, dropout=0.1):    
        super(relMSDiffModel, self).__init__()
        
        self.ndefects = ndefects
        self.defect_dim = 32
        self.defect_emb = nn.Embedding(ndefects, embedding_dim=self.defect_dim, padding_idx=0, max_norm=1.0)
        
        nominal_dim = 64
        step = 3
        nheads_1 = nnominals // step 
        self.nominal_emb_1 = relMultiheadLinear(self.defect_dim*nnominals, nominal_dim, nheads_1, mid_dim=128, dropout=0.1)
        step = 5
        nheads_2 = nheads_1 // step 
        self.nominal_emb_2 = relMultiheadLinear(nominal_dim*nheads_1, nominal_dim, nheads_2, mid_dim=128, dropout=0.1)
        
        self.hidden_dim = hidden_dim
        self.peak_emb = basic.relSequential(basic.relDropout(0.35),
                                      basic.relLinear(nominal_dim*nheads_2, hidden_dim))
        
        encoder_layer = relTransformerEncoderLayer(d_model=hidden_dim, 
                                                   nheads = 4, 
                                                   dim_feedforward=256,
                                                   dropout = dropout)
        self.transformer_encoder = relTransformerEncoder(encoder_layer, num_layers=2)
        self.pooling = basic.relIndexSelect()
                
    def forward(self, x, mask):
        x = self.defect_emb(x)
        x = torch.flatten(x, start_dim=2) #N x L x m*d
        
        x = self.nominal_emb_1(x)
        x = self.nominal_emb_2(x)
        x = self.peak_emb(x) #N x L x D
    
        h = self.transformer_encoder(x, src_key_padding_mask=mask) # N x L x D
        h = self.pooling(h, 1, torch.tensor(0, device=h.device))
        return h 
    
    def rel_prop(self, R, **kwargs):
        R = self.pooling.rel_prop(R, **kwargs)
    
        xR = self.transformer_encoder.rel_prop(R, **kwargs)
        
        xR = self.peak_emb.rel_prop(xR, **kwargs)
       
        xR = self.nominal_emb_2.rel_prop(xR, **kwargs)
        xR = self.nominal_emb_1.rel_prop(xR, **kwargs)

        bsz, L, _ = xR.shape 
        xR = xR.view(bsz, L, -1, self.defect_dim)
        
        return xR 
        
                   
class relMSSimilarityModel(nn.Module):
    def __init__(self, nnominals, defect_dim, hidden_dim=128, nclasses = 1, dropout=0.1):    
        super(relMSSimilarityModel, self).__init__()
        
        self.mz_encoder = relMSDiffModel(nnominals, defect_dim, hidden_dim, dropout)
        self.loss_encoder = relMSDiffModel(nnominals, defect_dim, hidden_dim, dropout)
        
        self.fc = basic.relSequential(basic.relLinear(2*hidden_dim, 128),
                                basic.relReLU(),
                                basic.relDropout(0.1),
                                basic.relLinear(128, 64),
                                basic.relLinear(64, nclasses))
        
        self.cat = basic.relCat()
        
    def forward(self, mz_f, mz_mask, loss_f, loss_mask):
        
        mz_x = self.mz_encoder(mz_f, mz_mask)
        loss_x = self.loss_encoder(loss_f, loss_mask)
        
        h = self.cat((mz_x, loss_x), dim=-1)
        h = self.fc(h)
        return h 
    
    def rel_prop(self, R, **kwargs):
      
        R = self.fc.rel_prop(R, **kwargs)
     
        mzR, lossR = self.cat.rel_prop(R, **kwargs)
        
        lossR = self.loss_encoder.rel_prop(lossR, **kwargs)
        mzR = self.mz_encoder.rel_prop(mzR, **kwargs)
      
        return mzR, lossR
        
    
    
    