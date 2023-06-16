'''
Created on 16 Sep 2022

@author: danhbuithi

This implementation is referenced from the implementation of Transformer Interpretability Beyond Attention Visualization 
Github: https://github.com/hila-chefer/Transformer-Explainability.git

'''

import torch 
from torch import nn 
import torch.nn.functional as F

def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())


def forward_hook(self, inputs, outputs):
    if type(inputs[0]) in (list, tuple):
        self.X = []
        for i in inputs[0]:
            x = i.detach()
            x.requires_grad = True
            self.X.append(x)
    else:
        self.X = inputs[0].detach()
        self.X.requires_grad = True

    self.Y = outputs


def backward_hook(self, grad_input, grad_output):
    self.grad_input = grad_input
    self.grad_output = grad_output


class RelProp(nn.Module):
    def __init__(self):
        super(RelProp, self).__init__()
        
    def set_register_forward_hook(self):
        self.register_forward_hook(forward_hook)
        
    def grad_prop(self, Z, X, S):
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C

    def rel_prop(self, R, **kwargs):
        return R 
    
class SimpleRelProp(RelProp):
    def rel_prop(self, R, **kwargs):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.grad_prop(Z, self.X, S)
        
        if torch.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * C[0]
        return outputs
        
class relDropout(nn.Dropout, RelProp):
    pass

class relReLU(nn.ReLU, RelProp):
    pass

class relSoftmax(nn.Softmax, RelProp):
    pass 

class relLayerNorm(nn.LayerNorm, RelProp):
    pass 


class relMatmul(SimpleRelProp):
    def forward(self, inputs):
        return torch.matmul(*inputs)
    
    
class relAdd(SimpleRelProp):
    def forward(self, inputs):
        return torch.add(*inputs)
    
    def relprop(self, R, **kwargs):
        
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        a = self.X[0] * C[0]
        b = self.X[1] * C[1]

        a_sum = a.sum()
        b_sum = b.sum()

        a_fact = safe_divide(a_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()
        b_fact = safe_divide(b_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()

        a = a * safe_divide(a_fact, a.sum())
        b = b * safe_divide(b_fact, b.sum())

        outputs = [a, b]

        return outputs
    

class relLinear(nn.Linear, RelProp):
    
    def rel_prop(self, R, **kwargs):
        alpha = kwargs['alpha']
        beta = alpha - 1
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        def f(w1, w2, x1, x2):
            Z1 = F.linear(x1, w1)
            Z2 = F.linear(x2, w2)
            S1 = safe_divide(R, Z1 + Z2)
            S2 = safe_divide(R, Z1 + Z2)
            C1 = x1 * self.grad_prop(Z1, x1, S1)[0]
            C2 = x2 * self.grad_prop(Z2, x2, S2)[0]

            return C1 + C2

        activator_relevances = f(pw, nw, px, nx)
        inhibitor_relevances = f(nw, pw, px, nx)

        R = alpha * activator_relevances - beta * inhibitor_relevances

        return R
    
def embedding_relpop(self, X, R, ndefects, **kwargs):
    alpha = kwargs['alpha']
    beta = alpha - 1
    
    weights = self.weight.transpose(0, 1) 
    pw = torch.clamp(weights, min=0)
    nw = torch.clamp(weights, max=0)
    
    X = F.one_hot(X, ndefects).double()
    X.requires_grad_(True)
    
    px = torch.clamp(X, min=0)
    nx = torch.clamp(X, max=0)

    def f(w1, w2, x1, x2):
        Z1 = F.linear(x1, w1)
        Z2 = F.linear(x2, w2)
        S1 = safe_divide(R, Z1 + Z2)
        S2 = safe_divide(R, Z1 + Z2)
        
        C1 = x1 * torch.autograd.grad(Z1, x1, S1, retain_graph=True)[0]
        C2 = x2 * torch.autograd.grad(Z2, x2, S2, retain_graph=True)[0]
        #C1 = x1 * self.grad_prop(Z1, x1, S1)[0]
        #C2 = x2 * self.grad_prop(Z2, x2, S2)[0]

        return C1 + C2

    activator_relevances = f(pw, nw, px, nx)
    inhibitor_relevances = f(nw, pw, px, nx)

    R = alpha * activator_relevances - beta * inhibitor_relevances

    return R
    
class relSequential(nn.Sequential):
    
    def rel_prop(self, R, **kwargs):
        for m in reversed(self._modules.values()):
            R = m.rel_prop(R, **kwargs)
        return R
    
       
class relClone(RelProp):
    def forward(self, inputs, num):
        self.__setattr__('num', num)
        outputs = []
        for _ in range(num):
            outputs.append(inputs)

        return outputs

    def rel_prop(self, R, **kwargs):
        Z = []
        for _ in range(self.num):
            Z.append(self.X)
        S = [safe_divide(r, z) for r, z in zip(R, Z)]
        C = self.grad_prop(Z, self.X, S)[0]

        R = self.X * C

        return R


class relCat(RelProp):
    def forward(self, inputs, dim):
        self.__setattr__('dim', dim)
        return torch.cat(inputs, dim)

    def rel_prop(self, R, **kwargs):
        Z = self.forward(self.X, self.dim)
        S = safe_divide(R, Z)
        C = self.grad_prop(Z, self.X, S)

        outputs = []
        for x, c in zip(self.X, C):
            outputs.append(x * c)

        return outputs 
    
    
class relIndexSelect(RelProp):
    
    def forward(self, inputs, dim, indices):
        self.__setattr__('dim', dim)
        self.__setattr__('indices', indices)

        return torch.index_select(inputs, dim, indices)

    def relprop(self, R, **kwargs):
        Z = self.forward(self.X, self.dim, self.indices)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * (C[0])
        return outputs 
    