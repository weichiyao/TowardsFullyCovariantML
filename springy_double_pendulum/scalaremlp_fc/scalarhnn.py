import jax.numpy as jnp
import objax.nn as nn
import objax.functional as F
import numpy as np  
from objax.module import Module 
from scalaremlp_fc.utils import Named,export

def Sequential(*args):
    """ Wrapped to mimic pytorch syntax"""
    return nn.Sequential(args)

def comp_inner_products_jax(x, take_sqrt=True):
    """
    INPUT: batch (q1, q2, p1, p2)
    N: number of datasets
    dim: dimension  
    x: numpy tensor of size [N, 4, dim] 
    """ 
    n = x.shape[0]
    scalars = jnp.einsum('bix,bjx->bij', x, x).reshape(n, -1) # (n, 16)
    if take_sqrt:
        xxsqrt = jnp.sqrt(jnp.einsum('bix,bix->bi', x, x)) # (n, 4)
        scalars = jnp.concatenate([xxsqrt, scalars], axis = -1)  # (n, 20)
    return scalars 

class BasicMLP_objax(Module):
    def __init__(
        self, 
        n_in, 
        n_out,
        n_hidden=100, 
        n_layers=2, 
    ):
        super().__init__()
        layers = [nn.Linear(n_in, n_hidden), F.relu]
        for _ in range(n_layers):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(F.relu)
        layers.append(nn.Linear(n_hidden, n_out))
        
        self.mlp = Sequential(*layers)
    
    def __call__(self,x,training=True):
        return self.mlp(x)

@export
class InvarianceLayer_objax(Module):
    def __init__(
        self,  
        n_hidden, 
        n_layers, 
        g_choice:str="default",
    ):  
        """
        g_choice: 'default', 'learn', 'ignore'
        """
        super().__init__() 
        n_in = 30
        self.g_choice = g_choice
        if self.g_choice=="learn":
            # set g to be learning variables
            self.g = nn.Linear(3, 1, use_bias=False)    
        elif self.g_choice=="default":
            self.g = jnp.array([0,0,-1])
        elif self.g_choice=="ignore":
            self.g = None 
            n_in = 26
        else:
            raise ValueError(f"Received g_choice={g_choice}. Only accepted 'default', 'learn' or 'ignore'.")
        
        self.mlp = BasicMLP_objax(
            n_in=n_in, n_out=1, n_hidden=n_hidden, n_layers=n_layers
        )

    def compute_scalars_jax(self, x):
        """Input x of dim [n, 4, 3]"""       
        xx = comp_inner_products_jax(x)  # (n,20) 
        if self.g_choice=="learn":
            xg = self.g(x).squeeze(axis=-1) # (n,4)
        elif self.g_choice=="default":
            xg = jnp.inner(self.g, x) # (n,4)
        else:
            xg = jnp.array([])

        y  = x[:,0,:] - x[:,1,:] # x1-x2 (n,3)
        yy = jnp.sum(y*y, axis = -1, keepdims=True) # <x1-x2, x1-x2> | (n,1)   
        yx = jnp.einsum('bx,bjx->bj', y, x) # <q1-q2, u>, u=q1-q0, q2-q0, p1, p2 | (n, 4)
        scalars = jnp.concatenate(
            [xx, xg, yx, yy, jnp.sqrt(yy)], 
            axis=-1
        ) # (n,30)
        return scalars  
    
    def H(self, x):  
        scalars = self.compute_scalars_jax(x)
        out = self.mlp(scalars)
        return out.sum()  
    
    def __call__(self, x:jnp.ndarray):
        x = x.reshape(-1,4,3) # (n,4,3)
        return self.H(x)
