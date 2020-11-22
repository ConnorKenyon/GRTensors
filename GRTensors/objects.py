import sympy
import numpy as np
import copy
import itertools

from . import functions

#---------------------------------------------------------
class Tensor():
    def __init__(self, indices, values):
        self.indices = indices[:]
        self.rank = len(indices)
        self.dims = int(len(values)**(1/self.rank))
        self.vals = sympy.Array(copy.deepcopy(values))
        return
    
    def __repr__(self):
        return repr(self.vals)
    
    def __add__(self,other):
        indmatch = [ind in other.indices for ind in self.indices]
        compatible = False not in indmatch
        
        final_shape = self.vals.shape
        
        value_count = len(sympy.flatten(self.vals))
        
        vals_self = sympy.flatten(self.vals)
        vals_other = sympy.flatten(other.vals)
        vals_final = vals_self[:]
        
        inds_other = [other.indices.index(ind) for ind in self.indices]
        for v_index in range(value_count):
            enum_other = [v_index//self.dims**((self.rank-1)-i)%self.dims for i in inds_other]
            i_other = sum([enum_other[i]*self.dims**(self.rank-1 - i) for i in range(self.rank)])
            vals_final[v_index] += vals_other[i_other]
        
        new_vals = sympy.Array(vals_final,shape=final_shape)
        return Tensor(self.indices,new_vals)
    
    def __sub__(self,other):
        indmatch = [ind in other.indices for ind in self.indices]
        compatible = False not in indmatch
        
        final_shape = self.vals.shape
        
        value_count = len(sympy.flatten(self.vals))
        
        vals_self = sympy.flatten(self.vals)
        vals_other = sympy.flatten(other.vals)
        vals_final = vals_self[:]
        
        inds_other = [other.indices.index(ind) for ind in self.indices]
        for v_index in range(value_count):
            enum_other = [v_index//self.dims**((self.rank-1)-i)%self.dims for i in inds_other]
            i_other = sum([enum_other[i]*self.dims**(self.rank-1 - i) for i in range(self.rank)])
            vals_final[v_index] -= vals_other[i_other]
        
        new_vals = sympy.Array(vals_final,shape=final_shape)
        return Tensor(self.indices,new_vals)
    
    def __mul__(self,other):
        if type(other)==Metric or type(other)==Tensor:
            return functions.tensor_product(self,other)
        elif type(other)==float or type(other)==int:
            return Tensor(self.indices,float(other)*self.vals)
    
    def __rmul__(self,other):
        if type(other)==Metric or type(other)==Tensor:
            return functions.tensor_product(self,other)
        elif type(other)==float or type(other)==int:
            return Tensor(self.indices,float(other)*self.vals)
    
    def copy(self):
        return Tensor(self.indices[:],copy.deepcopy(self.vals))
    
    def change_index(self,old_index,new_index):
        if old_index*new_index < 0:
            raise ValueError("Index types do not match")

        if old_index in self.indices:
            return Tensor([new_index if i==old_index else i for i in self.indices],self.vals)
        else:
            return self
    
    def swap_indices(self,ind1,ind2):
        if ind1 not in self.indices or ind2 not in self.indices: 
            raise AttributeError("Index not found in tensor indices")
            
        a1 = self.indices.index(ind1)
        a2 = self.indices.index(ind2)
        self.vals = sympy.Array(np.array(self.vals.tolist()).swapaxes(a1,a2))
        return Tensor(self.indices[:],copy.deepcopy(self.vals))
    
    def reset_indices(self,new_indices):
        if len(new_indices) != len(self.indices):
            raise ValueError
        return Tensor(new_indices,self.vals)
        
        
    def autocontract(self):
        
        tmp = self.copy()
        while [ind for ind in tmp.indices if -1*ind in tmp.indices] != []:
            for ind in self.indices:
                if -1*ind in self.indices:
                    return functions.tensor_contract(tmp,ind,-1*ind) 
#--------------------------------------------------------

#--------------------------------------------------------
class Metric(Tensor):
    def __init__(self,coords, indices, values):
        super().__init__(indices,values)
        self.coords = coords[:]
        self.dims = len(coords)
        self.vals = sympy.Matrix(self.vals)
        return
    
    def __repr__(self):
        return repr(self.vals)
    
    
    def __add__(self,other):
        new_tensor = super().__add__(other)
        return Metric(self.coords,new_tensor.indices,new_tensor.vals)
    
    def __sub__(self,other):
        new_tensor = super().__sub__(self,other)
        return Metric(self.coords,new_tensor.indices,new_tensor.vals)
    
    
    def get_state(self):
        i1 = self.indices[0].is_positive
        i2 = self.indices[1].is_positive
        state=""
        if i1==True and i2==True:
            state="++"
        elif i1==True and i2==False:
            state="+-"
        elif i1==False and i2==True:
            state="-+"
        elif i1==False and i2==False:
            state="--"
        return state
    
    def vals_lowered(self):
        state = self.get_state()
        metricvals = self.vals
        if state == '--':
            return metricvals
        elif state == '++':
            return metricvals.inv()
        else:
            return None
        
    def vals_raised(self):
        state = self.get_state()
        metricvals = self.vals
        if state == '++':
            return metricvals
        elif state == '--':
            return metricvals.inv()
        else:
            return None
        
    def raise_metric(self):
        if self.get_state() == '--':
            self.vals = self.vals.inv()
            self.indices = [-1*i for i in self.indices]
        else:
            pass
        return
    
    def lower_metric(self):
        if self.get_state() == '++':
            self.vals = self.vals.inv()
            self.indices = [-1*i for i in self.indices]
        else:
            pass
        return        
    
    def reset_indices(self,new_indices):
        if len(new_indices) != len(self.indices):
            raise ValueError
        return Metric(self.coords,new_indices,self.vals)
        
#--------------------------------------------------------
