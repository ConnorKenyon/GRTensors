import sympy
import numpy as np
import copy
import itertools

from . import functions

#---------------------------------------------------------
class Tensor():
    """ A class to represent a Tensor
    
    Attributes:
    indices (list) -- The list of indices for the tensor. Positivity represents upper vs lower indices.
    rank (int) -- The rank of the tensor.
    dims (int) -- The number of spacetime dimensions the tensor contains.
    vals (sympy.Array) -- The dims**rank size array of tensor values
    
    Methods:
    copy() -- Returns a copy of the tensor
    reset_indices(new_indices) -- Reset the indices while maintaining tensor 
        properties. this is intended to be used to prep the tensor for 
        calculations.
    """
    def __init__(self, indices, values):
        self.indices = indices[:]
        self.rank = len(indices)
        try:
            self.dims = int(len(values)**(1/self.rank))
        except:
            self.dims = 0
        if self.dims == 0:
            self.vals = values
        else:
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
        """Create a copy of the Tensor
        """
        return Tensor(self.indices[:],copy.deepcopy(self.vals))
    
    def reset_indices(self,new_indices):
        """ Safely reset the indices of the tensor.
        """
        if len(new_indices) != len(self.indices):
            raise ValueError
        if [ind > 0 for ind in self.indices] != [ind > 0 for ind in new_indices]:
            raise ValueError
        return Tensor(new_indices,self.vals)
        
#--------------------------------------------------------

#--------------------------------------------------------
class Metric(Tensor):
    """ A class to represent a Metric Tensor. Subclass of Tensor.
    
    Attributes:
    coords (list) -- The list of spacetime coordinates
    indices (list) -- The list of indices for the tensor. Positivity represents upper vs lower indices.
    rank (int) -- The rank of the tensor.
    dims (int) -- The number of spacetime dimensions the tensor contains.
    vals (sympy.Array) -- The dims**rank size array of tensor values
    
    Methods:
    vals_lowered() -- Returns the lowered values, regardless of state
    vals_raised() -- Returns the raised values, regardless of state
    copy() -- Returns a copy of the tensor
    reset_indices(new_indices) -- Reset the indices while maintaining tensor 
        properties. this is intended to be used to prep the tensor for 
        calculations.
    """
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
        """ Get the state of the metric (++, +-, -+, or --)
        """
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
        """ Get the lowered values of the tensor, 
            inverts values matrix if the metric is not already lowered.
        """
        state = self.get_state()
        metricvals = self.vals
        if state == '--':
            return metricvals
        elif state == '++':
            return metricvals.inv()
        else:
            return None
        
    def vals_raised(self):
        """ Get the raised values of the tensor, 
            inverts values matrix if the metric is not already raised.
        """
        state = self.get_state()
        metricvals = self.vals
        if state == '++':
            return metricvals
        elif state == '--':
            return metricvals.inv()
        else:
            return None
        
    def reset_indices(self,new_indices):
        """ Safely reset metric indices
        """
        if len(new_indices) != len(self.indices):
            raise ValueError
        if [ind > 0 for ind in self.indices] != [ind > 0 for ind in new_indices]:
            raise ValueError
        return Metric(self.coords,new_indices,self.vals)
        
#     def raise_metric(self):
#         if self.get_state() == '--':
#             self.vals = self.vals.inv()
#             self.indices = [-1*i for i in self.indices]
#         else:
#             pass
#         return
#     
#     def lower_metric(self):
#         if self.get_state() == '++':
#             self.vals = self.vals.inv()
#             self.indices = [-1*i for i in self.indices]
#         else:
#             pass
#         return        
    
#--------------------------------------------------------
