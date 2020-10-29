import sympy
import copy


#--------------------------------------------------------
class Tensor():
    def __init__(self, indices, values):
        self.indices = indices[:]
        self.rank = len(indices)
        self.vals = sympy.Array(copy.deepcopy(values))
        return
    
    def __repr__(self):
        return repr(self.vals)
    
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
        
#--------------------------------------------------------