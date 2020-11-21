import sympy
import copy
import itertools
from . import objects
# from GRTensors import Tensor
# from GRTensors import Metric


def make_index(ind_string):
    return sympy.symbols(ind_string, real=True, positive=True)

def make_coords(coords_string,dependent_coord=None):
    if dependent_coord:
        dependent_symbol = sympy.Symbol(dependent_coord,real=True)
        coords_list = sympy.symbols(coords_string,real=True)
        coord_funcs = [sympy.Function(coord)(dependent_symbol) for coord in coords_list]
        return [dependent_symbol] + coord_funcs
    else:
        return sympy.symbols(coords_string,real=True)
    
        
    return sympy.symbols(coords_string, real=True)

    
def tensor_product(T1, T2, contraction=None):
    new_ind = T1.indices[:] + T2.indices[:]
    new_vals = sympy.tensorproduct(copy.deepcopy(T1.vals),copy.deepcopy(T2.vals))
    Tf = objects.Tensor(new_ind,new_vals)
    if contraction:
        return tensor_contract(Tf,contraction[0],contraction[1])
    else:
        return Tf
        
    return objects.Tensor(new_ind,new_vals)

def tensor_contract(T,ind1,ind2,autocontract=False):
    if ind1*ind2 > 0:
        raise AttributeError("Index 1 and 2 cannot be in the same state")
        
        
    i1 = T.indices.index(ind1)
    i2 = T.indices.index(ind2)
    
    new_indices = T.indices
    new_indices.remove(ind1)
    new_indices.remove(ind2)
    new_vals = sympy.tensorcontraction(T.vals,(i1,i2))
    return objects.Tensor(new_indices,new_vals)

def ChristoffelFromMetric(metric):
    metric_upper = metric.vals_raised()
    metric_lower = metric.vals_lowered()

    tmp = []
    for i in range(metric.dims):
        for k in range(metric.dims):
            for l in range(metric.dims):
                term1 = sum([1/2. * metric_upper[i,m] * sympy.diff(metric_lower[m,k],metric.coords[l]) for m in range(metric.dims)])
                term2 = sum([1/2. * metric_upper[i,m] * sympy.diff(metric_lower[m,l],metric.coords[k]) for m in range(metric.dims)])
                term3 = sum([1/2. * metric_upper[i,m] * sympy.diff(metric_lower[k,l],metric.coords[m]) for m in range(metric.dims)])
                tmp.append(term1 + term2 - term3)
    ch = sympy.Array(tmp,(metric.dims,metric.dims,metric.dims))
    a,b,c = sympy.symbols("a b c",real=True,positive=True)
    
    return GRobjects.Tensor([a,-b,-c],ch)


def split_list(inds,sep):
    if sep in inds:
        inds1 = inds[:inds.index(sep)]
        inds2 = inds[inds.index(sep)+1:]
    else:
        inds1 = inds[:]
        inds2 = []
    return inds1,inds2
        
        
# def diff(target,metric,index):
#     if type(target) == Tensor or type(target) == Metric:
#         new_indices = target.indices[:] + [index]
#         coords = metric.coords[:] 
#         new_vals = sympy.derive_by_array(target.vals,coords)
#         return objects.Tensor(new_indices,new_vals)
#     else:
#         new_indices = [index]
#         coords = metric.coords[:]
#         new_vals = sympy.derive_by_array(target,coords)
#         return objects.Tensor(new_indices,new_vals)

def diff(target,coords,new_index):
    new_vals = sympy.derive_by_array(target.vals,coords)
    new_inds = target.indices[:] + [new_index]
    return objects.Tensor(new_inds,new_vals)

    
def add(T1,T2):
    if type(T1) != Metric and type(T1) != Tensor:
        raise AttributeError("T1 is not a valid type")
    if type(T2) != Metric and type(T2) != Tensor:
        raise AttributeError("T2 is not a valid type")
    
    return objects.Tensor(T1.indices[:],T1.vals+T2.vals)
        
    
def subtract(T1,T2):
    if type(T1) != Metric and type(T1) != Tensor:
        raise AttributeError("T1 is not a valid type")
    if type(T2) != Metric and type(T2) != Tensor:
        raise AttributeError("T2 is not a valid type")
    
    return objects.Tensor(T1.indices[:],T1.vals-T2.vals)
    
    
def cov_diff(target,metric,index):
    new_indices = target.indices[:] + [index]
    coords = metric.coords[:]
    return target
        
            
#     return sympy.tensorcontraction(sympy.tensorproduct(tensor1.vals,tensor2.vals),i)

def Riemann4FromMetric(metric):
    return

def Ricci2FromMetric(metric):
    return

def RicciScalarFromMetric(metric):
    return
