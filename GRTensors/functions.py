import sympy
import copy
import itertools
from . import objects


def make_index(ind_string):
    """Wrapper for sympy.symobls for a real and positive tensor index"""
    return sympy.symbols(ind_string, real=True, positive=True)

def make_coords(coords_string,dependent_coord=None):
    """Wrapper for sympy.symbols for a real coordinate"""
    if dependent_coord:
        dependent_symbol = sympy.Symbol(dependent_coord,real=True)
        coords_list = sympy.symbols(coords_string,real=True)
        coord_funcs = [sympy.Function(coord)(dependent_symbol) for coord in coords_list]
        return [dependent_symbol] + coord_funcs
    else:
        return sympy.symbols(coords_string,real=True)
    return sympy.symbols(coords_string, real=True)

    
def tensor_product(T1, T2, contraction=None): 
    """Take the tensor product of two tensors.
    
    Arguments:
    T1 (Tensor) --  The first tensor to be multiplied
    T2 (Tensor) --  The second tensor to be multiplied
    
    Keyword Arguments:
    contraction (tuple) -- Indices to contract, maximum length of 2 (default None)
    
    Return Type -- Tensor
    """
    new_ind = T1.indices[:] + T2.indices[:]
    new_vals = sympy.tensorproduct(copy.deepcopy(T1.vals),copy.deepcopy(T2.vals))
    Tf = objects.Tensor(new_ind,new_vals)
    if contraction:
        Tf = tensor_contract(Tf,contraction[0],contraction[1])
    return Tf

def tensor_contract(T,ind1,ind2):
    """Contract two indices of a tensor.
    
    Arguments:
    T (Tensor) -- Tensor to contract
    ind1 (Symbol) -- First index to contract.
    ind2 (Symbol) -- Second index to contract.
    
    Return Type -- Tensor
    """
    if ind1*ind2 > 0:
        raise AttributeError("Index 1 and 2 cannot be in the same state")
        
    if ind1 not in T.indices or ind2 not in T.indices:
        raise AttributeError(f"Index {ind1} or {ind2} not found in tensor T")
        
    i1 = T.indices.index(ind1)
    i2 = T.indices.index(ind2)
    
    new_indices = T.indices
    new_indices.remove(ind1)
    new_indices.remove(ind2)
    new_vals = sympy.tensorcontraction(T.vals,(i1,i2))
    return objects.Tensor(new_indices,new_vals)

def christoffel_from_metric(metric,indices=None):
    """Compute Christoffel Symbols for a given metric.
    
    Arguments:
    metric (Metric) -- The metric to be used for calculations
    
    Return Type -- Tensor
    """
    gmn = metric.vals_raised()
    g_mn = metric.vals_lowered()
    dims = metric.dims
    coords = metric.coords

    tmp = []
    for i in range(dims):
        for k in range(dims):
            for l in range(dims):
                v1 = sum([0.5 * gmn[i,m] * sympy.diff(g_mn[m,k],coords[l]) for m in range(dims)])
                v2 = sum([0.5 * gmn[i,m] * sympy.diff(g_mn[m,l],coords[k]) for m in range(dims)])
                v3 = sum([0.5 * gmn[i,m] * sympy.diff(g_mn[k,l],coords[m]) for m in range(dims)])
                tmp.append(v1 + v2 - v3)
    ch = sympy.Array(tmp,(dims,dims,dims))
    a,b,c = sympy.symbols("a b c",real=True,positive=True)
    T_ch = objects.Tensor([a,-b,-c],ch)
    if indices:
        T_ch = T_ch.reset_indices(indices)
    return T_ch
    
    
def cov_diff(target,metric,index):
    """Take the covariant derivative of a tensor.
    
    Arguments:
    target (Tensor) -- The tensor to be differentiated.
    metric (Metric) -- The spacetime metric to differentiate with respect to
    index (Symbol) -- The index to add to the indices
    
    Return Type -- Tensor
    """
    new_indices = target.indices[:] + [index]
    a,b,c,d =  make_index("a b c d")
    coords = metric.coords[:]
    ch = christoffel_from_metric(metric)
    tmp = diff(target,coords,index)
    T_out = objects.Tensor(new_indices,tmp.vals)
    for i in range(target.rank):
        if target.indices[i] > 0:
            tmp2 = tensor_product(ch,target)
            tmp2 = tensor_contract(tmp2,ch.indices[2],target.indices[i])
            T_out.vals = T_out.vals + tmp2.vals
        if target.indices[i] < 0:
            tmp2 = tensor_product(ch,target)
            tmp2 = tensor_contract(tmp2,ch.indices[0],target.indices[i])
            T_out.vals = T_out.vals - tmp2.vals
    return T_out


def riemann_tensor_from_metric(metric):
    """ Calculate the Riemann curvature tensor of a metric
    """
    sigma, alpha, mu, nu = sympy.symbols(r'\sigma \alpha \mu \nu')
    ch = christoffel_from_metric(metric).vals
    R = sympy.Array([0 for i in range(metric.dims**4)])
    R = R.reshape(metric.dims,metric.dims,metric.dims,metric.dims).as_mutable()
    for a_ in range(metric.dims):
        for b_ in range(metric.dims):
            for c_ in range(metric.dims):
                for d_ in range(metric.dims):
                    R[a_,b_,c_,d_] += sympy.diff(ch[a_,b_,d_],metric.coords[c_])
                    R[a_,b_,c_,d_] -= sympy.diff(ch[a_,b_,c_],metric.coords[d_])
                    for e_ in range(metric.dims):
                        R[a_,b_,c_,d_] += ch[e_,b_,d_]*ch[a_,e_,c_]
                        R[a_,b_,c_,d_] -= ch[e_,b_,c_]*ch[a_,e_,d_]
    return objects.Tensor([sigma,-alpha,-mu,-nu],R)


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

    
# def add(T1,T2):
#     if type(T1) != Metric and type(T1) != Tensor:
#         raise AttributeError("T1 is not a valid type")
#     if type(T2) != Metric and type(T2) != Tensor:
#         raise AttributeError("T2 is not a valid type")
#     if T1.rank==1 and T2.rank==1:
#         for i in T1.dims:
#             T1.vals[i] = 
#     return objects.Tensor(T1.indices[:],T1.vals+T2.vals)
        
    
# def subtract(T1,T2):
#     if type(T1) != Metric and type(T1) != Tensor:
#         raise AttributeError("T1 is not a valid type")
#     if type(T2) != Metric and type(T2) != Tensor:
#         raise AttributeError("T2 is not a valid type")
#     
#     return objects.Tensor(T1.indices[:],T1.vals-T2.vals)
