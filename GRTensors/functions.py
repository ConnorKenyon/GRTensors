import sympy
import copy
from GRTensors import GRTensor

def Index(index_string):
    return sympy.Symbol(index_string, real=True, positive=True)

def Indices(index_string):
    return sympy.symbols(index_string, real=True, positive=True)

def TensorDerivative(index, coords, target):
    """ Takes a comma derivative of a tensor
    
    Args:
        index (sympy.Symbol): index to add to the tensor
        coords (list(sympy.Symbol)): list of the coordinates to take derivatives with respect to
        target (sympy expression or GRTensor.vals): Target to be differentiated
        
    Returns:
        GRTensor
    """
    return GRTensor([index], sympy.derive_by_array(target,coords))

def CovariantDerivative_old(index, coords, metric, target,targettype='upper'):
    """ Takes the covariant (semicolon) derivative of a tensor.
    
    Args: 
        index (sympy.Symbol): index to add to the tensor
        coords (list(sympy.Symbol)): list of the coordinates to take derivatives with respect to
        metric (GRMetric): Metric to be differentiated with respect to (connection term)
        target (GRTensor): Target to be differentiated
        targettype (string): upper or lower, represents which index type the tensor is
        
    Returns:
        GRTensor of the covariant derivative
    """
    if type(target) is GRTensor:
        if targettype=='upper':
            dims = len(coords)
            rank = target.rank + 1
            d_A = sympy.derive_by_array(target.vals,coords)
            ch_A = metric.connection().vals
            t2 = sympy.tensorcontraction(sympy.tensorproduct(ch_A,target.vals),(2,3))
            cov_A = d_A + t2
            tmp = target.ind + [index]
            return GRTensor(tmp, cov_A)

        elif targettype=='lower':
            dims = len(coords)
            rank = target.rank + 1
            d_A = sympy.derive_by_array(target.vals,coords)
            ch_A = metric.connection().vals
            t2 = sympy.tensorcontraction(sympy.tensorproduct(ch_A,target.vals),(2,3))
            cov_A += d_A - t2
            tmp = target.ind + [index]
            return GRTensor(tmp, cov_A)
    else:
        return TensorDerivative(index,coords,target)

def CovDiff(index, metric, target):
    coords = metric.coords
    tmp = sympy.derive_by_array(target.vals,coords)
    return

def Product(tensor1,tensor2):
    i = tensor1.rank
    j = tensor2.rank
    dims = tensor1.metric.dims
    tmp = np.zeros()
            
        
            
#     return sympy.tensorcontraction(sympy.tensorproduct(tensor1.vals,tensor2.vals),i)

def ChristoffelFromMetric(metric):
    metric_upper = metric.get_metric_upper()
    metric_lower = metric.get_metric_lower()

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
    
    return GRTensor([a,-b,-c],ch,metric)

def Riemann4FromMetric(metric):
    return

def Ricci2FromMetric(metric):
    return

def RicciScalarFromMetric(metric):
    return
