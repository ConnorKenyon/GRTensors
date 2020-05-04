import sympy
import copy
from GRTensors import GRTensor


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

def CovariantDerivative(index, coords, metric, target,targettype='upper'):
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
