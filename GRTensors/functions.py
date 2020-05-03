import sympy
import copy
from GRTensors import GRTensor


def TensorDerivative(index, coords, target):
    return GRTensor([index], sympy.derive_by_array(target,coords))

def CovariantDerivative(index, coords, metric, target,targettype='upper'):
    if type(target) is GRTensor:
        dims = len(coords)
        rank = target.rank + 1
        d_A = sympy.derive_by_array(target.vals,coords)
        ch_A = metric.connection().vals
        t2 = sympy.tensorcontraction(sympy.tensorproduct(ch_A,target.vals),(2,3))
        cov_A = 0
        if targettype=='upper':
            cov_A += d_A + t2
        elif targettype=='lower':
            cov_A += d_A - t2
        tmp = target.ind + [index]
        return GRTensor(tmp, cov_A)
    else:
        return TensorDerivative(index,coords,target)
