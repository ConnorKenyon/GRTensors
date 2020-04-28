import sympy
import copy

import .objects


def TensorDerivative(index, coords, target):
    return GRTensor(index, sympy.derive_by_array(target,coords))

def CovariantDerivative(index, coords, metric, target):
    dims = len(coords)
    rank = target.get_rank() + 1
    d_A = sympy.derive_by_array(target.vals(),coords)
    ch_A = sympy.tensorcontraction(sympy.tensorproduct(ch[:,a,b],target.vals()),(0,1))
    return
