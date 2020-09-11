import sympy
import copy


class GRTensor:
    """ A Tensor object that is used similarly to a data type.

    Args:
        indices (list): a list of sympy symbols to represent the indices of the tensor.
            The number of indices must match the rank of the tensor.
        tensor (sympy Array): an N-dimensional array containing the tensor values.

    Attributes:
        ind (list): a copy of the indices argument, containing the indices of the tensor.
        vals (sympy Array): an N-dimensional array containing the tensor values.
        rank (int): the rank of the tensor (number of indices)

    """

    def __init__(self,indices,tensor):
        """ class initialization

        Args: 
            indices (dict): keys are the tensor symbol, the values are upper or lower 
        """
        self.indices=indices.copy()
        self.vals=copy.deepcopy(tensor)
        self.rank=len(indices.keys())
        return

    def __repr__(self):
        return repr(self.vals)


######################################################################

class GRMetric:
    """ A spacetime metric data type to handle calculations and metric properties.

    Args:
        coords (list): a list of sympy symbols representing the coordinates of the system. This determines the size of the metric.
        metric (sympy Array): a 2D array containing the metric values.

    Attributes:
        coords (list): list of the coordinates involved.
        dims (int): size of the metric (dims x dims).
        lowered (sympy Array): values of the lowered metric
        raised (sympy Array): values of the raised metric

    """
    def __init__(self,coords, metric, state):
        self.coords = coords[:]
        self.dims = len(coords)
        self.metric = sympy.Matrix(metric)
        self.state = state
        return

    def __repr__(self):
        return repr(self.metric)

    def change_state(self,newstate):
        if self.state==newstate:
            return
        self.state = newstate
        self.metric = self.metric.inv().copy()
        return

