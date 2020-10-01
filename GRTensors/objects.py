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

    def __init__(self,indices,tensor_vals,associated_metric):
        """ class initialization

        Args: 
            indices (dict): keys are the tensor symbol, the values are upper or lower 
        """
        self.indices=indices[:]
        self.rank=len(indices)
        self.vals=copy.deepcopy(tensor_vals)
        self.metric=associated_metric
        return

    def __repr__(self):
        return repr(self.vals)

    
    def raise_index(self,index):
        if -index in self.indices:
            self.indices[self.indices.index(-index)] = index
            
            #>> Do a tensor product then contraction with metric tensor



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

    def _change_state_(self,newstate):
        if self.state==newstate:
            return
        self.state = newstate
        self.metric = self.metric.inv().copy()
        return

    def get_metric_lower(self):
        if self.state=='lower':
            return self.metric.copy()
        elif self.state=='upper':
            return self.metric.inv().copy()
        else:
            raise ValueError("Metric is in an invalid state")
        
    def get_metric_upper(self):
        if self.state=='upper':
            return self.metric.copy()
        elif self.state=='lower':
            return self.metric.inv().copy()
        else:
            raise ValueError("Metric is in an invalid state")


# Make GRIndex class as wrapper for sympy.symbols with real=True and positive=True?
