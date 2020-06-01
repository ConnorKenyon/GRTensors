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
        assert type(indices)==list
        assert len(indices)==len(tensor.shape)
        self.ind=indices.copy()
        self.vals=copy.deepcopy(tensor)
        self.rank=len(indices)
        return

    def lower_index(self,ind,metric):
        """Lower an index by contracting the tensor with the lowered spacetime metric.

        Args:
            ind (sympy symbol): The symbolic index to be lowered.
            metric (GRMetric): The spacetime metric of the system.

        """
        sigma, alpha, mu, nu = sympy.symbols(r'\sigma \alpha \mu \nu')
        tmp = sympy.tensorproduct(metric.lowered,self.vals)
        # self.ind[self.ind.index(ind)] *= -1
        tmp2 = sympy.tensorcontraction(tmp,(0,self.ind.index(ind)+2))
        self.vals = copy.deepcopy(tmp2)
        return 

    def raise_index(self,ind,metric):
        """Raise an index by contracting the tensor with the raised spacetime metric.

        Args:
            ind (sympy symbol): The symbolic index to be raised.
            metric (GRMetric): The spacetime metric of the system.

        """
        sigma, alpha, mu, nu = sympy.symbols(r'\sigma \alpha \mu \nu')
        tmp = sympy.tensorproduct(metric.raised,self.vals)
        # self.ind[self.ind.index(ind)] *= -1
        tmp2 = sympy.tensorcontraction(tmp,(0,self.ind.index(ind)+2))
        self.vals = copy.deepcopy(tmp2)
        return 


######################################################################

class GRMetric:
    """ A spacetime metric data type to handle calculations and metric properties.

    Args:
        coords (list): a list of sympy symbols representing the coordinates of the system. This determines the size of the metric.
        metric (sympy Array): a 2D array containing the metric values.
        preset (string): an optional argument to generate a common metric.

    Attributes:
        coords (list): list of the coordinates involved.
        dims (int): size of the metric (dims x dims).
        lowered (sympy Array): values of the lowered metric
        raised (sympy Array): values of the raised metric
        Christoffel_symbols (GRTensor): Christoffel symbols of the metric
        Riemann_tensor (GRTensor): Riemann curvature tensor of the metric
        Ricci_tensor (GRTensor): Ricci tensor of the metric
        Ricci_scalar (sympy expression): Ricci scalar of the metric

    """
    def __init__(self,coords, metric=None, preset=None):
        if metric and not preset:
            self.lowered=metric.copy()
        elif not metric and preset:
            pass
        else:
            raise ValueError()

        self.coords = coords[:]
        self.dims = len(coords)
        self.lowered = metric.copy()
        self.raised = metric.copy().inv()
        self.Christoffel_symbols = self.connection()
        self.Riemann_tensor = self.curvature()
        self.Ricci_tensor = self.ricci_tensor()
        self.Ricci_scalar = self.ricci_scalar()
        return


    def connection(self):
        """ Calculate the Christoffel Symbols (Metric Connection) of a metric

        """
        ch = sympy.Array([0 for i in range(self.dims**3)])
        ch = ch.reshape(self.dims,self.dims,self.dims).as_mutable()
        alpha, mu, nu = sympy.symbols(r'\alpha, \mu, \nu')
        for a_ in range(self.dims):
            for m_ in range(self.dims):
                for n_ in range(self.dims):
                    for lam in range(self.dims):
                        ch[a_, m_, n_] += .5*self.raised[a_, lam]*(\
                        sympy.diff(self.lowered[m_, lam], self.coords[n_]) +\
                        sympy.diff(self.lowered[n_, lam], self.coords[m_]) -\
                        sympy.diff(self.lowered[m_, n_], self.coords[lam])  )
        return GRTensor([alpha, mu, nu],ch)

    def curvature(self):
        """ Calculate the Riemann curvature tensor of the metric

        """
        sigma, alpha, mu, nu = sympy.symbols(r'\sigma \alpha \mu \nu')
        ch = self.connection().vals
        R = sympy.Array([0 for i in range(self.dims**4)])
        R = R.reshape(self.dims,self.dims,self.dims,self.dims).as_mutable()
        for a_ in range(self.dims):
            for b_ in range(self.dims):
                for c_ in range(self.dims):
                    for d_ in range(self.dims):
                        R[a_,b_,c_,d_] += sympy.diff(ch[a_,b_,d_],self.coords[c_])
                        R[a_,b_,c_,d_] -= sympy.diff(ch[a_,b_,c_],self.coords[d_])
                        for e_ in range(self.dims):
                            R[a_,b_,c_,d_] += ch[e_,b_,d_]*ch[a_,e_,c_]
                            R[a_,b_,c_,d_] -= ch[e_,b_,c_]*ch[a_,e_,d_]
        return GRTensor([sigma, alpha, mu, nu],R)

    def geodesics(self):
        """ Calculate the Geodesic equations of the spacetime metric using the formula d^2x^a/dt^2 + gamma^a_bc dx^b/dt dx^c/dt = 0
        """
        ch = self.connection().vals

        t = sympy.Symbol('t')
        coord_funcs = [sympy.Function(str(f))(t) for f in self.coords]
        eqns = []

        for a_ in range(self.dims):
            for b_ in range(self.dims):
                for c_ in range(self.dims):
                    tmp1 = sympy.diff(sympy.diff(coord_funcs[a_],t),t)
                    tmp2 = ch[a_,b_,c_]*sympy.diff(coord_funcs[b_],t)*sympy.diff(coord_funcs[c_],t)
                    eqns.append(tmp1+tmp2)

        return sympy.Array(eqns).reshape(self.dims,self.dims,self.dims)
    


    def ricci_tensor(self):
        """ Calculate the Ricci curvature tensor of the metric

        """
        R = self.curvature()
        R2 = sympy.tensorcontraction(R.vals,(0,2))
        R.vals = R2
        return R

    def ricci_scalar(self):
        """ Calculate the Ricci scalar of the metric

        """
        R2 = self.ricci_tensor()
        R2_ = R2.vals
        R_ = sympy.tensorproduct(self.raised,R2_)
        R_1 = sympy.tensorcontraction(R_,(1,3))
        R_2 = sympy.tensorcontraction(R_1,(0,1))
        return R_2

