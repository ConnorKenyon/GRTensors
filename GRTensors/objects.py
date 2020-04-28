import sympy
import copy


class GRTensor:
    def __init__(self,indices,tensor):
        assert type(indices)==list
        # assert type(tensor)==sympy.Matrix
        self.ind=indices.copy()
        self.vals=copy.deepcopy(tensor)
        self.rank=len(indices)
        return

    def get_vals(self):
        return self.vals

    def get_indices(self):
        return self.ind

    def get_rank(self):
        return self.rank

    def lower_index(self,ind,metric):
        sigma, alpha, mu, nu = sympy.symbols(r'\sigma \alpha \mu \nu')
        tmp = sympy.tensorproduct(metric.lowered(),self.vals)
        # self.ind[self.ind.index(ind)] *= -1
        tmp2 = sympy.tensorcontraction(tmp,(0,self.ind.index(ind)+2))
        self.vals = copy.deepcopy(tmp2)
        return 

    def raise_index(self,ind,metric):
        sigma, alpha, mu, nu = sympy.symbols(r'\sigma \alpha \mu \nu')
        tmp = sympy.tensorproduct(metric.raised(),self.vals)
        # self.ind[self.ind.index(ind)] *= -1
        tmp2 = sympy.tensorcontraction(tmp,(0,self.ind.index(ind)+2))
        self.vals = copy.deepcopy(tmp2)
        return 
######################################################################

class GRMetric:
    def __init__(self,coords, metric=None, preset=None):
        self.coords = coords[:]
        self.dims = len(coords)
        if metric:
            self.metric=metric.copy()
        else:
            self.metric = sympy.zeros(self.dims)
        return

    def get_metric(self):
        return self.metric

    def lowered(self):
        return self.metric

    def raised(self):
        return self.metric.inv()

    def connection(self):
        ch = sympy.array.MutableDenseNDimArray([0 for i in range(self.dims**3)])
        ch = ch.reshape(self.dims,self.dims,self.dims)
        alpha, mu, nu = sympy.symbols(r'\alpha, \mu, \nu')
        for a_ in range(self.dims):
            for m_ in range(self.dims):
                for n_ in range(self.dims):
                    for lam in range(self.dims):
                        ch[a_, m_, n_] += .5*self.raised()[a_, lam]*(\
                        sympy.diff(self.lowered()[m_, lam], self.coords[n_]) +\
                        sympy.diff(self.lowered()[n_, lam], self.coords[m_]) -\
                        sympy.diff(self.lowered()[m_, n_], self.coords[lam])  )
        return GRTensor([alpha, mu, nu],ch)

    def curvature(self):
        sigma, alpha, mu, nu = sympy.symbols(r'\sigma \alpha \mu \nu')
        ch = self.connection().get_vals()
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

    def ricci_tensor(self):
        R = self.curvature()
        R2 = sympy.tensorcontraction(R.get_vals(),(0,2))
        R.vals = R2
        return R

    def ricci_scalar(self):
        R2 = self.ricci_tensor()
        R2_ = R2.get_vals()
        R_ = sympy.tensorproduct(self.raised(),R2_)
        R_1 = sympy.tensorcontraction(R_,(1,3))
        R_2 = sympy.tensorcontraction(R_1,(0,1))
        return R_2

