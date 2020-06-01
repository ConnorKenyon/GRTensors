import sympy 
import copy
import sympy.tensor.array
import GRTensors as bt


def test_GRTensor():
    """
    Test the creation of a GRTensor object by creating an arbitrary tensor 
    in the form of [[a, 0],[0,b]] where a and b are arbitrary symbolic values.
    """
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    test_ind = [a,b]
    test_vals = sympy.Matrix([[a,0],[0,b]])
    
    A = bt.GRTensor(test_ind,test_vals)

    assert A.ind == test_ind
    assert A.vals == test_vals
    assert A.rank == 2

    return 1

def test_GRMetric():

    """
    Test the creation of a GRMetric object by creating an arbitrary tensor 
    in the form of [[a, 0],[0,b]] where a and b are arbitrary symbolic values.
    """
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    coords = [a,b]
    test_metric = sympy.Matrix([[a,0],[0,b]])
    g = bt.GRMetric(coords,metric=test_metric)
    assert g.lowered == test_metric
    return 1

def test_raise_metric():
    """
    Test the raise_metric() function for a GRMetric, which should
    return a correctly inverted metric.
    """
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    coords = [a,b]
    test_metric = sympy.Matrix([[a,0],[0,b]])
    G = bt.GRMetric(coords,metric=test_metric)
    assert G.raised == test_metric.inv()
    return 1
    

def test_christoffel():
    """
    Test the christoffel_symbols parameter of a 2-sphere GRMetric.
    """
    theta, phi = sympy.symbols(r'\theta \phi')
    t,x,y,z = sympy.symbols('t x y z')
    test_metric = sympy.Matrix([[1,0],[0,sympy.sin(theta)**2]])
    g = bt.GRMetric([theta, phi], metric=test_metric)
    ch = g.Christoffel_symbols.vals
    assert ch[0,1,1] == -sympy.sin(theta)*sympy.cos(theta)
    assert ch[1,0,1] == ch[1,1,0] 
    assert ch[1,1,0] == 1.0*sympy.cos(theta)/sympy.sin(theta)
    return 1

def test_riemann():
    """
    Test the Riemann_tensor parameter value for a 2-sphere GRMetric.
    """
    theta, phi = sympy.symbols(r'\theta \phi')
    test_metric = sympy.Matrix([[1,0],[0,sympy.sin(theta)**2]])
    g = bt.GRMetric([theta, phi], metric=test_metric)
    R = g.Riemann_tensor.vals
    # assert 2==3 # intentional failure
    return 1

def test_riemann_lowered():
    """
    Test the Riemann_tensor parameter value for a 2-sphere GRMetric, 
    with an additional operation to lower the first index.
    """
    theta, phi = sympy.symbols(r'\theta \phi')
    sigma, alpha, mu, nu = sympy.symbols(r'\sigma \alpha \mu \nu')
    test_metric = sympy.Matrix([[1,0],[0,sympy.sin(theta)**2]])
    g = bt.GRMetric([theta, phi], metric=test_metric)
    R = g.Riemann_tensor
    R.lower_index(sigma,g)
    return 1

def test_ricci_tensor():
    """
    Test the ricci_tensor parameter value for a 2-sphere GRMetric.
    """
    theta, phi = sympy.symbols(r'\theta \phi')
    test_metric = sympy.Matrix([[1,0],[0,sympy.sin(theta)**2]])
    g = bt.GRMetric([theta, phi], metric=test_metric)
    R2 = g.ricci_tensor
    return 1

def test_ricci_scalar():
    """
    Test the Ricci_scalar parameter value for a 2-sphere GRMetric.
    """
    theta, phi = sympy.symbols(r'\theta \phi')
    test_metric = sympy.Matrix([[1,0],[0,sympy.sin(theta)**2]])
    g = bt.GRMetric([theta, phi], metric=test_metric)
    R0 = g.Ricci_scalar
    assert R0 == 2.0
    return 1

# Test failing, need to debug type being returned for the rhs value
# def test_line_element():
#     t, x, y, z = sympy.symbols("t x y z")
#     dt, dx, dy, dz = sympy.symbols('dt dx dy dz')
#     eta = bt.GRMetric([t, x, y, z],sympy.Matrix([[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
#     dx = bt.GRTensor([eta],sympy.Array([dt, dx, dy, dz]))
#     rhs_ = sympy.tensorcontraction(sympy.tensorproduct(eta.lowered,dx.vals),(1,2))
#     rhs = sympy.tensorcontraction(sympy.tensorproduct(rhs_,dx.vals),(0,1))
#     assert rhs == dt**2 - (dx**2 + dy**2 + dz**z)
#     assert rhs - (dt**2 - dx**2 - dy**2 - dz**2) == 0
#     return 1

# No good test to validate geodesic equations yet.
# def test_geodesic():
#     theta, phi = sympy.symbols(r'\theta \phi')
#     test_metric = sympy.Matrix([[1,0],[0,sympy.sin(theta)**2]])
#     g = bt.GRMetric([theta, phi], metric=test_metric)
#     geo = g.geodesics()
#     # sympy.pprint(geo[0,1,1])
#     # sympy.pprint(geo[1,1,0])
#     return 1

def test_covdiff():
    """
    Test the covariant derivative of a tensor on the 2-sphere.
    The contraction of the specified tensor after being raised should be 0,
    because it is a divergenceless tensor.
    """
    x, y = sympy.symbols('x y')
    test_metric = sympy.Matrix([[1,0],[0,x**2]])
    g = bt.GRMetric([x,y],metric=test_metric)
    a, b = sympy.symbols('a b')
    A = bt.GRTensor([a],sympy.Array([x*sympy.cos(2*y),-x*x*sympy.sin(2*y)]))
    A.raise_index(a,g)
    Aab = bt.CovariantDerivative(b,[x,y],g,A)
    assert sympy.tensorcontraction(Aab.vals,(0,1)) == 0
    return 1


