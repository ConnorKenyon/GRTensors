import sympy 
import copy

import GRTensors as bt


def InitTensor():
    print("Starting Test: Init Tensor...")
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    test_ind = [a,b]
    test_vals = sympy.Matrix([[a,0],[0,b]])
    
    A = bt.GRTensor(test_ind,test_vals)

    assert A.ind == test_ind
    assert A.vals == test_vals
    assert A.rank == 2

    print("Test: Init Tensor - Passed")
    return 1

def InitMetric():
    print("Starting Test: Init Metric...")
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    coords = [a,b]
    test_metric = sympy.Matrix([[a,0],[0,b]])
    g = bt.GRMetric(coords,metric=test_metric)
    assert g.lowered == test_metric
    print("Test: Init Metric - Passed")
    return 1

def RaiseMetric():
    print("Starting Test: Raise Metric...")
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    coords = [a,b]
    test_metric = sympy.Matrix([[a,0],[0,b]])
    G = bt.GRMetric(coords,metric=test_metric)
    assert G.raised == test_metric.inv()
    print("Test: Raise Metric - Passed")
    return 1
    

def ChristoffelTest():
    print("Starting Test: Christoffel Symbols...")
    theta, phi = sympy.symbols(r'\theta \phi')
    t,x,y,z = sympy.symbols('t x y z')
    test_metric = sympy.Matrix([[1,0],[0,sympy.sin(theta)**2]])
    g = bt.GRMetric([theta, phi], metric=test_metric)
    ch = g.Christoffel_symbols.vals
    assert ch[0,1,1] == -sympy.sin(theta)*sympy.cos(theta)
    assert ch[1,0,1] == ch[1,1,0] 
    assert ch[1,1,0] == 1.0*sympy.cos(theta)/sympy.sin(theta)
    print("Test: Christoffel Symbols - Passed")
    return 1

def RiemannTest():
    print("Starting Test: Riemann Curvature...")
    theta, phi = sympy.symbols(r'\theta \phi')
    test_metric = sympy.Matrix([[1,0],[0,sympy.sin(theta)**2]])
    g = bt.GRMetric([theta, phi], metric=test_metric)
    R = g.Riemann_tensor.vals
    print("Test: Riemann Curvature- Passed")
    return 1

def RiemannLowered():
    print("Starting Test: Riemann Lowered...")
    theta, phi = sympy.symbols(r'\theta \phi')
    sigma, alpha, mu, nu = sympy.symbols(r'\sigma \alpha \mu \nu')
    test_metric = sympy.Matrix([[1,0],[0,sympy.sin(theta)**2]])
    g = bt.GRMetric([theta, phi], metric=test_metric)
    R = g.Riemann_tensor
    R.lower_index(sigma,g)
    print("Test: Riemann Lowered - Passed")
    return 1

def RicciTensor():
    print("Starting Test: Ricci Tensor...")
    theta, phi = sympy.symbols(r'\theta \phi')
    test_metric = sympy.Matrix([[1,0],[0,sympy.sin(theta)**2]])
    g = bt.GRMetric([theta, phi], metric=test_metric)
    R2 = g.ricci_tensor
    print("Test: Ricci Tensor - Passed")
    return 1

def RicciScalar():
    print("Starting Test: Ricci Scalar...")
    theta, phi = sympy.symbols(r'\theta \phi')
    test_metric = sympy.Matrix([[1,0],[0,sympy.sin(theta)**2]])
    g = bt.GRMetric([theta, phi], metric=test_metric)
    R0 = g.Ricci_scalar
    assert R0 == 2.0
    print("Test: Ricci Scalar - Passed")
    return 1

def LineElement():
    print("Starting Test: Line Element...")
    try:
        t, x, y, z = sympy.symbols("t x y z")
        dt, dx, dy, dz = sympy.symbols('dt dx dy dz')
        eta = bt.GRMetric([t, x, y, z],sympy.Matrix([[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
        dx = bt.GRTensor([eta],sympy.Array([dt, dx, dy, dz]))
        rhs_ = sympy.tensorcontraction(sympy.tensorproduct(eta.lowered,dx.vals),(1,2))
        rhs = sympy.tensorcontraction(sympy.tensorproduct(rhs_,dx.vals),(0,1))
        assert rhs + dt**2 - dx**2 - dy**2 - dz**2 == 0
        print("Test: Line Element - Passed")
        return 1
    except:
        print(rhs)
        print(type(rhs))
        print("Test: Line Element - Failed")
        return 0

def GeodesicTest():
    print("Starting Test: Geodesics...")
    theta, phi = sympy.symbols(r'\theta \phi')
    test_metric = sympy.Matrix([[1,0],[0,sympy.sin(theta)**2]])
    g = bt.GRMetric([theta, phi], metric=test_metric)
    geo = g.geodesics()
    # sympy.pprint(geo[0,1,1])
    # sympy.pprint(geo[1,1,0])
    print("Test: Geodesics - Passed")
    return 1

def CovDeriv():
    print("Starting Test: Covariant Derivative...")
    x, y = sympy.symbols('x y')
    test_metric = sympy.Matrix([[1,0],[0,x**2]])
    g = bt.GRMetric([x,y],metric=test_metric)
    a, b = sympy.symbols('a b')
    A = bt.GRTensor([a],sympy.Array([x*sympy.cos(2*y),-x*x*sympy.sin(2*y)]))
    A.raise_index(a,g)
    Aab = bt.CovariantDerivative(b,[x,y],g,A)
    # print(Aab.vals)
    assert sympy.tensorcontraction(Aab.vals,(0,1)) == 0
    return 1


def UnitTests(tests):
    pass_count = 0
    num_tests = len(tests)
    for test in tests:
        pass_count += test()

    print("Unit Tests Complete: {} / {} Tests Passed".format(pass_count,num_tests))
    return

UnitTests([
    InitTensor,
    InitMetric,
    RaiseMetric,
    ChristoffelTest,
    RiemannTest, 
    RiemannLowered,
    RicciTensor,
    RicciScalar,
    LineElement, 
    GeodesicTest,
    CovDeriv
    ])
