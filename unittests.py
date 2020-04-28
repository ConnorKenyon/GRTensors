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
    assert A.get_indices() == test_ind
    assert A.get_vals() == test_vals
    assert A.get_rank() == 2
    print("Test: Init Tensor - Passed")
    return 1

def InitMetric():
    print("Starting Test: Init Metric...")
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    coords = [a,b]
    test_metric = sympy.Matrix([[a,0],[0,b]])
    g = bt.GRMetric(coords)
    assert g.lowered() == sympy.zeros(2)
    G = bt.GRMetric(coords,metric=test_metric)
    assert G.lowered() == test_metric
    print("Test: Init Metric - Passed")
    return 1

def RaiseMetric():
    print("Starting Test: Raise Metric...")
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    coords = [a,b]
    test_metric = sympy.Matrix([[a,0],[0,b]])
    G = bt.GRMetric(coords,metric=test_metric)
    assert G.raised() == test_metric.inv()
    print("Test: Raise Metric - Passed")
    return 1
    

def ChristoffelTest():
    print("Starting Test: Christoffel Symbols...")
    theta, phi = sympy.symbols(r'\theta \phi')
    t,x,y,z = sympy.symbols('t x y z')
    test_metric = sympy.Matrix([[1,0],[0,sympy.sin(theta)**2]])
    g = bt.GRMetric([theta, phi], metric=test_metric)
    ch = g.connection().get_vals()
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
    R = g.curvature().get_vals()
    print("Test: Riemann Curvature- Passed")
    return 1

def RiemannLowered():
    print("Starting Test: Riemann Lowered...")
    theta, phi = sympy.symbols(r'\theta \phi')
    sigma, alpha, mu, nu = sympy.symbols(r'\sigma \alpha \mu \nu')
    test_metric = sympy.Matrix([[1,0],[0,sympy.sin(theta)**2]])
    g = bt.GRMetric([theta, phi], metric=test_metric)
    R = g.curvature()
    R.lower_index(sigma,g)
    print("Test: Riemann Lowered - Passed")
    return 1

def RicciTensor():
    print("Starting Test: Ricci Tensor...")
    theta, phi = sympy.symbols(r'\theta \phi')
    test_metric = sympy.Matrix([[1,0],[0,sympy.sin(theta)**2]])
    g = bt.GRMetric([theta, phi], metric=test_metric)
    R2 = g.ricci_tensor()
    print("Test: Ricci Tensor - Passed")
    return 1

def RicciScalar():
    print("Starting Test: Ricci Scalar...")
    theta, phi = sympy.symbols(r'\theta \phi')
    test_metric = sympy.Matrix([[1,0],[0,sympy.sin(theta)**2]])
    g = bt.GRMetric([theta, phi], metric=test_metric)
    R0 = g.ricci_scalar()
    assert R0 == 2.0
    print("Test: Ricci Scalar - Passed")
    return 1

def LineElement():
    print("Starting Test: Line Element...")
    t, x, y, z = sympy.symbols("t x y z")
    dt, dx, dy, dz = sympy.symbols('dt dx dy dz')
    eta = bt.GRMetric([t, x, y, z],[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    dx = bt.GRTensor([eta],[dt, dx, dy, dz])
    rhs = sympy.tensorcontraction(sympy.tensorproduct(eta.lowered(),dx.get_vals()),(1,2))
    rhs = sympy.tensorcontraction(sympy.tensorproduct(rhs,dx.get_vals()),(1,1))
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
    LineElement
    ])
