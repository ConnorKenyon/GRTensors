# GRTensors
 

 ***GRTensors*** is a python package designed to make the symbolic tensor-based math needed for General Relativity, painless to calculate.

 ```python
>>> import GRTensors as grt
>>> import sympy
>>> 
>>> mu, nu = grt.make_index(r"\mu \nu")
>>> 
>>> # Metric for a unit radius 2-sphere
>>> theta, phi = sympy.symbols(r"\theta \phi")
>>> g_mn = grt.GRMetric([mu,nu], [[1, 0],[0,sympy.sin(theta)**2]])
 ```
