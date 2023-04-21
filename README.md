# GRTensors
 

 ***GRTensors*** is a python package designed for symbolic tensor-based
 calculations.  The primary design is for General Relativity, however the
 framework is broadly applicable to all fields that require tensor
 computations.

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
