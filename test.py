from fenics import *
set_log_level(30)
mesh = UnitSquareMesh(120, 120)
V = FunctionSpace(mesh, 'Lagrange', 1)
print("Mesh and function space created successfully!")