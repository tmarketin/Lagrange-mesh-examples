"""Module defining the Lagrange-Hermite coordinate basis"""

import numpy as np
import numpy.polynomial.hermite as npherm

# calculates the normalization factor 
def herm_factor(n):
  return np.sqrt(np.pi)*np.power(2.0,n)*np.math.factorial(n)

def mesh_points(mesh_size):
  """Determines the basis mesh points
  
  usage: mesh = mesh_points(mesh_size)"""
  coef = np.asarray([0]*mesh_size + [1])
  return npherm.hermroots(coef)

def lambdas(mesh_size):
  """Determines the lambda coefficients of a basis

  usage: l = lambdas(mesh_size)"""
  roots = mesh_points(mesh_size)
  coef = np.asarray([0]*(mesh_size - 1) + [1])
  eval_at_roots = npherm.hermval(roots,coef)

  nom = herm_factor(mesh_size - 1)*np.exp(np.multiply(roots,roots))
  denom = mesh_size*np.multiply(eval_at_roots,eval_at_roots)

  return np.divide(nom,denom)

def basis_func(out_mesh, j, base_mesh):
  """Returns the values of a single basis function on a given mesh.

  usage: f = basis_func(out_mesh, j, base_mesh)
  out_mesh - array of points for which the function is evaluated
  j - order of the basis function
  base_mesh - mesh on which the basis is defined, see mesh_points"""
  mesh_size = base_mesh.size
  coef = np.asarray([0]*mesh_size + [1])

  f1 = np.power(-1, mesh_size - j + 1)/np.sqrt(2*herm_factor(mesh_size))
  f2 = np.divide(npherm.hermval(out_mesh,coef), out_mesh - base_mesh[j])
  f3 = np.exp(-0.5*np.multiply(out_mesh, out_mesh))

  return f1*np.multiply(f2,f3)

def wave_function(eigvec, out_mesh, in_mesh, scale):
  """Returns the full wave function determined by the eigenvector in a given basis"""
  wf = np.zeros(out_mesh.size)

  for idx in range(0, in_mesh.size):
    wf = wf + eigvec[idx]*basis_func(out_mesh/scale, idx, in_mesh)
  
  return wf/np.sqrt(scale)

def me_dx_exact(i, j, mesh):
  """ Matrix element of the d/dx operator, Gauss approximation is exact """
  mesh_size = mesh.size

  if(i == j):
    return 0
  else:
    return np.power(-1, i - j)/(mesh[i] - mesh[j])

def me_ddx_gauss(i, j, mesh):
  """Matrix element of the d^2/dx^2 operator in the Gauss approximation"""
  mesh_size = mesh.size

  if(i == j):
    return -(2*mesh_size + 1 - np.power(mesh[i],2))/3.0
  else:
    fac1 = 2*np.power(-1, i-j)
    fac2 = np.power(mesh[i] - mesh[j],2)
    return -fac1/fac2

# exact kinetic matrix element (operator -d^2/dx^2)
def me_ddx_exact(i, j, mesh):
  """Matrix element of the d^2/dx^2 operator, exact"""
  return me_ddx_gauss(i, j, mesh) + 0.5*np.power(-1, i-j)