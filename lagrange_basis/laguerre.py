"""Module defining the Lagrange-Laguerre coordinate basis"""

import numpy as np
import numpy.polynomial.laguerre as nplag
import scipy.special as sp

# using Laguerre polynomials with \alpha = 0
def lag_factor(n):
  return 1.0

def mesh_points(mesh_size, alpha):
  """Determines the basis mesh points
  
  usage: mesh = mesh_points(mesh_size)"""
  if(alpha == 0):
    coef = np.asarray([0]*mesh_size + [1])
    return nplag.lagroots(coef)
  else:
    rmat = gen_polymatrix(mesh_size, alpha)
    return sorted(np.linalg.eigvals(rmat))


def lambdas(mesh_size, alpha):
  """Determines the lambda coefficients of a basis

  usage: l = lambdas(mesh_size)"""
  roots = mesh_points(mesh_size, alpha)
  coef = np.asarray([0]*(mesh_size - 1) + [1])
  eval_at_roots = nplag.lagval(roots, coef)

  f1 = np.multiply(roots,np.exp(roots))
  f2 = np.power(mesh_size,2)*np.multiply(eval_at_roots, eval_at_roots)

  return np.divide(f1, f2)

def basis_func(out_mesh, j, base_mesh):
  """Returns the values of a single basis function on a given mesh.

  usage: f = basis_func(out_mesh, j, base_mesh)
  out_mesh - array of points for which the function is evaluated
  j - order of the basis function
  base_mesh - mesh on which the basis is defined, see mesh_points"""
  mesh_size = base_mesh.size
  coef = np.asarray([0]*mesh_size + [1])

  f1 = np.power(-1,j + 1)*np.sqrt(base_mesh[j])/np.sqrt(lag_factor(mesh_size))
  f2 = np.divide(nplag.lagval(out_mesh, coef), out_mesh - base_mesh[j])
  f3 = np.exp(-0.5*out_mesh)

  return f1*np.multiply(f2, f3)

def me_gauss_dx(i, j, mesh, alpha):
  """Matrix element of the d/dx operator - <i|d/dx|j>"""
  if(i == j):
    return -0.5/mesh[i]
  else:
    return np.power(-1, i - j)*sqrt(mesh[j]/mesh[i])/(mesh[i] - mesh[j])

def me_exact_dx(i, j, mesh, alpha):
  """Exact matrix element of the d/dx operator
  provides correct value for alpha > 0 """
  if(i == j):
    return -0.5*delta_disc(alpha, 0)/mesh[i]
  else:
    fac1 = 0.5*np.power(-1, i - j)/np.sqrt(mesh[i]*mesh[j])
    fac2 = (mesh[i] + mesh[j])/(mesh[i] - mesh[j]) - delta_disc(alpha, 0)
    return fac1*fac2

def me_x2(i, j, mesh):
  """Matrix element of the x^2 operator - <i|x^2|j>"""
  retval = np.power(-1, i - j)*np.sqrt(mesh[i]*mesh[j])
  if(i == j):
    retval = retval + np.power(mesh[i],2)
  
  return retval

def me_xinv(i, j, mesh, alpha = 0):
  """Matrix element of the 1/x operator - <i|1/x|j>"""
  retval = np.power(-1, i - j)/np.sqrt(mesh[i]*mesh[j])
  if(i == j):
    retval = retval + 1.0/mesh[i]
  
  return retval

def me_x2inv(i, j, mesh):
  """Matrix element of the 1/x^2 operator - <i|1/x^2|j>"""
  retval = np.power(-1, i - j)*(1/mesh[i] + 1/mesh[j] - (2*mesh.size + 1))/np.sqrt(mesh[i]*mesh[j])
  if(i == j):
    retval = retval + 1/np.power(mesh[i],2)
  
  return retval

def me_ddx(i, j, mesh):
  """Matrix element of the d^2/dx^2 operator - <i|d^2/dx^2|j>"""
  mesh_size = mesh.size

  if(i == j):
    return (1 - 2*(2*mesh_size + 1 - 4.0/mesh[i])/mesh[i])/12.0
  else:
    f1 = np.power(-1, i - j)/np.sqrt(mesh[i]*mesh[j])
    f2 = 0.5*(1/mesh[i] + 1/mesh[j])
    f3 = (mesh[i] + mesh[j])/np.power(mesh[i] - mesh[j],2)

    return f1*(f2 - f3)

def basis_func_regularized_x(out_mesh, j, base_mesh):
  """Returns the values of a single basis function regularized by x on a given mesh.

  usage: f = basis_func_regularized_x(out_mesh, j, base_mesh)
  out_mesh - array of points for which the function is evaluated
  j - order of the basis function
  base_mesh - mesh on which the basis is defined, see mesh_points"""
  mesh_size = base_mesh.size
  coef = np.asarray([0]*mesh_size + [1])

  f1 = np.power(-1, j + 1)/np.sqrt(lag_factor(mesh_size)*base_mesh[j])
  f2 = np.divide(nplag.lagval(out_mesh, coef), out_mesh - base_mesh[j])
  f3 = np.multiply(out_mesh,np.exp(-0.5*out_mesh))

  return f1*np.multiply(f2, f3)

def me_dx_gauss_regularized_x(i, j, mesh, alpha):
  """Matrix element of the d/dx operator in a regularized basis, Gauss approximation"""
  if(i == j):
    return 0.5/mesh[i]
  else:
    return np.power(-1, i - j)*np.sqrt(mesh[i]/mesh[j])/(mesh[i] - mesh[j])

def me_dx_exact_regularized_x(i, j, mesh, alpha):
  """Matrix element of the d/dx operator in a regularized basis, exact"""
  return me_dx_gauss_regularized_x(i, j, mesh, alpha) - 0.5*np.power(-1, i - j)/np.sqrt(mesh[i]*mesh[j])

def me_ddx_gauss_regularized_x(i, j, mesh):
  """Matrix element of the d^2/dx^2 operator in a regularized basis, Gauss approximation"""
  mi = mesh[i]
  mj = mesh[j]

  if(i == j):
    return (np.power(mi, 2) - 2*(2*mesh.size + 1)*mi - 4)/(12.0*np.power(mi,2))
  else:
    return np.power(-1, i - j + 1)*(mi + mj)/(np.sqrt(mi*mj)*np.power(mi - mj,2))

def me_ddx_exact_regularized_x(i, j, mesh):
  """Matrix element of the d^2/dx^2 operator in a regularized basis, exact"""
  return me_ddx_gauss_regularized_x(i, j, mesh) + 0.25*np.power(-1, i - j)/np.sqrt(mesh[i]*mesh[j])

def wave_function(eigvec, out_mesh, in_mesh, bfunc, scale):
  """Returns the full wave function determined by the eigenvector in a given basis"""
  wf = np.zeros(out_mesh.size)

  for idx in range(0, in_mesh.size):
    wf = wf + eigvec[idx]*bfunc(out_mesh/scale, idx, in_mesh)
  
  return wf/np.sqrt(scale)

def gen_polymatrix(order, alpha):
  """ Returns a matrix eigenvalues of which are roots of n-th order
  generalized Laguerre polynomial"""
  genlag = sp.genlaguerre(order, alpha)
  retmat = np.zeros([order, order])
  for idx in range(order):
    retmat[0, idx] = -genlag[order - 1 - idx]/genlag[order]

  for idx in range(1, order):
    retmat[idx, idx - 1] = 1

  return retmat

def delta_disc(i, j):
  if(i == j):
    return 1
  else:
    return 0