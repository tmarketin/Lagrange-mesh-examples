"""Module defining the Lagrange basis based on second sine mesh"""

import numpy as np

def mesh_points(mesh_size):
  return np.asarray([2*k/(2*mesh_size + 1) for k in range(1,mesh_size + 1)])

def lambdas(mesh_size):
  return 2*np.ones([mesh_size])/(2*mesh_size + 1)

def me_dx_exact(i, j, mesh):
  ndim = mesh.size
  if(i == j):
    f1 = 2*ndim*np.pi*mesh[i]
    f2 = np.pi*mesh[i]

    nom = 2*ndim*np.cos(f1)*np.sin(f2) - np.sin(f1)*np.cos(f2)
    return -0.5*np.pi*nom/((2*ndim + 1)*np.power(np.sin(f2),2))
  else:
    fm1 = ndim*np.pi*(mesh[i] - mesh[j])
    fm2 = 0.5*np.pi*(mesh[i] - mesh[j])
    fp1 = ndim*np.pi*(mesh[i] + mesh[j])
    fp2 = 0.5*np.pi*(mesh[i] + mesh[j])

    nm = 2*ndim*np.cos(fm1)*np.sin(fm2) - np.sin(fm1)*np.cos(fm2)
    dm = np.power(np.sin(fm2),2)
    npl = 2*ndim*np.cos(fp1)*np.sin(fp2) - np.sin(fp1)*np.cos(fp2)
    dpl = np.power(np.sin(fp2),2)

    fac = 0.5*np.pi/(2*ndim + 1)
    return fac*(nm/dm - npl/dpl)

def me_ddx_exact(i, j, mesh):
  if(i == j):
    term1 = 2*mesh.size*(mesh.size + 1)/3 + 0.5
    term2 = 1/np.power(np.sin(np.pi*mesh[i]),2)
    return -1*0.5*np.power(np.pi,2)*(term1 - term2)
  else:
    term1 = 1/np.power(np.sin(0.5*np.pi*(mesh[i] - mesh[j])),2)
    term2 = 1/np.power(np.sin(0.5*np.pi*(mesh[i] + mesh[j])),2)
    return -1*0.5*np.power(-1, i - j)*np.power(np.pi,2)*(term1 - term2)

def basis_func(out_mesh, j, base_mesh):
  fac = 1/np.sqrt(4*base_mesh.size + 2)
  
  n1 = np.sin(base_mesh.size*np.pi*(out_mesh - base_mesh[j]))
  d1 = np.sin(0.5*np.pi*(out_mesh - base_mesh[j]))
  term1 = np.divide(n1, d1)

  n2 = np.sin(base_mesh.size*np.pi*(out_mesh + base_mesh[j]))
  d2 = np.sin(0.5*np.pi*(out_mesh + base_mesh[j]))
  term2 = np.divide(n2, d2)

  return fac*(term1 - term2)

def wave_function(eigvec, out_mesh, in_mesh, scale):
  """Returns the full wave function determined by the eigenvector in a given basis"""
  wf = np.zeros(out_mesh.size)

  for idx in range(0, in_mesh.size):
    wf = wf + eigvec[idx]*basis_func(out_mesh/scale, idx, in_mesh)
  
  return wf/np.sqrt(scale)
