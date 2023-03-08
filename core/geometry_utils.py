"""
Collection of geometric utility functions reusable across script.
Useful for exploration in GANs. 
"""
import numpy as np
from numpy.linalg import norm
from numpy import sqrt
#%% Utility functions for interpolation / exploration
def SLERP(code1, code2, steps, lim=(0,1)):
    """Spherical Linear Interpolation for numpy arrays"""
    code1, code2 = code1.squeeze(), code2.squeeze()
    cosang = np.dot(code1, code2) / norm(code1) / norm(code2)
    angle = np.arccos(cosang)
    ticks = angle * np.linspace(lim[0], lim[1], steps, endpoint=True)
    slerp_code = np.sin(angle - ticks)[:,np.newaxis] * code1[np.newaxis, :] + np.sin(ticks)[:,np.newaxis] * \
                 code2[np.newaxis, :]
    return slerp_code

def LERP(code1, code2, steps, lim=(0,1)):
    """ Linear Interpolation for numpy arrays"""
    code1, code2 = code1.reshape(1,-1), code2.reshape(1,-1)
    ticks = np.linspace(lim[0], lim[1], steps, endpoint=True)[:, np.newaxis]
    slerp_code = (1 - ticks) @ code1 + ticks @ code2
    return slerp_code

def LExpMap(refcode, tan_vec, steps, lim=(0,1)):
    """ Linear Exponential map for numpy arrays"""
    #print('ddd',refcode.shape,tan_vec.shape)
    
    refcode, tan_vec = refcode.reshape(1,-1), tan_vec.reshape(1,-1)
    ticks = np.linspace(lim[0], lim[1], steps, endpoint=True)[:, np.newaxis]
    #print(refcode.shape,tan_vec.shape, ticks.shape)
    # exit()
    #print(tan_vec,refcode)
    #exp_code = refcode + ticks*np.abs(ticks) @ tan_vec
    exp_code = refcode + (ticks-0) @ tan_vec
    return exp_code

def SExpMap(refcode, tan_vec, steps, lim=(0,1)):
    """ Spherical Exponential map for numpy arrays"""
    refcode, tan_vec = refcode.reshape(1,-1), tan_vec.reshape(1,-1)
    refnorm = np.linalg.norm(refcode)
    realtanv = tan_vec - (tan_vec@refcode.T)@refcode/refnorm**2
    tannorm = np.linalg.norm(realtanv)
    angles = np.linspace(lim[0], lim[1], steps, endpoint=True)[:, np.newaxis]
    exp_code = (np.cos(angles) @ refcode / refnorm + np.sin(angles) @ realtanv / tannorm) * refnorm
    return exp_code
    
#%% Geometric Utility Function
def ExpMap(x, tang_vec, EPS = 1E-4):
    angle_dist = sqrt((tang_vec ** 2).sum(axis=1))  # vectorized
    angle_dist = angle_dist[:, np.newaxis]
    # print("Angular distance for Exponentiation ", angle_dist[:,0])
    uni_tang_vec = tang_vec / angle_dist
    # x = repmat(x, size(tang_vec, 1), 1); # vectorized
    xnorm = np.linalg.norm(x)
    assert(xnorm > EPS, "Exponential Map from a basis point at origin is degenerate, examine the code. (May caused by 0 initialization)")
    y = (np.cos(angle_dist) @ (x[:] / xnorm) + np.sin(angle_dist) * uni_tang_vec) * xnorm
    return y

def VecTransport(xold, xnew, v):
    xold = xold / np.linalg.norm(xold)
    xnew = xnew / np.linalg.norm(xnew)
    x_symm_axis = xold + xnew
    v_transport = v - 2 * v @ x_symm_axis.T / np.linalg.norm(
        x_symm_axis) ** 2 * x_symm_axis  # Equation for vector parallel transport along geodesic
    # Don't use dot in numpy, it will have wierd behavior if the array is not single dimensional
    return v_transport

def radial_proj(codes, max_norm):
    if max_norm is np.inf:
        return codes
    else:
        max_norm = np.array(max_norm)
        assert np.all(max_norm >= 0)
        code_norm = np.linalg.norm(codes, axis=1)
        proj_norm = np.minimum(code_norm, max_norm)
        return codes / code_norm[:, np.newaxis] * proj_norm[:, np.newaxis]

def orthogonalize(basis, codes):
    if len(basis.shape) is 1:
        basis = basis[np.newaxis, :]
    assert basis.shape[1] == codes.shape[1]
    unit_basis = basis / norm(basis)
    codes_ortho = codes - (codes @ unit_basis.T) @ unit_basis
    return codes_ortho

def renormalize(codes, norms):
    norms = np.array(norms)
    assert codes.shape[0] == norms.size or norms.size == 1
    codes_renorm = norms.reshape([-1, 1]) * codes / norm(codes, axis=1).reshape([-1, 1])  # norms should be a 1d array
    return codes_renorm