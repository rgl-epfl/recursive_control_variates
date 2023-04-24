import mitsuba as mi
import drjit as dr
import numpy as np
from cholespy import CholeskySolverF, MatrixType
import scipy.sparse as sp

def laplacian_2d(tex, lambda_=19):
    h,w = tex.shape[:2]
    N = h*w
    idx = np.arange(N)

    i = idx[(idx+1)%w!=0]
    j = idx[idx%w!=0]
    k = idx[idx%(h*w)<w*(h-1)]
    l = idx[idx%(h*w)>=w]

    ij = np.stack((i,j))
    ji = np.stack((j,i))
    ii = np.stack((i,i))
    jj = np.stack((j,j))
    kl = np.stack((k,l))
    lk = np.stack((l,k))
    kk = np.stack((k,k))
    ll = np.stack((l,l))
    indices = np.concatenate((ij,ji,kl,lk,ii,jj,kk,ll), axis=1)
    values = np.ones(indices.shape[1])
    values[:-2*(ii.shape[1]+kk.shape[1])] = -1
    return sp.csc_matrix((values, indices)) * lambda_ + sp.identity(N, format="csc")

def laplacian_3d(vol, lambda_=19):
    d,h,w = vol.shape[:3]
    N = d*h*w
    idx = np.arange(N)

    i = idx[(idx+1)%w!=0]
    j = idx[idx%w!=0]
    k = idx[idx%(h*w)<w*(h-1)]
    l = idx[idx%(h*w)>=w]
    m = idx[idx<w*h*(d-1)]
    n = idx[idx>=w*h]

    ij = np.stack((i,j))
    ji = np.stack((j,i))
    ii = np.stack((i,i))
    jj = np.stack((j,j))
    kl = np.stack((k,l))
    lk = np.stack((l,k))
    kk = np.stack((k,k))
    ll = np.stack((l,l))
    mn = np.stack((m,n))
    nm = np.stack((n,m))
    mm = np.stack((m,m))
    nn = np.stack((n,n))
    indices = np.concatenate((ij,ji,kl,lk,mn,nm,ii,jj,kk,ll,mm,nn), axis=1)
    values = np.ones(indices.shape[1])
    values[:-2*(ii.shape[1]+kk.shape[1]+mm.shape[1])] = -1
    return sp.csc_matrix((values, indices)) * lambda_ + sp.identity(N, format="csc")

class CholeskySolve(dr.CustomOp):

    def eval(self, solver, u):
        self.solver = solver
        self.shape = u.shape
        self.shape_solver = (dr.prod(self.shape[:-1]), self.shape[-1])
        x = dr.zeros(mi.TensorXf, shape=self.shape_solver)
        u = mi.TensorXf(u.array, self.shape_solver)
        # u = np.array(u.array, dtype=np.float32).reshape(self.shape_solver)
        # x = np.zeros_like(u)
        solver.solve(u, x)
        return mi.TensorXf(x.array, self.shape)

    def forward(self):
        x = dr.zeros(mi.TensorXf, shape=self.grad_in('u').shape)
        self.solver.solve(self.grad_in('u'), x)
        self.set_grad_out(x)

    def backward(self):
        # x = np.zeros_like(self.grad_out())
        x = dr.zeros(mi.TensorXf, shape=self.shape_solver)
        self.solver.solve(mi.TensorXf(self.grad_out().array, self.solver), x)
        # self.set_grad_in('u', mi.TensorXf(x))
        self.set_grad_in('u', mi.TensorXf(x.array, self.shape))

    def name(self):
        return "Cholesky solve"


class CholeskySolver():
    def __init__(self, x, lambda_):
        self.shape = x.shape
        self.channels = x.shape[-1]
        assert len(self.shape) in (3,4)
        self.N = dr.prod(self.shape[:-1])
        if len(self.shape) == 3:
            L_csc = laplacian_2d(x, lambda_)
        else:
            L_csc = laplacian_3d(x, lambda_)

        self.solver = CholeskySolverF(self.N, mi.TensorXi(L_csc.indptr), mi.TensorXi(L_csc.indices), mi.TensorXd(L_csc.data), MatrixType.CSC)

    def solve(self, u):
        return dr.custom(CholeskySolve, self.solver, u)

    def precondition(self, u):
        return self.solve(self.solve(u))

def to_differential(tex, lambda_):
    if len(tex.shape) == 3:
        L_csc = laplacian_2d(tex, lambda_)
    elif len(tex.shape) == 4:
        L_csc = laplacian_3d(tex, lambda_)
    return mi.TensorXf(mi.TensorXf((L_csc @ tex.numpy().reshape((-1, tex.shape[-1])))).array, tex.shape)
