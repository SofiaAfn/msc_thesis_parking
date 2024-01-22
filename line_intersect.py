

import numpy as np

def lineIntersect3D(PA: np.ndarray, PB: np.ndarray):
    """Find the 3D intersection of two lines

    see: https://www.mathworks.com/matlabcentral/fileexchange/37192-intersection-point-of-lines-in-3d-space

    arguments:
        PA: np.ndarray[3]: The first point on the each line
        PB: np.ndarray[3]: The second point on the each line"""
    Si = PB - PA  # N lines described as vectors
    # Normalize vectors
    ni = Si / (np.sqrt(np.sum(Si**2, axis=1)).reshape(-1, 1))
    nx, ny, nz = ni[:, 0], ni[:, 1], ni[:, 2]
    SXX = np.sum(nx**2 - 1)
    SYY = np.sum(ny**2 - 1)
    SZZ = np.sum(nz**2 - 1)
    SXY = np.sum(nx*ny)
    SXZ = np.sum(nx*nz)
    SYZ = np.sum(ny*nz)
    S = np.array([[SXX, SXY, SXZ], [SXY, SYY, SYZ], [SXZ, SYZ, SZZ]])
    CX = np.sum(PA[:, 0]*(nx**2 - 1) + PA[:, 1]*(nx*ny) + PA[:, 2]*(nx*nz))
    CY = np.sum(PA[:, 0]*(nx*ny) + PA[:, 1]*(ny**2 - 1) + PA[:, 2]*(ny*nz))
    CZ = np.sum(PA[:, 0]*(nx*nz) + PA[:, 1]*(ny*nz) + PA[:, 2]*(nz**2 - 1))
    C = np.array([CX, CY, CZ]).reshape(-1, 1)
    P_intersect = np.linalg.solve(S, C).T[0]
    return P_intersect
