import numpy as np
import gmsh
import dolfinx
import ufl
import bempp_cl.api
import petsc4py
import scipy.sparse

from bempp_cl.api.external import fenicsx
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio
from mpi4py import MPI
from dolfinx import io

from dolfinx.geometry import compute_colliding_cells, compute_collisions_points
from scipy.sparse import coo_matrix, diags
from petsc4py import PETSc

from coupled_solver.utils.generate_mesh import load_dolfin_mesh
from coupled_solver.utils.generate_mesh import build_boundary_mesh
from coupled_solver.utils.check_charge_mesh_distance import check_charge_mesh_distance

from bempp_cl.api.operators.boundary import sparse, laplace, modified_helmholtz
from bempp_cl.api.assembly.blocked_operator import BlockedDiscreteOperator
from bempp_cl.api.assembly.discrete_boundary_operator import InverseSparseDiscreteBoundaryOperator
from bempp_cl.api.operators.potential.laplace import single_layer, double_layer

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres

def coulomb_potential(q, p, Q, xq, E):
    """
    Computes the electrostatic potential due to a point monopole, dipole, 
    quadrupole distribution at the position of the multipoles.
    See equation 29 of amoeba bem document. 
    Inputs:
    ------- 
        q: array size N with charges
        p: array size (Nx3) with dipoles
        Q: array size (Nx3x3) with quadrupoles
        xq: array size (Nx3) with positions of multipoles
        E: dielectric constant
    Returns:
    -------
        phi: electrostatic potential at the position of the multipoles
    """

    phi = np.zeros(len(xq))
    T2  = np.zeros((len(xq),3,3))
    for i in range(len(xq)):
        Ri = xq[i]-xq
        Rnorm = np.sqrt(np.sum(Ri*Ri, axis=1))

        for j in np.where(Rnorm>1e-10)[0]: #remove singularity
            T0 = 1/Rnorm[j]
            T1 = Ri[j,:]/Rnorm[j]**3
            T2[j,:,:] = np.ones((3,3))*Ri[j,:]*np.transpose(np.ones((3,3))*Ri[j,:])/Rnorm[j]**5

            phi[i] += q[j]*T0 + np.sum(T1[:]*p[j,:]) + 0.5*np.sum(np.sum(T2[j,:,:]*Q[j,:,:],axis=1),axis=0)

    phi /= (4*np.pi*E)

    return phi

def coulomb_potential_thole(p, alpha, xq, E):
    """
    Computes the electrostatic potential due to a point dipole 
    distribution at the position of the multipoles.
    See equation 29 of amoeba bem document. 
    Uses Thole damping
    Inputs:
    ------- 
        p: array size (Nx3) with polarizable dipoles
        alpha: array size (Nx3x3) with polarizabilities (tensor)
        xq: array size (Nx3) with positions of multipoles
        E: dielectric constant
    Returns:
    -------
        phi: electrostatic potential at the position of the multipoles
    """

    phi = np.zeros(len(xq))
    T2  = np.zeros((len(xq),3,3))
    for i in range(len(xq)):
        Ri = xq[i]-xq
        Rnorm = np.sqrt(np.sum(Ri*Ri, axis=1))

        for j in np.where(Rnorm>1e-10)[0]: #remove singularity

            # Thole damping for polarizable dipoles (valid for thole factor=1)
            damp = (alpha[i,0,0]*alpha[j,0,0])**(0.16666667)
            damp += 1e-12
            damp = -(Rnorm[j]/damp)**3
            expdamp = np.exp(damp)
            scale3 = 1 - expdamp
            scale5 = 1 - expdamp*(1-damp)

            T1 = Ri[j,:]/Rnorm[j]**3 * scale3

            phi[i] += np.sum(T1[:]*p[j,:]) 

    phi /= (4*np.pi*E)

    return phi

def coulomb_field(q, p, Q, xq, E):
    """
    Computes the electric field due to a point monopole, dipole, quadrupole distribution
    at the position of the multipoles. The field is defined as E=-nabla*phi.
    See equation 52 of kirkwood multipole, and Equation 30 of amoeba bem document.
    Inputs:
    ------- 
        q: array size N with charges
        p: array size (Nx3) with dipoles
        Q: array size (Nx3x3) with quadrupoles
        xq: array size (Nx3) with positions of multipoles
        E: dielectric constant
    Returns:
    -------
        Efield: electric field at the position of the multipoles
    """
    Efield = np.zeros((len(xq),3))
    T0 = np.zeros((len(xq),3))
    T1 = np.zeros((len(xq),3,3))
    T2 = np.zeros((len(xq),3,3,3))
    for i in range(len(xq)):
        Ri = xq[i]-xq
        Rnorm = np.sqrt(np.sum(Ri*Ri, axis=1))


        for j in np.where(Rnorm>1e-10)[0]: #remove singularity

            T0[j,:]   = -Ri[j,:]/Rnorm[j]**3
            T1[j,:,:] = np.identity(3)/Rnorm[j]**3 - 3*np.ones((3,3))*Ri[j,:]*np.transpose(np.ones((3,3))*Ri[j,:])/Rnorm[j]**5

            # the ordering in aux will be k,j,i looking at Eq 52 of kirkwood multipole
            aux = np.zeros((3,3,3))
            for k in range(3):
                aux[k,:,:] = np.ones((3,3))*Ri[j,:]*np.transpose(np.ones((3,3))*Ri[j,:])*Ri[j,k]
            aux *= -5/Rnorm[j]**7

            for k in range(3):
                aux[:,:,k] += np.identity(3)*Ri[j,k]/Rnorm[j]**5
            for k in range(3):
                aux[:,k,:] += np.identity(3)*Ri[j,k]/Rnorm[j]**5

            T2[j,:,:,:] = aux

            for k in range(3):
                Efield[i,k] += T0[j,k]*q[j] + np.sum(T1[j,k,:]*p[j,:]) + 0.5*np.sum(np.sum(T2[j,k,:,:]*Q[j,:,:],axis=1),axis=0)

    Efield /= -(4*np.pi*E)
    return Efield

def coulomb_field_thole(q, p, Q, alpha, xq, E):
    """
    Computes the electric field due to a point monpole, dipole and quadrupole
    at the position of the multipoles. The field is defined as E=-nabla*phi.
    Uses Thole damping.
    See equation 52 of kirkwood multipole, and Equation 30 of amoeba bem document.
    Inputs:
    ------- 
        q: array size N with charges
        p: array size (Nx3) with dipoles
        Q: array size (Nx3x3) with quadrupoles
        alpha: array size (Nx3x3) with polarizabilities
        xq: array size (Nx3) with positions of multipoles
        E: dielectric constant
    Returns:
    -------
        Efield: electric field at the position of the multipoles
    """
    Efield = np.zeros((len(xq),3))
    T0 = np.zeros((len(xq),3))
    T1 = np.zeros((len(xq),3,3))
    T2 = np.zeros((len(xq),3,3,3))
    for i in range(len(xq)):
        Ri = xq[i]-xq
        Rnorm = np.sqrt(np.sum(Ri*Ri, axis=1))


        for j in np.where(Rnorm>1e-10)[0]: #remove singularity

            # Thole damping for polarizable dipoles (valid for thole factor=1)
            damp = (alpha[i,0,0]*alpha[j,0,0])**(0.16666667)
            damp += 1e-12
            damp = -(Rnorm[j]/damp)**3
            expdamp = np.exp(damp)
            scale3 = 1 - expdamp
            scale5 = 1 - expdamp*(1-damp)
            scale7 = 1 - expdamp*(1-damp+0.6*damp*damp) 

            T0[j,:]   = -Ri[j,:]/Rnorm[j]**3 * scale3
            T1[j,:,:] = np.identity(3)/Rnorm[j]**3*scale3 - 3*np.ones((3,3))*Ri[j,:]*np.transpose(np.ones((3,3))*Ri[j,:])/Rnorm[j]**5*scale5

            # the ordering in aux will be k,j,i looking at Eq 52 of kirkwood multipole
            aux = np.zeros((3,3,3))
            for k in range(3):
                aux[k,:,:] = np.ones((3,3))*Ri[j,:]*np.transpose(np.ones((3,3))*Ri[j,:])*Ri[j,k]
            aux *= -5/Rnorm[j]**7*scale7

            for k in range(3):
                aux[:,:,k] += np.identity(3)*Ri[j,k]/Rnorm[j]**5*scale5
            for k in range(3):
                aux[:,k,:] += np.identity(3)*Ri[j,k]/Rnorm[j]**5*scale5

            T2[j,:,:,:] = aux

            for k in range(3):
                Efield[i,k] += T0[j,k]*q[j] + np.sum(T1[j,k,:]*p[j,:]) + 0.5*np.sum(np.sum(T2[j,k,:,:]*Q[j,:,:],axis=1),axis=0) 

    Efield /= -(4*np.pi*E)
    return Efield

def coulomb_ddpotential(q, p, Q, xq, E):
    """
    Computes the second derivative of the electrostatic potential due 
    to a point monopole, dipole, and quadrupole distribution at the 
    position of the multipoles. See equation 29 and 43 of amoeba bem document. 
    Inputs:
    ------- 
        q: array size N with charges
        p: array size (Nx3) with dipoles
        Q: array size (Nx3x3) with quadrupoles
        xq: array size (Nx3) with positions of multipoles
        E: dielectric constant
    Returns:
    -------
        ddphi: second derivative of electrostatic potential at the 
                position of the multipoles
    """
    ddphi = np.zeros((len(xq),3,3))
    T0 = np.zeros((len(xq),3,3))
    T1 = np.zeros((len(xq),3,3,3))
    T2 = np.zeros((len(xq),3,3,3,3))
    for i in range(len(xq)):
        Ri = xq[i]-xq
        Rnorm = np.sqrt(np.sum(Ri*Ri, axis=1))

        for j in np.where(Rnorm>1e-10)[0]: #remove singularity
            T0[j,:,:] = -np.identity(3)/Rnorm[j]**3 + 3*np.ones((3,3))*Ri[j,:]*np.transpose(np.ones((3,3))*Ri[j,:])/Rnorm[j]**5

            # the ordering in aux will be k,j,i looking at Eq 52 of kirkwood multipole
            aux = np.zeros((3,3,3))
            for k in range(3):
                aux[k,:,:] = np.ones((3,3))*Ri[j,:]*np.transpose(np.ones((3,3))*Ri[j,:])*Ri[j,k]
            aux *= 15/Rnorm[j]**7

            for k in range(3):
                aux[:,:,k] -= 3*np.identity(3)*Ri[j,k]/Rnorm[j]**5
                aux[:,k,:] -= 3*np.identity(3)*Ri[j,k]/Rnorm[j]**5
                aux[k,:,:] -= 3*np.identity(3)*Ri[j,k]/Rnorm[j]**5

            T1[j,:,:,:] = aux

            for k in range(3):
                for l in range(3):
                    for m in range(3):
                        for n in range(3):
                            dkl = (k==l)*1.0
                            dkm = (k==m)*1.0
                            dkn = (k==n)*1.0
                            dlm = (l==m)*1.0
                            dln = (l==n)*1.0

                            T2[j,k,l,m,n] = -7*Ri[j,k]*Ri[j,l]*Ri[j,m]*Ri[j,n]/Rnorm[j]**2  \
                                           + Ri[j,m]*Ri[j,n]*dkl + Ri[j,l]*Ri[j,n]*dkm      \
                                           + Ri[j,m]*Ri[j,l]*dkn + Ri[j,k]*Ri[j,n]*dlm      \
                                           + Ri[j,m]*Ri[j,k]*dln                            \
                                           - (dkm*dln + dlm*dkn)*Rnorm[j]**2/5
            T2 *= -5/Rnorm[j]**7

            for k in range(3):
                for l in range(3):
                    ddphi[i,k,l] += T0[j,k,l]*q[j] + np.sum(T1[j,k,l,:]*p[j,:]) + 0.5*np.sum(np.sum(T2[j,k,l,:,:]*Q[j,:,:],axis=1),axis=0)
    
    ddphi /= (4*np.pi*E)

    return ddphi

def coulomb_ddpotential_thole(p, alpha, xq, E):
    """
    Computes the second derivative of the electrostatic potential due 
    to a point dipole distribution at the 
    position of the multipoles. 
    Uses Thole damping.
    See equation 29 and 43 of amoeba bem document. 
    Inputs:
    ------- 
        p: array size (Nx3) with dipoles
        alpha: array size (Nx3x3) with polarizabilities
        xq: array size (Nx3) with positions of multipoles
        E: dielectric constant
    Returns:
    -------
        ddphi: second derivative of electrostatic potential at the 
                position of the multipoles
    """
    ddphi = np.zeros((len(xq),3,3))
    T1 = np.zeros((len(xq),3,3,3))
    for i in range(len(xq)):
        Ri = xq[i]-xq
        Rnorm = np.sqrt(np.sum(Ri*Ri, axis=1))

        for j in np.where(Rnorm>1e-10)[0]: #remove singularity

            # Thole damping for polarizable dipoles (valid for thole factor=1)
            damp = (alpha[i,0,0]*alpha[j,0,0])**(0.16666667)
            damp += 1e-12
            damp = -(Rnorm[j]/damp)**3
            expdamp = np.exp(damp)
            scale3 = 1 - expdamp
            scale5 = 1 - expdamp*(1-damp)
            scale7 = 1 - expdamp*(1-damp+0.6*damp*damp) 

            # the ordering in aux will be k,j,i looking at Eq 52 of kirkwood multipole
            aux = np.zeros((3,3,3))
            for k in range(3):
                aux[k,:,:] = np.ones((3,3))*Ri[j,:]*np.transpose(np.ones((3,3))*Ri[j,:])*Ri[j,k]
            aux *= 15/Rnorm[j]**7 * scale7

            for k in range(3):
                aux[:,:,k] -= 3*np.identity(3)*Ri[j,k]/Rnorm[j]**5 * scale5
                aux[:,k,:] -= 3*np.identity(3)*Ri[j,k]/Rnorm[j]**5 * scale5
                aux[k,:,:] -= 3*np.identity(3)*Ri[j,k]/Rnorm[j]**5 * scale5

            T1[j,:,:,:] = aux

            for k in range(3):
                for l in range(3):
                    ddphi[i,k,l] += np.sum(T1[j,k,l,:]*p[j,:]) 
    
    ddphi /= (4*np.pi*E)

    return ddphi

def solvation_energy(q, p, Q, phi, dphi, ddphi):
    solv_energy = 2*np.pi*332.064*(np.sum(q*phi) + np.sum(p*dphi) + np.sum(Q*ddphi)/6)
    return solv_energy

def coulomb_energy_multipole(q, p_per, p_pol, Q, alpha, xq, E):
    """
    Computes the Coulomb energy from a collection of point
    multipoles, according to equation 38 of amoeba bem document.
    
    Inputs:
    ------ 
        q : array size N with charges of multipoles
        p_per : array size (Nx3) with permanent dipoles of multipoles
        p_pol : array size (Nx3) with polarizable dipoles of multipoles
        Q : array size (Nx3x3) with quadrupoles of multipoles
        xq: array size Nx3 with positions of multipoles
    p_perm: array size (Nx3) with permanent dipoles of multipoles
        E : float, dielectric constant
    Outputs:
    -------
        E_coul: (float) free energy  
    """
    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184 

    phi   = coulomb_potential(q, p_per, Q, xq, E)
    dphi  = -1*coulomb_field(q, p_per, Q, xq, E)
    ddphi = coulomb_ddpotential(q, p_per, Q, xq, E)

    dummy1 = np.zeros((len(q)))        # dummy charges
    dummy2 = np.zeros((len(q),3,3))    # dummy quadrupoles

    phi += coulomb_potential_thole(p_pol, alpha, xq, E)
    dphi += -1*coulomb_field_thole(dummy1, p_pol, dummy2, alpha, xq, E)
    ddphi += coulomb_ddpotential_thole(p_pol, alpha, xq, E)

    cons = qe**2*Na*1e-3*1e10/(cal2J*E_0)
    E_coul = 0.5*cons*(np.sum(q*phi) + np.sum(np.sum(p_per*dphi,axis=1)) + np.sum(np.sum(np.sum(Q*ddphi,axis=2),axis=1))/6)

    return E_coul

def get_local_coordinates(vertices, point):
    """Get the local coordinates of the point in the cell."""
    origin = vertices[0]
    axes = [v - origin for v in vertices[1:]]
    tdim = 3
    if len(axes) == 2:
        axes.append(np.cross(axes[0], axes[1]))
        tdim = 2
    
    assert len(axes) == 3
    
    return np.linalg.solve(np.array(axes).T, point - origin)[:tdim]

def locate_cell(point, mesh, tree):

    '''
    Encuentra que tetrahedro contiene al punto ingresado

    point = Arreglo 3x1 con el punto buscado.
    mesh = Malla volumétrica donde buscar.

    Retorna:
        cell = N° del tetrahedro que contiene el punto
    '''
    cell_candidates = compute_collisions_points(tree, point)
    if len(compute_colliding_cells(mesh, cell_candidates, point).array)==0:
        return None
    else:
        cell = compute_colliding_cells(mesh, cell_candidates, point).array[0]

        return cell

def compute_reaction_potential(phi_rf, x_q, fem_space):
    """
    Compute the reaction potential in the multipoles location
    
    """
    N = x_q.shape[0]
    phi = np.zeros(N)
    mesh = fem_space.mesh

    tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)

    basix_el = fem_space.element.basix_element
    dofmap = fem_space.dofmap

    D000 = 0 # Indice de funciones de forma base

    for k in range(N):
        x = x_q[k]

        cell = locate_cell(x, mesh, tree)

        if cell == None:
            continue

        verts_idx = mesh.geometry.dofmap[cell] # dof del tetrahedro que contiene el multipolo
        verts = np.array([mesh.geometry.x[i] for i in verts_idx], dtype=np.float64)  # Coordenadas de los vertices. (4,3)
        xi = get_local_coordinates(verts, x)  # Coordenadas locales de los vertices (4,3)

        tab = basix_el.tabulate(0, np.array([xi], dtype=np.float64)) # Funciones de forma

        shape_fun = tab[D000, 0, :, 0]
        dofs = fem_space.dofmap.cell_dofs(cell)

        for d in range(len(dofs)):
            phi[k] += phi_rf[dofs[d]] * shape_fun[d]
            
    return phi

def compute_solvent_derivatives(fem_space, x_q, phi_rf):
    """
    
    Compute the first and second derivative of the reaction potential in the multipoles location
    
    """
    N     = x_q.shape[0]
    dphi  = np.zeros((N, 3))
    ddphi = np.zeros((N, 3, 3))
    mesh  = fem_space.mesh

    fem_ndof = fem_space.dofmap.index_map.size_global

    tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)

    basix_el = fem_space.element.basix_element
    dofmap = fem_space.dofmap

    # Indices de derivadas
    # (1,0,0)=1, (0,1,0)=2, (0,0,1)=3
    # (2,0,0)=4, (1,1,0)=5, (1,0,1)=6, (0,2,0)=7, (0,1,1)=8, (0,0,2)=9

    D100 = 1; D010 = 2; D001 = 3 # Indices de primeras derivadas
    D200 = 4; D110 = 5; D101 = 6; D020 = 7; D011 = 8; D002 = 9 # Indices de segundas derivadas

    for k in range(N):
        x = x_q[k]
        cell = locate_cell(x, mesh, tree)

        if cell == None:
            continue

        verts_idx = mesh.geometry.dofmap[cell] # dof del tetrahedro que contiene el multipolo       
        verts = np.array([mesh.geometry.x[i] for i in verts_idx], dtype=np.float64)  # Coordenadas de los vertices. (4,3)
        xi = get_local_coordinates(verts, x)  # Coordenadas locales de los vertices (4,3)

        tab = basix_el.tabulate(2, np.array([xi], dtype=np.float64)) # Funciones de forma hasta la segunda derivada
        dofs = fem_space.dofmap.cell_dofs(cell)

        #--- Calculo de Jacobiano ---#

        X0 = verts[0]
        J = np.column_stack((verts[1]-X0, verts[2]-X0, verts[3]-X0))  # Jacobiano. (3,3)
        invJ = np.linalg.inv(J) # J^-1
        invJT = invJ.T # J^-T

        g_ref = np.vstack([         # Valor de la primera derivada de las funciones de forma. (ndofs, 3)
            tab[D100, 0, :, 0],
            tab[D010, 0, :, 0],
            tab[D001, 0, :, 0],
        ])

        H_ref = np.zeros((3, 3, len(dofs)), dtype=np.float64)   # Segunda derivada de las funciones de forma. (ndofs, 3, 3)
        H_ref[0, 0, :] = tab[D200, 0, :, 0]
        H_ref[0, 1, :] = tab[D110, 0, :, 0]
        H_ref[0, 2, :] = tab[D101, 0, :, 0]
        H_ref[1, 0, :] = H_ref[0, 1, :]
        H_ref[1, 1, :] = tab[D020, 0, :, 0]
        H_ref[1, 2, :] = tab[D011, 0, :, 0]
        H_ref[2, 0, :] = H_ref[0, 2, :]
        H_ref[2, 1, :] = H_ref[1, 2, :]
        H_ref[2, 2, :] = tab[D002, 0, :, 0]

        g_phys = invJT @ g_ref  # (3, ndofs)
        H_phys = np.zeros_like(H_ref)
        for a in range(len(dofs)):
            H_phys[:, :, a] = invJT @ H_ref[:, :, a] @ invJ

        H_phys = np.transpose(H_phys, (2, 0, 1))
        for d in range(len(dofs)):
            dphi[k]  += phi_rf[dofs[d]] * g_phys.T[d, :]
            ddphi[k] += phi_rf[dofs[d]] * H_phys[d, :]

    return dphi, ddphi

def coulomb_polarizable_dipole(q, p_per, Q, alpha, xq, E, dtol):
    """
    Computes polarized dipole component of a collection of polarizabe multipoles
    Used in eq 56 of kirkwood multipole
    Inputs:
    ------
        q: array size N with charges of multipoles
        p_per: array size (Nx3) with permanent dipoles of multipoles
        Q: array size (Nx3x3) with quadrupoles of multipoles
        alpha: array size (Nqx3x3) with polarizability of dipoles (considered as a tensor)
        xq: array size Nx3 with positions of multipoles
        E : float, dielectric constant
    Returns:
    -------
        p_pol: array size (Nx3) with polarizable component of dipoles
        Efield: array size (Nx3) with electrostatic field that polarized the multipoles 
    """
    p_pol      = np.zeros((len(xq),3))
    dipole_diff= 1. 
    p_pol_prev = np.ones((len(xq),3))

    iteration = 0
    SOR = 0.7
    while dipole_diff>dtol:
        iteration += 1
        p_tot = p_per + p_pol
    
        Efield = coulomb_field_thole(q, p_tot, Q, alpha, xq, E)
        
        for k in range(len(q)):
            p_pol[k] = p_pol[k]*(1-SOR) + np.dot(alpha[k],4*np.pi*Efield[k])*SOR # 4*pi because alpha in Tinker
                                                                           # comes in atomic units that
                                                                           # that already include the 1/4pi

        dipole_diff = np.max(np.sqrt(np.sum((p_pol-p_pol_prev)**2, axis = 1)))
        p_pol_prev = p_pol.copy()

    print('Took %i iterations for vacuum induced dipole to converge'%iteration)
    return p_pol, Efield

def compute_trace_matrix(fem_space, bm_nodes):
    """
    Matriz T (Nv_bnd x Ndof_FEM) que extrae el valor nodal en vértices de frontera,
    con filas en el mismo orden que bm_nodes.
    """
    mesh = fem_space.mesh
    tdim = mesh.topology.dim

    bm_nodes = np.asarray(bm_nodes, dtype=np.int32)

    # Conectividades necesarias
    mesh.topology.create_connectivity(0, tdim)   # v -> cell
    mesh.topology.create_connectivity(tdim, 0)   # cell -> v

    v2c = mesh.topology.connectivity(0, tdim)
    c2v = mesh.topology.connectivity(tdim, 0)

    imap = fem_space.dofmap.index_map
    bs = fem_space.dofmap.index_map_bs
    num_fem_dofs = (imap.size_local + imap.num_ghosts) * bs

    dof_layout = fem_space.dofmap.dof_layout

    rows = np.empty(len(bm_nodes), dtype=np.int64)
    cols = np.empty(len(bm_nodes), dtype=np.int64)
    data = np.ones(len(bm_nodes), dtype=np.float64)

    for i, v in enumerate(bm_nodes):
        cells = v2c.links(int(v))
        if len(cells) == 0:
            raise RuntimeError(f"Vértice {v} no tiene celdas incidentes (¿id incorrecto?).")

        c = int(cells[0])
        cell_verts = c2v.links(c)

        # encontrar índice local del vértice dentro de la celda
        lv = int(np.where(cell_verts == v)[0][0])

        cell_dofs = fem_space.dofmap.cell_dofs(c)
        ldofs = dof_layout.entity_dofs(0, lv)  # dof(s) del vértice local lv
        if len(ldofs) != 1:
            raise RuntimeError("Esperaba 1 dof por vértice (Lagrange escalar).")

        dof = int(cell_dofs[ldofs[0]])

        rows[i] = i
        cols[i] = dof

    T = coo_matrix((data, (rows, cols)), shape=(len(bm_nodes), num_fem_dofs)).tocsc()
    return T

def gmres_callback(residual_norm):
        residuals.append(residual_norm)
        print(f"GMRES iter {len(residuals):4d} | residual = {residual_norm:.3e}")

def solvation_energy_sphere(x_q, q, d_scaled, Q_scaled, alpha, ep_in, ep_ex, kappa, external_mesh,
                            dtol=1e-3, maxiter=50, SOR=0.7, gmres_rtol=1e-5, gmres_atol=0., d_min=0.3,
                            d_max=2.0, h_min=0.2, h_max=0.5, probe_radius=1.4, stern_thickness=2.0,
                            algorithm=10):
    
    bohr = 0.52917721067 
    d = d_scaled * bohr
    Q = Q_scaled * 2 * bohr**2

    global residuals

    N = x_q.shape[0]
    mu = np.zeros((N,3))

    mesh, cell_tags, facet_tags = load_dolfin_mesh(external_mesh, d_min, d_max, h_min, h_max, probe_radius, stern_thickness, algorithm)
    #check_charge_mesh_distance(mesh, x_q, cell_tags=cell_tags, solute_marker=1, verbose=True)

    tdim = mesh.topology.dim
    assert tdim == 3
    fdim = tdim - 1

    mesh.topology.create_connectivity(0, tdim)
    mesh.topology.create_connectivity(tdim, 0)
    mesh.topology.create_entities(fdim)
    mesh.topology.create_connectivity(tdim, fdim)  # cell -> facet
    mesh.topology.create_connectivity(fdim, tdim)  # facet -> cell
    mesh.topology.create_connectivity(fdim, 0)     # facet -> vertex 
    mesh.topology.create_entity_permutations()

    ses_facets = facet_tags.find(1)
    stern_facets = facet_tags.find(2)

    solute_cells = cell_tags.find(1)
    stern_cells = cell_tags.find(2)

    ses_mesh, ses_nodes = build_boundary_mesh(mesh, ses_facets, cell_tags=cell_tags, inner_marker=1)
    stern_mesh, stern_nodes = build_boundary_mesh(mesh, stern_facets)

    fenics_space = fem.functionspace(mesh, ('Lagrange', 2)) # Espacio de funciones para FEM (H1)
    P1_space = fem.functionspace(mesh, ('Lagrange', 1)) # Espacio para la condición de Neumann
    dirichlet_space_ses = bempp_cl.api.function_space(ses_mesh, 'P', 1)
    neumann_space_ses = bempp_cl.api.function_space(ses_mesh, 'P', 1)
    dirichlet_space_stern = bempp_cl.api.function_space(stern_mesh, 'P', 1) # Espacios de funciones para BEM (H1/2)
    neumann_space_stern = bempp_cl.api.function_space(stern_mesh, 'P', 1)
    trace_matrix = compute_trace_matrix(fenics_space, stern_nodes)
    trace_ses = compute_trace_matrix(P1_space, ses_nodes)
    trace_stern = compute_trace_matrix(P1_space, stern_nodes)

    fem_ndof = fenics_space.dofmap.index_map.size_global
    bem_ndof = dirichlet_space_stern.global_dof_count
    total_ndof = fem_ndof + bem_ndof
    
    sol_prev = np.zeros(total_ndof)
    
    print("FEM dofs: {0}".format(fem_ndof))
    print("BEM dofs: {0}".format(bem_ndof))
    print("Total dofs: {0}".format(total_ndof))

    u = ufl.TrialFunction(fenics_space) # función de aproximación
    lam = ufl.TrialFunction(P1_space) # función P1 para los datos de frontera
    v = ufl.TestFunction(fenics_space) # Función de prueba

    Is = sparse.identity(dirichlet_space_ses, neumann_space_ses, neumann_space_ses)
    I = sparse.identity(dirichlet_space_stern, neumann_space_stern, neumann_space_stern)
    M = sparse.identity(neumann_space_stern, neumann_space_stern, dirichlet_space_stern)
    
    VL = laplace.single_layer(neumann_space_ses, dirichlet_space_ses, neumann_space_ses)
    KL = laplace.double_layer(dirichlet_space_ses, dirichlet_space_ses, neumann_space_ses)

    VYs = modified_helmholtz.single_layer(dirichlet_space_ses, dirichlet_space_ses, dirichlet_space_ses, kappa);
    KYs = modified_helmholtz.double_layer(neumann_space_ses, dirichlet_space_ses, dirichlet_space_ses, kappa);
        
    VY = modified_helmholtz.single_layer(dirichlet_space_stern, dirichlet_space_stern, dirichlet_space_stern, kappa);
    KY = modified_helmholtz.double_layer(neumann_space_stern, dirichlet_space_stern, dirichlet_space_stern, kappa);
    
    trace_op = LinearOperator(trace_matrix.shape, lambda x:trace_matrix * x)

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags) # cell_tags almacena ambos volumenes
    ds = ufl.Measure("dS", domain=mesh, subdomain_data=facet_tags) # En caso de necesitarlo, dS es la frontera interna
    ds_ext = ufl.Measure('ds', domain=mesh, subdomain_data=facet_tags) # ds es la frontera exterior, pinche fenics

    A = fenicsx.FenicsOperator(ep_in*(ufl.inner(ufl.nabla_grad(u), ufl.nabla_grad(v)))*dx(1) + ep_ex*(ufl.inner(ufl.nabla_grad(u), ufl.nabla_grad(v)))*dx(2)
                                + ep_ex*kappa*kappa*ufl.inner(u, v)*dx(2))

    mass = lam * v * ds_ext(2)
    mass_matrix = fem.petsc.assemble_matrix(fem.form(mass))
    mass_matrix.assemble()

    iv, jv, kv = mass_matrix.getValuesCSR()
    mass_matrix_sparse = scipy.sparse.csr_matrix((kv, jv, iv), shape=mass_matrix.getSize())

    blocks = [[None,None],[None,None]]
    
    blocks[0][0] = A.weak_form()
    blocks[0][1] = -ep_ex * mass_matrix_sparse @ trace_stern.T
    blocks[1][0] = (.5 * I - KY).weak_form() * trace_op
    blocks[1][1] = VY.weak_form()
    
    cterm_lhs = BlockedDiscreteOperator(np.array(blocks)) # Lado izquierdo de la ecuación
    
    P1 = diags(1./cterm_lhs[0,0].to_sparse().diagonal())
    
    P2 = InverseSparseDiscreteBoundaryOperator(
            bempp_cl.api.operators.boundary.sparse.identity(
            neumann_space_stern, neumann_space_stern, neumann_space_stern).weak_form())
    
    def apply_prec(x):
    
        m1 = P1.shape[0]
        m2 = P2.shape[0]
        n1 = P1.shape[1]
        n2 = P2.shape[1]
    
        res1 = P1.dot(x[:n1])
        res2 = P2.dot(x[n1:])
    
        return np.concatenate([res1, res2])
    
    p_shape = (P1.shape[0] + P2.shape[0], P1.shape[1] + P2.shape[1])
    P = LinearOperator(p_shape, apply_prec, dtype=np.dtype('float64'))

    #--- Parte invariable del lado derecho ---# 
    
    @bempp_cl.api.real_callable
    def charges_fun_perm(x, n, i, result):
        T2 = np.zeros((len(x_q),3,3))
        dist = x - x_q
        norm = np.sqrt(np.sum((dist*dist), axis = 1))
        T0 = 1/norm[:]
        T1 = np.transpose(dist.transpose()/norm**3)
        T2[:,:,:] = np.ones((len(x_q),3,3))[:]*dist.reshape((len(x_q),1,3))*np.transpose(np.ones((len(x_q),3,3))*dist.reshape((len(x_q),1,3)), (0,2,1))/norm.reshape((len(x_q),1,1))**5
        phi_c = np.sum(q[:]*T0[:]) + np.sum(T1[:]*d[:]) + 0.5*np.sum(np.sum(T2[:]*Q[:],axis=1))
        result[0] = phi_c/(4*np.pi*ep_in)
    
    G_fun_perm = bempp_cl.api.GridFunction(dirichlet_space_ses, fun = charges_fun_perm)

    @bempp_cl.api.real_callable
    def lambda_fun_perm(x, n, i, result):
        dist = x - x_q
        norm = np.sqrt(np.sum((dist*dist), axis = 1))
        dphi = np.zeros((3))
    
        T2 = np.zeros((3, 3, 3))
    
        for j in np.where(norm > 1e-10)[0]:
            T0 = -dist[j,:] / norm[j]**3    
            T1 = np.identity(3)/norm[j]**3 - 3*np.ones((3,3)) * dist[j,:] * np.transpose(np.ones((3,3)) * dist[j,:])/norm[j]**5
    
            aux = np.zeros((3,3,3))
    
            for k in range(3):
                aux[k,:,:] = np.ones((3,3)) * dist[j,:] * np.transpose(np.ones((3,3)) * dist[j,:])*dist[j,k]
            aux *= -5/norm[j]**7
    
            for k in range(3):
                aux[:,:,k] += np.identity(3) * dist[j,k] / norm[j]**5
    
            for k in range(3):
                aux[:,k,:] += np.identity(3) * dist[j,k] / norm[j]**5
    
            T2 = aux
    
            for k in range(3):
                dphi[k] += T0[k]*q[j] + np.sum(T1[k,:] * d[j,:]) + 0.5 * np.sum(np.sum(T2[k,:,:] * Q[j,:,:], axis = 1), axis = 0)
    
        dphi /=  (4*np.pi*ep_in)
        result[0] =  np.dot(n, dphi) # Derivada direccional en la dirección normal

    dGdn_fun_perm = bempp_cl.api.GridFunction(dirichlet_space_ses, fun = lambda_fun_perm)

    #--- RHS permanente componente armónica ---#
    
    hterm_rhs_perm = -(0.5*Is + KL) * G_fun_perm

    #--- RHS permanente componente de corrección ---#
    bem_rhs = np.zeros(bem_ndof)

    mass_ses = lam('+') * v('+') * ds(1)
    mass_ses_matrix = fem.petsc.assemble_matrix(fem.form(mass_ses))
    mass_ses_matrix.assemble()
    iv, jv, kv = mass_ses_matrix.getValuesCSR()
    mass_ses_sparse = scipy.sparse.csr_matrix((kv, jv, iv), shape=mass_ses_matrix.getSize())

    cterm_rhs_perm = -(ep_in) * dGdn_fun_perm  # elimine el /ep_ex # Lo volví a incluir porque como no va a ir po felipe # no va po, estaba bien al principio
    neumann_values_perm = cterm_rhs_perm.coefficients

    fem_rhs_perm = mass_ses_sparse @ (trace_ses.T @ neumann_values_perm)

    for iter_number in range(maxiter):
        print(F"------- Dipole iteration {iter_number + 1} -------")

        #--- 1. Resolver la ecuación para la componente armónica ---#

        @bempp_cl.api.real_callable
        def charges_fun_var(x, n, i, result):
            dist = x - x_q
            norm = np.sqrt(np.sum((dist*dist), axis = 1))
            T1 = np.transpose(dist.transpose()/norm**3)
            phi_c = np.sum(T1[:]*mu[:]) # Solo considera la componente polarizable, la componente permanente se considera en el lado constante del RHS
            result[0] = phi_c/(4*np.pi*ep_in)

        G_fun_var = bempp_cl.api.GridFunction(dirichlet_space_ses, fun = charges_fun_var)
        hterm_rhs_var = -(0.5*Is + KL) * G_fun_var
        hterm_rhs = hterm_rhs_perm + hterm_rhs_var
        dphi0dn, info = bempp_cl.api.linalg.cg(VL, hterm_rhs, tol=1e-5)

        #--- 2. Resolver el sistema para la componente de corrección ---#

        @bempp_cl.api.real_callable
        def lambda_fun_var(x, n, i, result):
            dist = x - x_q
            norm = np.sqrt(np.sum((dist*dist), axis = 1))
            dphi = np.zeros((3))
        
            for j in np.where(norm > 1e-10)[0]:  
                T1 = np.identity(3)/norm[j]**3 - 3*np.ones((3,3)) * dist[j,:] * np.transpose(np.ones((3,3)) * dist[j,:])/norm[j]**5
        
                for k in range(3):
                    dphi[k] += np.sum(T1[k,:] * mu[j,:])
        
            dphi /=  (4*np.pi*ep_in)
            result[0] =  np.dot(n, dphi) # Derivada direccional en la dirección normal

        dGdn_fun_var = bempp_cl.api.GridFunction(dirichlet_space_ses, fun = lambda_fun_var)
        
        cterm_rhs_var = -(ep_in)*(dGdn_fun_var + dphi0dn) # eliminé el /ep_ex # Lo volví a incluir porque como no va a ir po felipe # no va po, estaba bien al principio
        neumann_values_var = cterm_rhs_var.coefficients
        #neumann_values = neumann_values_perm + cterm_rhs_var.coefficients

        fem_rhs_var = mass_ses_sparse @ (trace_ses.T @ neumann_values_var)
        fem_rhs_array = fem_rhs_perm + fem_rhs_var
        cterm_rhs = np.concatenate([fem_rhs_array, bem_rhs])

        #g = fem.Function(P1_space)
        #values = g.x.array
        #values[:] = trace_ses.T @ neumann_values
        #g.x.scatter_forward()
        #L = f * v * dx + g('+') * v('+') * ds(1)
        #fem_rhs = fem.petsc.assemble_vector(fem.form(L))
        #fem_rhs.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD_VALUES, mode=petsc4py.PETSc.ScatterMode.REVERSE)
        #fem_rhs_array = fem_rhs.array # Condición de Neumann impuesta de forma debil (opción 1 del apunte)

        #fem_rhs = trace_ses.T @ neumann_values # aplicar directamente los valores a los nodos (opción 2 del apunte)

        residuals = []

        sol, info = gmres(
                    cterm_lhs,
                    cterm_rhs,
                    x0 = sol_prev,
                    rtol=gmres_rtol,
                    atol=gmres_atol,
                    M=P,
                    callback=gmres_callback,
                    callback_type="pr_norm",
                    restart=500,
                    maxiter=2000
                    )

        residual_real = np.linalg.norm(cterm_lhs @ sol - cterm_rhs) / np.linalg.norm(cterm_rhs)
        print(f"Residual relativo real: {residual_real:.3e}")
        
        sol_prev = sol.copy()
        
        phi_hat  = sol[:fem_ndof] # Componente de corrección en la ubicación de los dofs
        dphi_hat = sol[fem_ndof:]
        
        #--- 3. Determinar phi0 en la ubicación de los dofs y multipolos ---#
        
        G_fun = G_fun_perm + G_fun_var
        
        x_dofs = fenics_space.tabulate_dof_coordinates() # ndof x 3
        
        slpo = single_layer(neumann_space_ses, np.transpose(x_dofs))
        dlpo = double_layer(dirichlet_space_ses, np.transpose(x_dofs))
        phi0_dofs = slpo * dphi0dn + dlpo * G_fun # Componente armónica en los dofs
        
        slpo = single_layer(neumann_space_ses, np.transpose(x_q))
        dlpo = double_layer(dirichlet_space_ses, np.transpose(x_q))        
        phi0_mult = slpo * dphi0dn + dlpo * G_fun # Componente armónica en los multipolos
        
        #--- 4. Determinar la componente de corrección en la ubicación de los multipolos ---#
        
        phi_hat_mult = compute_reaction_potential(phi_hat, x_q, fenics_space)
        
        #--- 5. Se calcula el potencial de reacción en los dofs y multipolos ---#
        
        phi_rf_dofs = phi_hat + phi0_dofs[0]
        phi_rf_mult = phi_hat_mult + phi0_mult[0]
        
        #--- 6. Calcular derivadas con funciones de forma ---#
        
        dphi_rf, ddphi_rf = compute_solvent_derivatives(fenics_space, x_q, phi_rf_dofs)
        
        #--- Calculo de la componente variable del campo eléctrico ---#
        
        p = d + mu # Dipolo total
        
        #E_mult_pol  = coulomb_field_thole(np.zeros(N), mu, np.zeros((N, 3, 3)), alpha, x_q, ep_in)
        #E_mult      = E_mult_perm + E_mult_pol
        E_mult      = coulomb_field_thole(q, p, Q, alpha, x_q, ep_in)
        E_total     = E_mult - dphi_rf # Signo (-) porque E = -nabla phi
        
        #--- Comprobar convergencia ---#
        
        mu_prev = mu.copy()
        for i in range(N):
            mu[i] = mu[i]*(1-SOR) + np.dot(alpha[i], E_total[i]*4*np.pi)*SOR
        
        dipole_diff = np.max(np.sqrt(np.sum((mu-mu_prev)**2, axis = 1)))
        if dipole_diff < dtol:
            print(F"The result has converged in {iter_number+1} iterations")
            break
        print(F"Induced dipole residual: {dipole_diff}")
        
    G_diss_solv = solvation_energy(q, d, Q, phi_rf_mult, dphi_rf, ddphi_rf)
    print(F"Solvent contributión: {G_diss_solv}")
    G_diss_mult = coulomb_energy_multipole(q, d, mu, Q, alpha, x_q, ep_in)
    print(F"Multipoles contributión: {G_diss_mult}")
        
    #--- Cálculo de G_vacc ---#
        
    p_pol_vac, Epol_vac = coulomb_polarizable_dipole(q, d, Q, alpha, x_q, ep_in, dtol)
    G_vacc = coulomb_energy_multipole(q, d, p_pol_vac, Q, alpha, x_q, ep_in)
    print(F"Coulomb vacuum energy: {G_vacc}")
        
    total_energy = G_diss_solv + G_diss_mult - G_vacc
    print(F"Total solvation energy: {total_energy} [kcal/Mol]")
        
    return total_energy, total_ndof