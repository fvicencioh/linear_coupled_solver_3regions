import numpy as np
import gmsh
import dolfinx
import ufl
import bempp_cl.api
import time
import scipy.sparse
import os
import glob

from bempp_cl.api.external import fenicsx
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio
from mpi4py import MPI

from dolfinx.geometry import compute_colliding_cells, compute_collisions_points
from scipy.sparse import coo_matrix, diags
from petsc4py import PETSc

from coupled_solver.utils.generate_mesh import load_dolfin_mesh
from coupled_solver.utils.generate_mesh import build_boundary_mesh
from coupled_solver.utils.get_data import read_tinker
from coupled_solver.bem.direct import solvation_energy_solvent, get_coulomb_energy, compute_induced_dipole
from coupled_solver.coupling.johnson_nedelec import _compute_trace_matrix
from coupled_solver.fem.fem_assembly import _compute_reaction_potential, _compute_solvent_derivatives

from bempp_cl.api.operators.boundary import sparse, laplace, modified_helmholtz
from bempp_cl.api.assembly.blocked_operator import BlockedDiscreteOperator
from bempp_cl.api.assembly.discrete_boundary_operator import InverseSparseDiscreteBoundaryOperator
from bempp_cl.api.operators.potential.laplace import single_layer, double_layer

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres

def precon(lhs, neumann_space):

    P1 = diags(1./lhs[0,0].to_sparse().diagonal())

    P2 = InverseSparseDiscreteBoundaryOperator(
            bempp_cl.api.operators.boundary.sparse.identity(
            neumann_space, neumann_space, neumann_space).weak_form())

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
    return P

def gmres_callback(residual_norm):
    residuals.append(residual_norm)
    print(f"GMRES iter {len(residuals):4d} | residual = {residual_norm:.3e}")

def linear_solver(file, ep_in, ep_ex, k, maxiter=100, gmrs_maxiter=2000, mu='None', assembler='dense',
                       dtol=1e-3, gmres_rtol=1e-5, gmres_atol=0., external_mesh='None', SOR = 0.7,
                       gradation=0.1, probe_radius=1.4, stern_thickness=3.0,
                       algorithm=10, grid_scale=2.0, new_mesh=False):

    from coupled_solver import dir_name
    global residuals
    mu_flag = False
    time_init = time.time()

    x_q, q, d, Q, alpha, mass, polar_group, thole, \
               connections_12, connections_13, \
               pointer_connections_12, pointer_connections_13, \
               p12scale, p13scale, N = read_tinker('molecules/'+file, float)
    
    if external_mesh != 'None':
        mesh, cell_tags, facet_tags = load_dolfin_mesh(external_mesh, gradation, probe_radius, stern_thickness, algorithm, grid_scale)
    else:
        try:
            mol_name = file.split('/')[-1]
            mesh_path = os.path.join(dir_name, 'volumetric_mesh', f'{mol_name}.msh')
            boundary_mesh_path = os.path.join(dir_name, 'boundary_mesh', f'{mol_name}*')
            if new_mesh==True: 
                if os.path.exists(mesh_path):
                    os.remove(mesh_path)
                for archivo in glob.glob(boundary_mesh_path):
                    if os.path.isfile(archivo):
                        os.remove(archivo)
            mesh, cell_tags, facet_tags = load_dolfin_mesh(mol_name, gradation, probe_radius, stern_thickness, algorithm, grid_scale)
        except:
            raise ValueError(F'No se encontro malla {mol_name} ni archivos para generarla')

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

    #--- Dipolo inicial ---#

    if mu == 'None':
        mu = np.zeros((N,3))
    
    if mu == 'vacuum':
        mu_flag = True
        p_pol_vacc = np.zeros((N,3))
        dipole_diff_vacc = 1.
        iteration = 0
        while dipole_diff_vacc>dtol:
            
            iteration += 1
    
            p_pol_prev = p_pol_vacc.copy()
    
            p_pol_vacc = compute_induced_dipole(x_q, q, d, p_pol_vacc, Q, alpha, thole, polar_group,
                                                connections_12, pointer_connections_12, connections_13,
                                                pointer_connections_13, np.zeros((N,3)), ep_in)
    
            dipole_diff_vacc = np.max(np.sqrt(np.sum((p_pol_vacc-p_pol_prev)**2, axis = 1)))
    
            print(F"Induced dipole residual in vacuum: {dipole_diff_vacc}")
    
        print(F"{iteration} iterations for vacuum induced dipole to converge")
    
        G_vacc = get_coulomb_energy(x_q, q, d, p_pol_vacc, Q, alpha, ep_in, thole, polar_group, 
                                    connections_12, connections_13, pointer_connections_12, 
                                    pointer_connections_13, p12scale, p13scale)
        mu = p_pol_vacc.copy()
    if mu.shape != (N,3):
        mu = np.zeros((N,3))

    #--- Definir espacios funcionales ---#

    time_init_spaces = time.time()

    fenics_space = fem.functionspace(mesh, ('Lagrange', 2)) # Espacio de funciones para FEM (H1)
    P1_space = fem.functionspace(mesh, ('Lagrange', 1)) # Espacio para la condición de Neumann
    dirichlet_space_ses = bempp_cl.api.function_space(ses_mesh, 'P', 1)
    neumann_space_ses = bempp_cl.api.function_space(ses_mesh, 'P', 1)
    dirichlet_space_stern = bempp_cl.api.function_space(stern_mesh, 'P', 1) # Espacios de funciones para BEM (H1/2)
    neumann_space_stern = bempp_cl.api.function_space(stern_mesh, 'P', 1)
    trace_matrix = _compute_trace_matrix(fenics_space, stern_nodes)
    trace_ses = _compute_trace_matrix(P1_space, ses_nodes)
    trace_stern = _compute_trace_matrix(P1_space, stern_nodes)

    time_final_spaces = time.time()

    print(f'Time to define spaces: {(time_final_spaces - time_init_spaces):.2f} [s]')

    fem_ndof = fenics_space.dofmap.index_map.size_global
    bem_ndof = dirichlet_space_stern.global_dof_count
    total_ndof = fem_ndof + bem_ndof

    sol_prev = np.zeros(total_ndof)

    print("FEM dofs: {0}".format(fem_ndof))
    print("BEM dofs: {0}".format(bem_ndof))
    print("Total dofs: {0}".format(total_ndof))

    #--- Operadores potenciales ---#

    x_dofs = fenics_space.tabulate_dof_coordinates() # ndof x 3
    solute_dofs = set() # Para almacenar los dof del volumen 1 sin repetir.
    for cell_idx in solute_cells:
        cell_ndofs = fenics_space.dofmap.cell_dofs(cell_idx)
        for dof in cell_ndofs:
            solute_dofs.add(dof)
    solute_dofs = list(solute_dofs)
    x_dofs_solute = x_dofs[solute_dofs]

    phi0_dofs = np.zeros(fem_ndof)

    time_init_pot = time.time()
    
    slpo_dofs = single_layer(neumann_space_ses, np.transpose(x_dofs_solute))
    dlpo_dofs = double_layer(dirichlet_space_ses, np.transpose(x_dofs_solute))

    slpo_mult = single_layer(neumann_space_ses, np.transpose(x_q))
    dlpo_mult = double_layer(dirichlet_space_ses, np.transpose(x_q))

    time_final_pot = time.time()

    print(f'Time to define potential operators: {(time_final_pot - time_init_pot):.2f} [s]')

    #--- Lado izquierdo de la ecuación (Constante) ---#

    time_init_operators = time.time()

    u = ufl.TrialFunction(fenics_space) # función de aproximación
    v = ufl.TestFunction(fenics_space) # Función de prueba
    lam = ufl.TrialFunction(P1_space) # función P1 para los datos de frontera

    VL = laplace.single_layer(neumann_space_ses, dirichlet_space_ses, neumann_space_ses)
    KL = laplace.double_layer(dirichlet_space_ses, dirichlet_space_ses, neumann_space_ses)

    Is = sparse.identity(dirichlet_space_ses, neumann_space_ses, neumann_space_ses) # Espacios sobre la SES
    #VYs = modified_helmholtz.single_layer(dirichlet_space_ses, dirichlet_space_ses, dirichlet_space_ses, k);
    #KYs = modified_helmholtz.double_layer(neumann_space_ses, dirichlet_space_ses, dirichlet_space_ses, k);

    I = sparse.identity(dirichlet_space_stern, neumann_space_stern, neumann_space_stern) # Espacios sobre la capa stern
    M = sparse.identity(neumann_space_stern, neumann_space_stern, dirichlet_space_stern)
    VY = modified_helmholtz.single_layer(dirichlet_space_stern, dirichlet_space_stern, dirichlet_space_stern, k); 
    KY = modified_helmholtz.double_layer(neumann_space_stern, dirichlet_space_stern, dirichlet_space_stern, k);

    trace_op = LinearOperator(trace_matrix.shape, lambda x:trace_matrix * x)

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags) # cell_tags almacena ambos volumenes
    ds_int = ufl.Measure("dS", domain=mesh, subdomain_data=facet_tags) # dS es el diferencial sobre la frontera interna
    ds_ext = ufl.Measure('ds', domain=mesh, subdomain_data=facet_tags) # ds es la frontera exterior

    A = fenicsx.FenicsOperator(ep_in*(ufl.inner(ufl.nabla_grad(u), ufl.nabla_grad(v)))*dx(1) + ep_ex*(ufl.inner(ufl.nabla_grad(u), ufl.nabla_grad(v)))*dx(2)
                                    + ep_ex*k*k*ufl.inner(u, v)*dx(2))

    mass = lam * v * ds_ext(2)
    mass_matrix = fem.petsc.assemble_matrix(fem.form(mass))
    mass_matrix.assemble()

    iv, jv, kv = mass_matrix.getValuesCSR()
    mass_matrix_sparse = scipy.sparse.csr_matrix((kv, jv, iv), shape=mass_matrix.getSize())

    time_final_operators = time.time()

    print(f'Time to define boundary and FEM operators: {(time_final_operators - time_init_operators):.2f} [s]')

    time_init_lhs = time.time()

    blocks = [[None,None],[None,None]]

    blocks[0][0] = A.weak_form()
    blocks[0][1] = -ep_ex * mass_matrix_sparse @ trace_stern.T
    blocks[1][0] = (.5 * I - KY).weak_form() * trace_op
    blocks[1][1] = VY.weak_form()

    cterm_lhs = BlockedDiscreteOperator(np.array(blocks)) # Lado izquierdo de la ecuación

    time_final_lhs = time.time()
    print(f'Time to assemble LHS: {(time_final_lhs - time_init_lhs):.2f} [s]')
    P = precon(cterm_lhs, neumann_space_stern)

    #--- Parte constante del lado derecho ---#
    #--- Grid functions ---#

    time_init_rhs = time.time()

    #@bempp_cl.api.real_callable
    #def charges_fun_perm(x, n, i, result):
    #    T2 = np.zeros((len(x_q),3,3))
    #    dist = x - x_q
    #    norm = np.sqrt(np.sum((dist*dist), axis = 1))
    #    T0 = 1/norm[:]
    #    T1 = np.transpose(dist.transpose()/norm**3)
    #    T2[:,:,:] = np.ones((len(x_q),3,3))[:]*dist.reshape((len(x_q),1,3))*np.transpose(np.ones((len(x_q),3,3))*dist.reshape((len(x_q),1,3)), (0,2,1))/norm.reshape((len(x_q),1,1))**5
    #    phi_c = np.sum(q[:]*T0[:]) + np.sum(T1[:]*d[:]) + 0.5*np.sum(np.sum(T2[:]*Q[:],axis=1))
    #    result[0] = phi_c/(4*np.pi*ep_in)
    @bempp_cl.api.real_callable
    def charges_fun_perm(x, n, i, result):
        phi_c = 0.0
        for j in range(len(x_q)):
            dx = x[0] - x_q[j, 0]
            dy = x[1] - x_q[j, 1]
            dz = x[2] - x_q[j, 2]
            norm = np.sqrt(dx*dx + dy*dy + dz*dz)
            if norm > 1e-10:
                T0 = 1/norm
                phi_c += q[j] * T0
                
                T1_x = dx/(norm**3)
                T1_y = dy/(norm**3)
                T1_z = dz/(norm**3)
                phi_c += T1_x*d[j,0] + T1_y*d[j,1] + T1_z*d[j,2]
                
                T2_xx = (dx * dx)/(2*norm**5)
                T2_xy = (dx * dy)/(2*norm**5)
                T2_xz = (dx * dz)/(2*norm**5)
                T2_yx = (dy * dx)/(2*norm**5)
                T2_yy = (dy * dy)/(2*norm**5)
                T2_yz = (dy * dz)/(2*norm**5)
                T2_zx = (dz * dx)/(2*norm**5)
                T2_zy = (dz * dy)/(2*norm**5)
                T2_zz = (dz * dz)/(2*norm**5)
                phi_c += T2_xx*Q[j,0,0] + T2_xy*Q[j,0,1] + T2_xz*Q[j,0,2]
                phi_c += T2_yx*Q[j,1,0] + T2_yy*Q[j,1,1] + T2_yz*Q[j,1,2]
                phi_c += T2_zx*Q[j,2,0] + T2_zy*Q[j,2,1] + T2_zz*Q[j,2,2]
        result[0] = phi_c/(4*np.pi*ep_in)

    G_fun_perm = bempp_cl.api.GridFunction(dirichlet_space_ses, fun = charges_fun_perm)

    #@bempp_cl.api.real_callable
    #def lambda_fun_perm(x, n, i, result):
    #    dist = x - x_q
    #    norm = np.sqrt(np.sum((dist*dist), axis = 1))
    #    dphi = np.zeros((3))
    #
    #    T2 = np.zeros((3, 3, 3))
    #
    #    for j in np.where(norm > 1e-10)[0]:
    #        T0 = -dist[j,:] / norm[j]**3    
    #        T1 = np.identity(3)/norm[j]**3 - 3*np.ones((3,3)) * dist[j,:] * np.transpose(np.ones((3,3)) * dist[j,:])/norm[j]**5
    #
    #        aux = np.zeros((3,3,3))
    #
    #        for k in range(3):
    #            aux[k,:,:] = np.ones((3,3)) * dist[j,:] * np.transpose(np.ones((3,3)) * dist[j,:])*dist[j,k]
    #        aux *= -5/norm[j]**7
    #
    #        for k in range(3):
    #            aux[:,:,k] += np.identity(3) * dist[j,k] / norm[j]**5
    #
    #        for k in range(3):
    #            aux[:,k,:] += np.identity(3) * dist[j,k] / norm[j]**5
    #
    #        T2 = aux
    #
    #        for k in range(3):
    #            dphi[k] += T0[k]*q[j] + np.sum(T1[k,:] * d[j,:]) + 0.5 * np.sum(np.sum(T2[k,:,:] * Q[j,:,:], axis = 1), axis = 0)
    #
    #    dphi /=  (4*np.pi*ep_in)
    #    result[0] =  np.dot(n, dphi) # Derivada direccional en la dirección normal

    @bempp_cl.api.real_callable
    def lambda_fun_perm(x, n, i, result):
        dphi_x = 0.0
        dphi_y = 0.0
        dphi_z = 0.0
        for j in range(len(x_q)):
            dx = x[0] - x_q[j, 0]
            dy = x[1] - x_q[j, 1]
            dz = x[2] - x_q[j, 2]
            norm = np.sqrt(dx*dx + dy*dy + dz*dz)
            if norm > 1e-10:
                r2 = norm**2
                r3 = norm**3
                r5 = r3 * r2
                r7 = r5 * r2

                T0_x = -dx/r3
                T0_y = -dy/r3
                T0_z = -dz/r3
                dphi_x += T0_x * q[j]
                dphi_y += T0_y * q[j]
                dphi_z += T0_z * q[j]

                dot_rd = dx*d[i,0] + dy*d[i,1] + dz*d[i,2]
                T1_x = d[i, 0]/r3 - 3.0 * dx * dot_rd / r5
                T1_y = d[i, 1]/r3 - 3.0 * dy * dot_rd / r5
                T1_z = d[i, 2]/r3 - 3.0 * dz * dot_rd / r5

                dphi_x += T1_x
                dphi_y += T1_y
                dphi_z += T1_z

                Q_rr = (dx*dx*Q[j,0,0] + dx*dy*Q[j,0,1] + dx*dz*Q[j,0,2] +
                        dy*dx*Q[j,1,0] + dy*dy*Q[j,1,1] + dy*dz*Q[j,1,2] +
                        dz*dx*Q[j,2,0] + dz*dy*Q[j,2,1] + dz*dz*Q[j,2,2])

                Q_rx = dx * Q[j,0,0] + dy * Q[j,0,1] + dz * Q[j,0,2]
                Q_ry = dx * Q[j,1,0] + dy * Q[j,1,1] + dz * Q[j,1,2]
                Q_rz = dx * Q[j,2,0] + dy * Q[j,2,1] + dz * Q[j,2,2]

                T2_x = Q_rx/r5 - 2.5 * dx * Q_rr / r7
                T2_y = Q_ry/r5 - 2.5 * dy * Q_rr / r7
                T2_z = Q_rz/r5 - 2.5 * dz * Q_rr / r7

                dphi_x += T2_x
                dphi_y += T2_y
                dphi_z += T2_z

        dphi_x /= (4.0 * np.pi * ep_in)
        dphi_y /= (4.0 * np.pi * ep_in)
        dphi_z /= (4.0 * np.pi * ep_in)
        result[0] = n[0]*dphi_x + n[1]*dphi_y + n[2]*dphi_z

    dGdn_fun_perm = bempp_cl.api.GridFunction(dirichlet_space_ses, fun = lambda_fun_perm)

    #--- RHS componente armónica ---#

    hterm_rhs_perm = -(0.5*Is + KL) * G_fun_perm

    #--- RHS componente de corrección ---#

    bem_rhs = np.zeros(bem_ndof)
    mass_ses = lam('+') * v('+') * ds_int(1)
    mass_ses_matrix = fem.petsc.assemble_matrix(fem.form(mass_ses))
    mass_ses_matrix.assemble()
    iv, jv, kv = mass_ses_matrix.getValuesCSR()
    mass_ses_sparse = scipy.sparse.csr_matrix((kv, jv, iv), shape=mass_ses_matrix.getSize())

    cterm_rhs_perm = -(ep_in) * dGdn_fun_perm

    neumann_values_perm = cterm_rhs_perm.coefficients
    fem_rhs_perm = mass_ses_sparse @ (trace_ses.T @ neumann_values_perm)

    time_final_rhs = time.time()

    print(f'Time to assembly permanent part of the RHS: {(time_final_rhs - time_init_rhs):.2f} [s]')

    for iter_number in range(maxiter):
        print(F"------- Dipole iteration {iter_number + 1} -------")
    
        #--- 1. Resolver la ecuación para la componente armónica ---#

        #@bempp_cl.api.real_callable
        #def charges_fun_var(x, n, i, result):
        #    dist = x - x_q
        #    norm = np.sqrt(np.sum((dist*dist), axis = 1))
        #    T1 = np.transpose(dist.transpose()/norm**3)
        #    phi_c = np.sum(T1[:]*mu[:]) # Solo considera la componente polarizable, la componente permanente se considera en el lado constante del RHS
        #    result[0] = phi_c/(4*np.pi*ep_in)
        @bempp_cl.api.real_callable
        def charges_fun_var(x, n, i, result):
            phi_c = 0.0
            for j in range(len(x_q)):
                dx = x[0] - x_q[j, 0]
                dy = x[1] - x_q[j, 1]
                dz = x[2] - x_q[j, 2]
                norm = np.sqrt(dx*dx + dy*dy + dz*dz)
                if norm > 1e-10:
            
                    T1_x = dx/(norm**3)
                    T1_y = dy/(norm**3)
                    T1_z = dz/(norm**3)
                    phi_c += T1_x*mu[j,0] + T1_y*mu[j,1] + T1_z*mu[j,2]
            
            result[0] = phi_c/(4*np.pi*ep_in)

        G_fun_var = bempp_cl.api.GridFunction(dirichlet_space_ses, fun = charges_fun_var)
        hterm_rhs_var = -(0.5*Is + KL) * G_fun_var
        hterm_rhs = hterm_rhs_perm + hterm_rhs_var
        dphi0dn, info = bempp_cl.api.linalg.cg(VL, hterm_rhs, tol=gmres_rtol) # Considera la misma tolerancia que gmres

        #--- 2. Resolver el sistema para la componente de corrección ---#

        #@bempp_cl.api.real_callable
        #def lambda_fun_var(x, n, i, result):
        #    dist = x - x_q
        #    norm = np.sqrt(np.sum((dist*dist), axis = 1))
        #    dphi = np.zeros((3))
        #    
        #    for j in np.where(norm > 1e-10)[0]:  
        #        T1 = np.identity(3)/norm[j]**3 - 3*np.ones((3,3)) * dist[j,:] * np.transpose(np.ones((3,3)) * dist[j,:])/norm[j]**5
        #    
        #        for k in range(3):
        #            dphi[k] += np.sum(T1[k,:] * mu[j,:])
        #    
        #    dphi /=  (4*np.pi*ep_in)
        #    result[0] =  np.dot(n, dphi) # Derivada direccional en la dirección normal
        @bempp_cl.api.real_callable
        def lambda_fun_var(x, n, i, result):
            dphi_x = 0.0
            dphi_y = 0.0
            dphi_z = 0.0
            for j in range(len(x_q)):
                dx = x[0] - x_q[j, 0]
                dy = x[1] - x_q[j, 1]
                dz = x[2] - x_q[j, 2]
                norm = np.sqrt(dx*dx + dy*dy + dz*dz)
                if norm > 1e-10:
                    r3 = norm**3
                    r5 = r3 * norm**2
            
                    dot_rd = dx*mu[i,0] + dy*mu[i,1] + dz*mu[i,2]
                    T1_x = mu[i, 0]/r3 - 3.0 * dx * dot_rd / r5
                    T1_y = mu[i, 1]/r3 - 3.0 * dy * dot_rd / r5
                    T1_z = mu[i, 2]/r3 - 3.0 * dz * dot_rd / r5
            
                    dphi_x += T1_x
                    dphi_y += T1_y
                    dphi_z += T1_z
            
            dphi_x /= (4.0 * np.pi * ep_in)
            dphi_y /= (4.0 * np.pi * ep_in)
            dphi_z /= (4.0 * np.pi * ep_in)
            result[0] = n[0]*dphi_x + n[1]*dphi_y + n[2]*dphi_z
            
        dGdn_fun_var = bempp_cl.api.GridFunction(dirichlet_space_ses, fun = lambda_fun_var)
            
        cterm_rhs_var = -(ep_in)*(dGdn_fun_var + dphi0dn) # eliminé el /ep_ex # Lo volví a incluir porque como no va a ir po felipe # no va po, estaba bien al principio
        neumann_values_var = cterm_rhs_var.coefficients
            
        fem_rhs_var = mass_ses_sparse @ (trace_ses.T @ neumann_values_var)
        fem_rhs_array = fem_rhs_perm + fem_rhs_var
        cterm_rhs = np.concatenate([fem_rhs_array, bem_rhs])

        residuals = []

        time_gmres_init = time.time()
            
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

        time_gmres_final = time.time()
            
        residual_real = np.linalg.norm(cterm_lhs @ sol - cterm_rhs) / np.linalg.norm(cterm_rhs)
        print(f"Residual relativo real: {residual_real:.3e}")
        print(F"Compution time for linear system: {time_gmres_final - time_gmres_init} [s]")

        sol_prev = sol.copy()

        phi_hat  = sol[:fem_ndof] # Componente de corrección en la ubicación de los dofs
        #dphi_hat = sol[fem_ndof:] # No necesito esta parte de la solución

        #--- 3. Determinar phi_0 en la ubicación de los dofs y multipolos ---#

        time_g0_init = time.time()

        G_fun = G_fun_perm + G_fun_var

        phi0_dofs_solvent = slpo_dofs * dphi0dn + dlpo_dofs * G_fun # Componente armónica en los dofs
        phi0_dofs[solute_dofs] = phi0_dofs_solvent[0]
       
        #phi0_mult = slpo_mult * dphi0dn + dlpo_mult * G_fun # Componente armónica en los multipolos
        # No necesito el valor sobre los multipolos en el esquema iterativo
        time_g0_final = time.time()
        print(F"Harmonic component computation time for the DOFs: {time_g0_final - time_g0_init} [s]")

        #--- 4. Determinar la componente de corrección en la ubicación de los multipolos ---#
        # No necesito esto en el esquema iterativo, solo para el calculo de energía
        #time_cterm_init = time.time()
        
        #phi_hat_mult = _compute_reaction_potential(phi_hat, x_q, fenics_space)

        #time_cterm_final = time.time()
        #print(F"Correction term computation time for the mmultipoles: {time_cterm_final - time_cterm_init} [s]")

        #--- 5. Se calcula el potencial de reacción en los dofs y multipolos ---#

        phi_rf_dofs = phi_hat + phi0_dofs
        #phi_rf_mult = phi_hat_mult + phi0_mult[0]
        # Same caso anterior, no hay necesidad de tener esto dentro del esquema iterativo
        #--- 6. Calcular derivadas con funciones de forma ---#

        time_der_init = time.time()

        dphi_rf, ddphi_rf = _compute_solvent_derivatives(fenics_space, x_q, phi_rf_dofs)

        time_der_final = time.time()
        print(F"Computation time for calculating the derivative of the reaction potential: {time_der_final - time_der_init} [s]")

        mu_prev = mu.copy()
        time_mu_init = time.time()

        mu = compute_induced_dipole(x_q, q, d, mu, Q, alpha, thole, polar_group,
                                    connections_12, pointer_connections_12, connections_13,
                                    pointer_connections_13, dphi_rf, ep_in)

        time_mu_final = time.time()

        print(F"Compution time for induced dipole: {time_mu_final - time_mu_init} [s]")

        dipole_diff = np.max(np.sqrt(np.sum((mu-mu_prev)**2, axis = 1)))

        if dipole_diff<dtol:
            print(F"The induced dipole in dissolved state has converged in {iter_number+1} iterations")
            break

        print(F"Induced dipole residual: {dipole_diff}")

    # Ahora si necesito el valor sobre los multipolos
    phi_hat_mult = _compute_reaction_potential(phi_hat, x_q, fenics_space)
    phi0_mult = slpo_mult * dphi0dn + dlpo_mult * G_fun
    phi_rf_mult = phi_hat_mult + phi0_mult[0]

    G_diss_solv = solvation_energy_solvent(q, d, Q, phi_rf_mult, dphi_rf, ddphi_rf)
    time_energy_init = time.time()
    G_diss_mult = get_coulomb_energy(x_q, q, d, mu, Q, alpha, ep_in, thole, polar_group, connections_12, \
                                         connections_13, pointer_connections_12, pointer_connections_13, 
                                         p12scale, p13scale)

    time_energy_final = time.time()
    #--- Induced dipole in Vacuum ---#
    if not mu_flag:
        p_pol_vacc = np.zeros((N,3))


        dipole_diff_vacc = 1.
        iteration = 0

        while dipole_diff_vacc>dtol:
        
            iteration += 1
        
            p_pol_prev = p_pol_vacc.copy()
        
            p_pol_vacc = compute_induced_dipole(x_q, q, d, p_pol_vacc, Q, alpha, thole, polar_group,
                                                connections_12, pointer_connections_12, connections_13,
                                                pointer_connections_13, np.zeros((N,3)), ep_in)
        
            dipole_diff_vacc = np.max(np.sqrt(np.sum((p_pol_vacc-p_pol_prev)**2, axis = 1)))
        
            print(F"Induced dipole residual in vacuum: {dipole_diff_vacc}")
        
        print(F"{iteration} iterations for vacuum induced dipole to converge")
        
        G_vacc = get_coulomb_energy(x_q, q, d, p_pol_vacc, Q, alpha, ep_in, thole, polar_group, 
                                    connections_12, connections_13, pointer_connections_12, 
                                    pointer_connections_13, p12scale, p13scale)
    
    time_final = time.time()
    
    print(F"Solvent contribution: {G_diss_solv} [kcal/Mol]")
    print(F"Multipoles contribution: {G_diss_mult} [kcal/Mol]")
    print(F"Coulomb vacuum energy: {G_vacc}")
    
    print("These values consider the polarization energy")
    
    total_energy = G_diss_solv + G_diss_mult - G_vacc
    
    print(F"Total solvation energy: {total_energy} [kcal/Mol]")
    print(F"Compution time for Coulomb Energy: {time_energy_final - time_energy_init} seconds")
    print(F"Total time: {time_final - time_init} seconds.")
    
    return total_energy, total_ndof
    