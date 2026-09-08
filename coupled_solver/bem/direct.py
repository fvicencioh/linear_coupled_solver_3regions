import numpy as np
import numba
import bempp_cl

from numba import jit

@jit(nopython=True)
def coulomb_phi_multipole(xq, q, p, Q):
    
    N = len(xq)
    eps = 1e-15
    phi = np.zeros(N)
    
    T2 = np.zeros((N-1,3,3))
    
    for i in range(N):
        
        Ri = xq[i] - xq
        Rnorm = np.sqrt(np.sum((Ri*Ri), axis = 1) + eps*eps)
        
        Ri = np.delete(Ri, (3*i, 3*i+1, 3*i+2)).reshape((N-1,3))
        Rnorm = np.delete(Rnorm, i)
        q_temp = np.delete(q, i)
        p_temp = np.delete(p, (3*i, 3*i+1, 3*i+2)).reshape((N-1,3))
        Q_temp = np.delete(Q, (9*i, 9*i+1, 9*i+2, 9*i+3, 9*i+4, 9*i+5, 9*i+6, 9*i+7, 9*i+8)).reshape((N-1,3,3))
        
        T0 = 1./Rnorm[:]
        T1 = np.transpose(Ri.transpose()/Rnorm**3)
        T2[:,:,:] = np.ones((N-1,3,3))[:] * Ri.reshape((N-1,1,3)) * \
                    np.transpose(np.ones((N-1,3,3))*Ri.reshape((N-1,1,3)), (0,2,1))/ \
                    Rnorm.reshape((N-1,1,1))**5
        phi[i] = np.sum(q_temp[:]*T0[:]) + np.sum(T1[:]*p_temp[:]) + 0.5*np.sum(np.sum(T2[:]*Q_temp[:],axis=1))
            
    return phi

@jit(nopython=True, parallel=False, error_model="numpy", fastmath=True)
def coulomb_dphi_multipole(xq, q, p, Q, alpha, thole, polar_group, flag_polar_group):
    
    N = len(xq)
    T1 = np.zeros((3))
    T2 = np.zeros((3,3))
    eps = 1e-15
    
    scale3 = 1.0
    scale5 = 1.0
    scale7 = 1.0
    
    dphi = np.zeros((N,3))
    
    for i in range(N):
        
        aux = np.zeros((3))
        
        Ri = xq[i] - xq
        Rnorm = np.sqrt(np.sum((Ri*Ri), axis = 1) + eps*eps)
        
        for j in np.where(Rnorm>1e-12)[0]:
            
            R3 = Rnorm[j]**3
            R5 = Rnorm[j]**5
            R7 = Rnorm[j]**7
            
            if flag_polar_group==False:
                
                not_same_polar_group = True
                
            else:
                
                gamma = min(thole[i], thole[j])
                damp = (alpha[i]*alpha[j])**0.16666667
                damp += 1e-12
                damp = -1*gamma * (R3/(damp*damp*damp))
                expdamp = np.exp(damp)
                
                scale3 = 1 - expdamp
                scale5 = 1 - expdamp*(1-damp)
                scale7 = 1 - expdamp*(1-damp+0.6*damp*damp)
                
                if polar_group[i]!=polar_group[j]:
                    
                    not_same_polar_group = True
                    
                else:
                    
                    not_same_polar_group = False
                    
            if not_same_polar_group==True:
                
                for k in range(3):
                    
                    T0 = -Ri[j,k]/R3 * scale3
                    
                    for l in range(3):
                        
                        dkl = (k==l)*1.0
                        
                        T1[l] = dkl/R3 * scale3 - 3*Ri[j,k]*Ri[j,l]/R5 * scale5
                        
                        for m in range(3):
                            
                            dkm = (k==m)*1.0
                            T2[l][m] = (dkm*Ri[j,l]+dkl*Ri[j,m])/R5 * scale5 - 5*Ri[j,l]*Ri[j,m]*Ri[j,k]/R7 * scale7
         
                    
                    aux[k] += T0*q[j] + np.sum(T1*p[j]) + 0.5*np.sum(np.sum(T2[:,:]*Q[j,:,:], axis = 1), axis = 0)
                
        dphi[i,:] += aux[:]
        
    return dphi

@jit(nopython=True, parallel=False, error_model="numpy", fastmath=True)
def coulomb_ddphi_multipole(xq, q, p, Q):
    
    T1 = np.zeros((3))
    T2 = np.zeros((3,3))
    
    eps = 1e-15
    
    N = len(xq)
    
    ddphi = np.zeros((N,3,3))
    
    for i in range(N):
        
        aux = np.zeros((3,3))
        
        Ri = xq[i] - xq
        Rnorm = np.sqrt(np.sum((Ri*Ri), axis = 1) + eps*eps)
        
        for j in np.where(Rnorm>1e-12)[0]:
            
            R3 = Rnorm[j]**3
            R5 = Rnorm[j]**5
            R7 = Rnorm[j]**7
            R9 = R3**3
            
            for k in range(3):
                
                for l in range(3):
                    
                    dkl = (k==l)*1.0
                    T0 = -dkl/R3 + 3*Ri[j,k]*Ri[j,l]/R5
                    
                    for m in range(3):
                        
                        dkm = (k==m)*1.0
                        dlm = (l==m)*1.0
                        
                        T1[m] = -3*(dkm*Ri[j,l]+dkl*Ri[j,m]+dlm*Ri[j,k])/R5 + 15*Ri[j,l]*Ri[j,m]*Ri[j,k]/R7
                        
                        for n in range(3):
                            
                            dkn = (k==n)*1.0
                            dln = (l==n)*1.0
                            
                            T2[m][n] = 35*Ri[j,k]*Ri[j,l]*Ri[j,m]*Ri[j,n]/R9 - 5*(Ri[j,m]*Ri[j,n]*dkl \
                                                                          + Ri[j,l]*Ri[j,n]*dkm \
                                                                          + Ri[j,m]*Ri[j,l]*dkn \
                                                                          + Ri[j,k]*Ri[j,n]*dlm \
                                                                          + Ri[j,m]*Ri[j,k]*dln)/R7 + (dkm*dln + dlm*dkn)/R5
                            
                    aux[k][l] += T0*q[j] + np.sum(T1[:]*p[j,:]) +  0.5*np.sum(np.sum(T2[:,:]*Q[j,:,:], axis = 1), axis = 0)
                    
        ddphi[i,:,:] += aux[:,:]
        
    return ddphi

@jit(nopython=True, parallel=False, error_model="numpy", fastmath=True)
def coulomb_phi_multipole_Thole(xq, p, alpha, thole, polar_group, connections_12, pointer_connections_12, \
                                connections_13, pointer_connections_13, p12scale, p13scale):
    
    eps = 1e-15
    T1 = np.zeros((3))
    
    N = len(xq)
    
    phi = np.zeros((N))
    
    for i in range(N):
        
        aux = 0.
        start_12 = pointer_connections_12[i]
        stop_12 = pointer_connections_12[i+1]
        start_13 = pointer_connections_13[i]
        stop_13 = pointer_connections_13[i+1]
        
        Ri = xq[i] - xq
        
        r = 1./np.sqrt(np.sum((Ri*Ri), axis = 1) + eps*eps)
        
        for j in np.where(r<1e12)[0]:
            
            pscale = 1.0
            
            for ii in range(start_12, stop_12):
                
                if connections_12[ii]==j:
                    
                    pscale = p12scale
                    
            for ii in range(start_13, stop_13):
                
                if connections_13[ii]==j:
                    
                    pscale = p13scale
                    
            r3 = r[j]**3
            
            gamma = min(thole[i], thole[j])
            damp = (alpha[i]*alpha[j])**0.16666667
            damp += 1e-12
            damp = -gamma * (1/(r3*damp**3))
            expdamp = np.exp(damp)
            
            scale3 = 1 - expdamp
            
            for k in range(3):
                
                T1[k] = Ri[j,k]*r3*scale3*pscale
                
            aux += np.sum(T1[:]*p[j,:])
            
        phi[i] += aux
    
    return phi

@jit(nopython=True, parallel=False, error_model="numpy", fastmath=True)
def coulomb_dphi_multipole_Thole(xq, p, alpha, thole, polar_group, connections_12, pointer_connections_12, \
                                connections_13, pointer_connections_13, p12scale, p13scale):
    
    eps = 1e-15
    T1 = np.zeros((3))
    
    N = len(xq)
    
    dphi = np.zeros((N,3))
    
    for i in range(N):
        
        aux = np.zeros((3))
        
        start_12 = pointer_connections_12[i]
        stop_12 = pointer_connections_12[i+1]
        start_13 = pointer_connections_13[i]
        stop_13 = pointer_connections_13[i+1]
        
        Ri = xq[i] - xq
        r = 1./np.sqrt(np.sum((Ri*Ri), axis = 1) + eps*eps)
        
        for j in np.where(r<1e12)[0]:
            
            pscale = 1.0
            
            for ii in range(start_12, stop_12):
                
                if connections_12[ii]==j:
                    
                    pscale = p12scale
                    
            for ii in range(start_13, stop_13):
                
                if connections_13[ii]==j:
                    
                    pscale = p13scale
                    
            r3 = r[j]**3
            r5 = r[j]**5
            
            gamma = min(thole[i], thole[j])
            damp = (alpha[i]*alpha[j])**0.16666667
            damp += 1e-12
            damp = -gamma * (1/(r3*damp**3))
            expdamp = np.exp(damp)
            
            scale3 = 1 - expdamp
            scale5 = 1 - expdamp*(1 - damp)
            
            for k in range(3):
                
                for l in range(3):
                    
                    dkl = (k==l)*1.0
                    T1[l] = scale3*dkl*r3*pscale - scale5*3*Ri[j,k]*Ri[j,l]*r5*pscale
                    
                aux[k] += np.sum(T1[:] * p[j,:])
                
        dphi[i,:] += aux[:]
        
    return dphi

@jit(nopython=True, parallel=False, error_model="numpy", fastmath=True)
def coulomb_ddphi_multipole_Thole(xq, p, alpha, thole, polar_group, connections_12, pointer_connections_12, \
                                 connections_13, pointer_connections_13, p12scale, p13scale):
    
    eps = 1e-15
    T1 = np.zeros((3))
    
    N = len(xq)
    
    ddphi = np.zeros((N,3,3))
    
    for i in range(N):
        
        aux = np.zeros((3,3))
        
        start_12 = pointer_connections_12[i]
        stop_12 = pointer_connections_12[i+1]
        start_13 = pointer_connections_13[i]
        stop_13 = pointer_connections_13[i+1]
        
        Ri = xq[i] - xq
        r = 1./np.sqrt(np.sum((Ri*Ri), axis = 1) + eps*eps)
        
        for j in np.where(r<1e12)[0]:
            
            pscale = 1.0
            
            for ii in range(start_12, stop_12):
                
                if connections_12[ii]==j:
                    
                    pscale = p12scale
                    
            for ii in range(start_13, stop_13):
                
                if connections_13[ii]==j:
                    
                    pscale = p13scale
                    
            r3 = r[j]**3
            r5 = r[j]**5
            r7 = r[j]**7
            
            gamma = min(thole[i], thole[j])
            damp = (alpha[i]*alpha[j])**0.16666667
            damp += 1e-12
            damp = -gamma * (1/(r3*damp**3))
            expdamp = np.exp(damp)
            
            scale5 = 1 - expdamp*(1 - damp)
            scale7 = 1 - expdamp*(1 - damp + 0.6*damp**2)
            
            for k in range(3):
                
                for l in range(3):
                    
                    dkl = (k==l)*1.0
                    
                    for m in range(3):
                        
                        dkm = (k==m)*1.0
                        dlm = (l==m)*1.0
                        
                        T1[m] = -3*(dkm*Ri[j,l] + dkl*Ri[j,m] + dlm*Ri[j,k])*r5*scale5*pscale \
                        + 15*Ri[j,l]*Ri[j,m]*Ri[j,k]*r7*scale7*pscale
                        
                    aux[k][l] += np.sum(T1[:]*p[j,:])
                    
        ddphi[i,:,:] += aux[:,:]
        
    return ddphi


def coulomb_energy_multipole(xq, q, p, p_pol, Q, alphaxx, thole, polar_group, \
                             connections_12, pointer_connections_12, \
                             connections_13, pointer_connections_13, \
                             p12scale, p13scale):
    
    N = len(xq)
    
    point_energy = np.zeros((N))
    
    flag_polar_group = False
    
    dummy = np.zeros((N))
    
    #phi, dphi and ddphi from permanent multipoles
    
    phi = coulomb_phi_multipole(xq, q, p, Q)
    
    dphi = coulomb_dphi_multipole(xq, q, p, Q, dummy, dummy, dummy, flag_polar_group)
    
    ddphi = coulomb_ddphi_multipole(xq, q, p, Q)
    
    #phi, dphi and ddphi from induced dipoles
    
    phi_thole = coulomb_phi_multipole_Thole(xq, p_pol, alphaxx, thole, polar_group, \
                                            connections_12, pointer_connections_12, \
                                            connections_13, pointer_connections_13, \
                                            p12scale, p13scale)
    
    dphi_thole = coulomb_dphi_multipole_Thole(xq, p_pol, alphaxx, thole, polar_group, \
                                              connections_12, pointer_connections_12, \
                                              connections_13, pointer_connections_13, \
                                              p12scale, p13scale)
    
    ddphi_thole = coulomb_ddphi_multipole_Thole(xq, p_pol, alphaxx, thole, polar_group, \
                                                connections_12, pointer_connections_12, \
                                                connections_13, pointer_connections_13, \
                                                p12scale, p13scale)
    
    phi += phi_thole
    dphi += dphi_thole
    ddphi += ddphi_thole
    
    point_energy[:] = q[:]*phi[:] + np.sum(p[:] * dphi[:], axis = 1) + (np.sum(np.sum(Q[:]*ddphi[:], axis = 1), axis = 1))/6.
    
    return point_energy

def get_coulomb_energy(x_q, q, p, p_pol, Q, alpha, ep_in, thole, polar_group, connections_12, connections_13, pointer_connections_12, pointer_connections_13, p12scale, p13scale):
    
    
    cal2J = 4.184
    ep_vacc = 8.854187818e-12
    qe = 1.60217646e-19
    Na = 6.0221415e+23
    C0 = qe**2*Na*1e-3*1e10/(cal2J*ep_vacc)
    
    alphaxx = alpha[:,0,0]
    
    point_energy = coulomb_energy_multipole(x_q, q, p, p_pol, Q, alphaxx, thole, np.int32(polar_group), \
                                            np.int32(connections_12), np.int32(pointer_connections_12), \
                                            np.int32(connections_13), np.int32(pointer_connections_13), \
                                            p12scale, p13scale)
    
    coulomb_energy = np.sum(point_energy) * 0.5*C0/(4*np.pi*ep_in)
    
    return coulomb_energy

def compute_induced_dipole(xq, q, p, p_pol, Q, alpha, thole, polar_group, \
                           connections_12, pointer_connections_12, \
                           connections_13, pointer_connections_13, \
                           dphi_reac, E, SOR = 0.7):
    
    N = len(xq)
    
    u12scale = 1.0
    u13scale = 1.0
    
    flag_polar_group = True
    
    alphaxx = alpha[:,0,0]
    
    dphi_coul = coulomb_dphi_multipole(xq, q, p, Q, alphaxx, thole, polar_group, flag_polar_group)
    
    dphi_coul_thole = coulomb_dphi_multipole_Thole(xq, p_pol, alphaxx, thole, polar_group, \
                                                   connections_12, pointer_connections_12, \
                                                   connections_13, pointer_connections_13, \
                                                   u12scale, u13scale)
    
    dphi_coul += dphi_coul_thole
    
    for i in range(N):
        
        E_total = (dphi_coul[i]/E + 4*np.pi*dphi_reac[i])*-1
        p_pol[i] = p_pol[i]*(1 - SOR) + np.dot(alpha[i], E_total)*SOR
    
    return p_pol

#def solvation_energy_solvent(q, p, Q, phi, dphi, ddphi):
    
    #cal2J = 4.184
    #qe = 1.60217646e-19
    #Na = 6.0221415e+23
    #ep_vacc = 8.854187818e-12
    #C0 = qe**2*Na*1e-3*1e10/(cal2J*ep_vacc)
    
    #q_aux = 0
    #p_aux = 0
    #Q_aux = 0
    
    #for i in range(len(q)):
        #q_aux += q[i]*phi[i]
        
        #for j in range(3):
            #p_aux += p[i,j]*dphi[i,j]
            
            #for k in range(3):
                #Q_aux += Q[i,j,k]*ddphi[i,j,k]/6.
                
    #solvent_energy = 0.5 * C0 * (q_aux + p_aux + Q_aux)
    
    #return solvent_energy

def solvation_energy_solvent(q, p, Q, phi, dphi, ddphi):
    solv_energy = 2*np.pi*332.064*(np.sum(q*phi) + np.sum(p*dphi) + np.sum(Q*ddphi)/6)
    return solv_energy

def solvent_potential_first_derivate(xq, h, neumann_space, dirichl_space, solution_neumann, solution_dirichl):
    
    """
    Compute the first derivate of potential due to solvent
    in the position of the points
    Inputs:
    -------
        xq: Array size (Nx3) whit positions to calculate the derivate.
        h: Float number, distance for the central difference.

    Return:

        dpdr: Derivate of the potential in the positions of points.
    """

    dpdr = np.zeros([len(xq), 3])
    dist = np.array(([h,0,0],[0,h,0],[0,0,h]))
    # x axis derivate
    dx = xq[:] + dist[0]
    dx = np.concatenate((dx, xq[:] - dist[0]))
    slpo = bempp_cl.api.operators.potential.laplace.single_layer(neumann_space, dx.transpose())
    dlpo = bempp_cl.api.operators.potential.laplace.double_layer(dirichl_space, dx.transpose())
    phi = slpo.evaluate(solution_neumann) - dlpo.evaluate(solution_dirichl)
    dpdx = 0.5*(phi[0,:len(xq)] - phi[0,len(xq):])/h
    dpdr[:,0] = dpdx

    #y axis derivate
    dy = xq[:] + dist[1]
    dy = np.concatenate((dy, xq[:] - dist[1]))
    slpo = bempp_cl.api.operators.potential.laplace.single_layer(neumann_space, dy.transpose())
    dlpo = bempp_cl.api.operators.potential.laplace.double_layer(dirichl_space, dy.transpose())
    phi = slpo.evaluate(solution_neumann) - dlpo.evaluate(solution_dirichl)
    dpdy = 0.5*(phi[0,:len(xq)] - phi[0,len(xq):])/h
    dpdr[:,1] = dpdy

    #z axis derivate
    dz = xq[:] + dist[2]
    dz = np.concatenate((dz, xq[:] - dist[2]))
    slpo = bempp_cl.api.operators.potential.laplace.single_layer(neumann_space, dz.transpose())
    dlpo = bempp_cl.api.operators.potential.laplace.double_layer(dirichl_space, dz.transpose())
    phi = slpo.evaluate(solution_neumann) - dlpo.evaluate(solution_dirichl)
    dpdz = 0.5*(phi[0,:len(xq)] - phi[0,len(xq):])/h
    dpdr[:,2] = dpdz

    return dpdr

def solvent_potential_second_derivate(x_q, h, neumann_space, dirichl_space, solution_neumann, solution_dirichl):
    
    """
    Compute the second derivate of potential due to solvent
    in the position of the points

    xq: Array size (Nx3) whit positions to calculate the derivate.
    h: Float number, distance for the central difference.

    Return:

    ddphi: Second derivate of the potential in the positions of points.
    """
    ddphi = np.zeros((len(x_q),3,3))
    dist = np.array(([h,0,0],[0,h,0],[0,0,h]))
    for i in range(3):
        for j in np.where(np.array([0, 1, 2]) >= i)[0]:
            if i==j:
                dp = np.concatenate((x_q[:] + dist[i], x_q[:], x_q[:] - dist[i]))
                slpo = bempp_cl.api.operators.potential.laplace.single_layer(neumann_space, dp.transpose())
                dlpo = bempp_cl.api.operators.potential.laplace.double_layer(dirichl_space, dp.transpose())
                phi = slpo.evaluate(solution_neumann) - dlpo.evaluate(solution_dirichl)
                ddphi[:,i,j] = (phi[0,:len(x_q)] - 2*phi[0,len(x_q):2*len(x_q)] + phi[0, 2*len(x_q):])/(h**2)
      
            else:
                dp = np.concatenate((x_q[:] + dist[i] + dist[j], x_q[:] - dist[i] - dist[j], x_q[:] + \
                                     dist[i] - dist[j], x_q[:] - dist[i] + dist[j]))
                slpo = bempp_cl.api.operators.potential.laplace.single_layer(neumann_space, dp.transpose())
                dlpo = bempp_cl.api.operators.potential.laplace.double_layer(dirichl_space, dp.transpose())
                phi = slpo.evaluate(solution_neumann) - dlpo.evaluate(solution_dirichl)
                ddphi[:,i,j] = (phi[0,:len(x_q)] + phi[0,len(x_q):2*len(x_q)] - \
                                phi[0, 2*len(x_q):3*len(x_q)] - phi[0, 3*len(x_q):])/(4*h**2)
                ddphi[:,j,i] = (phi[0,:len(x_q)] + phi[0,len(x_q):2*len(x_q)] - \
                                phi[0, 2*len(x_q):3*len(x_q)] - phi[0, 3*len(x_q):])/(4*h**2)
  
    return ddphi