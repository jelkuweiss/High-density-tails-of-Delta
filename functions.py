# A repository of most of the functions that we use for our main code
import numpy as np
from math import *
from scipy import integrate
from classy import Class


## Spherical Collapse
# for now all of them are provided by the approximation that Alex provided.

def approximate_collapse_alex(delta):
    nu = 21/13
    return np.power((1 -  delta/nu ),-nu) - 1

def inverse_of_approximate_collapse_alex(delta):
    nu = 21/13
    inv_nu = 13/21
    return (- np.power((delta + 1), -inv_nu) + 1)*nu

def derivative_of_approximate_collapse_alex(delta):
    nu = 21/13
    return np.power((1 -  delta/nu ),-nu-1)

def derivative_of_inverse_of_approximate_collapse_alex(delta):
    inv_nu = 13/21
    return np.power( delta+1,-inv_nu-1)

## Variances

def top_hat_filter_in_fourier_space(x):
    return 3*( (np.sin(x)/np.power(x,3)) - (np.cos(x)/np.power(x,2)) )

def sigma_from_power_spectrum(k,P,R):
    integrand = (1/(2*(pi**2))) * P * np.square(k) * np.square(top_hat_filter_in_fourier_space(k*R))
    
    return np.sqrt(integrate.simpson(integrand,x=k))



## Valageas Functions

def derivative_of_eta_from_ValIIB8(del_R,kl,pl,R):
    RL = np.power(1 + del_R,1/3)*R
    del_l_RL = inverse_of_approximate_collapse_alex(del_R)
    sR = sigma_from_power_spectrum(kl,pl,R)
    RL = np.power(1 + del_R,1/3)*R
    if RL.shape == ():
        sRL = sigma_from_power_spectrum(kl,pl,RL)
    else:
        sRL = np.array([sigma_from_power_spectrum(kl,pl,r) for r in RL])
    
    int1 = 1 / (2 * np.square(pi))
    int2 = pl * np.square(kl) * 2 * top_hat_filter_in_fourier_space(kl*RL)
    int3 = (((np.square(kl*RL) - 3)*np.sin(kl*RL)) + (3*kl*RL*np.cos(RL*kl))) / (np.power(kl*RL,3)*RL)
    int_all = int1*np.multiply(int2,int3)
    Integration = integrate.simpson(int_all, kl)
    
    # factors divided as clumps from above equation:
    a1 = 1 / np.square(sRL) 
    a2 = derivative_of_inverse_of_approximate_collapse_alex(del_R) * sRL
    a3 = (R * del_l_RL) / (2 * sRL * np.power(1 + del_R,2/3))
    a4 = Integration
    return a1*(a2 - (a3*a4))

def high_density_tail_ValIIB9(del_R,kl,pl,R):
    del_l_RL = inverse_of_approximate_collapse_alex(del_R)

    sR = sigma_from_power_spectrum(kl,pl,R)
    RL = np.power(1 + del_R,1/3)*R
    if RL.shape == ():
        sRL = sigma_from_power_spectrum(kl,pl,RL)
    else:
        sRL = np.array([sigma_from_power_spectrum(kl,pl,r) for r in RL])
    
    h1 = 1/np.sqrt(2*pi)
    h2 = 1/(1+del_R)
    h3 = np.array([derivative_of_eta_from_ValIIB8(d,kl,pl,R) for d in del_R])
    h4 = np.exp(-np.square(del_l_RL) / (2*np.square(sRL)))
    return h1*h2*h3*h4

## Power Spectra
# Since the power spectra generation takes a long time, I will write one function that generates all the power spectra we need 
# so that it can be compiled at the very start of the work and then used later without any waiting time

def generate_all_power_spectra(Omega_m, Omega_b, h, n_s, sigma8, z, kmax_pk = 1e2, sta_k = -6, end_k = 2, class_reso = 2**6):
    print('generating power spectra ... (this will take time) ...')
    z_max_pk = z + 0.1
    # generate the linear power spectra
    commonsettings = {
        'output' :'mPk',
        'P_k_max_1/Mpc' :kmax_pk,
        'omega_b' :Omega_b * h**2,
        'h' :h,
        'n_s' :n_s,
        'sigma8' :sigma8,
        'omega_cdm' :(Omega_m - Omega_b) * h**2,
        'Omega_k' :0.0,
        'Omega_fld' :0.0,
        'Omega_scf' :0.0,
        'YHe' :0.24,
        'z_max_pk' :z_max_pk,
        'write warnings':'yes'
    }
    Cosmo = Class()
    Cosmo.set(commonsettings)
    Cosmo.compute()
    k_class_l  = np.logspace(sta_k,end_k,class_reso)
    pk_class_l = np.array([Cosmo.pk(k,z) for k in k_class_l])
    k_class_l = k_class_l / h
    pk_class_l = pk_class_l * (h**3)

    # generate the non-linear power spectra
    commonsettings = {
        'N_ur' :3.046,
        'N_ncdm' :0,
        'output' :'mPk',
        'P_k_max_1/Mpc' :kmax_pk,
        'omega_b' :Omega_b * h**2,
        'h' :h,
        'n_s' :n_s,
        'sigma8' :sigma8,
        'omega_cdm' :(Omega_m - Omega_b) * h**2,
        'Omega_k' :0.0,
        'Omega_fld' :0.0,
        'Omega_scf' :0.0,
        'YHe' :0.24,
        'z_max_pk' :z_max_pk,
        'non linear' :"Halofit",
        'write warnings':'yes'
    }
    Cosmo = Class()
    Cosmo.set(commonsettings)
    Cosmo.compute()
    k_class_nl  = np.logspace(sta_k,end_k,class_reso)
    pk_class_nl = np.array([Cosmo.pk(k,z) for k in k_class_nl])
    k_class_nl = k_class_nl / h
    pk_class_nl = pk_class_nl * (h**3)
    
    print('Done!')
    return k_class_l,pk_class_l,k_class_nl,pk_class_nl