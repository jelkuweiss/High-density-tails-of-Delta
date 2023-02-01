# A repository of most of the functions that we use for our main code
import numpy as np
from math import *
from scipy import integrate
from classy import Class


## Spherical Collapse
# for now all of them are provided by the approximation that Alex provided.

def approximate_collapse_alex(delta):
    nu = 21 / 13
    return np.power((1 - delta / nu), -nu) - 1


def inverse_of_approximate_collapse_alex(delta):
    nu = 21 / 13
    inv_nu = 13 / 21
    return (- np.power((delta + 1), -inv_nu) + 1) * nu


def derivative_of_approximate_collapse_alex(delta):
    nu = 21 / 13
    return np.power((1 - delta / nu), -nu - 1)


def derivative_of_inverse_of_approximate_collapse_alex(delta):
    inv_nu = 13 / 21
    return np.power(delta + 1, -inv_nu - 1)


## Variances

def top_hat_filter_in_fourier_space(x):
    return 3 * ((np.sin(x) / np.power(x, 3)) - (np.cos(x) / np.power(x, 2)))


def sigma_from_power_spectrum(k, pk, R):
    integrand = (1 / (2 * (pi ** 2))) * pk * np.square(k) * np.square(top_hat_filter_in_fourier_space(k * R))

    return np.sqrt(integrate.simpson(integrand, x=k))


## Valageas Functions

def derivative_of_eta_from_ValIIB8(del_R, kl, pl, R):
    RL = np.power(1 + del_R, 1 / 3) * R
    del_l_RL = inverse_of_approximate_collapse_alex(del_R)
    sR = sigma_from_power_spectrum(kl, pl, R)
    RL = np.power(1 + del_R, 1 / 3) * R
    if RL.shape == ():
        sRL = sigma_from_power_spectrum(kl, pl, RL)
    else:
        sRL = np.array([sigma_from_power_spectrum(kl, pl, r) for r in RL])

    int1 = 1 / (2 * np.square(pi))
    int2 = pl * np.square(kl) * 2 * top_hat_filter_in_fourier_space(kl * RL)
    int3 = (((np.square(kl * RL) - 3) * np.sin(kl * RL)) + (3 * kl * RL * np.cos(RL * kl))) / (
            np.power(kl * RL, 3) * RL)
    int_all = int1 * np.multiply(int2, int3)
    Integration = integrate.simpson(int_all, kl)

    # factors divided as clumps from above equation:
    a1 = 1 / np.square(sRL)
    a2 = derivative_of_inverse_of_approximate_collapse_alex(del_R) * sRL
    a3 = (R * del_l_RL) / (2 * sRL * np.power(1 + del_R, 2 / 3))
    a4 = Integration
    return a1 * (a2 - (a3 * a4))


def high_density_tail_ValIIB9(del_R, kl, pl, R):
    del_l_RL = inverse_of_approximate_collapse_alex(del_R)

    sR = sigma_from_power_spectrum(kl, pl, R)
    RL = np.power(1 + del_R, 1 / 3) * R
    if RL.shape == ():
        sRL = sigma_from_power_spectrum(kl, pl, RL)
    else:
        sRL = np.array([sigma_from_power_spectrum(kl, pl, r) for r in RL])

    h1 = 1 / np.sqrt(2 * pi)
    h2 = 1 / (1 + del_R)
    h3 = np.array([derivative_of_eta_from_ValIIB8(d, kl, pl, R) for d in del_R])
    h4 = np.exp(-np.square(del_l_RL) / (2 * np.square(sRL)))
    return h1 * h2 * h3 * h4


## Power Spectra
# Since the power spectra generation takes a long time, I will write one function that generates all the power spectra we need 
# so that it can be compiled at the very start of the work and then used later without any waiting time

def generate_all_power_spectra(Omega_m, Omega_b, h, n_s, sigma8, z, kmax_pk=1e2, sta_k=-6, end_k=2, class_reso=2 ** 6):
    print('generating power spectra ... (this will take time) ...')
    z_max_pk = z + 0.1
    # generate the linear power spectra
    commonsettings = {
        'output': 'mPk',
        'P_k_max_1/Mpc': kmax_pk,
        'omega_b': Omega_b * h ** 2,
        'h': h,
        'n_s': n_s,
        'sigma8': sigma8,
        'omega_cdm': (Omega_m - Omega_b) * h ** 2,
        'Omega_k': 0.0,
        'Omega_fld': 0.0,
        'Omega_scf': 0.0,
        'YHe': 0.24,
        'z_max_pk': z_max_pk,
        'write warnings': 'yes'
    }
    Cosmo = Class()
    Cosmo.set(commonsettings)
    Cosmo.compute()
    k_class_l = np.logspace(sta_k, end_k, class_reso)
    pk_class_l = np.array([Cosmo.pk(k, z) for k in k_class_l])
    k_class_l = k_class_l / h
    pk_class_l = pk_class_l * (h ** 3)

    # generate the non-linear power spectra
    commonsettings = {
        'N_ur': 3.046,
        'N_ncdm': 0,
        'output': 'mPk',
        'P_k_max_1/Mpc': kmax_pk,
        'omega_b': Omega_b * h ** 2,
        'h': h,
        'n_s': n_s,
        'sigma8': sigma8,
        'omega_cdm': (Omega_m - Omega_b) * h ** 2,
        'Omega_k': 0.0,
        'Omega_fld': 0.0,
        'Omega_scf': 0.0,
        'YHe': 0.24,
        'z_max_pk': z_max_pk,
        'non linear': "Halofit",
        'write warnings': 'yes'
    }
    Cosmo = Class()
    Cosmo.set(commonsettings)
    Cosmo.compute()
    k_class_nl = np.logspace(sta_k, end_k, class_reso)
    pk_class_nl = np.array([Cosmo.pk(k, z) for k in k_class_nl])
    k_class_nl = k_class_nl / h
    pk_class_nl = pk_class_nl * (h ** 3)

    print('Done!')
    return k_class_l, pk_class_l, k_class_nl, pk_class_nl


## Cumulant Generating Functions

def del_top_hat(k, R):
    # Expression of the derivative of the k-space radial filter wrt to the radius
    h1 = (3 * k) / np.power(R * k, 6)
    h2 = 3 * np.power(R * k, 3) * np.cos(R * k)
    h3 = (np.square(R * k) - 3) * np.power(R * k, 2) * np.sin(R * k)
    return h1 * (h2 + h3)


def del_variance(k, pk, R):
    # Expression for the derivative of the sigma^2_{L,R} wrt to the radius R
    integrand = (1 / (pi ** 2)) * pk * np.square(k) * top_hat_filter_in_fourier_space(k * R) * del_top_hat(k, R)
    return integrate.simpson(integrand, x=k)


def RCGF_at_the_saddle(kl, pl, knl, pnl, R, delta_min=-1.5, delta_max=1.4):
    # Expression of the CGF equated at the saddle point of the action (equation [5] of https://arxiv.org/abs/1912.06621)
    Lam = []  # lambda values
    cgf = []  # CGF values

    # The actual delta values that we return
    delta_L, delta_step = np.linspace(delta_min, delta_max, num=2 ** 12, retstep=True)
    # The extended delta values needed to calculate the derivatives of the CGF
    delta_L_buffed = np.linspace(delta_min - (2 * delta_step), delta_max + (2 * delta_step), num=(2 ** 10) + 4)

    for delta in delta_L_buffed:
        Rl = np.power(1 + approximate_collapse_alex(delta), 1 / 3) * R
        # collapses
        F = approximate_collapse_alex(delta)
        Fp = derivative_of_approximate_collapse_alex(delta)
        # sigmas
        sRl = sigma_from_power_spectrum(kl, pl, Rl)
        # sigma squares
        sRl2 = np.square(sRl)

        # jstar as a function of detla
        j = delta / sRl2

        # value of the derivative of the CGF wrt R'
        delCGF = 0.5 * (np.square(j)) * del_variance(kl, pl, Rl)
        # value of the lambda
        R_factor = np.power(R, 3) / (2 * np.power(Rl, 2))
        lam = +(j / Fp) - (delCGF * R_factor)

        Lam += [lam]
        cgf += [+ (lam * F) - (j * delta) + (0.5 * np.square(j) * sRl2)]

    Lam = np.array(Lam)
    var_ratio = np.square(sigma_from_power_spectrum(kl, pl, R)) / np.square(
        sigma_from_power_spectrum(knl, pnl, R))

    cgf = cgf * var_ratio
    Lam = Lam * var_ratio

    cgf_p = np.diff(cgf) / np.diff(Lam)
    lam_p = 0.5 * (Lam[1:] + Lam[:-1])
    cgf_pp = np.diff(cgf_p) / np.diff(lam_p)
    lam_pp = 0.5 * (lam_p[1:] + lam_p[:-1])

    Lam_final = np.interp(delta_L, delta_L_buffed, Lam)
    cgf_final = np.interp(Lam_final, Lam, cgf)
    cgfp_final = np.interp(Lam_final, lam_p, cgf_p)
    cgfpp_final = np.interp(Lam_final, lam_pp, cgf_pp)

    return delta_L, Lam_final, cgf_final, cgfp_final, cgfpp_final
