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
    delta_L_buffed = np.linspace(delta_min - (10 * delta_step), delta_max + (10 * delta_step), num=(2 ** 10) + 20)

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
    cgf = np.array(cgf)

    var_ratio = np.square(sigma_from_power_spectrum(kl, pl, R)) / np.square(
        sigma_from_power_spectrum(knl, pnl, R))
    # Rescaling both the cgf values and the lambda values to get the RCGF
    cgf = cgf * var_ratio
    Lam = Lam * var_ratio

    cgf_p = np.diff(cgf) / np.diff(Lam)
    lam_p = 0.5 * (Lam[1:] + Lam[:-1])
    cgf_pp = np.diff(cgf_p) / np.diff(lam_p)
    lam_pp = 0.5 * (lam_p[1:] + lam_p[:-1])

    # delta values for which we have all derivatives
    delta_L_semiUnbuffed = np.linspace(delta_min - (9 * delta_step), delta_max + (9 * delta_step), num=(2 ** 10) + 18)
    Lam_new = np.interp(delta_L_semiUnbuffed, delta_L_buffed, Lam)
    cgf = np.interp(Lam_new, Lam, cgf)
    cgf_p = np.interp(Lam_new, lam_p, cgf_p)
    cgf_pp = np.interp(Lam_new, lam_pp, cgf_pp)

    # Now we evaluate the values of the Psi function
    rho_new = cgf_p
    Psi = Lam_new * rho_new - cgf
    Psi_1 = Lam_new
    Psi_2 = np.diff(Psi_1) / np.diff(rho_new)
    rho_2 = 0.5 * (rho_new[1:] + rho_new[:-1])
    Psi_3 = np.diff(Psi_2) / np.diff(rho_2)
    rho_3 = 0.5 * (rho_2[1:] + rho_2[:-1])
    Psi_4 = np.diff(Psi_3) / np.diff(rho_3)
    rho_4 = 0.5 * (rho_3[1:] + rho_3[:-1])
    Psi_5 = np.diff(Psi_4) / np.diff(rho_4)
    rho_5 = 0.5 * (rho_4[1:] + rho_4[:-1])

    # All the needed values interpolated at the exact same lambda values
    Lam_final = np.interp(delta_L, delta_L_buffed, Lam)
    cgf_final = np.interp(Lam_final, Lam_new, cgf)
    cgfp_final = np.interp(Lam_final, Lam_new, cgf_p)
    cgfpp_final = np.interp(Lam_final, Lam_new, cgf_pp)

    # and now to reduce the psi's too
    rho_final = cgfp_final
    Psi = np.interp(rho_final, rho_new, Psi)
    Psi_1 = np.interp(rho_final, rho_new, Psi_1)
    Psi_2 = np.interp(rho_final, rho_2, Psi_2)
    Psi_3 = np.interp(rho_final, rho_3, Psi_3)
    Psi_4 = np.interp(rho_final, rho_4, Psi_4)
    Psi_5 = np.interp(rho_final, rho_5, Psi_5)
    Psi_final = np.array([Psi, Psi_1, Psi_2, Psi_3, Psi_4, Psi_5])

    return delta_L, Lam_final, cgf_final, cgfp_final, cgfpp_final, Psi_final

def RCGF_at_the_saddle2(kl, pl, knl, pnl, R, delta_min=-10, delta_max=1.5):
    # Expression of the CGF equated at the saddle point of the action (equation [5] of https://arxiv.org/abs/1912.06621)
    Lam = []  # lambda values
    cgf = []  # CGF values

    # The actual delta values that we return
    delta_L, delta_step = np.linspace(delta_min, delta_max, num=2 ** 13, retstep=True)

    for delta in delta_L:
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
    cgf = np.array(cgf)

    var_ratio = np.square(sigma_from_power_spectrum(kl, pl, R)) / np.square(
        sigma_from_power_spectrum(knl, pnl, R))
    # Rescaling both the cgf values and the lambda values to get the RCGF
    cgf = cgf * var_ratio
    Lam = Lam * var_ratio

    cgf_p = np.diff(cgf) / np.diff(Lam)
    lam_p = 0.5 * (Lam[1:] + Lam[:-1])
    cgf_pp = np.diff(cgf_p) / np.diff(lam_p)
    lam_pp = 0.5 * (lam_p[1:] + lam_p[:-1])

    # delta values for which we have all derivatives
    Lam_new = lam_pp
    cgf = np.interp(Lam_new, Lam, cgf)
    cgf_p = np.interp(Lam_new, lam_p, cgf_p)
    # cgf_pp = np.interp(Lam_new, lam_pp, cgf_pp)

    # Now we evaluate the values of the Psi function
    rho_new = cgf_p
    Psi = Lam_new * rho_new - cgf
    Psi_1 = Lam_new
    Psi_2 = np.diff(Psi_1) / np.diff(rho_new)
    rho_2 = 0.5 * (rho_new[1:] + rho_new[:-1])
    Psi_3 = np.diff(Psi_2) / np.diff(rho_2)
    rho_3 = 0.5 * (rho_2[1:] + rho_2[:-1])
    Psi_4 = np.diff(Psi_3) / np.diff(rho_3)
    rho_4 = 0.5 * (rho_3[1:] + rho_3[:-1])
    Psi_5 = np.diff(Psi_4) / np.diff(rho_4)
    rho_5 = 0.5 * (rho_4[1:] + rho_4[:-1])

    # and now to reduce the psi's too
    rho_final = rho_5
    Psi = np.interp(rho_final, rho_new, Psi)
    Psi_1 = np.interp(rho_final, rho_new, Psi_1)
    Psi_2 = np.interp(rho_final, rho_2, Psi_2)
    Psi_3 = np.interp(rho_final, rho_3, Psi_3)
    Psi_4 = np.interp(rho_final, rho_4, Psi_4)
    # Psi_5 = np.interp(rho_final, rho_5, Psi_5)
    Psi_final = np.array([rho_final, Psi, Psi_1, Psi_2, Psi_3, Psi_4, Psi_5])

    return delta_L, Lam_new, cgf, cgf_p, cgf_pp, Psi_final


def characteristic_function_from_definition(del_R, kl, pl, knl, pnl, R, order=10):
    Psi_0 = []
    var_ratio = np.square(sigma_from_power_spectrum(kl, pl, R)) / np.square(
        sigma_from_power_spectrum(knl, pnl, R))
    for d in del_R:
        RL = np.power(1 + d, 1 / 3) * R
        Psi_0 += [(1 / (2 * np.square(sigma_from_power_spectrum(kl, pl, RL)))) * np.square(
            inverse_of_approximate_collapse_alex(d)) * var_ratio]

    # Getting the final array of rhos where are derivatives would be defined
    rho_final = del_R
    for i in range(order):
        rho_final = 0.5 * (rho_final[1:] + rho_final[:-1])

    # Getting the derivates and the rho's at which they are defined and then interpolating on the rho_final
    Psi = np.array([np.interp(rho_final, del_R, Psi_0)])
    psi_deriv = Psi_0
    rho = del_R
    for i in range(order):
        psi_deriv = np.diff(psi_deriv) / np.diff(rho)
        rho = 0.5 * (rho[1:] + rho[:-1])
        Psi = np.append(Psi, np.array([np.interp(rho_final, rho, psi_deriv)]), axis=0)

    # Getting the critical
    return rho_final, Psi


## General Utility Functions

def accurate_first_order_derivative(f, x):
    # assuming uniform grid, otherwise provided an x, the f will be interpolated onto a uniform grid
    uniform_x, step = np.linspace(np.nanmin(x), np.nanmax(x), num=2 * len(x), retstep=True)
    f = np.interp(uniform_x, x, f)
    x = uniform_x

    fp = []
    xp = []
    for i in range(len(f)):
        if (i > 3) & (i < len(f) - 4):
            fp += [(((1 / 280) * (f[i - 4] - f[i + 4])) + ((4 / 105) * (f[i + 3] - f[i - 3])) + (
                        (1 / 5) * (f[i - 2] - f[i + 2])) + ((4 / 5) * (f[i + 1] - f[i - 1]))) / step]
            xp += [x[i]]
    return np.array(fp), np.array(xp)
