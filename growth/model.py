import numpy as np
import scipy
import pandas as pd

def self_replicator(params,
                    time,
                    gamma_max,
                    nu_max,
                    omega,
                    phi_Rb,
                    phi_Mb,
                    Kd_cpc=0.025,
                    Kd_cnt=5E-4,
                    phi_O = 0,
                    dil_approx=False):
    """
    Defines the system of ordinary differenetial equations (ODEs) which describe 
    the self-replicator ribosomal allocation model.

    Parameters
    ----------
    params: list, [Mr, Mp, c_AA, c_N]
        A list of the parameters whose dynamics are described by the ODEs.
        M_r : positive float, must be < M 
            Ribosomal protein biomass of the system
        M_p : positive float, must be < M
            Metabolic protein biomass of the system 
        c_AA : positive float
            Concentration of precursors in the culture. This is normalized to 
            total protein biomass.
        c_N : positive float
            Concentration of nutrients in the culture. This is in units of molar.
    time : float
        Evaluated time step of the system. This is only needed if using
        `scipy.integrate.odeint` or `scipy.integrate.solve_ivp` to evaluate 
        the system of equations.
    gamma_max: positive float 
        The maximum translational capacity in units of inverse time.
    nu_max : positive float
        The maximum nutritional capacity in units of inverse time. 
    omega: positive float
        The yield coefficient of the nutrient source in mass of amino acid 
        produced per mass of nutrient.
    phi_Rb : float, [0, 1]
        The ribosomal allocation factor.
    phi_Mb : float, [0, 1]
        The metabolic allocation factor.
    Kd_cAA : positive float 
        The effective dissociation constant of precursors to the elongating
        ribosome. This is in units of mass fraction.
    Kd_cN: positive float
        The effective dissociation constant for growth on the specific nutrient 
        source. This is in units of molar.
    dil_approx: bool
        If True, then the approximation is made that the dilution of charged-tRNAs
        with growing biomass is negligible

    Returns
    -------
    out: list, [ dM_Rb_dt, dM_Mb_dt, dc_pc_dt, dc_nt_dt]
        A list of the evaluated ODEs at the specified time step.
        dM_Rb_dt : The dynamics of the ribosomal protein biomass.
        dM_Mb_dt : the dynamics of the metabolic protein biomass.
        dc_pc_dt : The dynamics of the precursor concentration.
        dc_nt_dt :  The dynamics of the nutrient concentration in the growth medium
    """
    # Unpack the parameters
    if phi_O:
        M_Rb, M_Mb, M_O, c_pc, c_nt = params
    else:
        M_Rb, M_Mb, c_pc, c_nt = params

    # Compute the capacities
    gamma = gamma_max * (c_pc / (c_pc + Kd_cpc))
    nu = nu_max * (c_nt / (c_nt + Kd_cnt))

    # Biomass accumulation
    dM_dt = gamma * M_Rb

    # Resource allocation
    dM_Rb_dt = phi_Rb * dM_dt
    dM_Mb_dt = phi_Mb * dM_dt
         
    # Precursor dynamics
    if dil_approx:
        dc_pc_dt = (nu * M_Mb - dM_dt) / (M_Rb + M_Mb)
    else:
        dc_pc_dt = (nu * M_Mb - (1 + c_pc) * dM_dt) / (M_Rb + M_Mb)
    dc_nt_dt = -nu * M_Mb / omega

    # Pack and return the output
    if phi_O: 
        dM_O_dt = phi_O * dM_dt
        out = [dM_Rb_dt, dM_Mb_dt, dM_O_dt, dc_pc_dt, dc_nt_dt]
    else:
        out = [dM_Rb_dt, dM_Mb_dt, dc_pc_dt, dc_nt_dt]
    return out

def steady_state_precursors(gamma_max, phi_Rb, nu_max, Kd_cpc, phi_O=0):
    """
    Computes the steady state value of the charged-tRNA abundance.

    Parameters
    ----------
    gamma_max: positive float
        The maximum translational efficiency in units of inverse time.
    phi_Rb: float [0, 1]
        The fraction of the proteome occupied by ribosomal proteins.
    nu_max : positive float 
        The maximum nutritional capacity in units of inverse time. 
    Kd_cpc : positive float
        The effective dissociation constant of the precursors to the ribosome. 

    Returns
    -------
    c_pc : float
        The charged tRNA abunadance given the model parameters. This is defined
        as the mass of the charged-tRNA relative to the total biomass.

    Notes
    -----
    This function assumes that in steady state, the nutrients are in such abundance 
    that the nutritional capacy is equal to its maximal value. 

    """
    ss_lam = steady_state_growth_rate(gamma_max, phi_Rb, nu_max, Kd_cpc, phi_O=phi_O)
    cpc = (nu_max * (1 - phi_Rb - phi_O) / ss_lam) - 1
    return cpc

def steady_state_growth_rate(gamma_max, phi_Rb, nu_max, Kd_cpc, phi_O=0):
    """
    Computes the steady-state growth rate of the self-replicator model. 

    Parameters
    ----------
    gamma_max : positive float 
        The maximum translational capacity in units of inverse time.
    phi_Rb : float [0, 1]
        The fraction of the proteome occupied by ribosomal protein mass
    nu_max : positive float 
        The maximum nutritional capacity in units of inverse time. 
    Kd_cpc :  positive float
        The effective dissociation constant of charged tRNA to the elongating
        ribosome.

    Returns
    -------
    lam : float 
        The physically meaningful growth rate (in units of inverse time) given 
        the provided model parameeters.

    Notes
    -----
    This function assumes that in steady state, the nutrients are in such abundance 
    that the nutritional capacy is equal to its maximal value. 
    """
    Nu = nu_max * (1 - phi_Rb - phi_O)
    Gamma = gamma_max * phi_Rb
    numer = Nu + Gamma - np.sqrt((Nu + Gamma)**2 - 4 * (1 - Kd_cpc) * Nu * Gamma)
    denom = 2 * (1 - Kd_cpc)
    lam = numer / denom
    return lam

def steady_state_gamma(gamma_max, phi_Rb, nu_max, Kd_cpc, phi_O=0):
    """
    Computes the steady-state translational efficiency, gamma.

    Parameters 
    -----------
    gamma_max : positive float
        The maximum translational capacity in units of inverse time.
    phi_Rb : float [0, 1]
        The fraction of the proteome occupied by ribosomal protein mass.
    nu_max : positive float 
        The maximum nutritional capacity in units of inverse time.
    Kd_cpc : positive float 
        The effective dissociation constant of charged tRNA to the elongating
        ribosome.

    Returns
    -------
    gamma : positive float
        The translational efficiency in units of inverse time
    """

    c_pc = steady_state_precursors(gamma_max, phi_Rb, nu_max, Kd_cpc, phi_O=phi_O)
    return gamma_max * (c_pc / (c_pc + Kd_cpc))


def phi_R_optimal_allocation(gamma_max, nu_max, Kd_cpc, phi_O=0):
    """
    Computes the optimal fraction of proteome that is occupied by ribosomal 
    proteins which maximizes the growth rate. 

    Parameters
    ----------
    gamma_max : positive float 
        The maximum translational efficiency in units of inverse time.
    nu_max : positive float
        The maximum nutritional capacity in units of inverse time.
    Kd_cpc: positive float 
        The effective dissociation constant of charged tRNA to the elongating 
        ribosome.

    Returns
    -------
    phi_Rb_opt : positive float [0, 1]
        The optimal allocation to ribosomes.
    """
    numer = nu_max * (-2 * Kd_cpc * gamma_max + gamma_max + nu_max) +\
        np.sqrt(Kd_cpc * gamma_max * nu_max) * (gamma_max - nu_max)
    denom = -4 * Kd_cpc * gamma_max * nu_max + gamma_max**2 + 2 * gamma_max * nu_max + nu_max**2
    phi_Rb_opt = (1 - phi_O) * numer / denom
    return phi_Rb_opt


def batch_culture_self_replicator_ppGpp(params,
                                  time,
                                  gamma_max,
                                  nu_max, 
                                  tau = 1, 
                                  Kd_TAA_star = 0.025,
                                  Kd_TAA = 0.025,
                                  kappa_max = 0.01,
                                  phi_Rb = 0.1,
                                  k_Rb = 1E-3,
                                  dil_approx = False,
                                  dynamic_phiRb = True,
                                  tRNA_regulation = False,
                                  ribosome_maturation = False,
                                  ):
    """
    Defines the system of ordinary differenetial equations (ODEs) which describe 
    the self-replicator model in batch culture conditions.

    Parameters
    ----------
    params: list, [Mr, Mp, T_AA, T_AA_star]
        A list of the parameters whose dynamics are described by the ODEs.
        M_r : positive float, must be < M 
            Ribosomal protein biomass of the system
        M_p : positive float, must be < M
            Metabolic protein biomass of the system 
        T_AA_star : positive float
            Concentration of charged tRNAs in the culture. This is normalized to 
            total protein biomass.
        T_AA : positive float
            Concentration of uncharged tRNAs in the culture. This is normalized to 
            total protein biomass.
    time : float
        Evaluated time step of the system.
    gamma_max: positive float 
        The maximum translational capacity in units of inverse time.
    nu_max : positive float
        The maximum nutritional capacity in units of inverse time. 
    omega: positive float
        The yield coefficient of the nutrient source in mass of amino acid 
        produced per mass of nutrient.
    phi_R : float, [0, 1]
        The fraction of the proteome occupied by ribosomal protein mass
    Kd_cAA : positive float 
        The effective dissociation constant of precursors to the elongating
        ribosome. This is in units of mass fraction.
    Kd_cN: positive float
        The effective dissociation constant for growth on the specific nutrient 
        source. This is in units of molar.
    dil_approx: bool
        If True, then the approximation is made that the dilution of charged-tRNAs
        with growing biomass is negligible.
    num_muts: int
        The number of mutants whose dynamics need to be tracked.

    Returns
    -------
    out: list, [dM_dt, dMr_dt, dMp_dt, dcAA_dt, dcN_dt]
        A list of the evaluated ODEs at the specified time step.

        dMr_dt : The dynamics of the ribosomal protein biomass.
        dMp_dt : the dynamics of the metabolic protein biomass.
        dcAA_dt : The dynamics of the precursor concentration.
        dcN_dt :  The dynamics of the nutrient concentration in the growth medium
    """
    # Unpack the parameters
    if ribosome_maturation:
        M_Rb, M_Rb_star, M_Mb, T_AA, T_AA_star = params
    else:
        M_Rb, M_Mb, T_AA, T_AA_star = params

    # Compute the capacities
    gamma = gamma_max * (T_AA_star / (T_AA_star + Kd_TAA_star))
    nu = nu_max * (T_AA / (T_AA + Kd_TAA))

    # Compute the active fraction
    ratio = T_AA_star / T_AA

    # Biomass accumulation
    dM_dt = gamma * M_Rb

    # Resource allocation
    if dynamic_phiRb:
        phi_Rb = ratio / (ratio + tau)

    if ribosome_maturation: 
        dM_Rb_star_dt = phi_Rb * dM_dt - k_Rb * M_Rb_star
        dM_Rb_dt = k_Rb * M_Rb_star
    else:
        dM_Rb_dt = phi_Rb * dM_dt
    dM_Mb_dt = (1 - phi_Rb) * dM_dt


    if ribosome_maturation:
        M = M_Mb + M_Rb + M_Rb_star
    else:
        M = M_Mb + M_Rb

    # tRNA dynamics
    dT_AA_star_dt = (nu * M_Mb - dM_dt) / M
    dT_AA_dt = (dM_dt - nu * M_Mb) / M

    if dil_approx == False:
        dT_AA_star_dt -= T_AA_star * dM_dt / M
        if tRNA_regulation:
            kappa = kappa_max * phi_Rb
        else:
            kappa = kappa_max
        dT_AA_dt += kappa - (T_AA * dM_dt) / M

    # Pack and return the output.
    if ribosome_maturation:
        out = [dM_Rb_dt, dM_Rb_star_dt, dM_Mb_dt, dT_AA_dt, dT_AA_star_dt]
    else:
        out = [dM_Rb_dt, dM_Mb_dt, dT_AA_dt, dT_AA_star_dt]
    return out

def batch_culture_self_replicator_ppGpp_phi_O(params,
                                  time,
                                  gamma_max,
                                  nu_max, 
                                  tau = 1, 
                                  Kd_TAA_star = 0.025,
                                  Kd_TAA = 0.025,
                                  phi_O = 0,
                                  kappa_max = 0.01,
                                  phi_Rb = 0.1,
                                  dil_approx = False,
                                  dynamic_phiRb = True,
                                  tRNA_regulation = False,
                                  ):
    """
    Defines the system of ordinary differenetial equations (ODEs) which describe 
    the self-replicator model in batch culture conditions.

    Parameters
    ----------
    params: list, [Mr, Mp, T_AA, T_AA_star]
        A list of the parameters whose dynamics are described by the ODEs.
        M_r : positive float, must be < M 
            Ribosomal protein biomass of the system
        M_p : positive float, must be < M
            Metabolic protein biomass of the system 
        T_AA_star : positive float
            Concentration of charged tRNAs in the culture. This is normalized to 
            total protein biomass.
        T_AA : positive float
            Concentration of uncharged tRNAs in the culture. This is normalized to 
            total protein biomass.
    time : float
        Evaluated time step of the system.
    gamma_max: positive float 
        The maximum translational capacity in units of inverse time.
    nu_max : positive float
        The maximum nutritional capacity in units of inverse time. 
    omega: positive float
        The yield coefficient of the nutrient source in mass of amino acid 
        produced per mass of nutrient.
    phi_R : float, [0, 1]
        The fraction of the proteome occupied by ribosomal protein mass
    Kd_cAA : positive float 
        The effective dissociation constant of precursors to the elongating
        ribosome. This is in units of mass fraction.
    Kd_cN: positive float
        The effective dissociation constant for growth on the specific nutrient 
        source. This is in units of molar.
    dil_approx: bool
        If True, then the approximation is made that the dilution of charged-tRNAs
        with growing biomass is negligible.
    num_muts: int
        The number of mutants whose dynamics need to be tracked.

    Returns
    -------
    out: list, [dM_dt, dMr_dt, dMp_dt, dcAA_dt, dcN_dt]
        A list of the evaluated ODEs at the specified time step.

        dMr_dt : The dynamics of the ribosomal protein biomass.
        dMp_dt : the dynamics of the metabolic protein biomass.
        dcAA_dt : The dynamics of the precursor concentration.
        dcN_dt :  The dynamics of the nutrient concentration in the growth medium
    """
    # Unpack the parameters
    M, M_Rb, M_Mb, T_AA, T_AA_star = params


    # Compute the capacities
    gamma = gamma_max * (T_AA_star / (T_AA_star + Kd_TAA_star))
    nu = nu_max * (T_AA / (T_AA + Kd_TAA))

    # Compute the active fraction
    ratio = T_AA_star / T_AA

    # Biomass accumulation
    dM_dt = gamma * M_Rb

    # Resource allocation
    if dynamic_phiRb:
        phi_Rb = ratio / (ratio + tau)

    dM_Rb_dt = phi_Rb * dM_dt
    dM_Mb_dt = (1 - phi_Rb - phi_O) * dM_dt

    # tRNA dynamics
    dT_AA_star_dt = (nu * M_Mb - dM_dt) / M
    dT_AA_dt = (dM_dt - nu * M_Mb) / (M_Rb + M_Mb)
    if dil_approx == False:
        dT_AA_star_dt -= T_AA_star * dM_dt / M
        if tRNA_regulation:
            kappa = kappa_max * phi_Rb
        else:
            kappa = kappa_max
        dT_AA_dt += kappa - (T_AA * dM_dt) / M

    # Pack and return the output.
    out = [dM_dt, dM_Rb_dt, dM_Mb_dt, dT_AA_dt, dT_AA_star_dt]
    return out






def nutrient_shift_ppGpp(nu_preshift, 
                         nu_postshift, 
                         shift_time, 
                         init_params, 
                         init_args,
                         total_time,
                         postshift_args = False,
                         dt=0.0001):
                        
    cols = ['M', 'M_Rb', 'M_Mb_1', 'M_Mb_2', 'T_AA', 'T_AA_star']

    # Set the timespans
    preshift_time = np.arange(0, shift_time, dt)
    postshift_time = np.arange(shift_time - dt, total_time, dt)

    # Integrate the preshift
    preshift_out = scipy.integrate.odeint(batch_culture_self_replicator_ppGpp_shift,
                                          init_params, 
                                          preshift_time, 
                                          args=init_args)
    
    preshift_df = pd.DataFrame(preshift_out,
                               columns=cols)
    preshift_df['nu'] = nu_preshift
    preshift_df['phase'] = 'preshift'
    preshift_df['time'] =  preshift_time
    preshift_df['tRNA_balance'] = preshift_df['T_AA_star'].values / preshift_df['T_AA'].values

    if init_args[-2]:
        preshift_df['prescribed_phiR'] = preshift_df['tRNA_balance'].values / (preshift_df['tRNA_balance'].values + init_args[3])

    else:
        preshift_df['prescribed_phiR'] = init_args[-4]
   
    preshift_df['prescribed_phiMb1'] = (1 - preshift_df['prescribed_phiR'] - init_args[-6])
    preshift_df['prescribed_phiMb2'] = init_args[-6]

    postshift_params = preshift_out[-1]

    if postshift_args == False:
        postshift_args = [init_args[i] for i in range(len(init_args))]
        postshift_args[1] = nu_postshift
        postshift_args = tuple(postshift_args)
    postshift_out = scipy.integrate.odeint(batch_culture_self_replicator_ppGpp_shift,
                                                 postshift_params, 
                                                 postshift_time, 
                                                 args=postshift_args) 
    postshift_df = pd.DataFrame(postshift_out[1:], columns=cols)
    postshift_df['nu'] = nu_postshift
    postshift_df['phase'] = 'postshift'
    postshift_df['time'] = postshift_time[1:] 
    postshift_df['tRNA_balance'] = postshift_df['T_AA_star'].values / postshift_df['T_AA'].values
    if init_args[-2]:
        postshift_df['prescribed_phiR'] = postshift_df['tRNA_balance'].values / (postshift_df['tRNA_balance'].values + init_args[3])
    else:
        postshift_df['prescribed_phiR'] = postshift_args[-4]
    postshift_df['prescribed_phiMb1'] = 0
    postshift_df['prescribed_phiMb2'] = (1 - postshift_df['prescribed_phiR'])
 
    shift_df = pd.concat([preshift_df, postshift_df])

    # Compute properties
    shift_df['total_biomass'] = shift_df['M'].values
    shift_df['relative_biomass'] = shift_df['total_biomass'].values / (preshift_out[0][0])
    shift_df['realized_phiR'] = shift_df['M_Rb'].values / shift_df['total_biomass'].values
    shift_df['realized_phiMb1'] = shift_df['M_Mb_1'].values / shift_df['total_biomass'].values
    shift_df['realized_phiMb2'] = shift_df['M_Mb_2'].values / shift_df['total_biomass'].values
    shift_df['gamma'] = init_args[0] * shift_df['T_AA_star'].values / (shift_df['T_AA_star'].values + init_args[3])
    return shift_df

def batch_culture_self_replicator_ppGpp_shift(params,
                                  time,
                                  gamma_max,
                                  nu_max, 
                                  prefactor,
                                  tau = 1, 
                                  Kd_TAA_star = 0.025,
                                  Kd_TAA = 0.025,
                                  phi_O = 0,
                                  kappa_max = 0.01,
                                  phi_Rb = 0.1,
                                  dil_approx = False,
                                  dynamic_phiRb = True,
                                  tRNA_regulation = False,
                                  ):
    """
    """
    # Unpack the parameters
    M, M_Rb, M_Mb_1, M_Mb_2, T_AA, T_AA_star = params


    # Compute the capacities
    gamma = gamma_max * (T_AA_star / (T_AA_star + Kd_TAA_star))
    nu_1 = nu_max[0] * (T_AA / (T_AA + Kd_TAA))
    nu_2 = nu_max[1] * (T_AA / (T_AA + Kd_TAA))

    # Compute the active fraction
    ratio = T_AA_star / T_AA

    # Biomass accumulation
    dM_dt = gamma * M_Rb

    # Resource allocation
    if dynamic_phiRb:
        phi_Rb = ratio / (ratio + tau)

    dM_Rb_dt = phi_Rb * dM_dt
    
    dM_Mb_1_dt = prefactor[0] * (1 - phi_Rb - phi_O) * dM_dt
    if prefactor[1] == 0:
        dM_Mb_2_dt = phi_O * dM_dt
    else:
        dM_Mb_2_dt = (1 - phi_Rb) * dM_dt

    # tRNA dynamics
    dT_AA_star_dt = (prefactor[0] * nu_1 * M_Mb_1  + prefactor[1] * nu_2 * M_Mb_2- dM_dt) / M
    dT_AA_dt = (dM_dt - prefactor[0] * nu_1 * M_Mb_1 - prefactor[1] * nu_2 * M_Mb_2) / M
    if dil_approx == False:
        dT_AA_star_dt -= T_AA_star * dM_dt / M
        if tRNA_regulation:
            kappa = kappa_max * phi_Rb
        else:
            kappa = kappa_max
        dT_AA_dt += kappa - (T_AA * dM_dt) / M

    # Pack and return the output.
    out = [dM_dt, dM_Rb_dt, dM_Mb_1_dt, dM_Mb_2_dt, dT_AA_dt, dT_AA_star_dt]
    return out




