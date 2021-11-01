import numpy as np
import scipy
import pandas as pd

def load_constants(organism='ecoli'):
    """Returns constants frequently used in this work"""
    if organism == 'ecoli':
       params =  {'vtl_max': 20 ,  #Max translation speed in AA/s
                  'm_Rb': 7459, # Proteinaceous mass of ribosome  in AA
                  'Kd_cpc':  0.01, # precursor dissociation constant in abundance units 
                  'Kd_cnt': 5E-4, # Nutrient monod constant in M
                  'Y': 2.95E19, # Yield coefficient in  precursor mass per nutrient mass nutrient per L 
                  'OD_conv': 1.5E17, # Conversion factor from OD to AA mass.
                  'Kd_TAA': 2E-5, # uncharged tRNA dissociation constant in abundance units
                  'Kd_TAA_star': 2E-5, # Charged tRNA dissociation constant in abundance units
                  'kappa_max': (88 * 5 * 3600) / 1E9, # Maximum tRNA synthesis rate  in abundance units per unit time
                  'tau': 3 # ppGpp threshold parameter for charged/uncharged tRNA balance
                } 
    params['gamma_max'] = params['vtl_max'] * 3600 / params['m_Rb']
    return params


def self_replicator(params,
                    time,
                    gamma_max,
                    nu_max,
                    Y,
                    phi_Rb,
                    phi_Mb,
                    Kd_cpc,
                    Kd_cnt,
                    dil_approx=False):
    """
    Defines the system of ordinary differenetial equations (ODEs) which describe 
    the self-replicator ribosomal allocation model.

    Parameters
    ----------
    params: list, [M, M_Rb, M_Mb, c_pc, c_nt]
        A list of the parameters whose dynamics are described by the ODEs.
        M: positive float
            Total protein biomass of the system.
        M_Rb : positive float, must be < M 
            Ribosomal protein biomass of the system
        M_Mb : positive float, must be < M
            Metabolic protein biomass of the system 
        c_pc : positive float
            Concentration of precursors in the culture. This is normalized to 
            total protein biomass.
        c_nt : positive float
            Concentration of nutrients in the culture. This is in units of molar.
    time : float
        Evaluated time step of the system. This is only needed if using
        `scipy.integrate.odeint` or `scipy.integrate.solve_ivp` to evaluate 
        the system of equations.
    gamma_max: positive float 
        The maximum translational capacity in units of inverse time.
    nu_max : positive float
        The maximum nutritional capacity in units of inverse time. 
    Y: positive float
        The yield coefficient of the nutrient source in mass of amino acid 
        produced per mass of nutrient.
    phi_Rb : float, [0, 1]
        The ribosomal allocation factor.
    phi_Mb : float, [0, 1]
        The metabolic allocation factor.
    Kd_cpc : positive float 
        The effective dissociation constant of precursors to the elongating
        ribosome. This is in units of abundance.
    Kd_cnt: positive float
        The effective dissociation constant for growth on the specific nutrient 
        source. This is in units of molar.
    dil_approx: bool
        If True, then the approximation is made that the dilution of charged-tRNAs
        with growing biomass is negligible

    Returns
    -------
    out: list, [dM_dt,  dM_Rb_dt, dM_Mb_dt, dc_pc_dt, dc_nt_dt]
        A list of the evaluated ODEs at the specified time step.
        dM_dt : the dynamics of the total protein biomass.
        dM_Rb_dt : The dynamics of the ribosomal protein biomass.
        dM_Mb_dt : the dynamics of the metabolic protein biomass.
        dc_pc_dt : The dynamics of the precursor concentration.
        dc_nt_dt :  The dynamics of the nutrient concentration in the growth medium
    """
    # Unpack the parameters
    M, M_Rb, M_Mb, c_pc, c_nt = params

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
        dc_pc_dt = (nu * M_Mb - dM_dt) / M
    else:
        dc_pc_dt = (nu * M_Mb - (1 + c_pc) * dM_dt) / M
    dc_nt_dt = -nu * M_Mb / Y

    # Pack and return the output
    out = [dM_dt, dM_Rb_dt, dM_Mb_dt, dc_pc_dt, dc_nt_dt]
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


def self_replicator_ppGpp(params,
                          time,
                          gamma_max,
                          nu_max, 
                          tau, 
                          Kd_TAA,
                          Kd_TAA_star,
                          kappa_max = 0.01,
                          phi_O = 0,
                          phi_Rb = 0.1,
                          dil_approx = False,
                          dynamic_phiRb = True,
                          tRNA_regulation = True
                                  ):
    """
    Defines the system of ordinary differenetial equations (ODEs) which describe 
    the self-replicator model with ppGpp regulation.

    Parameters
    ----------
    params: list, [M, Mr, Mp, T_AA, T_AA_star]
        A list of the parameters whose dynamics are described by the ODEs.
        M : positive float 
            Total biomass of the system
        M_Rb : positive float, must be < M 
            Ribosomal protein biomass of the system
        M_Mb : positive float, must be < M
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
    Kd_TAA : positive float
        The effective dissociation constant for uncharged tRNA to the metabolic 
        machinery. In units of abundance.
    Kd_TAA_star: positive float
        The effective dissociation constant for charged tRNA to the ribosome complex.
        In units of abundance
    kappa_max : positive float
        The maximum rate of uncharged tRNA synthesis in abundance units per unit time.
    phi_O : float, [0, 1], optional
        The fraction of the proteome occupied by 'other' protein mass.
    phi_Rb : float, [0, 1], optional
        The prescribed value of phi_Rb to use. This only holds if 'dyanamic_phiRb' is True.
    dil_approx: bool
        If True, then the approximation is made that the dilution of charged-tRNAs
        with growing biomass is negligible.
    dynamic_phiRb: bool
        If True, phiRb will dynamically adjusted in reponse to charged/uncharged
        tRNA balance.
    tRNA_regulation: bool
        if True, tRNA abundance will be regulated the same way as dynamic_phiRb. 
    Returns
    -------
    out: list, [dM_dt, dM_Rb_dt, dM_Mb_dt, dT_AA_dt, dT_AA_star_dt]
        A list of the evaluated ODEs at the specified time step.

        dM_dt : The dynamics of the total protein biomass.
        dM_Rb_dt : The dynamics of the ribosomal protein biomass.
        dM_Mb_dt : the dynamics of the metabolic protein biomass.
        dT_AA_dt : The dynamics of the uncharged tRNA concentration.
        dT_AA_star_dt : The dynamics of the uncharged tRNA concentration.
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
    dT_AA_dt = (dM_dt - nu * M_Mb) / M
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
                        
    cols = ['M', 'M_Rb', 'M_Mb', 'T_AA', 'T_AA_star']

    # Set the timespans
    preshift_time = np.arange(0, shift_time, dt)
    postshift_time = np.arange(shift_time - dt, total_time, dt)

    # Integrate the preshift
    preshift_out = scipy.integrate.odeint(self_replicator_ppGpp,
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
        preshift_df['prescribed_phiR'] = preshift_df['tRNA_balance'].values / (preshift_df['tRNA_balance'].values + init_args[2])

    else:
        preshift_df['prescribed_phiR'] = init_args[-4]
   
    preshift_df['prescribed_phiMb'] = (1 - preshift_df['prescribed_phiR'] - init_args[-6])
    postshift_params = preshift_out[-1]

    if postshift_args == False:
        postshift_args = [init_args[i] for i in range(len(init_args))]
        postshift_args[1] = nu_postshift
        postshift_args = tuple(postshift_args)
    postshift_out = scipy.integrate.odeint(self_replicator_ppGpp,
                                                 postshift_params, 
                                                 postshift_time, 
                                                 args=postshift_args) 
    postshift_df = pd.DataFrame(postshift_out[1:], columns=cols)
    postshift_df['nu'] = nu_postshift
    postshift_df['phase'] = 'postshift'
    postshift_df['time'] = postshift_time[1:] 
    postshift_df['tRNA_balance'] = postshift_df['T_AA_star'].values / postshift_df['T_AA'].values
    if init_args[-2]:
        postshift_df['prescribed_phiR'] = postshift_df['tRNA_balance'].values / (postshift_df['tRNA_balance'].values + init_args[2]) 
    else:
        postshift_df['prescribed_phiR'] = postshift_args[-4]

    postshift_df['prescribed_phiMb'] = (1 - postshift_df['prescribed_phiR'] - postshift_args[-6])
 
    shift_df = pd.concat([preshift_df, postshift_df])

    # Compute properties
    shift_df['total_biomass'] = shift_df['M'].values
    shift_df['relative_biomass'] = shift_df['total_biomass'].values / (preshift_out[0][0])
    shift_df['realized_phiR'] = shift_df['M_Rb'].values / shift_df['total_biomass'].values
    shift_df['realized_phiMb'] = shift_df['M_Mb'].values / shift_df['total_biomass'].values
    shift_df['total_tRNA'] = shift_df['T_AA'].values + shift_df['T_AA_star'].values
    shift_df['gamma'] = init_args[0] * shift_df['T_AA_star'].values / (shift_df['T_AA_star'].values + init_args[3])
    return shift_df
    