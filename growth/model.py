import numpy as np 

def batch_culture_self_replicator(params, 
                           time, 
                           gamma_max, 
                           nu_max, 
                           omega,
                           phi_R, 
                           phi_P,
                           Kd=0.025, 
                           Km=5E-4,
                           num_muts=1,
                           vol=1E-3):
    """
    Defines the system of ordinary differenetial equations (ODEs) which describe 
    the self-replicator model in batch culture conditions.

    Parameters
    ----------
    params: list, [M, Mr, Mp, m_AA, m_N]
        A list of the parameters whose dynamics are described by the ODEs.
        M : positive float
            Total protein biomass of the system
        M_r : positive float, must be < M 
            Ribosomal protein biomass of the system
        M_p : positive float, must be < M
            Metabbolic protein biomass of the system 
        m_AA : positive float
            Total mass of precursors in the culture. This is normalized to 
            total protein biomass when calculating the translational 
            capacity.
        m_N : positive float
            Total mass of nutrients in the system.
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
    phi_P : float, [0, 1] 
        The fraction of the proteome occupied by metabolic protein mass  
    Kd : positive float 
        The effective dissociation constant of charged tRNA to the elongating
        ribosome. This is in units of mass fraction.
    Km : positive float
        The Monod constant for growth on the specific nutrient source. 
        This is in units of molar.
    num_muts: int
        The number of mutants whose dynamics need to be tracked.
    vol: float, default 1 mL
        The volume of the system for calculation of the nutrient concentration.

    Returns
    -------
    out: list, [dM_dt, dMr_dt, dMp_dt, dmAA_dt, dmN_dt]
        A list of the evaluated ODEs at the specified time step.

        dM_dt : The dynamics of the total protein biomass.
        dMr_dt : The dynamics of the ribosomal protein biomass.
        dMp_dt : the dynamics of the metabolic protein biomass.
        dmAA_dt : The dynamics of the charged-tRNA pool.
        dmN_dt :  The dynamics of the nutrient mass in the growth medium
    """

    # Define constants
    AVO = 6.022E23 # Avogadro's number

    # Unpack the parameters
    if num_muts > 1:
        m_N = params[-1]
        M, M_r, M_p, m_A = np.reshape(params[:-1], (4, num_muts))
    else: 
        M, M_r, M_p, m_AA, m_N = params

    # Compute the number of precursors and the number of nutrients
    c_AA = m_AA / M
    c_N = m_N / (AVO * vol) 

    # Compute the capacities
    gamma = gamma_max * (c_AA / (c_AA + Kd))
    nu = nu_max * (c_N / (c_N + Km))

    # Biomass accumulation
    dM_dt = gamma * M_r

    # Resource allocation
    dMr_dt = phi_R * dM_dt
    dMp_dt = phi_P * dM_dt

    # Precursor dynamics
    dmAA_dt = nu * M_p - (1 + c_AA) * dM_dt
    dmN_dt = -nu * M_p / omega

    # Pack and return the output.
    out = [dM_dt, dMr_dt, dMp_dt, dmAA_dt]
    if num_muts > 1:
        dmN_dt = np.sum(dmN_dt)
        out = [value for deriv in out for value in deriv]
    out.append(dmN_dt)
    return out

def steady_state_growth_rate(gamma_max,
                             nu_max, 
                             phi_R, 
                             phi_P, 
                             Kd, 
                             f_a=1):
    """
    Computes the steady-state growth rate of the self-replicator model. 

    Parameters
    ----------
    gamma_max : positive float 
        The maximum translational capacity in units of inverse time.
    nu_max : positive float 
        The maximum nutritional capacity in units of inverse time. 
    phi_R : float [0, 1]
        The fraction of the proteome occupied by ribosomal protein mass
    phi_P : float [0, 1]
        The fraction of the proteome occupied by metabolic protein mass  
    Kd :  positive float
        The effective dissociation constant of charged tRNA to the elongating
        ribosome.
    f_a : float [0, 1], optional
        The fraction of the ribosome pool which is actively translating. Default 
        is 1.0.
    
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
    term_a = Kd - 1
    term_b = nu_max * phi_P + gamma_max * phi_R * f_a
    term_c = nu_max * phi_P * gamma_max * phi_R * f_a
    lam  = (-term_b + np.sqrt(term_b**2 + 4 * term_a * term_c)) / (2 * term_a)
    return lam

def steady_state_tRNA_balance(nu_max, 
                 phi_P, 
                 growth_rate):
    """
    Computes the steady state value of the charged-tRNA abundance.

    Parameters
    ----------
    nu_max : positive float 
        The maximum nutritional capacity in units of inverse time. 
    phi_P : float [0, 1]
        The fraction of the proteome occupied by metabolic protein mass  
    growth_rate : positive float
        The steady-state growth rate given the parameters

    Returns
    -------
    c_AA : float
        The charged tRNA abunadance given the model parameters. This is defined
        as the mass of the charged-tRNA relative to the total biomass.

    Notes
    -----
    This function assumes that in steady state, the nutrients are in such abundance 
    that the nutritional capacy is equal to its maximal value. 
    
    """ 
    return (nu_max * phi_P / growth_rate) - 1

def sstRNA_balance(nu_max, phi_P, gamma_max, phi_R, Kd, f_a=1):
    alpha = (f_a * phi_R  * gamma_max) / (nu_max * phi_P)
    return ((1 - alpha) + np.sqrt((alpha - 1)**2 + 4 * Kd * alpha))  / (2 * alpha) 

def translation_rate(gamma_max, c_AA, Kd):
    return gamma_max * c_AA  / (c_AA + Kd)

def phi_R_optimal_allocation(gamma_max, nu_max, Kd, phi_O, f_a=1):
    term_a = phi_O - 1
    term_b = -nu_max * (-2 * Kd * gamma_max * f_a  + gamma_max * f_a + nu_max)
    term_c = np.sqrt(Kd * gamma_max * f_a * nu_max) * (-gamma_max * f_a + nu_max)
    denom = -4 * Kd * gamma_max * f_a * nu_max + (f_a * gamma_max)**2 + 2 * gamma_max * f_a * nu_max + nu_max**2
    return (term_a * (term_b + term_c)) / denom

def phi_R_max_translation(gamma_max, nu_max, phi_O, f_a=1):
    numer = nu_max * (phi_O - 1)
    denom = f_a * gamma_max + nu_max
    return -numer / denom


