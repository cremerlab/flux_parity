import numpy as np


def self_replicator(params,
                    time,
                    gamma_max,
                    nu_max,
                    omega,
                    phi_Rb,
                    phi_Mb,
                    Kd_cAA=0.025,
                    Kd_cN=5E-4,
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
    out: list, [dM_dt, dMr_dt, dMp_dt, dcAA_dt, dcN_dt]
        A list of the evaluated ODEs at the specified time step.
        dMr_dt : The dynamics of the ribosomal protein biomass.
        dMp_dt : the dynamics of the metabolic protein biomass.
        dcAA_dt : The dynamics of the precursor concentration.
        dcN_dt :  The dynamics of the nutrient concentration in the growth medium
    """
    # Unpack the parameters
    M_r, M_p, c_AA, c_N = params

    # Compute the capacities
    gamma = gamma_max * (c_AA / (c_AA + Kd_cAA))
    nu = nu_max * (c_N / (c_N + Kd_cN))

    # Biomass accumulation
    dM_dt = gamma * M_r

    # Resource allocation
    dMr_dt = phi_Rb * dM_dt
    dMp_dt = phi_Mb * dM_dt

    # Precursor dynamics
    if dil_approx:
        dcAA_dt = (nu * M_p - dM_dt) / (M_r + M_p)
    else:
        dcAA_dt = (nu * M_p - (1 + c_AA) * dM_dt) / (M_r + M_p)
    dcN_dt = -nu * M_p / omega

    # Pack and return the output
    return [dMr_dt, dMp_dt, dcAA_dt, dcN_dt]

def steady_state_cAA(gamma_max, phi_Rb, nu_max, Kd_cAA):
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
    Kd_cAA : positive float
        The effective dissociation constant of the precursors to the ribosome. 

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
    ss_mu = steady_state_mu(gamma_max, phi_Rb, nu_max, Kd_cAA)
    return (nu_max * (1 - phi_Rb) / ss_mu) - 1

def steady_state_mu(gamma_max, phi_Rb, nu_max, Kd_cAA):
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
    Kd_cAA :  positive float
        The effective dissociation constant of charged tRNA to the elongating
        ribosome.

    Returns
    -------
    mu : float 
        The physically meaningful growth rate (in units of inverse time) given 
        the provided model parameeters.

    Notes
    -----
    This function assumes that in steady state, the nutrients are in such abundance 
    that the nutritional capacy is equal to its maximal value. 
    """
    Nu = nu_max * (1 - phi_Rb)
    Gamma = gamma_max * phi_Rb
    numer = Nu + Gamma - np.sqrt((Nu + Gamma)**2 - 4 * (1 - Kd_cAA) * Nu * Gamma)
    denom = 2 * (1 - Kd_cAA)
    return numer / denom 

def steady_state_gamma(gamma_max, phi_Rb, nu_max, Kd_cAA):
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
    Kd_cAA : positive float 
        The effective dissociation constant of charged tRNA to the elongating
        ribosome.

    Returns
    -------
    gamma : positive float
        The translational efficiency in units of inverse time
    """

    c_AA = steady_state_cAA(gamma_max, phi_Rb, nu_max, Kd_cAA)
    return gamma_max * (c_AA / (c_AA + Kd_cAA))


def phi_R_optimal_allocation(gamma_max, nu_max, Kd_cAA):
    """
    Computes the optimal fraction of proteome that is occupied by ribosomal 
    proteins which maximizes the growth rate. 

    Parameters
    ----------
    gamma_max : positive float 
        The maximum translational efficiency in units of inverse time.
    nu_max : positive float
        The maximum nutritional capacity in units of inverse time.
    Kd_cAA: positive float 
        The effective dissociation constant of charged tRNA to the elongating 
        ribosome.

    Returns
    -------
    phi_Rb_opt : positive float [0, 1]
        The optimal allocation to ribosomes.
    """
    numer = nu_max * (-2 * Kd_cAA * gamma_max + gamma_max + nu_max) +\
        np.sqrt(Kd_cAA * gamma_max * nu_max) * (gamma_max - nu_max)
    denom = -4 * Kd_cAA * gamma_max * nu_max + gamma_max**2 + 2 * gamma_max * nu_max + nu_max**2
    phi_Rb_opt = numer / denom
    # prefix = (4 * Kd_cAA * gamma_max * nu_max - (nu_max + gamma_max)**2)**-1
    # bracket = 2 * Kd_cAA * gamma_max * nu_max - gamma_max * nu_max +\
                # np.sqrt(Kd_cAA * gamma_max * nu_max)*(nu_max - gamma_max) -nu_max**2
    return phi_Rb_opt


def batch_culture_self_replicator_ppGpp(params,
                                  time,
                                  gamma_max,
                                  nu_max, 
                                  tau = 1,
                                  Kd_TAA_star = 0.025,
                                  Kd_TAA = 0.025,
                                  dil_approx=False,
                                  num_muts=1):
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
    M_r, M_p, T_AA, T_AA_star = params

    # Compute the capacities
    gamma = gamma_max * (T_AA_star / (T_AA_star + Kd_TAA_star))
    nu = nu_max * (T_AA / (T_AA + Kd_TAA))

    # Compute the active fraction
    ratio = T_AA_star / T_AA

    # Biomass accumulation
    dM_dt = gamma * M_r

    # Resource allocation
    phi_R = ratio / (ratio + tau)
    dMr_dt = phi_R * dM_dt
    dMp_dt = (1 - phi_R) * dM_dt

    dT_AA_star_dt = (nu * M_p - dM_dt) / (M_r + M_p)
    dT_AA_dt = (dM_dt - nu * M_p) / (M_r + M_p)

    # Pack and return the output.
    out = [dMr_dt, dMp_dt, dT_AA_dt, dT_AA_star_dt]
    return out




def batch_culture_self_replicator_cAA(params,
                                  time,
                                  gamma_max,
                                  nu_max, 
                                  Kd_cAA = 0.025,
                                  dil_approx=False,
                                  num_muts=1):
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
        c_AA
    time : float
        Evaluated time step of the system.
    gamma_max: positive float 
        The maximum translational capacity in units of inverse time.
    nu_max : positive float
        The maximum nutritional capacity in units of inverse time. 
    Kd_cAA : positive float 
        The effective dissociation constant of precursors to the elongating
        ribosome. This is in units of mass fraction.

    Returns
    -------
    out: list, [dM_dt, dMr_dt, dMp_dt, dcAA_dt]
        A list of the evaluated ODEs at the specified time step.

        dMr_dt : The dynamics of the ribosomal protein biomass.
        dMp_dt : the dynamics of the metabolic protein biomass.
        dcAA_dt : The dynamics of the precursor concentration.
    """
    # Unpack the parameters
    M_r, M_p, c_AA = params

    # Compute the capacities
    gamma = gamma_max * (c_AA / (c_AA + Kd_cAA))


    # Biomass accumulation
    dM_dt = gamma * M_r

    # Resource allocation
    phi_R = c_AA / (c_AA + Kd_cAA)
    dMr_dt = phi_R * dM_dt
    dMp_dt = (1 - phi_R) * dM_dt

    dcAA_dt = (nu_max * M_p - dM_dt * (1 + c_AA)) / (M_r + M_p)

    # Pack and return the output.
    out = [dMr_dt, dMp_dt, dcAA_dt]
    return out