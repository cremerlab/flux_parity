import numpy as np
import scipy
import pandas as pd


def load_constants():
    """Returns constants frequently used in this work"""
    params = {'vtl_max': 20,  # Max translation speed in AA/s
              'm_Rb': 7459,  # Proteinaceous mass of ribosome  in AA
              'Kd_cpc': 0.03,  # precursor dissociation constant in abundance units
              'Kd_cnt': 5E-4,  # Nutrient monod constant in M
              'Y': 2.95E19,  # Yield coefficient
              'OD_conv': 1.5E17,  # Conversion factor from OD to AA mass.
              'Kd_TAA': 3E-5,  # uncharged tRNA dissociation constant in abundance units
              'Kd_TAA_star': 3E-5,  # Charged tRNA dissociation constant in abundance units
              # Maximum tRNA synthesis rate  in abundance units per unit time
              'kappa_max': (64 * 5 * 3600) / 1E9,
              'tau': 1,  # ppGpp threshold parameter for charged/uncharged tRNA balance
              # Fraction of proteome deoveted to `other` proteins for E. coli.
              'phi_O': 0.55,
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
                    f_a=1,
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
    f_a : float, [0,1]
        The fraction of the ribosomes actively translating. 
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
    dM_dt = f_a * gamma * M_Rb

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


def steady_state_precursors(gamma_max,
                            phi_Rb,
                            nu_max,
                            Kd_cpc,
                            phi_O):
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
    phi_O : float [0, 1]
        Allocation towards other proteins.
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
    ss_lam = steady_state_growth_rate(
        gamma_max, phi_Rb, nu_max, Kd_cpc, phi_O=phi_O)
    cpc = (nu_max * (1 - phi_Rb - phi_O) / ss_lam) - 1
    return cpc


def steady_state_growth_rate(gamma_max,
                             phi_Rb,
                             nu_max,
                             Kd_cpc,
                             phi_O):
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
    phi_O : float [0, 1]
        Allocation towards other proteins.

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
    numer = Nu + Gamma - \
        np.sqrt((Nu + Gamma)**2 - 4 * (1 - Kd_cpc) * Nu * Gamma)
    denom = 2 * (1 - Kd_cpc)
    lam = numer / denom
    return lam


def steady_state_gamma(gamma_max,
                       phi_Rb,
                       nu_max,
                       Kd_cpc,
                       phi_O=0):
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
    phi_O : float [0, 1]
        Allocation towards other proteins

    Returns
    -------
    gamma : positive float
        The translational efficiency in units of inverse time
    """

    c_pc = steady_state_precursors(
        gamma_max, phi_Rb, nu_max, Kd_cpc, phi_O=phi_O)
    return gamma_max * (c_pc / (c_pc + Kd_cpc))


def phiRb_optimal_allocation(gamma_max,
                             nu_max,
                             Kd_cpc,
                             phi_O):
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
    phi_O : float [0, 1]
        Allocation towards other proteins.
    Returns
    -------
    phi_Rb_opt : positive float [0, 1]
        The optimal allocation to ribosomes.
    """
    numer = nu_max * (-2 * Kd_cpc * gamma_max + gamma_max + nu_max) +\
        np.sqrt(Kd_cpc * gamma_max * nu_max) * (gamma_max - nu_max)
    denom = -4 * Kd_cpc * gamma_max * nu_max + \
        gamma_max**2 + 2 * gamma_max * nu_max + nu_max**2
    phi_Rb_opt = (1 - phi_O) * numer / denom
    return phi_Rb_opt


def phiRb_constant_translation(gamma_max,
                               nu_max,
                               cpc_Kd,
                               Kd_cpc,
                               phi_O):
    """
    Computes the ribosomal allocation which maintains a high translation rate. 

    Parameters
    ----------
    gamma_max : positive float 
        The maximum translational efficiency in units of inverse time.
    nu_max : positive float
        The maximum nutritional capacity in units of inverse time.
    phi_O : float [0, 1]
        Allocation towards other proteins. 

    Returns
    -------
    phi_Rbt : positive float [0, 1]
        The ribosomal allocation for constant translation.
    """
    c_pc = cpc_Kd * Kd_cpc
    return (1 - phi_O) * nu_max * (c_pc + Kd_cpc) / (nu_max * (c_pc + Kd_cpc) + gamma_max * c_pc * (c_pc + 1))


def self_replicator_FPM(params,
                        time,
                        args):
    """
    Defines the system of ordinary differenetial equations (ODEs) which describe 
    the self-replicator model with ppGpp regulation.

    Parameters
    ----------
    params: list, [M, Mr, Mp, (c_nt), T_AA, T_AA_star]
        A list of the parameters whose dynamics are described by the ODEs.
        M : positive float 
            Total biomass of the system
        M_Rb : positive float, must be < M 
            Ribosomal protein biomass of the system
        M_Mb : positive float, must be < M
            Metabolic protein biomass of the system 
        c_nt : positive float, optional
            The nutrient concentration in the environment. This should only 
            be provided if 'nutrients' is not False in the supplied arguments.
        T_AA_star : positive float
            Concentration of charged tRNAs in the culture. This is normalized to 
            total protein biomass.
        T_AA : positive float
            Concentration of uncharged tRNAs in the culture. This is normalized to 
            total protein biomass.
    time : float
        Evaluated time step of the system.
    args: dict 
        Dictionary of argument terms as follows
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
        nutrients: bool or dict
            If False, nutrients will not be explicitly modeled and will be taken to 
            be saturating. If a dictionary is supplied, nutrients will be modeled 
            with following parameters
            Kd_cnc : float [0, inf)
                The effective dissociation constant of nutrients in the 
                to the metabolic machinery. 
            Y : float [0, inf)
                The yield coefficient of turning nutrients into precursors.

        dynamic_phiRb: bool or dict
            If True, phiRb will dynamically adjusted in reponse to charged/uncharged
            tRNA balance. If a dictionary is provided, seeded phiRb will be used.  
                phiRb: float [0, 1]
                    The seeded phiRb to be used.
        tRNA_regulation: bool
            if True, tRNA abundance will be regulated the same way as dynamic_phiRb.
            If False, kappa_max will be used directly. 
        antibiotic: bool
            If False, antiboitic presence will not be modeld and the fraction 
            of active ribosomes will be taken to be unity. If a dictionary is 
            provided with the following terms, the influence of antibiotics will
            be explicitly modeled.
                drug_conc : float [0, inf)
                    The concentration of the applied antibiotic
                Kd_drug : float [0, inf)
                    The effective dissociation constant of the drug to the ribosome.
        f_a : float, [0, 1]
            The faraction of ribosomes actively translating. 
        dil_approx: bool
            If True, then the approximation is made that the dilution of charged-tRNAs
            with growing biomass is negligible.

    Returns
    -------
    out: list, [dM_dt, dM_Rb_dt, dM_Mb_dt, (dc_nt_dt), dT_AA_dt, dT_AA_star_dt]
        A list of the evaluated ODEs at the specified time step.

        dM_dt : The dynamics of the total protein biomass.
        dM_Rb_dt : The dynamics of the ribosomal protein biomass.
        dM_Mb_dt : the dynamics of the metabolic protein biomass.
        dc_nt_dt : The dynamics of the nutrients in the environment, if modeled.
        dT_AA_dt : The dynamics of the uncharged tRNA concentration.
        dT_AA_star_dt : The dynamics of the uncharged tRNA concentration.
    """

    # Unpack the parameters
    if 'nutrients' in args.keys():
        M, M_Rb, M_Mb, c_nt, T_AA, T_AA_star = params
    else:
        M, M_Rb, M_Mb, T_AA, T_AA_star = params

    # Compute the capacities
    gamma = args['gamma_max'] * (T_AA_star / (T_AA_star + args['Kd_TAA_star']))
    if 'nutrients' in args.keys():
        pref = c_nt / (c_nt + args['nutrients']['Kd_cnt'])
    else:
        pref = 1
    nu = pref * args['nu_max'] * (T_AA / (T_AA + args['Kd_TAA']))

    # Compute the active fraction
    ratio = T_AA_star / T_AA

    fa = 1
    if 'antibiotic' in args.keys():
        fa -= args['antibiotic']['c_drug'] / \
            (args['antibiotic']['c_drug'] + args['antibiotic']['Kd_drug'])

    # Biomass accumulation
    if 'f_a' not in args.keys():
        f_a = 1
    else:
        f_a = args['f_a']
    dM_dt = f_a * gamma * M_Rb

    # Resource allocation
    if 'ansatz' in args.keys():
        if args['ansatz'] == 'binding':
            allocation = T_AA_star / (T_AA_star + T_AA)
    else:
        allocation = ratio / (ratio + args['tau'])

    if 'phiRb' not in args.keys():
        phiRb = (1 - args['phi_O']) * allocation
        kappa = args['kappa_max'] * allocation
    else:
        phiRb = args['phiRb']
        kappa = phiRb * args['kappa_max'] / (1 - args['phi_O'])

    dM_Rb_dt = phiRb * dM_dt
    dM_Mb_dt = (1 - phiRb - args['phi_O']) * dM_dt

    # Core tRNA dynamics
    dT_AA_star_dt = (nu * M_Mb - dM_dt) / M
    dT_AA_dt = (dM_dt - nu * M_Mb) / M

    # Dilution terms
    dT_AA_star_dt -= T_AA_star * dM_dt / M
    dT_AA_dt += kappa - (T_AA * dM_dt) / M

    if 'nutrients' in args.keys():
        dcnt_dt = -nu * M_Mb / args['nutrients']['Y']
        out = [dM_dt, dM_Rb_dt, dM_Mb_dt, dcnt_dt, dT_AA_dt, dT_AA_star_dt]
    else:
        out = [dM_dt, dM_Rb_dt, dM_Mb_dt, dT_AA_dt, dT_AA_star_dt]
    return out
