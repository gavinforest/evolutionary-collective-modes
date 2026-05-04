import numpy as np
import cvxpy as cp
import pandas as pd
from qpsolvers import Problem, solve_problem
from scipy.integrate import cumulative_trapezoid
import gurobipy as gb
from gurobipy import GRB

# %% Save matrices in file


def mat2file(fname, Au, Al, S, Gu, Gl, beta, Sigmau, Sigmal, uimmutable, limmutable):
    np.savez(
        fname,
        Au=Au,
        Al=Al,
        S=S,
        Gu=Gu,
        Gl=Gl,
        beta=beta,
        Sigmau=Sigmau,
        Sigmal=Sigmal,
        uimmutable=uimmutable,
        limmutable=limmutable,
    )


# %% Random matrics

# %%% Biomass


def random_genes_biomass(c, g, p=1):

    # Want each constraint to have at least one gene and vice versa
    # p is the minimum average number of genes per constraint (OR-ness)

    G = np.zeros((c, g))

    if g >= c:  # There is some sharing of genes
        for i in range(c):
            G[i, i] = 1

        for i in range(c, g):
            G[np.random.randint(c), i] = 1

    else:  # Some genes code for multiple constraints
        for i in range(g):
            G[i, i] = 1
        for i in range(g, c):  # These are the extra constraints
            G[i, np.random.randint(g)] = 1

    G = G[:, np.random.choice(np.arange(g), size=g, replace=False)]
    # Shuffle genes - ensures that each constraint still has a gene

    while G.sum(axis=1).mean() < p:
        G[np.random.randint(c), np.random.randint(g)] = 1

    return G


def random_network_biomass_beta(
    m, r, n_import=1, n_export=1, z_max=4, nu_max=2, biomass_rxt=None
):

    # z_max is the maximum number of metabolites in a reaction
    # nu_max is the maximum stoichiometric coefficient for a metabolite

    nu_max = nu_max + 1  # randint is upper bound exclusive

    while True:
        S = np.zeros((m, r), dtype="float")

        # Exchange reactions
        for i in range(n_import):
            S[np.random.choice(m, 1), i] = 1
        for i in range(n_export):
            S[np.random.choice(m, 1), i + n_import] = -1
        n_exchange = n_import + n_export

        n_internal = r - n_exchange - 1

        # First we want to ensure every metabolite is used, so number of
        # metabolites in a reaction until we'll have used every metabolite
        num_metabolites_reactions = []
        for i in range(n_internal):
            num_metabolites_reactions.append(np.random.randint(2, z_max))
            if sum(num_metabolites_reactions) >= m:
                break

        metabolites_not_used = np.arange(m)
        n_r = 0

        # Cycle over the number of metabolites in each reaction and pick metabolites and coefficients
        for nmr, i in zip(
            num_metabolites_reactions,
            range(n_exchange, n_exchange + len(num_metabolites_reactions)),
        ):

            n_r += 1  # Keep track of how many reactions we actually make

            if len(metabolites_not_used) >= nmr:
                # We have enough unused metabolites to fill the current reaction

                ri_idx = np.random.choice(len(metabolites_not_used), nmr, replace=False)
                # We have to pick indices, not just metabolites because we are deleting from the list
                ri = metabolites_not_used[
                    ri_idx
                ]  # Metabolites involved in this reaction
                metabolites_not_used = np.delete(metabolites_not_used, ri_idx)
                # Delete the metabolites we just picked

                # something must be consumed
                S[ri[0], i] = -np.random.randint(1, nu_max)
                # something must be produced
                S[ri[1], i] = np.random.randint(1, nu_max)

                for j in ri[2:]:  # Everything else is either consumed or produced
                    S[j, i] = (
                        np.random.choice([1, -1], 1) * np.random.randint(1, nu_max)
                    )[0]

            elif len(metabolites_not_used) == 2:

                # We have exactly two metabolites not used, so we don't need to
                # pick indices, we just know one is consumed and one is produced

                # something must be consumed
                S[metabolites_not_used[0], i] = -np.random.randint(1, nu_max)
                # something must be produced
                S[metabolites_not_used[1], i] = np.random.randint(1, nu_max)

                break  # all metabolites used

            elif len(metabolites_not_used) == 1:  # Only 1 metabolite is not used

                ri = np.random.choice(
                    m, 1
                )  # Pick another random metabolite to be in the reaction
                ri = np.append(ri, metabolites_not_used)

                # something must be consumed
                S[ri[0], i] = -np.random.randint(1, nu_max)
                # something must be produced
                S[ri[1], i] = np.random.randint(1, nu_max)

                break  # All metabolites will have been used

            elif len(metabolites_not_used) <= nmr:

                # This is the weird option where we have 3 metabolites, but our
                # reaction needs 4. So, just ignore nmr and use all the remaining
                # metabolites

                ri_idx = np.random.choice(
                    len(metabolites_not_used), len(metabolites_not_used), replace=False
                )
                # print(ri_idx)
                ri = metabolites_not_used[ri_idx]
                metabolites_not_used = np.delete(metabolites_not_used, ri_idx)

                # something must be consumed
                S[ri[0], i] = -np.random.randint(1, nu_max)
                # something must be produced
                S[ri[1], i] = np.random.randint(1, nu_max)

                for j in ri[2:]:  # Everything else is either consumed or produced
                    S[j, i] = (
                        np.random.choice([1, -1], 1) * np.random.randint(1, nu_max)
                    )[0]

        # For the remaining reactions, we can just do everything randomly
        for i in range(n_exchange + n_r, r - 1):

            ri = np.random.choice(m, max([2, np.random.randint(z_max)]), replace=False)
            # Pick metabolites to be used, use at least 2!

            # something must be consumed
            S[ri[0], i] = -np.random.randint(1, nu_max)
            # something must be produced
            S[ri[1], i] = np.random.randint(1, nu_max)

            for j in ri[2:]:  # Everything else is either consumed or produced
                S[j, i] = (np.random.choice([1, -1], 1) * np.random.randint(1, nu_max))[
                    0
                ]

        # Make the biomass reaction
        if biomass_rxt is None:

            biomass_rxt = int(
                m * 0.1
            )  # Won't use every reactant in the biomass reaction
            biomass_idx = np.random.choice(m, biomass_rxt, replace=False)
            S[biomass_idx, -1] = np.random.randn(biomass_rxt)
            S[biomass_idx, -1] = (
                S[biomass_idx, -1] - np.mean(S[biomass_idx, -1])
            ).round(3)
            # Mean center it so that we have metabolites consumed and produced

        else:

            biomass_idx = np.random.choice(m, biomass_rxt, replace=False)
            S[biomass_idx, -1] = np.random.randn(biomass_rxt)
            S[biomass_idx, -1] = (
                S[biomass_idx, -1] - np.mean(S[biomass_idx, -1])
            ).round(3)

        # print(S)
        beta = np.zeros(r)
        beta[-1] = 1

        # The only constraint we might not have is that the biomass reaction
        # might not be in Null S, so we need to check this
        good_S = True
        u, s, vt = np.linalg.svd(S)
        V = vt[len(s[s > 1e-6]) :].T
        if np.all(np.round(V[-1, :], 8) == 0):
            good_S = False
        if good_S:
            break

    return S, beta


# %%%% No sharing


def random_constraints_nosharing(c, r):

    # We want each reaction to at least one constraint (except the biomass)

    A = np.zeros((c, r))
    if c >= r:
        for i in range(r - 1):
            A[i, i] = 1

        for i in range(r - 1, c):
            A[i, np.random.randint(r - 1)] = 1

    else:  # Not every reaction will be constrained
        for i in range(c):
            A[i, i] = 1

    A[:, :-1] = A[:, np.random.choice(np.arange(r - 1), size=r - 1, replace=False)]
    # Shuffle the columns so there is some randomness in how reactions are constrained

    return A


# %% Initializing simulations


def random_start(
    Au, Al, S, Gu, Gl, beta, uimmutable=[], limmutable=[], scale=1, irreversible=False
):
    while True:
        ubounds = np.random.random(Gu.shape[1]) * scale
        ubounds[[i[0] for i in uimmutable]] = [i[1] for i in uimmutable]
        lbounds = -np.random.random(Gl.shape[1]) * scale
        lbounds[[i[0] for i in limmutable]] = [-i[1] for i in limmutable]
        flux = np.random.random(S.shape[1])
        flux = nearest_feasible_gene(
            ubounds, lbounds, flux, Au, Al, S, Gu, Gl, irreversible=irreversible
        )
        biomass = beta.dot(flux)
        if biomass > 0:
            break
    return ubounds, lbounds, flux, biomass


def fixed_start(
    Au, Al, S, Gu, Gl, beta, uimmutable=[], limmutable=[], scale=1, irreversible=False
):
    ubounds = np.ones(Gu.shape[1]) * scale
    ubounds[[i[0] for i in uimmutable]] = [i[1] for i in uimmutable]
    lbounds = -np.ones(Gl.shape[1]) * scale
    lbounds[[i[0] for i in limmutable]] = [-i[1] for i in limmutable]
    biomass, flux = FBA_gene(
        ubounds, lbounds, Au, Al, S, Gu, Gl, beta, irreversible=irreversible
    )

    return ubounds, lbounds, flux, biomass


def fixednoisy_start(
    Au,
    Al,
    S,
    Gu,
    Gl,
    beta,
    epi=0.005,
    uimmutable=[],
    limmutable=[],
    scale=1,
    irreversible=False,
):
    ubounds = np.ones(Gu.shape[1]) * scale + np.random.randn(Gu.shape[1]) * epi * scale
    ubounds[[i[0] for i in uimmutable]] = [i[1] for i in uimmutable]
    lbounds = -np.ones(Gl.shape[1]) * scale + np.random.randn(Gl.shape[1]) * epi * scale
    lbounds[[i[0] for i in limmutable]] = [-i[1] for i in limmutable]
    biomass, flux = FBA_gene(
        ubounds, lbounds, Au, Al, S, Gu, Gl, beta, irreversible=irreversible
    )

    return ubounds, lbounds, flux, biomass


def biological_start(
    Au,
    Al,
    S,
    Gu,
    Gl,
    beta,
    epi=0.005,
    uimmutable=[],
    limmutable=[],
    scale=1,
    irreversible=False,
):

    ubounds = np.ones(Gu.shape[1]) * scale
    ubounds[[i[0] for i in uimmutable]] = [i[1] for i in uimmutable]
    lbounds = -np.ones(Gl.shape[1]) * scale
    lbounds[[i[0] for i in limmutable]] = [-i[1] for i in limmutable]

    biomass0, flux0 = FBA_gene(
        ubounds, lbounds, Au, Al, S, Gu, Gl, beta, irreversible=irreversible
    )

    ubounds_opt = cp.Variable(Gu.shape[1])
    lbounds_opt = cp.Variable(Gl.shape[1])

    obj = cp.Minimize(cp.sum_squares(ubounds_opt) + cp.sum_squares(lbounds_opt))
    con = [Au @ flux0 <= Gu @ ubounds_opt, Al @ flux0 >= Gl @ lbounds_opt]

    for c in uimmutable:
        con += [ubounds_opt[c[0]] == c[1]]

    for c in limmutable:
        con += [lbounds_opt[c[0]] == -c[1]]

    prob = cp.Problem(obj, con)
    prob.solve()

    while (
        True
    ):  # Add some noise to the bounds, but check to make sure nothing is infeasible
        ubounds_bio = ubounds_opt.value + np.random.rand(Gu.shape[1]) * epi * scale
        ubounds_bio[[i[0] for i in uimmutable]] = [i[1] for i in uimmutable]
        lbounds_bio = lbounds_opt.value - np.random.rand(Gl.shape[1]) * epi * scale
        lbounds_bio[[i[0] for i in limmutable]] = [-i[1] for i in limmutable]

        biomass_bio, flux_bio = FBA_gene(
            ubounds_bio, lbounds_bio, Au, Al, S, Gu, Gl, beta, irreversible=irreversible
        )

        if biomass_bio > 0:
            break

    return ubounds_bio, lbounds_bio, flux_bio, biomass_bio


# %% Moving flux (nearest feasible and FBA)


def nearest_feasible_gene(u_g, l_g, v0, Au, Al, S, Gu, Gl, irreversible=False):
    # nearest feasible function reformatted as a quadratic program
    M = np.identity(len(v0))
    P = M.T @ M
    q = -(M @ v0)
    K = np.vstack((Au, -Al))
    h = np.hstack((Gu @ u_g, Gl @ -l_g))
    A = S
    A = A.astype("float")
    b = np.zeros(len(S)).T

    if irreversible:
        problem = Problem(P, q, K, h=h, A=A, b=b, lb=np.zeros(len(v0)))
    else:
        problem = Problem(P, q, K, h=h, A=A, b=b)

    try:
        solution = solve_problem(problem, solver="quadprog")
        if solution.x is None:
            raise ValueError

    except:
        solution = solve_problem(problem, solver="proxqp", initvals=v0, eps_abs=1e-8)

    return solution.x


def nearest_feasible_flux(u_c, l_c, v0, Au, Al, S, irreversible=False):
    # nearest feasible function reformatted as a quadratic program
    M = np.identity(len(v0))
    P = M.T @ M
    q = -(M @ v0)
    K = np.vstack((Au, -Al))
    h = np.hstack((u_c, -l_c))
    A = S
    A = A.astype("float")
    b = np.zeros(len(S)).T

    if irreversible:
        problem = Problem(P, q, K, h=h, A=A, b=b, lb=np.zeros(len(v0)))
    else:
        problem = Problem(P, q, K, h=h, A=A, b=b)

    try:
        solution = solve_problem(problem, solver="quadprog")
        if solution.x is None:
            raise ValueError

    except:
        solution = solve_problem(problem, solver="proxqp", initvals=v0, eps_abs=1e-8)

    return solution.x


def FBA_flux(u_c, l_c, Au, Al, S, beta, return_lagrange=False, irreversible=False):
    v = cp.Variable(S.shape[1])
    objective = cp.Minimize(-beta @ v)
    constraints = [S @ v == 0, Au @ v <= u_c, Al @ v >= l_c]
    if irreversible:
        constraints += [v >= 0]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    if return_lagrange:
        return (
            -objective.value,
            v.value,
            constraints[0].dual_value,
            constraints[1].dual_value,
            constraints[2].dual_value,
        )
    else:
        return -objective.value, v.value


class FBA_gene_container:
    def __init__(
        self, Au, Al, S, Gu, Gl, beta, return_lagrange=False, irreversible=False
    ):
        self.Au = Au
        self.Al = Al
        self.S = S
        self.Gu = Gu
        self.Gl = Gl
        self.beta = beta
        self.return_lagrange = return_lagrange
        self.irreversible = irreversible

        # Set up Gurobi environment
        gurobi_env = gb.Env(empty=True)
        gurobi_env.setParam("OutputFlag", 0)
        gurobi_env.start()
        self.fba_model = gb.Model("FBA", env=gurobi_env)

        # Unchanging structure and constants of the problem
        self.v = self.fba_model.addMVar(shape=S.shape[1], name="v")
        self.stoich_constr = self.fba_model.addMConstr(
            S, self.v, GRB.EQUAL, np.zeros((S.shape[0],)), name="stoich"
        )

        # Unchanging structure, substitute in upper and lower values later
        self.v_upper_constr = self.fba_model.addMConstr(
            Au, self.v, GRB.LESS_EQUAL, np.zeros((Au.shape[0],)), name="v_upper"
        )
        self.v_lower_constr = self.fba_model.addMConstr(
            Al, self.v, GRB.GREATER_EQUAL, np.zeros((Al.shape[0],)), name="v_lower"
        )

        self.fba_model.setMObjective(None, beta, 0.0, None, None, self.v, GRB.MAXIMIZE)

    def optimize(self, u_g, l_g):
        av_upper = self.Gu @ u_g
        av_lower = self.Gl @ l_g
        self.v_upper_constr.setAttr("RHS", av_upper)
        self.v_lower_constr.setAttr("RHS", av_lower)

        self.fba_model.optimize()

        obj_value = self.fba_model.ObjVal

        if obj_value is None:

            if self.return_lagrange:
                return (
                    0,
                    np.zeros(self.S.shape[1]),
                    np.zeros(self.S.shape[0]),
                    np.zeros(self.Au.shape[0]),
                    np.zeros(self.Al.shape[0]),
                )
            else:
                return 0, np.zeros(self.S.shape[1])
        else:

            if self.return_lagrange:
                return (
                    obj_value,
                    self.v.X,
                    self.stoich_constr.Pi,
                    self.v_upper_constr.Pi,
                    self.v_lower_constr.Pi,
                )
            else:
                return -obj_value, self.v.X


def FBA_gene_gurobi(
    u_g, l_g, Au, Al, S, Gu, Gl, beta, return_lagrange=False, irreversible=False
):
    av_upper = Gu @ u_g
    av_lower = Gl @ l_g

    fba_model = gb.Model("FBA", env=gurobi_env)
    v = fba_model.addMVar(shape=S.shape[1], name="v")
    fba_model.addMConstr(S, v, GRB.EQUAL, np.zeros((S.shape[0],)), name="stoich")
    fba_model.addMConstr(Au, v, GRB.LESS_EQUAL, av_upper, name="v_upper")
    fba_model.addMConstr(Al, v, GRB.GREATER_EQUAL, av_lower, name="v_lower")
    fba_model.setMObjective(None, beta, 0.0, None, None, v, GRB.MAXIMIZE)

    fba_model.optimize()

    obj_value = fba_model.ObjVal

    # v = cp.Variable(S.shape[1])
    # objective = cp.Minimize(-beta @ v)
    # constraints = [S @ v == 0, Au @ v <= Gu @ u_g, Al @ v >= Gl @ l_g]
    # if irreversible:
    #     constraints += [v >= 0]
    # prob = cp.Problem(objective, constraints)

    # try:
    #     result = prob.solve(solver=solver, warm_start=True)
    # except:
    #     pass

    if obj_value is None:

        if return_lagrange:
            return (
                0,
                np.zeros(S.shape[1]),
                np.zeros(S.shape[0]),
                np.zeros(Au.shape[0]),
                np.zeros(Al.shape[0]),
            )
        else:
            return 0, np.zeros(S.shape[1])
    else:

        if return_lagrange:
            return (
                obj_value,
                v.X,
                fba_model.getConstrByName("stoich").Pi,
                fba_model.getConstrByName("v_upper").Pi,
                fba_model.getConstrByName("v_lower").Pi,
            )
        else:
            return -obj_value, v.X


def FBA_gene(
    u_g,
    l_g,
    Au,
    Al,
    S,
    Gu,
    Gl,
    beta,
    return_lagrange=False,
    irreversible=False,
    solver="GUROBI",
):
    v = cp.Variable(S.shape[1])
    objective = cp.Minimize(-beta @ v)
    constraints = [S @ v == 0, Au @ v <= Gu @ u_g, Al @ v >= Gl @ l_g]
    if irreversible:
        constraints += [v >= 0]
    prob = cp.Problem(objective, constraints)

    try:
        result = prob.solve(solver=solver, warm_start=True)
    except:
        pass

    if objective.value is None:

        if return_lagrange:
            return (
                0,
                np.zeros(S.shape[1]),
                np.zeros(S.shape[0]),
                np.zeros(Au.shape[0]),
                np.zeros(Al.shape[0]),
            )
        else:
            return 0, np.zeros(S.shape[1])
    else:

        if return_lagrange:
            return (
                -objective.value,
                v.value,
                constraints[0].dual_value,
                constraints[1].dual_value,
                constraints[2].dual_value,
            )
        else:
            return -objective.value, v.value


# %% Evolution (mutation, sampling, and fixation)


def mutate_bounds(x, p, scale=0.1, immutable=[], Sigma_cholesky=None):
    # Involves samples from Multivariate Normal using prefactored (cholesky decomp) covariance (L)
    # where Cov = L @ L.T
    r = len(x)
    effects = np.random.standard_normal(size=(r,)) * scale
    if Sigma_cholesky is not None:
        effects = effects @ Sigma_cholesky.T

    # choose which bounds are mutated independently with probability p
    effects *= np.random.random(r) < p
    effects[[i[0] for i in immutable]] = 0
    Xmut = x + effects
    # set bounds that went negative to zero
    Xmut[Xmut < 0] = 0
    Xmut[[i[0] for i in immutable]] = [i[1] for i in immutable]
    return Xmut


def sample_flux_p(v0, std, p):
    # random samples with mean "v0" and stdev "std"
    effects = np.random.randn(len(v0)) * std
    effects *= np.random.random(len(v0)) < p
    return v0 + effects


def random_fixation(s, pop_size):
    # using frequency of mutant as 1/pop_size
    if s != 0:
        prob_fix = (1 - np.exp(-2 * s)) / (1 - np.exp(-2 * pop_size * s))
    else:
        prob_fix = 1 / pop_size
    return prob_fix


# %% Selective pressure (empirical and FBA)


def selective_pressure_flux(
    Au, Al, S, Gu, Gl, beta, v0, u_g, l_g, u_comp, l_comp, delta=1e-6, push=1
):

    # v0 is current flux
    # u is current upper bounds
    # l is current lower bounds
    # u_c are upper bound constraints to examine selective pressure of
    # l_c are lower bound constraints to examine selective pressure of
    # n is the number of wiggles to average over
    # delta is the amount constraints are perturbed by

    u_c = Gu @ u_g
    l_c = Gl @ l_g

    SP_u = -np.ones((len(u_c)))
    SP_l = -np.ones((len(l_c)))

    u, s, vt = np.linalg.svd(S)
    V = vt[len(s) :].T
    b = V @ np.linalg.inv(V.T @ V) @ V.T @ beta
    b[b > 0] = 1
    b[b < 0] = -1

    for i_u in range(len(u_c)):

        u_c[i_u] += delta
        new_flux = nearest_feasible_flux(u_c, l_c, v0 + push * delta * b, Au, Al, S)
        SP_u[i_u] = (beta.T @ new_flux - beta.T @ v0) / delta
        u_c[i_u] -= delta

    for i_l in range(len(l_c)):

        l_c[i_l] -= delta
        new_flux = nearest_feasible_flux(u_c, l_c, v0 + push * delta * b, Au, Al, S)
        SP_l[i_l] = (beta.T @ new_flux - beta.T @ v0) / delta
        l_c[i_l] += delta

    m = np.min(np.vstack((SP_u, SP_l)))
    SP_u = SP_u - m
    SP_l = SP_l - m
    SP_u = SP_u.dot(u_comp)
    SP_l = SP_l.dot(l_comp)

    return SP_u, SP_l


def selective_pressure_gene(
    Au, Al, S, Gu, Gl, beta, v0, u_g, l_g, u_comp, l_comp, delta=1e-6, push=1
):

    # v0 is current flux
    # u is current upper bounds
    # l is current lower bounds
    # u_c are upper bound constraints to examine selective pressure of
    # l_c are lower bound constraints to examine selective pressure of
    # n is the number of wiggles to average over
    # delta is the amount constraints are perturbed by

    u, s, vt = np.linalg.svd(S)
    V = vt[len(s) :].T
    b = V @ np.linalg.inv(V.T @ V) @ V.T @ beta
    b[b > 0] = 1
    b[b < 0] = -1

    SP_u = np.zeros((len(u_g)))
    SP_l = np.zeros((len(l_g)))

    for i_u in range(len(u_g)):

        u_g[i_u] += delta
        new_flux = nearest_feasible_gene(
            u_g, l_g, v0 + push * delta * b, Au, Al, S, Gu, Gl
        )
        SP_u[i_u] = (beta.T @ new_flux - beta.T @ v0) / delta
        u_g[i_u] -= delta

    for i_l in range(len(l_g)):

        l_g[i_l] -= delta
        new_flux = nearest_feasible_gene(
            u_g, l_g, v0 + push * delta * b, Au, Al, S, Gu, Gl
        )
        SP_l[i_l] = (beta.T @ new_flux - beta.T @ v0) / delta
        l_g[i_l] += delta

    m = np.min(np.vstack((SP_u, SP_l)))
    SP_u = SP_u - m
    SP_l = SP_l - m
    SP_u = SP_u.dot(u_comp)
    SP_l = SP_l.dot(l_comp)

    return SP_u, SP_l


def selective_pressure_FBA_gene(Au, Al, S, Gu, Gl, beta, v0, u_g, l_g, u_comp, l_comp):

    # v0 is current flux
    # u is current upper bounds
    # l is current lower bounds
    # u_c are upper bound constraints to examine selective pressure of
    # l_c are lower bound constraints to examine selective pressure of

    _, _, _, SP_u, SP_l = FBA_gene(
        u_g, l_g, Au, Al, S, Gu, Gl, beta, return_lagrange=True
    )

    SP_u = (SP_u @ Gu).dot(u_comp)
    SP_l = (SP_l @ Gl).dot(l_comp)

    return SP_u, SP_l


def selective_pressure_FBA_flux(Au, Al, S, Gu, Gl, beta, v0, u_g, l_g, u_comp, l_comp):

    # v0 is current flux
    # u is current upper bounds
    # l is current lower bounds
    # u_c are upper bound constraints to examine selective pressure of
    # l_c are lower bound constraints to examine selective pressure of

    _, _, _, SP_u, SP_l = FBA_gene(
        u_g, l_g, Au, Al, S, Gu, Gl, beta, return_lagrange=True
    )

    SP_u = SP_u.dot(u_comp)
    SP_l = SP_l.dot(l_comp)

    return SP_u, SP_l


def integrateSP(SP, mask=None):

    SP_pivot = pd.pivot(
        data=SP, columns="Vector", index="Time", values="Selective Pressure"
    )

    if mask is not None:

        SP_int = cumulative_trapezoid(
            SP_pivot[mask].to_numpy(), x=SP_pivot.index[mask], axis=0, initial=0
        )
        return pd.DataFrame(
            data=SP_int, index=SP_pivot.index[mask], columns=SP_pivot.columns
        )

    else:

        SP_int = cumulative_trapezoid(
            SP_pivot.to_numpy(), x=SP_pivot.index, axis=0, initial=0
        )
        return pd.DataFrame(data=SP_int, index=SP_pivot.index, columns=SP_pivot.columns)


# %% Other


def genes2constraints(df, G, labels=None):

    c = G @ pd.pivot(df, columns="Gene", index="Time", values="Bound").T

    if labels is not None:
        c.index = labels
        c.index.name = "Constraint"
    else:
        c.index = ["%d" % i for i in range(len(c.index))]
        c.index.name = "Constraint"

    return c


def optimal_fitness(df_flux, df_ubounds, df_lbounds, Au, Al, S, Gu, Gl, beta):

    np_ubounds = pd.pivot(
        data=df_ubounds, columns="Gene", index="Time", values="Bound"
    ).to_numpy()
    np_lbounds = pd.pivot(
        data=df_lbounds, columns="Gene", index="Time", values="Bound"
    ).to_numpy()
    np_flux = pd.pivot(
        data=df_flux, columns="Reaction", index="Time", values="Relative Flux"
    ).to_numpy()

    optimal_flux = np.zeros(np_flux.shape)

    for i in range(np_ubounds.shape[0]):

        _, optimal_flux[i, :] = FBA_gene(
            np_ubounds[i, :], np_lbounds[i, :], Au, Al, S, Gu, Gl, beta
        )

    fitness_dif = beta.T @ optimal_flux.T - beta.T @ np_flux.T

    return fitness_dif


def FBAlamvsest(Au, Al, S, Gu, Gl, beta, u_g, l_g, delta=0.001):

    f0, _, _, lam_u_FBA, lam_l_FBA = FBA_gene(
        u_g, l_g, Au, Al, S, Gu, Gl, beta, return_lagrange=True
    )

    lamg_u_est = np.zeros(u_g.shape)
    for i in range(len(u_g)):

        u_g[i] += delta
        fnew, _ = FBA_gene(u_g, l_g, Au, Al, S, Gu, Gl, beta)
        lamg_u_est[i] = (fnew - f0) / delta
        u_g[i] -= delta

    lamg_l_est = np.zeros(l_g.shape)
    for i in range(len(l_g)):

        l_g[i] -= delta
        fnew, _ = FBA_gene(u_g, l_g, Au, Al, S, Gu, Gl, beta)
        lamg_l_est[i] = (fnew - f0) / delta
        l_g[i] += delta

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2)

    ax[0].set_title("Upper")
    ax[0].scatter(lam_u_FBA @ Gu, lamg_u_est)
    ax[0].axline([0, 0], [1, 1], color="r")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_aspect("equal", "box")
    ax[0].set_xlabel("Dual")
    ax[0].set_ylabel("Estimated")
    ax[0].set_xlim([1e-15, 1e3])
    ax[0].set_ylim([1e-15, 1e3])

    ax[1].set_title("Lower")
    ax[1].scatter(lam_l_FBA @ Gl, lamg_l_est)
    ax[1].axline([0, 0], [1, 1], color="r")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_aspect("equal", "box")
    ax[1].set_xlabel("Dual")
    ax[1].set_xlim([1e-15, 1e3])
    ax[1].set_ylim([1e-15, 1e3])

    return lamg_u_est, lamg_l_est
