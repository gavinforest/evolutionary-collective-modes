import marimo

__generated_with = "0.23.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Example simulations of toy metabolic networks

    This notebook runs evolutionary simulation of two toy metabolic networks described in the text: the original toy network and the augmented toy network with five genes for reaction 2.

    The functions to run and analyze a simulation are included in the evcm module.
    """)
    return


@app.cell
def _():
    import evcm

    return (evcm,)


@app.cell
def _():
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import polars as pl

    return np, pl, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Run simulation of toy network

    The matrices for the original toy network are saved as a .npz file in the networks folder. $a_{max}$ is infinite for all bounds, so uimmutable and limmutable are empty lists.
    """)
    return


@app.cell
def _(np):
    toy_directory = "./networks/toynet.npz"

    toy_mats = np.load(toy_directory)
    S = toy_mats["S"]
    beta = toy_mats["beta"]
    Au = toy_mats["Au"]
    Al = toy_mats["Al"]
    Gu = toy_mats["Gu"]
    Gl = toy_mats["Gl"]
    uimmutable = [list(i) for i in toy_mats["uimmutable"]]
    limmutable = [list(i) for i in toy_mats["limmutable"]]
    return Al, Au, Gl, Gu, S, beta, limmutable, uimmutable


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The key hyperparameters mentioned in the supplement are assigned below.
    """)
    return


@app.cell
def _():
    # Hyperparameters mentioned in the supplement
    T = 30001  # Length of simulation
    N = 10000000  # Population size
    ms = 0.1  # Mutation scale
    em = 1  # Expected number of mutations per step
    ss = 10  # Simulation scale
    return N, T, em, ms, ss


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `run_sim()` is the main function that performs the mutation+fixation/extinction loop. Various utilities for the simulation (generating random matrices, generating mutations, etc.) are included in `evcm.utils`. `run_sim()` takes in more hyperparameters than indicated here. These hyperparameters are related to different variations of the evolutionary simulations that are not used in the paper and are described in `run_sim()`.
    """)
    return


@app.cell
def _(Al, Au, Gl, Gu, N, S, T, beta, em, evcm, limmutable, ms, ss, uimmutable):
    (
        df_flux,
        df_ubounds,
        df_lbounds,
        df_biomass,
        df_umutation,
        df_lmutation,
        df_neutral,
        *_,
    ) = evcm.sim.run_sim(
        T=T,
        Au=Au,
        Al=Al,
        S=S,
        Gu=Gu,
        Gl=Gl,
        beta=beta,
        pop_size=N,
        mutate_scale=ms,
        simulation_scale=ss,
        expected_mutations=em,
        uimmutable=uimmutable,
        limmutable=limmutable,
        Reaction_labels=["1", "2", "3", "4"],
        uGene_labels=["1", "2", "3", "4"],
        lGene_labels=["1", "2", "3", "4"],
    )
    return (df_flux,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Basic analysis

    We can perform some basic analysis on the the simulation results.
    """)
    return


@app.cell
def _(df_flux, pl):
    df_flux_1 = pl.from_pandas(df_flux)
    return (df_flux_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Flux as a function of time steps
    """)
    return


@app.cell
def _(df_flux_1, plt, sns):
    sns.lineplot(df_flux_1, x="Time", y="Flux", hue="Reaction")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Normalized, rolling average change in flux

    This shows that an EvCM exists, as the direction of evolution is approximately constant.
    """)
    return


@app.cell
def _(df_flux_1, pl, sns):
    sns.lineplot(
        df_flux_1.with_columns(
            pl.col("Flux").diff().rolling_mean(window_size=500).over("Reaction")
        )
        .with_columns((pl.col("Flux") / pl.col("Flux").max()).over("Time"))
        .drop_nulls(),
        x="Time",
        y="Flux",
        hue="Reaction",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Run simulation of toy network with 5 independent genes on reaction 2

    Now, we will run a simulation of the toy network when reaction 2 can be independently catalyzed by five genes, as described in the main text.
    """)
    return


@app.cell
def _(np):
    toy_5gene_directory = "./networks/toynet_5gene.npz"

    toy_5gene_mats = np.load(toy_5gene_directory)
    S_5gene = toy_5gene_mats["S"]
    beta_5gene = toy_5gene_mats["beta"]
    Au_5gene = toy_5gene_mats["Au"]
    Al_5gene = toy_5gene_mats["Al"]
    Gu_5gene = toy_5gene_mats["Gu"]
    Gl_5gene = toy_5gene_mats["Gl"]
    Sigmau_5gene = toy_5gene_mats["Sigmau"]
    Sigmal_5gene = toy_5gene_mats["Sigmal"]
    uimmutable_5gene = [list(i) for i in toy_5gene_mats["uimmutable"]]
    limmutable_5gene = [list(i) for i in toy_5gene_mats["limmutable"]]
    return Al_5gene, Au_5gene, Gl_5gene, Gu_5gene, S_5gene, beta_5gene


@app.cell
def _(
    Al_5gene,
    Au_5gene,
    Gl_5gene,
    Gu_5gene,
    N,
    S_5gene,
    T,
    beta_5gene,
    em,
    evcm,
    limmutable,
    ms,
    ss,
    uimmutable,
):
    (
        df_flux_5gene,
        df_ubounds_5gene,
        df_lbounds_5gene,
        df_biomass_5gene,
        df_umutation_5gene,
        df_lmutation_5gene,
        df_neutral_5gene,
        *_,
    ) = evcm.sim.run_sim(
        T=T,
        Au=Au_5gene,
        Al=Al_5gene,
        S=S_5gene,
        Gu=Gu_5gene,
        Gl=Gl_5gene,
        beta=beta_5gene,
        pop_size=N,
        mutate_scale=ms,
        simulation_scale=ss,
        expected_mutations=em,
        uimmutable=uimmutable,
        limmutable=limmutable,
        Reaction_labels=["1", "2", "3", "4"],
        uGene_labels=["1", "2a", "2b", "2c", "2d", "2e", "3", "4"],
        lGene_labels=["1", "2a", "2b", "2c", "2d", "2e", "3", "4"],
    )
    return (df_flux_5gene,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Basic analysis
    """)
    return


@app.cell
def _(df_flux_5gene, pl):
    df_flux_5gene_1 = pl.from_pandas(df_flux_5gene)
    return (df_flux_5gene_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fluxes as a function of time steps
    """)
    return


@app.cell
def _(df_flux_1, plt, sns):
    sns.lineplot(df_flux_1, x="Time", y="Flux", hue="Reaction")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Normalized, rolling average change in flux across evolution.

    This shows that an EvCM exists, as the direction of evolution is approximately constant.
    """)
    return


@app.cell
def _(df_flux_5gene_1, pl, plt, sns):
    sns.lineplot(
        df_flux_5gene_1.with_columns(
            pl.col("Flux").diff().rolling_mean(window_size=500).over("Reaction")
        )
        .with_columns((pl.col("Flux") / pl.col("Flux").max()).over("Time"))
        .drop_nulls(),
        x="Time",
        y="Flux",
        hue="Reaction",
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison of the original toy network and the five gene version
    """)
    return


@app.cell
def _(df_flux_1, df_flux_5gene_1, pl, plt, sns):
    ax = sns.lineplot(
        df_flux_5gene_1.with_columns(
            pl.col("Flux").diff().rolling_mean(window_size=500).over("Reaction")
        )
        .with_columns((pl.col("Flux") / pl.col("Flux").max()).over("Time"))
        .drop_nulls()
        .with_columns(pl.lit("5 Gene").alias("Sim")),
        x="Time",
        y="Flux",
        hue="Reaction",
        style="Sim",
        style_order=["Original", "5 Gene"],
        legend=False,
    )
    sns.lineplot(
        df_flux_1.with_columns(
            pl.col("Flux").diff().rolling_mean(window_size=500).over("Reaction")
        )
        .with_columns((pl.col("Flux") / pl.col("Flux").max()).over("Time"))
        .drop_nulls()
        .with_columns(pl.lit("Original").alias("Sim")),
        x="Time",
        y="Flux",
        hue="Reaction",
        style="Sim",
        style_order=["Original", "5 Gene"],
        ax=ax,
    )
    sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
