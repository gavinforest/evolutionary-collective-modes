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
    # Figure 2 - Analysis of toy metabolic network

    This notebook performs analysis of simulation results of the toy metabolic network to generate Figure 2 in the main text.
    """)
    return


@app.cell
def _():
    import os
    import polars as pl
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from evcm.analysis import load_sim_results, load_mats, normalize_by_norm, rolling_average_difference

    return (
        load_mats,
        load_sim_results,
        normalize_by_norm,
        os,
        pl,
        plt,
        rolling_average_difference,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, we will load the simulation data
    """)
    return


@app.cell
def _(load_sim_results, os):
    root_specifictoynets = r'./toynet_simulation_data/'
    sims_3211 = sorted(os.listdir(root_specifictoynets))

    __DEBUG__=False #If True, only load a the initial periods of the simulation, but does nothing here as there is only one period
    biomass_3211,ubounds_3211,lbounds_3211,flux_3211 = load_sim_results(root_specifictoynets,
                                               ['biomass','ubounds','lbounds','flux'],[3,4,4,4],
                                               sims_3211,dtypes=[{},{'Gene':str},{'Gene':str},{'Reaction':str}],debug=__DEBUG__)
    return (
        biomass_3211,
        flux_3211,
        lbounds_3211,
        root_specifictoynets,
        sims_3211,
        ubounds_3211,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Then, we load the toy network matrices
    """)
    return


@app.cell
def _(load_mats, root_specifictoynets, sims_3211):
    mats_3211 = load_mats(root_specifictoynets,sims_3211)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we load some already performed analysis. `SPu_3211` contains values of $\lambda$ throughout the simulation, and `outcomeSP_3211` contains the selective pressure on the estimated EvCM.
    """)
    return


@app.cell
def _(pl, root_specifictoynets, sims_3211):
    __DATE__ = '2025_12_24'
    SPu_3211 = pl.DataFrame()
    outcomeSP_3211 = pl.DataFrame()

    for f in sims_3211:

        SPu_3211.vstack(pl.read_csv(root_specifictoynets+f+'/'+f+'__'+__DATE__+'__SPu.csv'),in_place=True)
        outcomeSP_3211.vstack(pl.read_csv(root_specifictoynets+f+'/'+f+'__'+__DATE__+'__outcomeSP.csv'),in_place=True)
    return SPu_3211, outcomeSP_3211


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Details for plotting
    """)
    return


@app.cell
def _(flux_3211, lbounds_3211, pl, ubounds_3211):
    ubounds_3211_1 = ubounds_3211.with_columns(pl.col('Gene').replace({'0': '1', '1': '2', '2': '3', '3': '4'}))
    lbounds_3211_1 = lbounds_3211.with_columns(pl.col('Gene').replace({'0': '1', '1': '2', '2': '3', '3': '4'}))
    flux_3211_1 = flux_3211.with_columns(pl.col('Reaction').replace({'0': '1', '1': '2', '2': '3', '3': '4'}))
    return flux_3211_1, ubounds_3211_1


@app.cell
def _(sims_3211):
    simulation_palette=['#37B34A','#B2D732','#66B032','#347B98','#092834']
    simulation_hue_order = sims_3211

    ubounds_palette = list(reversed(["#EA202C",
    "#448D76",
    "#FB8604",
    "#4424D6"]))
    ubounds_hue_order = list(reversed(["4",
    "3",
    "2",
    "1"]))

    cm_ = 1/2.54
    onecolumn_ = 8.7*cm_
    sidelegend_ = 11.4*cm_
    widefig_ = 17.8*cm_
    return (
        simulation_hue_order,
        simulation_palette,
        ubounds_hue_order,
        ubounds_palette,
        widefig_,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Calculations for figure 2

    We often downsample the data for visual clarity
    """)
    return


@app.cell
def _():
    __TIME_DOWNSAMPLE__ = 1000
    __MARKER_DOWNSAMPLE__ = __TIME_DOWNSAMPLE__*50
    __SP_DOWNSAMPLE__ = 1 #Already downsampled by 100 when calculation was performed
    return __MARKER_DOWNSAMPLE__, __SP_DOWNSAMPLE__, __TIME_DOWNSAMPLE__


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We selected one simulation as the key simulation for Figure 2. You can select another to see how Figure 2 changes.
    """)
    return


@app.cell
def _(simulation_hue_order):
    key_sim_3211 = simulation_hue_order[0]
    return (key_sim_3211,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Select the biomass (growth rate) of the key simulation
    """)
    return


@app.cell
def _(__TIME_DOWNSAMPLE__, biomass_3211, key_sim_3211, pl):
    biomass_key_ds_3211 = biomass_3211.filter(pl.col('Sim').eq(key_sim_3211)).\
                                select(pl.all().gather_every(__TIME_DOWNSAMPLE__).over('Sim',mapping_strategy='explode'))
    return (biomass_key_ds_3211,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Select the fluxes of the key simulation
    """)
    return


@app.cell
def _(
    __MARKER_DOWNSAMPLE__,
    __TIME_DOWNSAMPLE__,
    flux_3211_1,
    key_sim_3211,
    pl,
):
    flux_key_ds_3211 = flux_3211_1.filter(pl.col('Sim').eq(key_sim_3211)).select(pl.all().gather_every(__TIME_DOWNSAMPLE__).over(['Sim', 'Reaction'], mapping_strategy='explode'))
    flux_key_ds_3211_formarkers1 = flux_3211_1.filter(pl.col('Sim').eq(key_sim_3211)).select(pl.all().gather_every(__MARKER_DOWNSAMPLE__).over(['Sim', 'Reaction'], mapping_strategy='explode'))
    flux_key_ds_3211_formarkers2 = flux_3211_1.filter(pl.col('Sim').eq(key_sim_3211)).select(pl.all().gather_every(__MARKER_DOWNSAMPLE__, offset=__MARKER_DOWNSAMPLE__ // 2).over(['Sim', 'Reaction'], mapping_strategy='explode'))
    return (
        flux_key_ds_3211,
        flux_key_ds_3211_formarkers1,
        flux_key_ds_3211_formarkers2,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Calculate $\cos\theta$ of the angle between the EvCM and the local direction of evolution
    """)
    return


@app.cell
def _(
    __TIME_DOWNSAMPLE__,
    key_sim_3211,
    normalize_by_norm,
    pl,
    rolling_average_difference,
    ubounds_3211_1,
):
    #Estimate the gene EvCM
    key_g_evcm_est_3211 = ubounds_3211_1.filter(pl.col('Time').eq(pl.col('Time').max()) | pl.col('Time').eq(pl.col('Time').min()), pl.col('Sim').eq(key_sim_3211)).select(pl.exclude('Bound'), pl.col('Bound').diff().over('Gene')).drop_nulls()
    key_g_evcm_est_3211 = key_g_evcm_est_3211.with_columns((pl.col('Bound') / (pl.col('Bound') ** 2).sum().sqrt()).over('Sim'))
    key_g_evcm_est_3211 = key_g_evcm_est_3211.rename({'Bound': 'Gene EvCM'})
    gene_key_normchange_3211 = normalize_by_norm(rolling_average_difference(ubounds_3211_1.filter(pl.col('Sim').eq(key_sim_3211)), 'Bound', 'Gene', 'Sim', window=5000), 'Time', 'Bound', 'Sim')
    deltagene_evcm_cos = gene_key_normchange_3211.join(key_g_evcm_est_3211.drop('Time'), on=['Gene', 'Sim']).group_by(['Time', 'Sim'], maintain_order=True).agg(pl.col('Bound').dot(pl.col('Gene EvCM')).alias('Gene-Gene EvCM Cos'))
    #Estimate the local change in genes
    #Calculate the angle between the EvCM and the local change in gene
    deltagene_evcm_cos_ds = deltagene_evcm_cos.select(pl.all().gather_every(__TIME_DOWNSAMPLE__).over(['Sim'], mapping_strategy='explode'))
    return (deltagene_evcm_cos_ds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Select the selective pressure on individual genes (components of $\lambda$) and the selective pressure on the simulation outcome. We only show a small portion of the simulation in Figure 2.
    """)
    return


@app.cell
def _(SPu_3211, __SP_DOWNSAMPLE__, key_sim_3211, outcomeSP_3211, pl):
    SPu_key_3211 = SPu_3211.filter(pl.col('Sim').eq(key_sim_3211))
    SPu_key_3211 = SPu_key_3211.with_columns(pl.col('Constraint').replace({'0___u':'1','1___u':'2','2___u':'3','3___u':'4'}))
    SPu_key_3211 = SPu_key_3211.select(pl.all().gather_every(__SP_DOWNSAMPLE__).over('Sim','Constraint',mapping_strategy='explode'))

    outcomeSP_key_3211 = outcomeSP_3211.filter(pl.col('Sim').eq(key_sim_3211))
    outcomeSP_key_3211 = outcomeSP_key_3211.select(pl.all().gather_every(__SP_DOWNSAMPLE__).over('Sim',mapping_strategy='explode'))

    SP_range_3211 = (1.05e6,1.1e6)

    SPu_key_zoom_3211 = SPu_key_3211.filter(pl.col('Time').ge(SP_range_3211[0]),pl.col('Time').le(SP_range_3211[1]))
    outcomeSP_key_zoom_3211 = outcomeSP_key_3211.filter(pl.col('Time').ge(SP_range_3211[0]),pl.col('Time').le(SP_range_3211[1]))
    return SPu_key_zoom_3211, outcomeSP_key_zoom_3211


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 2
    """)
    return


@app.cell
def _(
    SPu_key_zoom_3211,
    biomass_key_ds_3211,
    deltagene_evcm_cos_ds,
    flux_key_ds_3211,
    flux_key_ds_3211_formarkers1,
    flux_key_ds_3211_formarkers2,
    outcomeSP_key_zoom_3211,
    pl,
    plt,
    simulation_hue_order,
    simulation_palette,
    sns,
    ubounds_hue_order,
    ubounds_palette,
    widefig_,
):
    M = """
        AAAD
        AAAG
        BHHE
        CHHF
        """
    _width_=widefig_
    _height_=5.2
    _labelpad_ = 4
    fig,axd = plt.subplot_mosaic(M,figsize=(_width_,_height_),layout='constrained',
                                height_ratios=[1.8,1.0,1.2,1.2],width_ratios=[5/3,5/3,5/3,widefig_-5])

    axd['B'].sharex(axd['C'])

    # A ==========================================================================
    _a_ = 'A'
    axd[_a_].set_xticks([])
    axd[_a_].set_yticks([])

    # B ==========================================================================
    _a_ = 'B'

    sns.lineplot(biomass_key_ds_3211,x='Time',y='Biomass',hue='Sim',
                 palette=simulation_palette,hue_order=simulation_hue_order,
                 legend=False,
                 ax=axd[_a_])
    axd[_a_].set_ylabel('Growth Rate, $\\beta^Tv$')
    axd[_a_].tick_params(labelbottom=False)
    axd[_a_].set_xlabel('')
    axd[_a_].set_yticks([0,2e4,4e4],['0','2e4','4e4']) # DOUBLE CHECK IF MOVING

    # C ==========================================================================
    _a_ = 'C'
    sns.lineplot(flux_key_ds_3211,x='Time',y='Flux',hue='Reaction',
                 palette=ubounds_palette,hue_order=ubounds_hue_order,
                 legend=False,alpha=0.5,
                 ax=axd[_a_])
    sns.lineplot(flux_key_ds_3211_formarkers1.filter(pl.col('Reaction').is_in(['1','2','3'])),
                 x='Time',y='Flux',hue='Reaction',style='Reaction',markers=['^']*4,dashes=False,
                 palette=ubounds_palette,hue_order=ubounds_hue_order,linewidth=0,markeredgecolor='none',
                 legend=False,
                 ax=axd[_a_])
    sns.lineplot(flux_key_ds_3211_formarkers2.filter(pl.col('Reaction').is_in(['4'])),
                 x='Time',y='Flux',hue='Reaction',style='Reaction',markers=['^']*4,dashes=False,
                 palette=ubounds_palette,hue_order=ubounds_hue_order,linewidth=0,markeredgecolor='none',
                 legend=False,
                 ax=axd[_a_])


    axd[_a_].set_ylabel('Flux, $v$')
    axd[_a_].set_xlabel('Attempted\nMutations',labelpad=_labelpad_)
    axd[_a_].set_xticks([0,5e6],['0','5e6']) # DOUBLE CHECK IF MOVING

    axd[_a_].set_yticks([0,2e4,4e4],['0','2e4','4e4']) # DOUBLE CHECK IF MOVING


    # D ==========================================================================
    _a_ = 'D'
    sns.lineplot(deltagene_evcm_cos_ds,x='Time',y='Gene-Gene EvCM Cos',hue='Sim',
                 legend=False,palette=['#000000'],
                 ax=axd[_a_])


    axd[_a_].set_ylabel(r'cos $\Theta$')
    axd[_a_].set_xlabel('Attempted\nMutations',labelpad=_labelpad_)
    axd[_a_].set_ylim([0.86,1.05])

    axd[_a_].set_xticks([0,5e6],['0','5e6']) # DOUBLE CHECK IF MOVING
    axd[_a_].set_yticks([0.9,1.0],['0.9','1.0']) # DOUBLE CHECK IF MOVING

    # E/F ==========================================================================
    _a_ = 'E'
    sns.lineplot(SPu_key_zoom_3211,x='Time',y='Lambda',hue='Constraint',
                 palette=ubounds_palette,hue_order=ubounds_hue_order,
                 legend=False,
                 ax=axd[_a_])


    axd[_a_].set_ylabel('')
    axd[_a_].set_xlabel('')
    axd[_a_].set_ylim([-0.15,1.15])

    axd[_a_].tick_params(labelbottom=False)


    _a_ = 'F'
    sns.lineplot(outcomeSP_key_zoom_3211,x='Time',y='Selective Pressure',hue='Sim',
                 palette=['#76e3ff'],
                 legend=False,
                 ax=axd[_a_])


    axd[_a_].set_xlabel('Attempted\nMutations',labelpad=_labelpad_)
    axd[_a_].set_ylabel(r'Selective Pressure, $\lambda^TGg$')
    axd[_a_].set_ylim([-0.15,1.15])

    axd[_a_].set_xticks([1.05e6,1.1e6],['1.05','1.1e6']) # DOUBLE CHECK IF MOVING



    # G ==========================================================================
    _a_ = 'G'
    axd[_a_].set_xticks([])
    axd[_a_].set_yticks([])

    # H ==========================================================================
    _a_ = 'H'
    axd[_a_].set_xticks([])
    axd[_a_].set_yticks([])

    #   ==========================================================================

    for label, ax in axd.items():
        if label in ['B','C','D','E','F']:
            ax.set_title(label, x=-0.15,y=1.01,size='large',weight='bold',pad=0)
        elif label in ['A']:
            ax.set_title(label, x=-0.15/3,y=1.01,size='large',weight='bold',pad=0)

    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
