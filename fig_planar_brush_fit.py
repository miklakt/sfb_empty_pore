ph =8
pw = ph
#X=[1, 0]
X = [0.70585835, -0.31406453]
master = pd.read_pickle("pkl/reference_table_planar.pkl")
master_empty = pd.read_pickle("pkl/reference_table_planar_empty.pkl")
master = master.loc[
    master.chi_PS.isin([0.3, 0.4, 0.5, 0.6, 0.7, 0.75]) & 
    (master.ph==ph) & 
    (master.pw==pw)& (master.sigma==0.02)
    ]

gamma_f = gamma

g = sns.FacetGrid(
    master, 
    row = "chi_PC", col = "chi_PS", 
    hue = "ph", 
    sharey=False, 
    hue_kws=dict(marker = "s")
    )
g.map_dataframe(sns.scatterplot, x="pc", y="free_energy")
g.add_legend()

for (chi_PC, chi_PS), ax in g.axes_dict.items():
    empty_pore_data = utils.get_by_kwargs(master_empty, chi_PS = chi_PS)
    osm, sur = free_energy_cylinder(int(pw/2), empty_pore_data, chi_PS, chi_PC, gamma_f, X)
    tot = osm+sur
    x = list(range(len(tot)))
    ax.plot(x, tot, color = "red")

    osm, sur = free_energy_cylinder(int(pw/2), empty_pore_data, chi_PS, chi_PC, gamma_f, X, trunc =True)
    tot = osm+sur
    ax.plot(x, tot, color = "darkred", linestyle = "--")