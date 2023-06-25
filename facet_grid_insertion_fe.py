#%%
import pandas as pd
import seaborn as sns
master = pd.read_pickle("reference_table.pkl")
master = master.loc[master.chi_PC.isin([-1.5, -1.0, -0.5, 0])]
g = sns.FacetGrid(master, row = "chi_PC", col = "chi_PS", hue = "ph", sharey=False)
g.map_dataframe(sns.scatterplot, x="pc", y="free_energy")
g.add_legend()
# %%
