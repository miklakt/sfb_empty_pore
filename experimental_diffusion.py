#%%
import numpy as np

#%%
def diffusion_Rubinstein(phi, d, eta=0.001, T = 293, prefactor = 1, k=1):
    if prefactor==0:
        m = np.ones_like(phi)
        return m
    eps = np.where(phi==0, 0.0, 1/phi)
    m = eps * eps / (d * d)/prefactor
    m = m /(1.0 + m**k)**(1 / k)
    m = np.where(phi>0, m, 1.0)

    d_ = d*1e-9                                 #m^3
    k_B = 1.380649*1e-23                        #J/K
    D_0_ = k_B * T / (3 * np.pi * eta * d_)     #m^2/s

    D = D_0_*m                                  #m^2/s
    D = D*1e12                                  #microm^2/s

    return D

def diffusion_Rubinstein2(phi, d, eta=0.001, T = 293, prefactor = 1, k=1):
    if prefactor==0:
        m = np.ones_like(phi)
        return m
    eps = np.where(phi==0, 0.0, 1/phi)
    m = np.exp(eps / d)/prefactor
    m = m /(1.0 + m**k)**(1 / k)
    m = np.where(phi>0, m, 1.0)

    d_ = d*1e-9                                 #m^3
    k_B = 1.380649*1e-23                        #J/K
    D_0_ = k_B * T / (3 * np.pi * eta * d_)     #m^2/s

    D = D_0_*m                                  #m^2/s
    D = D*1e12                                  #microm^2/s

    return D

def diffusion_Phillies(phi, d, beta, nu, eta=0.001, T = 293):
    m = np.where(phi==0, 1.0, np.exp(-beta*np.power(phi, nu)))
    d_ = d*1e-9                                 #m^3
    k_B = 1.380649*1e-23                        #J/K
    D_0_ = k_B * T / (3 * np.pi * eta * d_)     #m^2/s

    D = D_0_*m                                  #m^2/s
    D = D*1e12                                  #microm^2/s

    return D

def diffusion_Krieger_Dougherty(
        phi, d, 
        intrinsic_viscosity=40.0, 
        phi_max=1.5, 
        eta=0.001,
        T = 293
        ):
    import numpy as np

    phi = np.asarray(phi)
    if np.any(phi >= phi_max):
        raise ValueError("Volume fraction must be less than phi_max.")

    exponent = -intrinsic_viscosity * phi_max
    eta_ = eta * (1 - phi / phi_max) ** exponent

    d_ = d*1e-9                                 #m^3
    k_B = 1.380649*1e-23                        #J/K
    D = k_B * T / (3 * np.pi * eta_ * d_)       #m^2/s
    D = D*1e12                                  #microm^2/s

    return D


#%%
from io import StringIO
import pandas as pd

csv_string = """
d,Concentration,Mobility
5,0.0,1.0
5,10.0,0.872
5,20.0,0.787
5,30.0,0.692
5,50.0,0.618
5,75.0,0.457
5,125.0,0.284
5,200.0,0.222
5,300.0,0.152
10,0.0,1.0
10,10,0.819
10,15.0,0.772
10,20.0,0.695
10,30.0,0.419
10,50.0,0.235
10,75.0,0.219
10,100.0,0.188
10,125.0,0.149
10,200.0,0.137
10,300.0,0.098
"""
# Use StringIO to simulate a file object
csv_data = StringIO(csv_string)

# Read the CSV string into a DataFrame
df = pd.read_csv(csv_data)

df["phi"] = df["Concentration"]*1e-6/1600
# Display the DataFrame
print(df)
# %%
import matplotlib.pyplot as plt

fig, ax =plt.subplots()
ax.scatter(df["phi"], df["Mobility"])

# for i, (x, y) in enumerate(zip(df["Concentration"], df["Mobility"])):
#     ax.text(x, y, str(i), fontsize=12, ha='right', va='bottom', color='red')

ax.set_xscale("log")
ax.set_yscale("log")
#ax.set_xlim(1e-9, 1e-6)
# %%
#%%
csv_data=StringIO("""
phi,d,D,D_err
0.0,100,4.60,0.68
0.0,200,2.15,0.21
0.0,1000,0.472,0.001
0.0069,25,0.19,0.005
0.02,25,0.029,0.002
""")
df = pd.read_csv(csv_data)

phi = np.linspace(0, 0.4)
from calculate_fields_in_pore import mobility_Rubinstein
fig, ax =plt.subplots()
ax.scatter(df["phi"], df["D"], label = "microsperes in PEO")
D = diffusion_Rubinstein(phi, 25, prefactor=1, k=1)
ax.plot(phi, D, label = 25)
D = diffusion_Rubinstein(phi, 100, prefactor=1, k=1)
ax.plot(phi, D, label = 100)
D = diffusion_Rubinstein(phi, 200, prefactor=1, k=1)
ax.plot(phi, D, label = 200)
D = diffusion_Rubinstein(phi, 1000, prefactor=1, k=1)
ax.plot(phi, D, label = 1000)

# D = diffusion_Rubinstein(phi, 25, prefactor=30, k=1)
# ax.plot(phi, D, label = "f2=30")

# D = diffusion_Phillies(phi, 25, beta = 8, nu = 0.76)
# ax.plot(phi, D, label = "Phillies")

D = diffusion_Rubinstein(phi, 5, prefactor=30, k=1)
ax.plot(phi, D, label = 5)

csv_data=StringIO("""
conc,d,D,D_err
0,5,90,3
100,5,6.5,1.9
500,5,1,nan
""")
df = pd.read_csv(csv_data)
df["phi"] = df["conc"]/1400

ax.scatter(df["phi"], df["D"], label = "GFPstd in NSP1")



ax.legend()

#ax.set_xscale("log")
ax.set_yscale("log")
# %%
phi = np.linspace(0, 0.5)

csv_data=StringIO("""
conc,d,D,D_err,Probe
0,4.6,90,3,GFP_STD
0,4.6,90,3,GFP_NTR
100,4.6,6.5,1.9,GFP_STD
500,4.6,1,0.5,GFP_NTR
100,4.6,1.6,0.2,GFP_NTR
""")
df = pd.read_csv(csv_data)
df["phi"] = df["conc"]/1400

d_particle = 4.6

fig, ax =plt.subplots()
f = 1
D = diffusion_Rubinstein(phi, d_particle, prefactor=f, k=1)
ax.plot(phi, D, label = f"{f=}, {d_particle=}", color = "black", linestyle = "--")

f = 30
D = diffusion_Rubinstein(phi, d_particle, prefactor=f, k=1)
ax.plot(phi, D, color = "red", label = f"{f=}, {d_particle=}")

D = diffusion_Krieger_Dougherty(phi, d=d_particle, intrinsic_viscosity=20, phi_max=1.0)
ax.plot(phi, D, color = "black", label = f"Krieger-Dougherty, {d_particle=}")

f = 30
d_particle=6.5
D = diffusion_Rubinstein(phi, d_particle, prefactor=f, k=1)
ax.plot(phi, D, color = "green", label = f"{f=}, {d_particle=}")


df_ = df.loc[df.Probe=="GFP_STD"]
ax.scatter(
    df_["phi"], 
    df_["D"], 
    label = r"$\text{GFP}^{\text{STD}}$ in NSP1 brush",
    linewidth = 0,
    color = "tab:blue"
    )
ax.errorbar(
    df_["phi"], 
    df_["D"], 
    df_["D_err"],
    ls = "none",
    color = "tab:blue"
    )

# df_ = df.loc[df.Probe=="GFP_NTR"]
# ax.scatter(
#     df_["phi"],
#     df_["D"],
#     color = "red",
#     label = r"$\text{GFP}^{\text{NTR}}$ in NSP1"
#     )
# ax.errorbar(
#     df_["phi"], 
#     df_["D"], 
#     df_["D_err"],
#     ls = "none",
#     color = "red"
#     )

csv_data=StringIO("""
d,D,D_err,d_err
7,1.05,0.25,0.35
12.5,1.2,0.25,0.35
20.5,1.3,0.25,0.4
29.5,1.6,0.2,0.5
""")

d_min = 2.3
d_min_err = 0.5

df = pd.read_csv(csv_data)
df["phi"] = d_min/df["d"]


# Compute phi_err using error propagation formula
df["phi_err"] = df["phi"] * np.sqrt((d_min_err / d_min) ** 2 + (df["d_err"] / df["d"]) ** 2)
ax.scatter(
    df["phi"],
    df["D"],
    color = "red",
    label = r"$\text{GFP}^{\text{NTR}}$ in NSP1 brush"
    )
ax.errorbar(
    df["phi"], 
    df["D"], 
    yerr=df["D_err"],
    xerr=df["phi_err"],
    ls = "none",
    color = "red"
    )


def estimate_protein_diameter(MW_kDa, density=1.4):

    NA = 6.022e23

    # Partial specific volume (cm^3/g)
    v_bar = 1/density

    mw_g_per_mol = MW_kDa * 1000.0
    mass_one_molecule = mw_g_per_mol / NA
    volume_cm3 = mass_one_molecule * v_bar
    volume_nm3 = volume_cm3 * 1.0e21
    radius_nm = ((3.0 * volume_nm3) / (4.0 * np.pi)) ** (1.0 / 3.0)
    diameter_nm = 2.0 * radius_nm

    return diameter_nm

csv_data=StringIO("""
conc,Probe,MW,D,D_err
0,acRedStar,117,60,nan
0.27,acRedStar,117,45,5
2.2,acRedStar,117,0.6,0.4
2.2,GFP-impb,124,0.15,0.05
""")

MW = 64#kDa for Nsp1-fsFG
protein_density = 1400#g/l

df = pd.read_csv(csv_data)

df["phi"] = df["conc"]*MW/protein_density

df["d"] = estimate_protein_diameter(df["MW"])

df_ = df.loc[df.Probe=="acRedStar"]
ax.scatter(
    df_["phi"],
    df_["D"],
    color = "green",
    label = r"acRedStar in NSP1 gel"
    )
ax.errorbar(
    df_["phi"], 
    df_["D"], 
    df_["D_err"],
    ls = "none",
    color = "green"
    )

df_ = df.loc[df.Probe!="acRedStar"]
ax.scatter(
    df_["phi"],
    df_["D"],
    color = "magenta",
    label = r"GFP-impb in NSP1 gel"
    )
ax.errorbar(
    df_["phi"], 
    df_["D"], 
    df_["D_err"],
    ls = "none",
    color = "magenta"
    )


ax.legend(bbox_to_anchor = [1,1])

ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$D \, [\mu \text{m}^2/\text{s}]$")
#ax.set_xscale("log")
ax.set_yscale("log")

ax.set_ylim(1e-1, 1e2)
#ax.set_xlim(1e-2, 0.5)
#%%