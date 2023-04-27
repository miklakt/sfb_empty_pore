#%%
import sys
sys.path.append('..')

import pandas as pd
import numpy as np
from utils import get_by_kwargs
#from theory import D_eff, free_energy_phi
import scf_pb

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

#matplotlib settings
LAST_USED_COLOR = lambda: plt.gca().lines[-1].get_color()
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

fig_path = "../fig/"
a1, a2 = [ 0.18, -0.09]
#%%
df = pd.read_pickle("/home/ml/Studium/sfb_empty_pore/empty_brush.pkl")
df = get_by_kwargs(df, chi_PW = 0.0)
# %%
def phi_center(CHI_PS, s, r):
    fig, ax = plt.subplots()
    for chi in CHI_PS:
        plot_data = get_by_kwargs(df, chi_PS = chi, s=s, r=r)
        phi_0 = plot_data.phi.squeeze()[0]
        z = np.arange(len(phi_0)) - len(phi_0)/2
        ax.plot(z, phi_0, label = chi)
        y1 = -plot_data.s.squeeze()/2
        y2 = plot_data.s.squeeze()/2
        ax.axvline(y1, color = "grey", linestyle = "--")
        ax.axvline(y2, color = "grey", linestyle = "--")
        trans = transforms.blended_transform_factory(
            ax.transData, ax.transAxes
            )
        ax.arrow(
            y1, 0.1, y2-y1, 0, 
            transform = trans,
            shape = "full",
            #head_width=0.15,
            #head_length=0.1
        )
        ax.text(
            y1+(y2-y1)/2, 0.1, transform = trans,
            s = "in pore", ha= "center", va = "bottom"
            #head_width=0.15,
            #head_length=0.1
        )
    ax.set_ylabel("$\phi(r=0)$")
    ax.set_xlabel("$z$")
    ax.legend(title = "$\chi_{PS}$")
    return fig
chi_PSs = [0.0, 0.3, 0.6, 0.9, 1.1]
#chi_PSs=[ 0.5, 0.6, 0.7, 0.8, 0.9]
s = 52
r = 26
fig = phi_center(chi_PSs, s, r)
fig.savefig(fig_path+f"phi_center.pdf")
fig.savefig(fig_path+f"phi_center.png", dpi =300)
#fig.savefig("/home/ml/Studium/sfb_empty_pore/conference/phi_center.png", dpi=600)
#%%
# %%
def fe_center_on_chi_pc(CHI_PC, d, chi_ps, s, r):
    fig, ax = plt.subplots()
    for chi_pc in CHI_PC:
        plot_data = get_by_kwargs(df, chi_PS = chi_ps, s=s, r=r)
        phi_0 = plot_data.phi.squeeze()[0]
        #fe = [free_energy_phi(
        #        phi = phi_,
        #        a1=a1, a2=a2, 
        #        chi_PC=chi_pc, 
        #        chi_PS=chi_ps, 
        #        w=d
        #    ) for phi_ in phi_0]
        fe = [scf_pb.free_energy_external(
            phi = phi_0,
            a0=a1, a1=a2, 
            chi_PC=chi_pc, 
            chi=chi_ps, 
            particle_width = d,
            particle_height =d,
            z = z_
        ) for z_ in range(len(phi_0))]
        z = np.arange(len(phi_0)) - len(phi_0)/2
        ax.plot(z, fe, label = chi_pc)
        
        y1 = -plot_data.s.squeeze()/2
        y2 = plot_data.s.squeeze()/2
        ax.axvline(y1, color = "grey", linestyle = "--")
        ax.axvline(y2, color = "grey", linestyle = "--")
        trans = transforms.blended_transform_factory(
            ax.transData, ax.transAxes
            )
        ax.arrow(
            y1, 0.1, y2-y1, 0, 
            transform = trans,
            shape = "full",
            #head_width=0.15,
            #head_length=0.1
        )
        ax.text(
            y1+(y2-y1)/2, 0.1, transform = trans,
            s = "in pore", ha= "center", va = "bottom"
            #head_width=0.15,
            #head_length=0.1
        )
    ax.set_ylabel("$F(z)/k_BT$")
    ax.set_xlabel("$z$")
    ax.legend(title = "$\chi_{PC}$")
    ax.set_title("$\chi_{PS}=$"+f"{chi_ps} "+f"$d={d}$")
    return fig
CHI_PC=[0.5, 0, -1.0, -2.0, -3.0]
d=4
chi_ps=0.4
s=52
r=26
fig = fe_center_on_chi_pc(CHI_PC, d, chi_ps, s, r)
fig.savefig(fig_path+f"fe_center_on_chi_pc_{chi_ps}_{d}.pdf")
#%%
def fe_center_on_chi_ps(CHI_PS, d, chi_pc, s, r):
    fig, ax = plt.subplots()
    for chi_ps in CHI_PS:
        plot_data = get_by_kwargs(df, chi_PS = chi_ps, s=s, r=r)
        phi_0 = plot_data.phi.squeeze()[0]
        fe = [scf_pb.free_energy_external(
            phi = phi_0,
            a0=a1, a1=a2, 
            chi_PC=chi_pc, 
            chi=chi_ps, 
            particle_width = d,
            particle_height =d,
            z = z_
        ) for z_ in range(len(phi_0))]
        z = np.arange(len(phi_0)) - len(phi_0)/2
        ax.plot(z, fe, label = chi_ps)
        
        y1 = -plot_data.s.squeeze()/2
        y2 = plot_data.s.squeeze()/2
        ax.axvline(y1, color = "grey", linestyle = "--")
        ax.axvline(y2, color = "grey", linestyle = "--")
        trans = transforms.blended_transform_factory(
            ax.transData, ax.transAxes
            )
        ax.arrow(
            y1, 0.1, y2-y1, 0, 
            transform = trans,
            shape = "full",
            #head_width=0.15,
            #head_length=0.1
        )
        ax.text(
            y1+(y2-y1)/2, 0.1, transform = trans,
            s = "in pore", ha= "center", va = "bottom"
            #head_width=0.15,
            #head_length=0.1
        )
    ax.set_ylabel("$F(z)/k_BT$")
    ax.set_xlabel("$z$")
    ax.legend(title = "$\chi_{PS}$")
    ax.set_title("$\chi_{PC}=$"+f"{chi_pc} "+f"$d={d}$")
    return fig
CHI_PS=[0.0, 0.4, 0.6, 0.8, 0.9]
d=4
chi_pc=-1.0
s=52
r=26
fig = fe_center_on_chi_ps(CHI_PS, d, chi_pc, s, r)
fig.savefig(fig_path+f"fe_center_on_chi_ps_{chi_pc}_{d}.pdf")
#%%
def D_eff_on_chi_PS(d, chi_PCs, s, r):
    fig, ax = plt.subplots()
    tbl = get_by_kwargs(df, s=s, r=r)
    for chi_pc in chi_PCs:
        D = []
        chi_PSs = []
        for chi_ps, plot_data in tbl.groupby(by = "chi_PS"):
            print(chi_ps)
            phi_0 = plot_data.phi.squeeze()[0][:]
            D_eff = [scf_pb.D_eff_external(
                phi = phi_0,
                a0=a1, a1=a2, 
                chi_PC=chi_pc, 
                chi=chi_ps, 
                particle_width = d,
                particle_height =d,
                a = d/2,
                b = len(phi_0) - d/2,
                k_smooth = 4
                )]
            D.append(D_eff)
            chi_PSs.append(chi_ps)
        ax.plot(chi_PSs, D, "o-", label = chi_pc)
    ax.axhline(y=1, color="grey")
    ax.legend(title = "$\chi_{PC}$")
    ax.set_ylabel("$D_{pore}/D_s$")
    ax.set_xlabel("$\chi_{PS}$")
    ax.set_title(f"$d={d}$")
    #ax.set_yscale("log")
    return fig
d=4
chi_PCs = [0, -0.5, -1.0, -1.5, -2.0, -3.0]
fig = D_eff_on_chi_PS(d, chi_PCs, s, r)
fig.savefig(fig_path+f"D_eff_on_chi_ps_{d}_adj.pdf")
fig.savefig(fig_path+f"D_eff_on_chi_ps_{d}_adj.svg")
# %%
def D_eff_on_chi_PC(d, s, r):
    fig, ax = plt.subplots()
    chi_PCs = np.arange(-3,1,0.1)
    tbl = get_by_kwargs(
            df, 
            s=s, r=r
        )
    for chi_ps, plot_data in tbl.groupby(by = "chi_PS"):
        phi_0 = plot_data.phi.squeeze()[0][60:-60]
        D = []
        for chi_pc in chi_PCs:
            print(chi_pc)
            phi_0 = plot_data.phi.squeeze()[0][:]
            D_eff = [scf_pb.D_eff_external(
                phi = phi_0,
                a0=a1, a1=a2, 
                chi_PC=chi_pc, 
                chi=chi_ps, 
                particle_width = d,
                particle_height =d,
                a = d/2,
                b = len(phi_0) - d/2,
                k_smooth = 4
                )]
            D.append(D_eff)
        ax.plot(chi_PCs, D, "-", label = chi_ps)

    ax.axhline(y=1, color="grey")
    ax.legend(title = "$\chi_{PS}$")
    ax.set_ylabel("$D_{eff}/D_s$")
    ax.set_xlabel("$\chi_{PC}$")
    ax.set_title("$d=$"+f"{d}")
    return fig
d=4
fig = D_eff_on_chi_PC(d, s, r)
fig.savefig(fig_path+f"D_eff_on_chi_pc_{d}.pdf")
# %%
s=52
shifts = [0, 5, 10, 20, 26, 30, 40]
fig, axs = plt.subplots(nrows = int(len(shifts)/2), ncols=2, sharex=True, sharey=True)
for chi in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]:
    for ax, shift in zip(axs.reshape(-1), shifts):
        plot_data = get_by_kwargs(df, chi_PS = chi, s=s, r=r)
        r = plot_data.r.squeeze()
        phi_2d = plot_data.phi.squeeze()
        y1 = plot_data.l1.squeeze()-1
        y2 = plot_data.l1.squeeze()+plot_data.s.squeeze()
        phi = phi_2d[:,y1+int((y2-y1+1)/2-shift)]
        ax.plot(phi, label = chi)
        ax.set_title(f"shift={shift}")
        ax.axvline(r, color = "black")
ax.text(r, 0.3, '$r>R_{pore}$', va='center', rotation='vertical')
#ax.set_ylabel("$\phi$")
#ax.set_xlabel("$x$")
ax.set_xlim(0,r+10)

fig.text(0.5, 0.04, '$r$', ha='center')
fig.text(0.04, 0.5, '$\phi$', va='center', rotation='vertical')

ax.legend(title = "$\chi_{PS}$", bbox_to_anchor=[1,1])
fig.set_size_inches(7,7)
fig.savefig(f"phi_cross.pdf", bbox_inches='tight')
# %%
