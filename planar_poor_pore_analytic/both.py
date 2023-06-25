#%%
from symbolic_convex_planar import Y as Y_convex
from symbolic_concave_planar import Y as Y_concave
from symbolic_convex_planar import S as S_convex
from symbolic_concave_planar import S as S_concave
from symbolic_concave_planar import yr_V, V_yr
from symbolic_saddle_planar import Y as Y_saddle
from symbolic_saddle_planar import S as S_saddle
#from symbolic_concave_planar import Y as Y_concave
import numpy as np
import matplotlib.pyplot as plt
from symbolic_convex_planar import h_V, V_h
# %%
s = 1
r = 0.5
V = 0.4
# %%
x = np.linspace(0, r, 400, endpoint=True)
y_convex = [float(Y_convex(s, V, r)(x_)) for x_ in x]
try:
    y_convex[np.where(np.isnan(y_convex))[0][0]] = 0 #make first nan 0
except:
    pass
y_concave = [float(Y_concave(s, r, V)(x_)) for x_ in x]

y_saddle = [float(Y_saddle(s, r, V)(x_)) for x_ in x]
#%%
x = np.linspace(0, 2*r, 800, endpoint=True)
y_convex = np.hstack([y_convex, np.flip(y_convex)])
y_concave = np.hstack([y_concave, np.flip(y_concave)])
y_saddle = np.hstack([y_saddle, np.flip(y_saddle)])
#%%
s_concave = S_concave(s,r,V)
s_convex = S_convex(s, V, r)
s_saddle = S_saddle(s, r, V)
# %%
fig, ax = plt.subplots()

ax.plot(x, y_convex, label ="convex", color = "blue")
ax.plot(x, -y_convex, color = "blue")
ax.fill_between(x, y_convex, -y_convex, alpha =0.05)

ax.plot(x, y_concave, label = "concave", color = "green")
ax.plot(x, -y_concave, color = "green")
ax.fill_between(x, y_concave, -y_concave, alpha =0.05, color = "green")

ax.plot(x, y_saddle, label = "saddle", color = "red", linestyle = "--")
ax.plot(x, -y_saddle, color = "red", linestyle = "--")
ax.fill_between(x, y_saddle, -y_saddle, alpha =0.001, color = "red")

ax.set_xlim(0, 2*r)

text1 = f"slit pore in poor solvent, s = {s}, r = {r}, V = {V}\n"
text2 = r"$S_{concave} = " +\
    f"{'{:.3f}'.format(s_concave)}$ " +\
    r"$S_{convex} = " +\
    f"{'{:.3f}'.format(s_convex)}$ " +\
    r"$S_{saddle} = " +\
    f"{'{:.3f}'.format(s_saddle)}$"
fig.suptitle(text1+text2)
ax.legend()
ax.set_aspect('equal', 'box')
ax.set_xlabel("z")
ax.set_ylabel("y")
# %%
s=1
r=0.5
Vmin_convex  = 0
Vmax_convex = float(V_h(s, r))# maximum volume a pore can accommodate with concave geometry
Vmax_concave =s*r # maximum volume a pore can accommodate with concave geometry
Vmin_concave = float(V_yr(s, r, 0))
# %%
v_min = min(Vmin_convex, Vmin_concave)
v_max = max(Vmax_convex, Vmax_concave)
v = np.linspace(v_min, v_max)

#v_concave = np.linspace(Vmin_concave, Vmax_concave)
#v_concave = np.clip(v, Vmin_concave, Vmax_concave)
s_concave = [float(S_concave(s, r, V_)) if ((V_>Vmin_concave) and (V_>Vmin_concave)) else np.nan for V_ in v]
#v_convex = np.linspace(Vmin_convex, Vmax_convex)
#v_convex = np.clip(v, Vmin_convex, Vmax_convex)
s_convex = [float(S_convex(s, V_, r)) if ((V_>Vmin_convex) and (V_<Vmax_convex)) else np.nan for V_ in v ]

v_min_common, v_max_common =  max(Vmin_convex, Vmin_concave), min(Vmax_convex, Vmax_concave)
s_saddle = [float(S_saddle(s, r, V_)) if ((V_>v_min_common) and (V_<v_max_common)) else np.nan for V_ in v]
# %%
fig, ax = plt.subplots()

ax.plot(v, s_convex, label = "convex", color = "blue")
ax.plot(v, s_concave, label = "concave", color = "green")
ax.plot(v, s_saddle, label = "saddle", color = "red", linestyle = "--")

ax.fill_between(v, s_saddle, np.maximum(s_convex, s_concave), hatch = "|", color = "red", alpha = 0.05)

ax.set_xlabel("Volume") 
ax.set_ylabel("Surface (Energy)")
ax.legend()
# %%