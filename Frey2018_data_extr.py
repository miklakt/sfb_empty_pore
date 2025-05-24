#%%
from io import StringIO
import itertools
import numpy as np
import pandas as pd
#%%
table_S1_csv = """Plasmid,Probe,Partition Coefficient
pSF779,mCherry,0.40
pDG2049,EGFP wt,1.05
pDG2050,EGFP L7E,1.11
pDG2051,EGFP V11E,1.10
pDG2052,EGFP Y39E,0.96
pDG2053,EGFP F99E,0.90
pDG2055,EGFP Y151R,0.83
pDG2056,EGFP M153E,0.87
pDG2054,EGFP V176H,1.06
pDG2057,EGFP Y182K,0.69
pDG2058,EGFP I188K,1.10
pDG2059,EGFP F223R,1.16
pDG2060,"EGFP L231K, M233E, L236R, Y237D",0.84
pSF1310,"shGFP1 =EGFP+Y39E, F99E, Y151, M153R, Y182K, L231K, M233E, L236R, Y237D",0.42
pSF1438,shGFP2,0.24
"""
table_S2_csv="""
Plasmid,Probe,Partition Coefficient
pSF1526,EGFP wt,1.04
pSF1538,EGFP D36W,1.49
pSF1553,EGFP T38W,1.41
pSF1539,EGFP K41W,2.89
pSF1540,EGFP R73W,1.47
pSF1534,EGFP T97W,3.73
pSF1533,EGFP D117W,1.43
pSF1532,EGFP E132W,1.30
pSF1541,EGFP N149W,1.67
pSF1531,EGFP N164W,3.23
pSF1530,EGFP D180W,2.02
pSF1529,EGFP D190W,1.39
pSF1542,EGFP N198W,1.72
pSF1552,EGFP S202W,1.69
pSF1528,EGFP Q204W,2.19
pSF1554,EGFP A206K,0.99
pSF1527,EGFP K209W,1.40
pSF1550,EGFP K221W,1.60
pSF1551,EGFP T225W,1.50
"""

table_S3_csv = """Plasmid,Protein,MALS Mw,Oligomerisation State
pSF1526,EGFP,"28 kDa ±0.6%",1
pSF1438,shGFP2,"29 kDa ±1.4%",1
pSF2893,sinGFP1,"28 kDa ±1.5%",1
pDG2754,sinGFP4a,"28 kDa ±0.5%",1
pSF2646,efGFP_0W,"28 kDa ±1.6%",1
pSF2647,efGFP_3W,"33 kDa ±0.5%",1.2
pSF2648,efGFP_5W,"56 kDa ±0.3%",2
pSF2649,efGFP_8W,ND,≥4*
pSF2654,efGFP_8Y,ND,≥2*
pSF2653,efGFP_8F,"170 kDa ±1.9%",6
pSF2651,efGFP_8L,"59 kDa ±0.2%",2.2
pSF2650,efGFP_8I,"56 kDa ±0.2%",2.1
pSF2652,efGFP_8M,"48 kDa ±0.5%",1.7
pDG2939,efGFP_8V,"52 kDa ±0.2%",1.8
pDG2931,efGFP_8C,"37 kDa ±3.8%",1.2
pDG2940,efGFP_8A,"28 kDa ±0.7%",1
pDG2934,efGFP_8H,"38 kDa ±0.5%",1.3
pSF2892,efGFP_8R,"30 kDa ±0.7%",1
pDG2936,efGFP_8Q,"28 kDa ±0.5%",1
pDG2938,efGFP_8T,"31 kDa ±0.5%",1
pDG2937,efGFP_8S,"28 kDa ±0.5%",1
pDG2932,efGFP_8E,"28 kDa ±0.4%",1
pDG2935,efGFP_8K,"28 kDa ±0.5%",1
pDG2966,efGFP_8N,"28 kDa ±0.5%",1
pDG2805,sffrGFP4,"29 kDa ±1.8%",1
pSF2884,"sffrGFP4 18x R → K","28 kDa ±2.2%",1
pSF2885,"sffrGFP4 25x R → K","27 kDa ±1.8%",1
pDG2805,sffrGFP4,"29 kDa ±0.3%",1
pDG2806,sffrGFP5,"28 kDa ±1.0%",1
pDG2713,sffrGFP6,"29 kDa ±1.6%",1
pDG2715,sffrGFP7,"29 kDa ±1.4%",1
pSF2902,GFP_MaxR_3W,ND,≥1.5*
pSF2903,GFP_MaxR_5W,"140 kDa ±0.4%",5
pSF2905,GFP_MaxR_8i,"47 kDa ±1.6%",1.7
pDG2718,GFPNTR _2B7,"65 kDa ±1.7%",2.4
pDG2798,GFPNTR_7B3,"42 kDa ±2.0%",1.6
pDG2721,3B1,"114 kDa ±1.2%",4
pDG2722,3B7,"110 kDa ±0.6%",4
pDG2779,3B7C,"110 kDa ±0.5%",4
pDG2723,3B8,"110 kDa ±0.5%",4
pDG2724,3B9,"113 kDa ±0.7%",4
"""

table_1_csv="""
Protein,Plasmid,Arg/Lys,Charge at pH 7.4,NPC Passage Rate Norm. to mCherry,FG Partition Coefficients Mac98A,FG Partition Coefficients Nup116,FG Partition Coefficients Nsp1
yNTF2 (dimer),pSF360,10R/6K,-15.0,600,290,1400,800
rNTF2 (dimer),pDG2121,4R/8K,-14.6,680,3200,13000,890
mCherry,pSF779,8R/24K,-6.4,1,≤0.05,0.09,0.4
EGFP,pSF1526,6R/21K,-8.1,3.5,0.11,0.33,0.98
shGFP2,pSF1438,11R/24K,-6.2,0.47,≤0.05,≤0.05,0.24
sinGFP1,pSF2893,1R/34K,-6.4,0.23,≤0.05,≤0.05,0.18
sinGFP4a,pDG2754,1R/34K,-11.4,0.1,≤0.05,≤0.05,ND
efGFP_0W,pSF2646,6R/20K,-10.1,6,0.09,0.42,3
efGFP_3W,pSF2647,6R/20K,-10.1,82,1.5,14,104
efGFP_5W,pSF2648,6R/19K,-11.1,87,2.2,15,170
efGFP_8W,pSF2649,6R/18K,-11.0,620,160,330,1300
efGFP_8Y,pSF2654,6R/18K,-11.0,290,14,82,790
efGFP_8F,pSF2653,6R/18K,-11.0,43,8.3,51,102
efGFP_8L,pSF2651,6R/18K,-11.0,33,1.8,10,90
efGFP_8I,pSF2650,6R/18K,-11.0,68,2.9,23,102
efGFP_8M,pSF2652,6R/18K,-11.0,110,3,17,92
efGFP_8R,pSF2892,14R/18K,-3.0,19,0.5,1.9,5.8
sffrGFP4,pDG2805,26R/1K,-7.7,160,14,50,9.8
sffrGFP4 18xR → K,pSF2884,8R/19K,-8.1,3.8,0.12,0.31,0.76
sffrGFP4 25xR → K,pSF2885,1R/26K,-9.2,1.6,0.06,0.1,0.54
sffrGFP4,pDG2805,26R/1K,-7.7,160,14,50,ND
sffrGFP5,pDG2806,26R/1K,-15.7,64,0.67,5.5,ND
sffrGFP6,pDG2713,26R/1K,-1.7,280,100,160,ND
sffrGFP7,pDG2715,26R/1K,0.3,310,200,200,ND
GFP_MaxR_3W,pSF2902,24R/4K,-7.8,440,120,400,136
GFP_MaxR_5W,pSF2903,23R/4K,-7.8,830,2100,4000,690
GFP_MaxR_8i,pSF2905,25R/1K,-8.7,1300,2000,4100,640
GFPNTR_2B7,pDG2718,25R/1K,-10.7,1600,1200,1600,ND
GFPNTR_7B3,pDG2798,25R/1K,-9.7,1700,1700,1400,ND
Sin_tCherry2,pDG2804,12R/88K,-49.2,≤0.02,≤0.05,ND,ND
GFPNTR_3B1,pDG2721,96R/4K,-46.8,430,290,ND,ND
GFPNTR_3B7,pDG2722,96R/4K,-42.8,760,3000,ND,ND
GFPNTR_3B8,pDG2723,96R/4K,-42.8,670,1800,ND,ND
GFPNTR_3B9,pDG2724,96R/4K,-38.8,870,4800,ND,ND
MBP,pSF1864,5R/36K,-11.1,0.08,ND,ND,ND
MBP K→R,pSF2895,34R/7K,-10.5,3.2,ND,ND,ND
Importin β 1-493,pSF2911,18R/22K,-41.1,340,ND,ND,ND
Importin β 1-493_R→K,pSF2912,2R/38K,-41.5,89,ND,ND,ND
Importin β 1-493_K→R,pSF2913,34R/6K,-40.8,470,ND,ND,ND
"""
#%%
categories = {
    "Protein":["pSF360", "pDG2121"],
    "Standard, Hydrophilic, and Super-Inert FP Monomers" : 
                    ["pSF779", "pSF1526", "pSF1438", "pSF2893", "pDG2754"],
    "Superhydrophobic GFPs" : ["pSF2646", "pSF2647", "pSF2648", "pSF2649", 
                               "pSF2654", "pSF2653","pSF2651", "pSF2650", 
                               "pSF2652", "pSF2892"],
    "Superfast, Arginine-Rich GFPs and R → K Revertants" :["pDG2805", "pSF2884", "pSF2885"],
    "Charge Series Based on sffrGFP4" : ["pDG2805", "pDG2806", "pDG2713", "pDG2715"],
    "GFPNTR-Variants Combining Surface Hydrophobicity with Translocation-Promoting Arginines":
    ["pSF2902", "pSF2903", "pSF2905", "pDG2718", "pDG2798"],
    "FP Tetramers":
    ["pDG2804", "pDG2721", "pDG2722", "pDG2723", "pDG2724"],
    "Other Scaffolds":["pSF1864", "pSF2895", "pSF2911", "pSF2912", "pSF2913"]
}
#%%
#%%
table_1 = pd.read_csv(StringIO(table_1_csv))
table_S3 = pd.read_csv(StringIO(table_S3_csv))
del table_S3["Protein"]
# %%
table_S3[['Mw (kDa)', 'Uncertainty (%)']] = table_S3['MALS Mw'].str.extract(r'(\d+)\s*kDa\s*±(\d+\.?\d*)%')
# %%
plasmid_to_category = {}
for i, (cat_name, plasmids) in enumerate(categories.items()):
    short = chr(ord('A') + i)
    for p in plasmids:
        plasmid_to_category[p] = (cat_name, short)

#%%
cat_desc = {chr(ord('A') + i):k for i, k in enumerate(categories)}
#%%

# Map new columns into DataFrame
table_1['Category Long'] = table_1['Plasmid'].map(lambda x: plasmid_to_category.get(x, (None, None))[0])
table_1['Category'] = table_1['Plasmid'].map(lambda x: plasmid_to_category.get(x, (None, None))[1])
#%%
table_1 = pd.merge(table_1, table_S3, on = "Plasmid", how = "left")
# %%
table_1.loc[table_1["Protein"]=="yNTF2 (dimer)", "Mw (kDa)"] = 28
table_1.loc[table_1["Protein"]=="rNTF2 (dimer)", "Mw (kDa)"] = 28
##The R to K or K to R substitution change the molar weight only by about 20 kDa
#"https://www.uniprot.org/uniprotkb/P0AG82/entry"
table_1.loc[table_1["Protein"]=="MBP", "Mw (kDa)"] = 43
table_1.loc[table_1["Protein"]=="MBP K→R", "Mw (kDa)"] = 43
#calculated from the sequence for 1-143 with https://web.expasy.org/cgi-bin/protparam/protparam
table_1.loc[table_1["Protein"]=="Importin β 1-493", "Mw (kDa)"] = 54.8
table_1.loc[table_1["Protein"]=="Importin β 1-493_R→K", "Mw (kDa)"] = 54.8
table_1.loc[table_1["Protein"]=="Importin β 1-493_K→R", "Mw (kDa)"] = 54.8
# %%
table_1.replace("ND", np.nan, inplace=True)
short_table = table_1.loc[table_1["FG Partition Coefficients Nup116"] != "≤0.05"][[
    "Protein", 
    "NPC Passage Rate Norm. to mCherry", 
    "FG Partition Coefficients Mac98A",
    "FG Partition Coefficients Nup116",
    "FG Partition Coefficients Nsp1",
    "Category",
    "Mw (kDa)",
    "Oligomerisation State"
    ]]
#%%


short_table.to_csv("perm_rates_experimental.csv", index = False)
# %%
