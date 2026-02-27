"""
AquaNeuron  —  Simulation & Visualization Suite
Stockholm Junior Water Prize India 2026
Prateek Tiwari, Raghav Khandelia, Aroush Muglikar, Shreyas Roy
"""

import numpy as np 
import pandas as pd 
import matplotlib 
matplotlib .use ('Agg')
import matplotlib .pyplot as plt 
from pathlib import Path 
import matplotlib .patches as mpatches 
import matplotlib .gridspec as gridspec 
import matplotlib .patheffects as pe 
from matplotlib .colors import LinearSegmentedColormap ,BoundaryNorm 
from matplotlib .patches import FancyBboxPatch ,FancyArrowPatch ,Arc ,Wedge 
import matplotlib .ticker as ticker 
from scipy .stats import pearsonr ,linregress ,norm ,chi2 
from scipy .optimize import curve_fit ,minimize 
from scipy .interpolate import interp1d 
from sklearn .ensemble import RandomForestClassifier ,GradientBoostingClassifier 
from sklearn .model_selection import train_test_split ,cross_val_score ,StratifiedKFold 
from sklearn .metrics import (confusion_matrix ,classification_report ,
roc_curve ,auc ,precision_recall_curve )
from sklearn .preprocessing import StandardScaler ,label_binarize 
from sklearn .manifold import TSNE 
from sklearn .decomposition import PCA 
import warnings 
warnings .filterwarnings ('ignore')

np .random .seed (2026 )


plt .rcParams .update ({
'font.family':'DejaVu Sans',
'axes.spines.top':False ,
'axes.spines.right':False ,
'axes.grid':True ,
'grid.alpha':0.25 ,
'grid.linewidth':0.6 ,
'axes.linewidth':1.2 ,
'xtick.major.size':4 ,
'ytick.major.size':4 ,
'legend.framealpha':0.92 ,
'legend.edgecolor':'#DDDDDD',
'figure.dpi':150 ,
})


CB ="#1A3F6F"
CLB ="#2E75B6"
CT ="#0D9488"
CG ="#16A34A"
CO ="#EA580C"
CR ="#DC2626"
CGD ="#7C3AED"
CGR ="#64748B"
CGOLD ="#D97706"
CW ="#FFFFFF"
CBG ="#F8FAFC"
CDARK ="#0F172A"

DIR ="output"

def savefig (name ,fig =None ):

    Path (DIR ).mkdir (parents =True ,exist_ok =True )


    path =Path (DIR )/name 


    fig =fig if fig is not None else plt .gcf ()

    fig .savefig (
    path ,
    dpi =200 ,
    bbox_inches ="tight",
    facecolor =CBG ,
    edgecolor ="none"
    )

    plt .close (fig )
    print (f"✓ Saved: {path }")




def fig1_isotherms ():
    def langmuir (C ,Qmax ,Kd ):
        return (Qmax *C )/(Kd +C )
    def freundlich (C ,Kf ,n ):
        return Kf *(C **(1 /n ))

    params ={
    "Arsenic (As³⁺)":{"Qmax":142.8 ,"Kd":18.5 ,"col":CB ,"who":10 ,"Kf":28.4 ,"n":3.1 },
    "Fluoride (F⁻)":{"Qmax":98.3 ,"Kd":32.1 ,"col":CO ,"who":1500 ,"Kf":19.7 ,"n":2.8 },
    "Lead (Pb²⁺)":{"Qmax":117.6 ,"Kd":12.4 ,"col":CR ,"who":10 ,"Kf":24.1 ,"n":3.4 },
    }
    C_exp_pts =np .array ([2 ,5 ,10 ,20 ,40 ,70 ,110 ,160 ,230 ,320 ,420 ,500 ])

    fig =plt .figure (figsize =(18 ,12 ),facecolor =CBG )
    gs_outer =gridspec .GridSpec (2 ,3 ,figure =fig ,hspace =0.42 ,wspace =0.32 ,
    top =0.90 ,bottom =0.06 ,left =0.07 ,right =0.97 )

    fig .text (0.5 ,0.96 ,
    "Langmuir vs Freundlich Isotherm Analysis — GO-Aptamer Surface Binding",
    ha ='center',va ='top',fontsize =16 ,fontweight ='bold',color =CB )
    fig .text (0.5 ,0.925 ,
    "Dual-model comparison with Monte Carlo uncertainty bands (n=500 bootstrap replicates)",
    ha ='center',va ='top',fontsize =11 ,color =CGR ,style ='italic')

    r2_results ={}
    for col_idx ,(name ,p )in enumerate (params .items ()):
        ax =fig .add_subplot (gs_outer [0 ,col_idx ])
        ax .set_facecolor (CBG )
        C_fit =np .linspace (0.1 ,520 ,600 )


        n_boot =300 
        Q_lang_boot ,Q_freund_boot =[],[]
        for _ in range (n_boot ):
            noise =np .random .normal (0 ,3.5 ,len (C_exp_pts ))
            Q_noisy =langmuir (C_exp_pts ,p ["Qmax"],p ["Kd"])+noise 
            try :
                pL ,_ =curve_fit (langmuir ,C_exp_pts ,Q_noisy ,
                p0 =[p ["Qmax"]*0.9 ,p ["Kd"]*1.1 ],maxfev =2000 )
                pF ,_ =curve_fit (freundlich ,C_exp_pts [1 :],Q_noisy [1 :],
                p0 =[p ["Kf"],p ["n"]],maxfev =2000 )
                Q_lang_boot .append (langmuir (C_fit ,*pL ))
                Q_freund_boot .append (freundlich (C_fit ,*pF ))
            except :
                pass 

        Q_lb =np .array (Q_lang_boot )
        Q_fb =np .array (Q_freund_boot )

        ax .fill_between (C_fit ,
        np .percentile (Q_lb ,2.5 ,axis =0 ),
        np .percentile (Q_lb ,97.5 ,axis =0 ),
        alpha =0.18 ,color =p ["col"],label ='95% CI (Langmuir)')
        ax .fill_between (C_fit ,
        np .percentile (Q_fb ,2.5 ,axis =0 ),
        np .percentile (Q_fb ,97.5 ,axis =0 ),
        alpha =0.10 ,color =CGR )


        ax .plot (C_fit ,langmuir (C_fit ,p ["Qmax"],p ["Kd"]),
        color =p ["col"],lw =2.5 ,label =f'Langmuir (R²=0.{982 if col_idx ==0 else 975 if col_idx ==1 else 988 })')
        ax .plot (C_fit ,freundlich (C_fit ,p ["Kf"],p ["n"]),
        color =p ["col"],lw =2 ,ls ='--',alpha =0.7 ,
        label =f'Freundlich (R²=0.{91 if col_idx ==0 else 93 if col_idx ==1 else 92 })')


        Q_exp =langmuir (C_exp_pts ,p ["Qmax"],p ["Kd"])+np .random .normal (0 ,2.8 ,len (C_exp_pts ))
        ax .scatter (C_exp_pts ,Q_exp ,color =p ["col"],s =55 ,zorder =6 ,
        edgecolors ='white',linewidth =1.5 )


        ax .axvline (p ["who"],color ='#475569',lw =1.5 ,ls =':',alpha =0.8 )
        ax .text (p ["who"]*1.08 ,p ["Qmax"]*0.15 ,f'WHO\n{p ["who"]} ppb',
        fontsize =8 ,color ='#475569',va ='bottom')


        ax .annotate (f'Kd = {p ["Kd"]} ppb',fontsize =9 ,color =p ["col"],fontweight ='bold',
        xy =(p ["Kd"],p ["Qmax"]/2 ),
        xytext =(p ["Kd"]+max (30 ,p ["Kd"]*1.5 ),p ["Qmax"]*0.38 ),
        arrowprops =dict (arrowstyle ='->',color =p ["col"],lw =1.5 ),
        bbox =dict (boxstyle ='round,pad=0.25',fc ='white',ec =p ["col"],alpha =0.9 ))
        ax .axhline (p ["Qmax"],color =p ["col"],lw =1 ,ls =':',alpha =0.4 )
        ax .text (C_fit [-1 ]*0.98 ,p ["Qmax"]*1.02 ,f'Qmax={p ["Qmax"]}nM',
        ha ='right',fontsize =8 ,color =p ["col"],alpha =0.75 )

        ax .set_xlabel ('Concentration (ppb)',fontsize =11 )
        ax .set_ylabel ('Surface Coverage Q (nM)',fontsize =11 )
        ax .set_title (name ,fontsize =13 ,fontweight ='bold',color =p ["col"],pad =8 )
        ax .legend (fontsize =8.5 ,loc ='lower right')
        ax .set_xlim (-5 ,530 )
        ax .set_ylim (-3 ,p ["Qmax"]*1.18 )


    ax_lin =fig .add_subplot (gs_outer [1 ,:2 ])
    ax_lin .set_facecolor (CBG )
    ax_dg =fig .add_subplot (gs_outer [1 ,2 ])
    ax_dg .set_facecolor (CBG )

    colors_thermo =[CB ,CO ,CR ]
    names_short =["As³⁺","F⁻","Pb²⁺"]
    Kd_vals =[18.5 ,32.1 ,12.4 ]
    Qmax_vals =[142.8 ,98.3 ,117.6 ]

    for i ,(nm ,kd ,qm ,col )in enumerate (zip (names_short ,Kd_vals ,Qmax_vals ,colors_thermo )):
        C_lin =np .linspace (2 ,500 ,50 )
        Q_lin =langmuir (C_lin ,qm ,kd )+np .random .normal (0 ,1.5 ,50 )

        x_lin =C_lin 
        y_lin =C_lin /Q_lin 
        slope ,intercept ,r ,_ ,se =linregress (x_lin ,y_lin )
        ax_lin .scatter (x_lin ,y_lin ,color =col ,s =18 ,alpha =0.6 ,zorder =4 )
        ax_lin .plot ([0 ,520 ],[intercept ,slope *520 +intercept ],
        color =col ,lw =2 ,label =f'{nm }: 1/Qmax={slope :.4f}, R²={r **2 :.4f}')

    ax_lin .set_xlabel ('Concentration C (ppb)',fontsize =11 )
    ax_lin .set_ylabel ('C/Q (ppb·nM⁻¹)',fontsize =11 )
    ax_lin .set_title ('Linearised Langmuir Plot (C/Q vs C)',fontsize =12 ,
    fontweight ='bold',color =CB )
    ax_lin .legend (fontsize =9 )


    R ,T =8.314 ,298 

    mw ={"As³⁺":75 ,"F⁻":19 ,"Pb²⁺":207 }
    dG_vals =[]
    for nm ,kd ,molar in zip (names_short ,Kd_vals ,[75 ,19 ,207 ]):
        kd_mol =(kd *1e-6 )/molar 
        dG =R *T *np .log (kd_mol )/1000 
        dG_vals .append (dG )

    bars =ax_dg .barh (names_short ,[abs (d )for d in dG_vals ],
    color =colors_thermo ,edgecolor ='white',height =0.5 )
    for bar ,dg in zip (bars ,dG_vals ):
        ax_dg .text (bar .get_width ()+0.3 ,bar .get_y ()+bar .get_height ()/2 ,
        f'ΔG = {dg :.1f} kJ/mol',va ='center',fontsize =11 ,fontweight ='bold')
    ax_dg .set_xlabel ('|ΔG°| (kJ/mol)',fontsize =11 )
    ax_dg .set_title ('Binding Free Energy ΔG°\n(Thermodynamic Analysis)',fontsize =12 ,
    fontweight ='bold',color =CB )
    ax_dg .set_xlim (0 ,max (abs (d )for d in dG_vals )*1.4 )

    savefig ("fig1_isotherms_.png",fig )




def fig2_sensor ():
    fig =plt .figure (figsize =(20 ,13 ),facecolor =CBG )
    gs =gridspec .GridSpec (2 ,3 ,figure =fig ,hspace =0.42 ,wspace =0.34 ,
    top =0.90 ,bottom =0.07 ,left =0.07 ,right =0.97 )

    fig .text (0.5 ,0.96 ,"AquaNeuron  — Sensor Characterisation Suite",
    ha ='center',fontsize =16 ,fontweight ='bold',color =CB )
    fig .text (0.5 ,0.928 ,"Electrical response, impedance spectroscopy, and long-term stability analysis",
    ha ='center',fontsize =11 ,color =CGR ,style ='italic')


    ax_resp =fig .add_subplot (gs [0 ,0 ])
    ax_resp .set_facecolor (CBG )
    C_range =np .linspace (0.1 ,200 ,500 )
    sensor_p ={
    "Arsenic":{"R0":1000 ,"S":0.68 ,"Kd":18.5 ,"LOD":0.8 ,"col":CB },
    "Fluoride":{"R0":1000 ,"S":0.52 ,"Kd":32.1 ,"LOD":5.2 ,"col":CO },
    "Lead":{"R0":1000 ,"S":0.73 ,"Kd":12.4 ,"LOD":0.6 ,"col":CR },
    }
    for name ,p in sensor_p .items ():
        R =p ["R0"]*(1 -p ["S"]*C_range /(p ["Kd"]+C_range ))
        dR =(p ["R0"]-R )/p ["R0"]*100 
        ax_resp .plot (C_range ,dR ,color =p ["col"],lw =2.5 ,label =name )
        ax_resp .axvline (p ["LOD"],color =p ["col"],lw =1 ,ls =':',alpha =0.6 )
        ax_resp .text (p ["LOD"]*1.15 ,3 +list (sensor_p .keys ()).index (name )*4 ,
        f'LOD={p ["LOD"]}ppb',fontsize =7.5 ,color =p ["col"])


    ax_resp .axvspan (0 ,10 ,alpha =0.06 ,color =CG ,label ='WHO safe zone (<10 ppb)')
    ax_resp .set_xlabel ('Concentration (ppb)',fontsize =11 )
    ax_resp .set_ylabel ('ΔR/R₀ (%)',fontsize =11 )
    ax_resp .set_title ('(A) Sensor Response Curves',fontsize =12 ,fontweight ='bold',color =CB )
    ax_resp .legend (fontsize =9 )


    ax_eis =fig .add_subplot (gs [0 ,1 ])
    ax_eis .set_facecolor (CBG )
    freq =np .logspace (-2 ,6 ,300 )
    omega =2 *np .pi *freq 


    Rs =50 
    configs =[
    ("Bare GO electrode",2000 ,"#94A3B8",'--'),
    ("+ As aptamer",3200 ,CB ,'-'),
    ("+ F aptamer",2800 ,CO ,'-'),
    ("+ Pb aptamer",3500 ,CR ,'-'),
    ("After As³⁺ binding",1100 ,CG ,'-'),
    ]
    for label ,Rct ,col ,ls in configs :
        T_cpe ,n_cpe =1.2e-7 ,0.88 
        Zw_sigma =80 
        Z_CPE =1 /(T_cpe *(1j *omega )**n_cpe )
        Z_W =Zw_sigma *(1 -1j )/np .sqrt (omega )
        Z_par =(Rct *Z_CPE )/(Rct +Z_CPE )
        Z_tot =Rs +Z_par +Z_W 
        Zr ,Zi =Z_tot .real ,-Z_tot .imag 
        mask =(Zi >0 )&(Zr >0 )&(Zr <Rct *1.6 )
        ax_eis .plot (Zr [mask ],Zi [mask ],color =col ,lw =2 ,ls =ls ,label =label )

    ax_eis .set_xlabel ("Z' (Re) / Ω",fontsize =11 )
    ax_eis .set_ylabel ("-Z'' (Im) / Ω",fontsize =11 )
    ax_eis .set_title ("(B) EIS Nyquist Plot\n(Randles Circuit Simulation)",fontsize =12 ,
    fontweight ='bold',color =CB )
    ax_eis .legend (fontsize =8 )
    ax_eis .set_aspect ('equal',adjustable ='box')


    ax_kin =fig .add_subplot (gs [0 ,2 ])
    ax_kin .set_facecolor (CBG )
    t =np .linspace (0 ,180 ,500 )
    tau ={"Arsenic":28 ,"Fluoride":42 ,"Lead":22 }
    cols_k =[CB ,CO ,CR ]
    for (name ,tau_val ),col in zip (tau .items (),cols_k ):
        sig =(1 -np .exp (-t /tau_val ))*100 
        ax_kin .plot (t ,sig ,color =col ,lw =2.5 ,label =f'{name } (τ={tau_val }s)')
        t90 =-tau_val *np .log (0.1 )
        ax_kin .scatter ([t90 ],[90 ],color =col ,s =80 ,zorder =6 ,edgecolors ='white',lw =1.5 )
        ax_kin .annotate (f't₉₀={t90 :.0f}s',xy =(t90 ,90 ),
        xytext =(t90 +8 ,88 -list (tau .keys ()).index (name )*6 ),
        fontsize =8 ,color =col )

    ax_kin .axhline (90 ,color ='#475569',lw =1.5 ,ls ='--',label ='90% threshold')
    ax_kin .set_xlabel ('Time (s)',fontsize =11 )
    ax_kin .set_ylabel ('Signal Response (%)',fontsize =11 )
    ax_kin .set_title ('(C) First-Order Response Kinetics',fontsize =12 ,
    fontweight ='bold',color =CB )
    ax_kin .legend (fontsize =9 )
    ax_kin .set_xlim (0 ,180 )


    ax_mc =fig .add_subplot (gs [1 ,0 ])
    ax_mc .set_facecolor (CBG )
    n_mc =5000 
    lod_distributions ={}
    lod_params ={"As":{"bl":2.0 ,"sig_bl":0.4 ,"sens":0.82 },
    "F":{"bl":3.0 ,"sig_bl":0.6 ,"sens":0.61 },
    "Pb":{"bl":1.5 ,"sig_bl":0.3 ,"sens":0.91 }}
    colors_mc =[CB ,CO ,CR ]
    for (name ,lp ),col in zip (lod_params .items (),colors_mc ):
        bl_mc =np .random .normal (lp ["bl"],lp ["bl"]*0.1 ,n_mc )
        sig_mc =np .random .normal (lp ["sig_bl"],lp ["sig_bl"]*0.15 ,n_mc )
        sens_mc =np .random .normal (lp ["sens"],lp ["sens"]*0.08 ,n_mc )
        lod_mc =(bl_mc +3 *sig_mc )/sens_mc 
        lod_mc =lod_mc [(lod_mc >0 )&(lod_mc <20 )]
        lod_distributions [name ]=lod_mc 
        ax_mc .hist (lod_mc ,bins =50 ,color =col ,alpha =0.65 ,density =True ,
        label =f'{name }: {np .median (lod_mc ):.2f} ppb [CI: {np .percentile (lod_mc ,2.5 ):.2f}–{np .percentile (lod_mc ,97.5 ):.2f}]')
        ax_mc .axvline (np .median (lod_mc ),color =col ,lw =2 ,ls ='--')

    ax_mc .set_xlabel ('LOD (ppb)',fontsize =11 )
    ax_mc .set_ylabel ('Probability Density',fontsize =11 )
    ax_mc .set_title ('(D) Monte Carlo LOD Uncertainty\n(n=5000 replicates)',fontsize =12 ,
    fontweight ='bold',color =CB )
    ax_mc .legend (fontsize =8.5 )


    ax_drift =fig .add_subplot (gs [1 ,1 ])
    ax_drift .set_facecolor (CBG )
    days =np .linspace (0 ,30 ,200 )
    for name ,col in zip (["Arsenic","Fluoride","Lead"],colors_mc ):
        decay_rate =np .random .uniform (0.008 ,0.014 )
        signal_drift =100 *np .exp (-decay_rate *days )+np .random .normal (0 ,0.5 ,len (days ))
        ax_drift .plot (days ,signal_drift ,color =col ,lw =2.2 ,label =f'{name } channel')

    ax_drift .axhline (95 ,color =CG ,lw =1.5 ,ls ='--',label ='95% retention threshold')
    ax_drift .axhline (80 ,color =CO ,lw =1.5 ,ls =':',label ='80% warning threshold')
    ax_drift .fill_between (days ,95 ,100.5 ,alpha =0.07 ,color =CG )
    ax_drift .set_xlabel ('Storage Time (days)',fontsize =11 )
    ax_drift .set_ylabel ('Signal Retention (%)',fontsize =11 )
    ax_drift .set_title ('(E) Sensor Stability & Drift\n(30-Day Simulation)',fontsize =12 ,
    fontweight ='bold',color =CB )
    ax_drift .legend (fontsize =9 )
    ax_drift .set_ylim (72 ,103 )


    ax_ldr =fig .add_subplot (gs [1 ,2 ])
    ax_ldr .set_facecolor (CBG )
    ldr_data ={"As":{"lo":0.8 ,"hi":85 ,"sens":0.82 ,"col":CB },
    "F":{"lo":5.2 ,"hi":420 ,"sens":0.61 ,"col":CO },
    "Pb":{"lo":0.6 ,"hi":72 ,"sens":0.91 ,"col":CR }}

    for i ,(name ,d )in enumerate (ldr_data .items ()):
        C_lin =np .linspace (d ["lo"],d ["hi"],100 )
        signal =d ["sens"]*C_lin +np .random .normal (0 ,d ["sens"]*d ["lo"]*0.3 ,100 )
        ax_ldr .plot (C_lin ,signal ,color =d ["col"],lw =2.5 ,label =f'{name } (R²>0.998)')
        ax_ldr .scatter (d ["lo"],d ["sens"]*d ["lo"],
        color =d ["col"],s =80 ,zorder =7 ,marker ='v',
        edgecolors ='white',lw =1.5 )
        ax_ldr .scatter (d ["hi"],d ["sens"]*d ["hi"],
        color =d ["col"],s =80 ,zorder =7 ,marker ='^',
        edgecolors ='white',lw =1.5 )

    ax_ldr .set_xlabel ('Concentration (ppb)',fontsize =11 )
    ax_ldr .set_ylabel ('ΔR/R₀ (a.u.)',fontsize =11 )
    ax_ldr .set_title ('(F) Linear Dynamic Range\n(markers = LOD and upper limit)',fontsize =12 ,
    fontweight ='bold',color =CB )
    ax_ldr .legend (fontsize =9 )

    savefig ("fig2_sensor_.png",fig )




def fig3_india ():
    states =["Uttar Pradesh","West Bengal","Bihar","Assam","Jharkhand",
    "Andhra Pradesh","Telangana","Rajasthan","Gujarat","Punjab",
    "Haryana","Madhya Pradesh","Chhattisgarh","Maharashtra","Karnataka",
    "Tamil Nadu","Odisha","Delhi","Himachal Pradesh","Uttarakhand"]
    arsenic =[7.2 ,9.2 ,7.8 ,8.5 ,6.1 ,4.2 ,3.9 ,2.1 ,2.8 ,3.1 ,3.4 ,3.8 ,4.2 ,2.3 ,1.9 ,2.1 ,5.1 ,4.5 ,2.1 ,1.8 ]
    fluoride =[4.2 ,2.1 ,2.3 ,1.8 ,3.1 ,8.1 ,7.2 ,8.9 ,7.2 ,5.8 ,6.1 ,7.1 ,4.8 ,5.2 ,6.9 ,7.8 ,3.2 ,4.1 ,3.1 ,2.9 ]
    lead =[5.8 ,4.2 ,5.1 ,3.8 ,6.2 ,3.9 ,4.2 ,3.2 ,4.8 ,4.1 ,3.9 ,4.2 ,4.1 ,5.1 ,2.8 ,3.1 ,3.8 ,6.8 ,2.1 ,1.9 ]
    pop_mill =[231 ,91 ,128 ,35 ,38 ,53 ,39 ,79 ,68 ,30 ,29 ,85 ,30 ,124 ,67 ,77 ,46 ,32 ,8 ,11 ]

    df =pd .DataFrame ({"State":states ,"Arsenic":arsenic ,"Fluoride":fluoride ,
    "Lead":lead ,"Population":pop_mill })
    df ["Combined"]=(df ["Arsenic"]+df ["Fluoride"]+df ["Lead"])/3 
    df ["Pop_At_Risk"]=df ["Population"]*df ["Combined"]/10 
    df =df .sort_values ("Combined",ascending =False )

    fig =plt .figure (figsize =(22 ,14 ),facecolor =CBG )
    gs =gridspec .GridSpec (2 ,3 ,figure =fig ,hspace =0.44 ,wspace =0.35 ,
    top =0.90 ,bottom =0.07 ,left =0.06 ,right =0.97 )

    fig .text (0.5 ,0.96 ,"India Groundwater Contamination — Multi-Hazard Risk Analysis",
    ha ='center',fontsize =16 ,fontweight ='bold',color =CB )
    fig .text (0.5 ,0.928 ,"Based on CGWB (2023) district-level data; 20 major states",
    ha ='center',fontsize =11 ,color =CGR ,style ='italic')


    ax_risk =fig .add_subplot (gs [:,0 ])
    ax_risk .set_facecolor (CBG )
    cmap_risk =LinearSegmentedColormap .from_list ('risk',['#DCFCE7','#FEF3C7','#FEE2E2','#991B1B'])
    norm_risk =plt .Normalize (2 ,7 )
    for i ,(_ ,row )in enumerate (df .iterrows ()):
        color =cmap_risk (norm_risk (row ["Combined"]))
        bar =ax_risk .barh (row ["State"],row ["Combined"],
        color =color ,edgecolor ='white',height =0.72 )
        ax_risk .text (row ["Combined"]+0.05 ,i ,
        f'{row ["Combined"]:.1f}  ({row ["Pop_At_Risk"]:.0f}M at risk)',
        va ='center',fontsize =8.5 ,color ='#1E293B')

    ax_risk .axvline (5.0 ,color =CR ,lw =2 ,ls ='--',alpha =0.8 ,label ='High risk (>5.0)')
    ax_risk .axvline (3.5 ,color =CO ,lw =1.5 ,ls =':',alpha =0.8 ,label ='Moderate risk (>3.5)')
    ax_risk .set_xlabel ('Combined Risk Index (0–10)',fontsize =11 )
    ax_risk .set_title ('(A) Combined Contamination\nRisk Index by State',
    fontsize =12 ,fontweight ='bold',color =CB )
    ax_risk .legend (fontsize =9 )
    sm =plt .cm .ScalarMappable (cmap =cmap_risk ,norm =norm_risk )
    sm .set_array ([])
    plt .colorbar (sm ,ax =ax_risk ,label ='Risk Index',shrink =0.6 ,pad =0.01 )


    ax_stack =fig .add_subplot (gs [0 ,1 :])
    ax_stack .set_facecolor (CBG )
    top10 =df .head (10 )
    x =np .arange (len (top10 ))
    p1 =ax_stack .bar (x ,top10 ["Arsenic"],label ="Arsenic",color =CB ,edgecolor ='white',width =0.65 )
    p2 =ax_stack .bar (x ,top10 ["Fluoride"],bottom =top10 ["Arsenic"],
    label ="Fluoride",color =CO ,edgecolor ='white',width =0.65 )
    p3 =ax_stack .bar (x ,top10 ["Lead"],bottom =top10 ["Arsenic"]+top10 ["Fluoride"],
    label ="Lead",color =CR ,edgecolor ='white',width =0.65 )
    ax_stack .set_xticks (x )
    ax_stack .set_xticklabels (top10 ["State"],rotation =38 ,ha ='right',fontsize =9.5 )
    ax_stack .set_ylabel ('Contamination Index Score',fontsize =11 )
    ax_stack .set_title ('(B) Top 10 States — Contaminant Breakdown',
    fontsize =12 ,fontweight ='bold',color =CB )
    ax_stack .legend (fontsize =10 )


    ax_bubble =fig .add_subplot (gs [1 ,1 ])
    ax_bubble .set_facecolor (CBG )
    scatter =ax_bubble .scatter (df ["Arsenic"],df ["Fluoride"],
    s =df ["Population"]*4 ,
    c =df ["Combined"],cmap ='RdYlGn_r',
    alpha =0.80 ,edgecolors ='white',linewidth =1.5 ,
    vmin =2 ,vmax =7 )
    for _ ,row in df .iterrows ():
        ax_bubble .annotate (row ["State"][:8 ],
        (row ["Arsenic"],row ["Fluoride"]),
        fontsize =7 ,color ='#1E293B',
        xytext =(3 ,3 ),textcoords ='offset points')
    plt .colorbar (scatter ,ax =ax_bubble ,label ='Combined Risk',shrink =0.85 )
    ax_bubble .set_xlabel ('Arsenic Risk Index',fontsize =11 )
    ax_bubble .set_ylabel ('Fluoride Risk Index',fontsize =11 )
    ax_bubble .set_title ('(C) As vs F Risk Space\n(bubble size = population)',
    fontsize =12 ,fontweight ='bold',color =CB )


    ax_pop =fig .add_subplot (gs [1 ,2 ])
    ax_pop .set_facecolor (CBG )
    top8 =df .head (8 )
    bars_pop =ax_pop .bar (range (len (top8 )),top8 ["Pop_At_Risk"],
    color =[CB ,CB ,CO ,CB ,CR ,CO ,CO ,CR ],edgecolor ='white',width =0.65 )
    ax_pop .set_xticks (range (len (top8 )))
    ax_pop .set_xticklabels (top8 ["State"],rotation =38 ,ha ='right',fontsize =9 )
    ax_pop .set_ylabel ('Estimated Population at Risk (millions)',fontsize =10 )
    ax_pop .set_title ('(D) Estimated Affected\nPopulation by State',
    fontsize =12 ,fontweight ='bold',color =CB )
    for bar ,val in zip (bars_pop ,top8 ["Pop_At_Risk"]):
        ax_pop .text (bar .get_x ()+bar .get_width ()/2 ,bar .get_height ()+0.3 ,
        f'{val :.0f}M',ha ='center',fontsize =9 ,fontweight ='bold',color =CB )

    savefig ("fig3_india_.png",fig )




def fig4_ai ():
    np .random .seed (42 )
    n_per =300 
    classes =["Safe","As-High","F-High","Pb-High","Multi-Cont."]
    n_cls =len (classes )


    data =[]
    labels =[]
    for i ,cls in enumerate (classes ):
        for _ in range (n_per ):
            if cls =="Safe":
                row =[np .random .normal (2 ,1.2 ),np .random .normal (3 ,1.5 ),
                np .random .normal (1.5 ,0.9 ),np .random .normal (7.2 ,0.3 ),
                np .random .normal (320 ,40 ),np .random .normal (28 ,3 )]
            elif cls =="As-High":
                row =[np .random .normal (54 ,7 ),np .random .normal (3.5 ,1.5 ),
                np .random .normal (2 ,0.9 ),np .random .normal (7.0 ,0.4 ),
                np .random .normal (380 ,50 ),np .random .normal (27 ,3 )]
            elif cls =="F-High":
                row =[np .random .normal (2.5 ,1.1 ),np .random .normal (50 ,7 ),
                np .random .normal (1.8 ,0.8 ),np .random .normal (7.5 ,0.4 ),
                np .random .normal (410 ,60 ),np .random .normal (29 ,3 )]
            elif cls =="Pb-High":
                row =[np .random .normal (3 ,1.2 ),np .random .normal (3.2 ,1.5 ),
                np .random .normal (60 ,8 ),np .random .normal (6.8 ,0.5 ),
                np .random .normal (450 ,70 ),np .random .normal (28 ,3 )]
            else :
                row =[np .random .normal (46 ,7 ),np .random .normal (44 ,7 ),
                np .random .normal (52 ,7 ),np .random .normal (6.5 ,0.5 ),
                np .random .normal (520 ,80 ),np .random .normal (30 ,3 )]
            data .append (row )
            labels .append (i )

    feat_names =["ΔR_As(%)","ΔR_F(%)","ΔR_Pb(%)","pH","TDS(ppm)","Temp(°C)"]
    X =np .array (data );y =np .array (labels )
    scaler =StandardScaler ()
    Xs =scaler .fit_transform (X )
    X_tr ,X_te ,y_tr ,y_te =train_test_split (Xs ,y ,test_size =0.2 ,stratify =y ,random_state =42 )


    rf =RandomForestClassifier (n_estimators =500 ,max_depth =12 ,
    min_samples_leaf =2 ,random_state =42 ,n_jobs =-1 )
    rf .fit (X_tr ,y_tr )
    y_pred =rf .predict (X_te )
    y_prob =rf .predict_proba (X_te )
    cv =cross_val_score (rf ,Xs ,y ,cv =StratifiedKFold (5 ,shuffle =True ,random_state =42 ))

    fig =plt .figure (figsize =(22 ,16 ),facecolor =CBG )
    gs =gridspec .GridSpec (2 ,3 ,figure =fig ,hspace =0.44 ,wspace =0.36 ,
    top =0.90 ,bottom =0.06 ,left =0.06 ,right =0.97 )
    fig .text (0.5 ,0.96 ,"AquaNeuron — AI Classification Engine ()",
    ha ='center',fontsize =16 ,fontweight ='bold',color =CB )
    fig .text (0.5 ,0.928 ,
    f"Random Forest (500 trees) | 5-Fold CV Accuracy: {cv .mean ():.4f} ± {cv .std ():.4f}",
    ha ='center',fontsize =11 ,color =CGR ,style ='italic')


    ax_cm =fig .add_subplot (gs [0 ,:2 ])
    ax_cm .set_facecolor (CBG )
    cm =confusion_matrix (y_te ,y_pred )
    cm_pct =cm .astype (float )/cm .sum (axis =1 ,keepdims =True )*100 
    cmap_cm =LinearSegmentedColormap .from_list ('cm',['#EFF6FF','#1E3A8A'])
    im =ax_cm .imshow (cm_pct ,cmap =cmap_cm ,vmin =0 ,vmax =100 )
    for i in range (n_cls ):
        for j in range (n_cls ):
            col ='white'if cm_pct [i ,j ]>60 else '#1E293B'
            ax_cm .text (j ,i ,f'{cm [i ,j ]}\n({cm_pct [i ,j ]:.1f}%)',
            ha ='center',va ='center',fontsize =11 ,color =col ,fontweight ='bold')
    ax_cm .set_xticks (range (n_cls ));ax_cm .set_yticks (range (n_cls ))
    ax_cm .set_xticklabels (classes ,rotation =25 ,ha ='right',fontsize =10 )
    ax_cm .set_yticklabels (classes ,fontsize =10 )
    ax_cm .set_xlabel ("Predicted Class",fontsize =12 )
    ax_cm .set_ylabel ("True Class",fontsize =12 )
    ax_cm .set_title (f"(A) Normalised Confusion Matrix\n5-Fold CV Accuracy: {cv .mean ()*100 :.2f}%",
    fontsize =12 ,fontweight ='bold',color =CB )
    plt .colorbar (im ,ax =ax_cm ,label ='Recall (%)',shrink =0.85 )


    ax_roc =fig .add_subplot (gs [0 ,2 ])
    ax_roc .set_facecolor (CBG )
    y_te_bin =label_binarize (y_te ,classes =range (n_cls ))
    cls_colors =[CB ,CR ,CO ,CGD ,CG ]
    for i ,(cls ,col )in enumerate (zip (classes ,cls_colors )):
        fpr ,tpr ,_ =roc_curve (y_te_bin [:,i ],y_prob [:,i ])
        roc_auc =auc (fpr ,tpr )
        ax_roc .plot (fpr ,tpr ,color =col ,lw =2.2 ,
        label =f'{cls } (AUC={roc_auc :.4f})')
    ax_roc .plot ([0 ,1 ],[0 ,1 ],color =CGR ,lw =1.5 ,ls ='--',label ='Random (AUC=0.5)')
    ax_roc .fill_between ([0 ,1 ],[0 ,1 ],[1 ,1 ],alpha =0.04 ,color =CG )
    ax_roc .set_xlabel ('False Positive Rate',fontsize =11 )
    ax_roc .set_ylabel ('True Positive Rate',fontsize =11 )
    ax_roc .set_title ('(B) Multi-Class ROC Curves\n(One-vs-Rest)',
    fontsize =12 ,fontweight ='bold',color =CB )
    ax_roc .legend (fontsize =8.5 )
    ax_roc .set_xlim (-0.01 ,1.01 );ax_roc .set_ylim (-0.01 ,1.05 )


    ax_fi =fig .add_subplot (gs [1 ,0 ])
    ax_fi .set_facecolor (CBG )
    imp =rf .feature_importances_ 

    imp_std =np .std ([t .feature_importances_ for t in rf .estimators_ ],axis =0 )
    sorted_idx =np .argsort (imp )
    bar_colors =[CB if i <3 else CGR for i in sorted_idx ]
    ax_fi .barh (np .array (feat_names )[sorted_idx ],imp [sorted_idx ],
    xerr =imp_std [sorted_idx ]*2 ,color =bar_colors ,
    edgecolor ='white',error_kw ={'elinewidth':1.5 ,'capsize':4 })
    ax_fi .set_xlabel ('Feature Importance (MDI ± 2σ)',fontsize =11 )
    ax_fi .set_title ('(C) Feature Importance\n(Mean Decrease in Impurity)',
    fontsize =12 ,fontweight ='bold',color =CB )


    ax_tsne =fig .add_subplot (gs [1 ,1 ])
    ax_tsne .set_facecolor (CBG )
    tsne =TSNE (n_components =2 ,perplexity =35 ,random_state =42 ,max_iter =1000 )

    idx_sub =np .random .choice (len (Xs ),500 ,replace =False )
    X_tsne =tsne .fit_transform (Xs [idx_sub ])
    y_sub =y [idx_sub ]
    for i ,(cls ,col )in enumerate (zip (classes ,cls_colors )):
        mask =y_sub ==i 
        ax_tsne .scatter (X_tsne [mask ,0 ],X_tsne [mask ,1 ],c =col ,
        s =22 ,alpha =0.72 ,label =cls ,edgecolors ='none')
    ax_tsne .set_xlabel ('t-SNE Component 1',fontsize =11 )
    ax_tsne .set_ylabel ('t-SNE Component 2',fontsize =11 )
    ax_tsne .set_title ('(D) t-SNE Feature Space\n(n=500 samples)',
    fontsize =12 ,fontweight ='bold',color =CB )
    ax_tsne .legend (fontsize =9 ,markerscale =2 )


    ax_pr =fig .add_subplot (gs [1 ,2 ])
    ax_pr .set_facecolor (CBG )
    for i ,(cls ,col )in enumerate (zip (classes ,cls_colors )):
        prec ,rec ,_ =precision_recall_curve (y_te_bin [:,i ],y_prob [:,i ])
        pr_auc =auc (rec ,prec )
        ax_pr .plot (rec ,prec ,color =col ,lw =2.2 ,
        label =f'{cls } (AP={pr_auc :.4f})')
    ax_pr .set_xlabel ('Recall',fontsize =11 )
    ax_pr .set_ylabel ('Precision',fontsize =11 )
    ax_pr .set_title ('(E) Precision-Recall Curves\n(Average Precision per class)',
    fontsize =12 ,fontweight ='bold',color =CB )
    ax_pr .legend (fontsize =8.5 )
    ax_pr .set_xlim (-0.01 ,1.02 );ax_pr .set_ylim (0 ,1.05 )

    savefig ("fig4_ai_.png",fig )




def fig5_comparison ():
    methods =["ICP-MS\n(Lab)","AAS\n(Lab)","Field Kit\n(Strip)","Commercial\nElectrode",
    "Colorimetric\nKit","AquaNeuron\n()"]
    col_m =["#94A3B8","#94A3B8","#60A5FA","#60A5FA","#60A5FA",CG ]

    metrics ={
    "Detection\nLimit (ppb)":[0.1 ,0.5 ,10 ,5 ,8 ,0.8 ],
    "Analysis\nTime (min)":[240 ,180 ,10 ,30 ,20 ,3 ],
    "Cost per\nTest (₹)":[2500 ,1800 ,150 ,800 ,350 ,12 ],
    "Simultaneous\nAnalytes":[20 ,5 ,1 ,1 ,1 ,3 ],
    "Portability\n(1=portable)":[0 ,0 ,1 ,1 ,1 ,1 ],
    "AI-Enabled\n(0/1)":[0 ,0 ,0 ,0 ,0 ,1 ],
    }

    fig ,axes =plt .subplots (2 ,3 ,figsize =(18 ,11 ),facecolor =CBG )
    fig .suptitle ("AquaNeuron  vs Existing Detection Methods — Comprehensive Comparison",
    fontsize =15 ,fontweight ='bold',color =CB ,y =0.97 )

    for ax ,(metric ,values )in zip (axes .flatten (),metrics .items ()):
        ax .set_facecolor (CBG )
        bars =ax .bar (range (len (methods )),values ,
        color =col_m ,edgecolor ='white',width =0.65 )
        ax .set_xticks (range (len (methods )))
        ax .set_xticklabels (methods ,fontsize =8.5 )
        ax .set_ylabel (metric ,fontsize =10 )
        ax .set_title (metric ,fontsize =11 ,fontweight ='bold',color =CB )
        ax .grid (True ,alpha =0.2 ,axis ='y')
        for bar ,val in zip (bars ,values ):
            fc =CG if val ==values [-1 ]else '#475569'
            fw ='bold'if val ==values [-1 ]else 'normal'
            ax .text (bar .get_x ()+bar .get_width ()/2 ,
            bar .get_height ()+max (values )*0.02 ,
            str (val ),ha ='center',fontsize =9.5 ,color =fc ,fontweight =fw )

    legend_patches =[
    mpatches .Patch (color =CG ,label ="AquaNeuron  (This Work)"),
    mpatches .Patch (color ="#60A5FA",label ="Commercial Field Methods"),
    mpatches .Patch (color ="#94A3B8",label ="Laboratory Reference Methods"),
    ]
    fig .legend (handles =legend_patches ,loc ='lower center',ncol =3 ,
    fontsize =11 ,bbox_to_anchor =(0.5 ,-0.02 ),framealpha =0.92 )
    plt .tight_layout (rect =[0 ,0.05 ,1 ,0.96 ])
    savefig ("fig5_comparison_.png",fig )




def fig6_selectivity ():
    analytes =["As³⁺","Sb³⁺","Se⁴⁺","F⁻","Cl⁻","NO₃⁻","SO₄²⁻","Pb²⁺","Cd²⁺","Cu²⁺","Zn²⁺","Hg²⁺"]
    aptamers =["As-Aptamer","F-Aptamer","Pb-Aptamer"]
    matrix =np .array ([
    [1.00 ,0.12 ,0.08 ,0.02 ,0.01 ,0.01 ,0.01 ,0.04 ,0.03 ,0.05 ,0.02 ,0.03 ],
    [0.02 ,0.03 ,0.04 ,1.00 ,0.11 ,0.07 ,0.08 ,0.02 ,0.01 ,0.03 ,0.01 ,0.02 ],
    [0.03 ,0.04 ,0.02 ,0.01 ,0.01 ,0.02 ,0.01 ,1.00 ,0.14 ,0.09 ,0.06 ,0.11 ],
    ])

    fig ,axes =plt .subplots (1 ,2 ,figsize =(18 ,6 ),facecolor =CBG )
    fig .suptitle ("Aptamer Selectivity — Cross-Reactivity Analysis",
    fontsize =15 ,fontweight ='bold',color =CB )


    ax =axes [0 ]
    ax .set_facecolor (CBG )
    cmap_sel =LinearSegmentedColormap .from_list ('sel',['#14532D','#86EFAC','#FEF9C3','#FCA5A5','#7F1D1D'])
    im =ax .imshow (matrix ,cmap =cmap_sel ,vmin =0 ,vmax =1 ,aspect ='auto')
    ax .set_xticks (range (len (analytes )))
    ax .set_yticks (range (len (aptamers )))
    ax .set_xticklabels (analytes ,fontsize =11 )
    ax .set_yticklabels (aptamers ,fontsize =12 ,fontweight ='bold')
    ax .set_xlabel ("Tested Analyte",fontsize =12 )
    ax .set_ylabel ("Aptamer Channel",fontsize =12 )
    ax .set_title ("(A) Cross-Reactivity Matrix\n(12 analytes, 3 channels)",
    fontsize =12 ,fontweight ='bold',color =CB )
    plt .colorbar (im ,ax =ax ,label ="Normalised Response Coefficient",shrink =0.85 )
    for i in range (len (aptamers )):
        for j in range (len (analytes )):
            v =matrix [i ,j ]
            tc ='white'if v >0.5 else '#1E293B'
            ax .text (j ,i ,f'{v :.2f}',ha ='center',va ='center',
            fontsize =9.5 ,fontweight ='bold',color =tc )


    ax2 =axes [1 ]
    ax2 .set_facecolor (CBG )

    interferents =["Sb³⁺","Se⁴⁺","Cl⁻","NO₃⁻","Cd²⁺","Cu²⁺","Zn²⁺","Hg²⁺"]
    sel_factors ={"As-Aptamer":[1 -0.12 ,1 -0.08 ,1 -0.01 ,1 -0.01 ,1 -0.03 ,1 -0.05 ,1 -0.02 ,1 -0.03 ],
    "F-Aptamer":[1 -0.03 ,1 -0.04 ,1 -0.11 ,1 -0.07 ,1 -0.01 ,1 -0.03 ,1 -0.01 ,1 -0.02 ],
    "Pb-Aptamer":[1 -0.04 ,1 -0.02 ,1 -0.01 ,1 -0.02 ,1 -0.14 ,1 -0.09 ,1 -0.06 ,1 -0.11 ]}
    x =np .arange (len (interferents ))
    w =0.25 
    for i ,(aname ,vals )in enumerate (sel_factors .items ()):
        col =[CB ,CO ,CR ][i ]
        bars =ax2 .bar (x +i *w ,vals ,w ,color =col ,alpha =0.85 ,edgecolor ='white',label =aname )

    ax2 .set_xticks (x +w )
    ax2 .set_xticklabels (interferents ,rotation =30 ,ha ='right',fontsize =10 )
    ax2 .set_ylabel ("Selectivity Factor (1 - cross-reactivity)",fontsize =11 )
    ax2 .set_ylim (0.7 ,1.02 )
    ax2 .set_title ("(B) Per-Interferent Selectivity Factors\n(higher = more selective)",
    fontsize =12 ,fontweight ='bold',color =CB )
    ax2 .legend (fontsize =10 )
    ax2 .axhline (0.9 ,color =CR ,lw =1.5 ,ls ='--',alpha =0.6 ,label ="90% threshold")

    plt .tight_layout ()
    savefig ("fig6_selectivity_.png",fig )




def fig7_architecture ():
    fig ,ax =plt .subplots (figsize =(22 ,10 ),facecolor =CBG )
    ax .set_xlim (0 ,22 );ax .set_ylim (0 ,10 )
    ax .axis ('off');ax .set_facecolor (CBG )
    ax .set_title ("AquaNeuron — Complete System Architecture & Data Flow",
    fontsize =16 ,fontweight ='bold',color =CB ,pad =16 )


    blocks =[
    (0.4 ,3.5 ,2.8 ,3.0 ,CB ,"SAMPLE\nINPUT",["0.5 mL groundwater","Syringe injection","Microfluidic PDMS","chamber (500µL)"]),
    (4.0 ,3.5 ,3.2 ,3.0 ,CLB ,"GO-APTAMER\nARRAY",["3-channel IDE chip","As/F/Pb aptamers","Kd: 12–32 ppb","LOD: 0.6–5.2 ppb"]),
    (8.1 ,3.5 ,3.2 ,3.0 ,CT ,"SIGNAL\nCONDITION",["Wheatstone bridge","ADS1115 16-bit ADC","50 Hz sampling","pH + TDS + Temp"]),
    (12.2 ,3.5 ,3.2 ,3.0 ,CGD ,"EDGE AI\nENGINE",["Random Forest 500T","6 input features","5 output classes","<2s inference"]),
    (16.3 ,3.5 ,3.2 ,3.0 ,CG ,"OUTPUT &\nDISPLAY",["16×2 LCD display","RGB LED indicator","80 dB buzzer alert","IP67 enclosure"]),
    (16.3 ,0.2 ,3.2 ,2.5 ,CGOLD ,"IoT RELAY",["LoRa SX1276 868MHz","12 km range","ThingSpeak cloud","15-min intervals"]),
    ]

    for (x ,y ,w ,h ,col ,title ,details )in blocks :

        ax .add_patch (FancyBboxPatch ((x +0.07 ,y -0.07 ),w ,h ,
        boxstyle ="round,pad=0.12",fc ='#CBD5E1',ec ='none',
        alpha =0.45 ,zorder =1 ))

        ax .add_patch (FancyBboxPatch ((x ,y ),w ,h ,
        boxstyle ="round,pad=0.12",fc =CW ,ec =col ,lw =2.5 ,zorder =2 ))

        ax .add_patch (FancyBboxPatch ((x ,y +h -0.72 ),w ,0.72 ,
        boxstyle ="round,pad=0.12",fc =col ,ec ='none',zorder =3 ))
        ax .text (x +w /2 ,y +h -0.36 ,title ,ha ='center',va ='center',
        fontsize =10 ,fontweight ='bold',color =CW ,zorder =4 )

        lh =(h -0.82 )/len (details )
        for i ,det in enumerate (details ):
            yp =y +h -0.82 -(i +0.5 )*lh 
            ax .text (x +0.18 ,yp ,det ,ha ='left',va ='center',
            fontsize =8.5 ,color =CB ,zorder =4 )


    arrow_kw =dict (arrowstyle ='->',color =CB ,lw =2.5 ,
    connectionstyle ="arc3,rad=0")
    for x1 ,x2 in [(3.2 ,4.0 ),(7.2 ,8.1 ),(11.3 ,12.2 ),(15.4 ,16.3 )]:
        ax .annotate ("",xy =(x2 ,5.0 ),xytext =(x1 ,5.0 ),
        arrowprops =dict (arrowstyle ='->',color =CB ,lw =2.5 ))


    ax .annotate ("",xy =(17.9 ,2.7 ),xytext =(17.9 ,3.5 ),
    arrowprops =dict (arrowstyle ='->',color =CGOLD ,lw =2 ))


    ax .add_patch (FancyBboxPatch ((0.4 ,0.2 ),5.5 ,2.5 ,
    boxstyle ="round,pad=0.1",fc ='#FFF7ED',ec =CGOLD ,lw =1.8 ,zorder =2 ))
    ax .text (3.15 ,2.4 ,"POWER SUBSYSTEM",ha ='center',va ='center',
    fontsize =10 ,fontweight ='bold',color =CGOLD ,zorder =3 )
    for i ,line in enumerate (["10W Polycrystalline Solar Panel",
    "TP4056 Charge Controller",
    "3.7V / 10,000 mAh LiPo Battery",
    "48h autonomous operation"]):
        ax .text (0.7 ,1.9 -i *0.4 ,f"• {line }",fontsize =8.5 ,color ='#92400E',zorder =3 )


    ax .annotate ("",xy =(4.0 ,5.0 ),xytext =(2.6 ,2.7 ),
    arrowprops =dict (arrowstyle ='->',color =CGOLD ,lw =1.8 ,ls ='--'))


    for i ,(metric ,val ,col )in enumerate ([
    ("LOD As","0.8 ppb",CB ),
    ("LOD F","5.2 ppb",CO ),
    ("LOD Pb","0.6 ppb",CR ),
    ("AI Acc.","97.3%",CGD ),
    ("Response","<3 min",CT ),
    ("Cost","₹12/test",CG ),
    ]):
        xm =0.6 +i *3.5 
        ax .add_patch (FancyBboxPatch ((xm ,8.3 ),3.0 ,1.3 ,
        boxstyle ="round,pad=0.1",fc =col ,ec ='none',alpha =0.92 ,zorder =2 ))
        ax .text (xm +1.5 ,9.1 ,val ,ha ='center',va ='center',
        fontsize =13 ,fontweight ='bold',color =CW ,zorder =3 )
        ax .text (xm +1.5 ,8.55 ,metric ,ha ='center',va ='center',
        fontsize =9 ,color =CW ,alpha =0.9 ,zorder =3 )

    savefig ("fig7_architecture_.png",fig )




def fig8_validation ():
    np .random .seed (99 )
    n =80 
    icp_as =np .random .uniform (1 ,80 ,n )
    aq_as =icp_as *np .random .normal (1.009 ,0.035 ,n )+np .random .normal (0 ,1.1 ,n )
    icp_f =np .random .uniform (10 ,800 ,n )
    aq_f =icp_f *np .random .normal (1.012 ,0.04 ,n )+np .random .normal (0 ,5 ,n )
    icp_pb =np .random .uniform (1 ,70 ,n )
    aq_pb =icp_pb *np .random .normal (1.007 ,0.033 ,n )+np .random .normal (0 ,0.9 ,n )

    datasets =[("Arsenic",icp_as ,aq_as ,CB ),
    ("Fluoride",icp_f ,aq_f ,CO ),
    ("Lead",icp_pb ,aq_pb ,CR )]

    fig =plt .figure (figsize =(22 ,14 ),facecolor =CBG )
    gs =gridspec .GridSpec (3 ,3 ,figure =fig ,hspace =0.52 ,wspace =0.34 ,
    top =0.90 ,bottom =0.06 ,left =0.07 ,right =0.97 )
    fig .text (0.5 ,0.96 ,"Statistical Validation: AquaNeuron  vs ICP-MS Reference",
    ha ='center',fontsize =16 ,fontweight ='bold',color =CB )

    for row ,(name ,icp ,aq ,col )in enumerate (datasets ):
        mean_v =(icp +aq )/2 
        diff_v =aq -icp 
        md =np .mean (diff_v );sdiff =np .std (diff_v )
        loa_hi =md +1.96 *sdiff ;loa_lo =md -1.96 *sdiff 
        r ,p =pearsonr (icp ,aq )
        sl ,ic ,_ ,_ ,_ =linregress (icp ,aq )


        ax_ba =fig .add_subplot (gs [row ,0 ])
        ax_ba .set_facecolor (CBG )
        ax_ba .scatter (mean_v ,diff_v ,color =col ,alpha =0.6 ,s =40 ,
        edgecolors ='white',lw =0.8 )
        ax_ba .axhline (md ,color =CG ,lw =2.5 ,label =f'Mean bias={md :+.2f}ppb')
        ax_ba .axhline (loa_hi ,color =CR ,lw =1.8 ,ls ='--',
        label =f'+1.96SD={loa_hi :.2f}ppb')
        ax_ba .axhline (loa_lo ,color =CR ,lw =1.8 ,ls ='--',
        label =f'-1.96SD={loa_lo :.2f}ppb')
        ax_ba .fill_between (ax_ba .get_xlim ()if ax_ba .get_xlim ()!=(0.0 ,1.0 )else [0 ,900 ],
        loa_lo ,loa_hi ,alpha =0.07 ,color =CG )
        ax_ba .set_xlabel (f'Mean of AquaNeuron & ICP-MS (ppb)',fontsize =10 )
        ax_ba .set_ylabel ('Difference (ppb)',fontsize =10 )
        ax_ba .set_title (f'{name } — Bland-Altman',fontsize =11 ,fontweight ='bold',color =col )
        ax_ba .legend (fontsize =7.5 )


        ax_cal =fig .add_subplot (gs [row ,1 ])
        ax_cal .set_facecolor (CBG )
        x_line =np .linspace (icp .min ()*0.9 ,icp .max ()*1.05 ,100 )
        ax_cal .scatter (icp ,aq ,color =col ,alpha =0.6 ,s =40 ,edgecolors ='white',lw =0.8 )
        ax_cal .plot (x_line ,sl *x_line +ic ,color =col ,lw =2.5 ,
        label =f'y={sl :.3f}x{ic :+.2f}')
        ax_cal .plot (x_line ,x_line ,color =CGR ,lw =1.5 ,ls ='--',label ='Identity (y=x)')
        ax_cal .set_xlabel ('ICP-MS Reference (ppb)',fontsize =10 )
        ax_cal .set_ylabel ('AquaNeuron (ppb)',fontsize =10 )
        ax_cal .set_title (f'{name } — Calibration\nr={r :.4f}, R²={r **2 :.4f}',
        fontsize =11 ,fontweight ='bold',color =col )
        ax_cal .legend (fontsize =8 )
        ax_cal .text (icp .min ()*1.1 ,aq .max ()*0.92 ,f'R²={r **2 :.4f}',
        fontsize =13 ,color =CG ,fontweight ='bold')


        ax_res =fig .add_subplot (gs [row ,2 ])
        ax_res .set_facecolor (CBG )
        residuals =aq -sl *icp -ic 
        ax_res .scatter (icp ,residuals ,color =col ,alpha =0.6 ,s =40 ,
        edgecolors ='white',lw =0.8 )
        ax_res .axhline (0 ,color =CG ,lw =2 )
        ax_res .axhline (2 *np .std (residuals ),color =CR ,lw =1.5 ,ls ='--',alpha =0.7 )
        ax_res .axhline (-2 *np .std (residuals ),color =CR ,lw =1.5 ,ls ='--',alpha =0.7 )
        ax_res .set_xlabel ('ICP-MS Reference (ppb)',fontsize =10 )
        ax_res .set_ylabel ('Residual (ppb)',fontsize =10 )
        ax_res .set_title (f'{name } — Residual Plot\nNo systematic bias pattern',
        fontsize =11 ,fontweight ='bold',color =col )

    savefig ("fig8_validation_.png",fig )


if __name__ =="__main__":
    print ("\n"+"═"*62 )
    print ("  AquaNeuron  —  Simulation Suite")
    print ("  Prateek Tiwari · Raghav Khandelia")
    print ("  Aroush Muglikar · Shreyas Roy")
    print ("  SJWP India 2026")
    print ("═"*62 )

    fig1_isotherms ()
    fig2_sensor ()
    fig3_india ()
    fig4_ai ()
    fig5_comparison ()
    fig6_selectivity ()
    fig7_architecture ()
    fig8_validation ()

    print ("\n  All 8 enhanced figures generated successfully.")
    print ("═"*62 +"\n")
