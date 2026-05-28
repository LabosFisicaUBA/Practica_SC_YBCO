import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

def process_multi_loops(archivo = "lazos_M_variasT_variasH.txt" , columns = ["H_Oe", "T_K", "m_emu", "reg_fit"], width = 0.1 , thickness = 0.0005 ):
    
    df = pd.read_csv(
        archivo,
        sep="\t+", #sep=r"\s+|\t+",
        engine="python"
    )
    
    # Renombrar columnas si hace falta
    df.columns = columns
    
    
    df["T_nom"] = 5 * np.round(df["T_K"] / 5)
    
    results = []
    
    for T_nom, g in df.groupby("T_nom", sort=True):
        g = g.reset_index(drop=True)
    
        # localizar el punto de campo máximo, donde cambia la rama
        idx_turn = g["H_Oe"].abs().idxmax()
    
        branch_up = g.iloc[:idx_turn + 1].copy()
        branch_down = g.iloc[idx_turn:].copy()
    
        # usar campo absoluto luego de separar ramas
        branch_up["H_abs"] = branch_up["H_Oe"].abs()
        branch_down["H_abs"] = branch_down["H_Oe"].abs()
    
        # ordenar ambas ramas por H creciente para PCHIP
        branch_up = branch_up.sort_values("H_abs")
        branch_down = branch_down.sort_values("H_abs")
    
        # eliminar posibles duplicados exactos de H, si aparecen
        branch_up = branch_up.groupby("H_abs", as_index=False)["m_emu"].mean()
        branch_down = branch_down.groupby("H_abs", as_index=False)["m_emu"].mean()
    
        # necesitamos al menos 3 puntos por rama para interpolación razonable
        if len(branch_up) < 3 or len(branch_down) < 3:
            continue
    
        p_up = PchipInterpolator(branch_up["H_abs"], branch_up["m_emu"])
        p_down = PchipInterpolator(branch_down["H_abs"], branch_down["m_emu"])
    
        H_min = max(branch_up["H_abs"].min(), branch_down["H_abs"].min())
        H_max = min(branch_up["H_abs"].max(), branch_down["H_abs"].max())
        
        H_grid = np.linspace(H_min, H_max, 500)
        
    
        m_up = p_up(H_grid)
        m_down = p_down(H_grid)
    
        delta_m = np.abs(m_down - m_up)
        delta_m_max = delta_m.max()
        delta_m_norm = delta_m/delta_m_max
        m_center = 0.5 * (m_down + m_up)
    
        for H, dm, dm_n, mc in zip(H_grid, delta_m, delta_m_norm,  m_center):
            results.append({
                "T_K": T_nom,
                "H_Oe": H,
                "delta_m_emu": dm,
                "m_center_emu": mc,
                "delta_m_norm_emu": dm_n
            })
    
    out = pd.DataFrame(results)
    
    V_cm3 = width**2 * thickness   # volumen de la muestra
    a_cm = width    # dimensión menor transversal
    b_cm = width    # dimensión mayor transversal
    
    out["delta_M_emu_cm3"] = out["delta_m_emu"] / V_cm3
    out["Jc_A_cm2"] = (
        20 * out["delta_M_emu_cm3"] /
        (a_cm * (1 - a_cm/(3*b_cm)))
    )
    
    
#    out["Jc_norm_A_cm2"] = (
#        20 * out["delta_m_norm_emu"] /
#        (V_cm3 * (a_cm * (1 - a_cm/(3*b_cm))))
#    )

    return out 