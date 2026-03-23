import os
import sqlite3
import wfdb
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def generate_banner_12leads_elite(db_path, raw_dir, output_path):
    plt.style.use('dark_background')
    
    # 1. Pegar um paciente MI aleatorio no banco
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT r.filename_hr, r.ecg_id FROM ecg_records r JOIN ecg_scp_diagnoses map ON r.ecg_id = map.ecg_id JOIN scp_codes d ON map.scp_code = d.scp_code WHERE d.diagnostic_class = 'MI' ORDER BY RANDOM() LIMIT 1")
    res = cur.fetchone()
    conn.close()

    if not res:
        print("❌ Nao encontrei paciente no banco!")
        return
    
    path, ecg_id = res
    full_path = (raw_dir / path).with_suffix("").as_posix()
    
    # 2. Carregar as 12 leads (500Hz)
    signal, fields = wfdb.rdsamp(full_path)
    leads = fields['sig_name']
    
    # Config do Grid (4 linhas x 3 colunas = 12 leads)
    fig, axes = plt.subplots(4, 3, figsize=(24, 14))
    plt.subplots_adjust(hspace=0.4, wspace=0.15)
    
    colors = ["#00FFCC", "#00CCFF", "#0088FF", "#FF3366", "#FFDD00", "#00FF88"] * 2
    
    for i, (ax, lead_name) in enumerate(zip(axes.flatten(), leads)):
        data = signal[1000:2000, i] # 2 segundos
        data = (data - np.mean(data)) / (np.std(data) + 1e-8)
        
        # Grid milimetrado 'Elite'
        ax.grid(True, which='both', color='#1a1a1a', linestyle='-', lw=0.6, alpha=0.9, zorder=1)
        ax.set_xticks(np.arange(0, 1000, 100))
        ax.set_yticks(np.arange(-3, 4, 1.5))
        
        # Plot
        ax.plot(data, color=colors[i], linewidth=2.0, alpha=0.85, zorder=5)
        ax.fill_between(range(len(data)), data, y2=-3, color=colors[i], alpha=0.02, zorder=3)
        
        # Labels e Ticks
        ax.set_title(f"LEAD: {lead_name}", color=colors[i], fontsize=12, fontweight='bold', family='monospace', pad=10)
        ax.tick_params(axis='both', which='both', length=0, labelsize=0)
        for spine in ax.spines.values():
            spine.set_color('#1a1a1a')
        
    plt.suptitle(f"12-LEAD CLINICAL MONITOR | PATIENT_ID: {ecg_id:05d}_HR", color='white', 
                 fontsize=22, fontweight='bold', family='monospace', y=0.96)

    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#050505')
    print(f"✅ BANNER 12-LEADS (MESMO PACIENTE) GERADO EM: {output_path}")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    db_path = project_root / "data" / "processed" / "ptbxl.db"
    raw_dir = project_root / "data" / "raw"
    output_f = project_root / "docs" / "banner_12leads.png"
    
    generate_banner_12leads_elite(db_path, raw_dir, output_f)
