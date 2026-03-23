import os
import sqlite3
import wfdb
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def get_record_paths(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT r.filename_hr FROM ecg_records r JOIN ecg_scp_diagnoses map ON r.ecg_id = map.ecg_id JOIN scp_codes d ON map.scp_code = d.scp_code WHERE d.diagnostic_class = 'MI' LIMIT 1")
    mi_p = cur.fetchone()[0]
    cur.execute("SELECT r.filename_hr FROM ecg_records r JOIN ecg_scp_diagnoses map ON r.ecg_id = map.ecg_id JOIN scp_codes d ON map.scp_code = d.scp_code WHERE d.diagnostic_class = 'NORM' LIMIT 1")
    norm_p = cur.fetchone()[0]
    conn.close()
    return mi_p, norm_p

def generate_comparison_elite(mi_path, norm_path, raw_dir, output_path):
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    plt.subplots_adjust(wspace=0.15)

    LEAD_IDX = 7 # Lead V2
    COLORS = ["#00FFCC", "#FF3366"] 
    LABELS = ["CONTROL: NORMAL SINUS RHYTHM", "DIAGNOSIS: MYOCARDIAL INFARCTION"]

    for ax, path, label, color in zip([ax1, ax2], [norm_path, mi_path], LABELS, COLORS):
        full_path = (raw_dir / path).with_suffix("").as_posix()
        signal, fields = wfdb.rdsamp(full_path)
        
        # Janela de sinal (1.5 segundos)
        data = signal[1000:2250, LEAD_IDX]
        data = (data - np.mean(data)) / (np.std(data) + 1e-8)
        
        # --- ESTÉTICA ELITE DO BANNER ---
        # Grid milimetrado real
        ax.grid(True, which='both', color='#222222', linestyle='-', lw=0.8, alpha=0.9, zorder=1)
        ax.set_xticks(np.arange(0, 1250, 50))
        ax.set_yticks(np.arange(-3, 4, 1))

        # Detalhes nos eixos (Ticks brancos pequenos conforme solicitado)
        ax.tick_params(axis='both', which='major', color='#444444', labelsize=0, length=5, width=1.5)
        
        # Plot Principal com Profundidade
        ax.plot(data, color=color, linewidth=2.8, alpha=0.9, zorder=5)
        ax.fill_between(range(len(data)), data, y2=-3, color=color, alpha=0.03, zorder=3)
        
        # Spines discretos (estilo painel)
        for spine in ax.spines.values():
            spine.set_color('#1a1a1a')
            spine.set_linewidth(1.5)
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Título Elite
        ax.set_title(label, color=color, fontsize=13, fontweight='bold', pad=20, family='monospace')
        ax.set_xlim(0, 1250)
        ax.set_ylim(-3, 4)

    plt.savefig(output_path, dpi=250, bbox_inches='tight', facecolor='#050505')
    print(f"\n🚀 COMPARATIVO ELITE (Estética Banner) GERADO EM: {output_path}")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    db_path = project_root / "data" / "processed" / "ptbxl.db"
    raw_dir = project_root / "data" / "raw"
    output_f = project_root / "docs" / "mi_vs_normal_trace.png"
    
    os.makedirs(output_f.parent, exist_ok=True)
    mi, norm = get_record_paths(db_path)
    generate_comparison_elite(mi, norm, raw_dir, output_f)
