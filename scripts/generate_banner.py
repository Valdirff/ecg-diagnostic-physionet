import os
import sqlite3
import wfdb
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def generate_banner():
    project_root = Path(__file__).parent.parent
    db_path = project_root / "data" / "processed" / "ptbxl.db"
    raw_dir = project_root / "data" / "raw"

    # 1. Puxar 6 record IDs diferentes via SQL
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT filename_hr FROM ecg_records ORDER BY RANDOM() LIMIT 6")
    records = [r[0] for r in cur.fetchall()]
    conn.close()

    # 2. Configurar FIGURA ELITE (3x2)
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 2, figsize=(20, 10))
    plt.subplots_adjust(hspace=0.5, wspace=0.15)
    
    colors = ["#00FFCC", "#00F5FF", "#00E2FF", "#00BFFF", "#008CFF", "#0040FF"]

    for i, ax in enumerate(axes.flatten()):
        record_path = (raw_dir / records[i]).with_suffix("")
        
        # Ler Lead II
        signal, fields = wfdb.rdsamp(str(record_path), channels=[1])
        data = signal[500:2500, 0] # Janela maior de 4 segundos
        
        # Normalização balanceada
        data = (data - np.mean(data)) / (np.std(data) + 1e-8)
        
        # --- ESTÉTICA ELITE: O GRID MILIMETRADO ---
        # Grade maior (0.5s) e menor (0.1s)
        ax.grid(True, which='both', color='#222222', linestyle='-', alpha=0.9, linewidth=0.6)
        ax.set_xticks(np.arange(0, 2000, 50))  # Milimetrado sutil
        ax.set_yticks(np.arange(-3, 4, 1))

        # Plot com efeito de "brilho" (Glow)
        ax.plot(data, color=colors[i], linewidth=2.5, alpha=0.9, zorder=5)
        # Sombra sutil abaixo da linha para profundidade
        ax.fill_between(range(len(data)), data, y2=-3, color=colors[i], alpha=0.03, zorder=3)

        # Labels e Estética
        patient_name = Path(records[i]).stem
        ax.set_title(f"PATIENT_ID: {patient_name.upper()}  |  DIAGNOSTIC SIGNAL: LEAD II", 
                    color=colors[i], fontsize=10, fontweight='bold', pad=10, loc='left', family='monospace')
        
        # Limpar desnecessários
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim(0, 2000)
        ax.set_ylim(-3, 4)
        for spine in ax.spines.values():
            spine.set_color('#1a1a1a') # Bordas discretas de painel

    # Acabamento Final
    output_path = project_root / "docs" / "banner_12leads.png"
    plt.savefig(output_path, dpi=250, facecolor='#050505', bbox_inches='tight')
    print(f"\n🚀 ELITE BANNER GERADO EM: {output_path}")

if __name__ == "__main__":
    generate_banner()
