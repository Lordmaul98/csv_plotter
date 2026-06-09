import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys

# -------------------------------------------------------
# Dateipfad hier anpassen:
DATEIPFAD = r"J:\ECS\02_Projekte\01_Projektentwicklung\2024-01_ÖgP_GenSYS\05_Daten\KW_Mischventil\2026_06_03_Daten\AufzeichnungSystem\ASI_System_20260603_093921_035_0.dat"
# -------------------------------------------------------


def plot_dat(dateipfad: str):
    if not os.path.exists(dateipfad):
        print(f"Fehler: Datei '{dateipfad}' nicht gefunden.")
        sys.exit(1)

    # DAT einlesen: 4 Kopfzeilen überspringen, Tab-getrennt
    df = pd.read_csv(dateipfad, encoding="cp1252", skiprows=4, sep="\t")

    # Zeitstempel parsen
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Gewünschte Spalten
    col_tc20 = "TC20_T.HF.i.CL.AI [°C]"
    col_tc21 = "TC21_T.HF.o.CL.AI [°C]"
    col_p_i  = "p_HF_i_CL_AI [bara]"
    col_p_o  = "p_HF_o_CL_AI [bara]"

    for col in [col_tc20, col_tc21, col_p_i, col_p_o]:
        if col not in df.columns:
            print(f"Fehler: Spalte '{col}' nicht gefunden.")
            sys.exit(1)

    # --- Diagramm ---
    fig, ax1 = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor("#d0d0d0")
    ax1.set_facecolor("#d0d0d0")

    # Y-Achse links: Temperaturen
    l1, = ax1.plot(df["Date"], df[col_tc20],
                   color="#1a1aff", linewidth=1.0, label="TC20 T.HF.i.CL – Eingang [°C]")
    l2, = ax1.plot(df["Date"], df[col_tc21],
                   color="#ff9800", linewidth=1.0, label="TC21 T.HF.o.CL – Ausgang [°C]")
    ax1.set_ylabel("Temperatur [°C]", fontsize=10)
    ax1.tick_params(axis="y")

    # Y-Achse rechts: Drücke
    ax2 = ax1.twinx()
    l3, = ax2.plot(df["Date"], df[col_p_i],
                   color="#00bcd4", linewidth=1.0, label="p_HF_i_CL – Eingang [bara]")
    l4, = ax2.plot(df["Date"], df[col_p_o],
                   color="#e91e63", linewidth=1.0, label="p_HF_o_CL – Ausgang [bara]")
    ax2.set_ylabel("Druck [bara]", fontsize=10)

    # X-Achse
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 15)))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)

    # Legende
    lines = [l1, l2, l3, l4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=9, framealpha=0.8)

    ax1.grid(True, linestyle="-", color="white", alpha=0.4, linewidth=0.5)
    plt.title("ASI System – HF Kühlkreis", fontsize=12, pad=10)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_dat(DATEIPFAD)