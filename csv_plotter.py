import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys

# -------------------------------------------------------
# Dateipfad hier anpassen:
DATEIPFAD = r"J:\ECS\02_Projekte\01_Projektentwicklung\2024-01_ÖgP_GenSYS\05_Daten\KW_Mischventil\2026_06_03_Daten\Tst1_0_100_Auf_06.CSV"
# -------------------------------------------------------


def plot_csv(dateipfad: str):
    if not os.path.exists(dateipfad):
        print(f"Fehler: Datei '{dateipfad}' nicht gefunden.")
        sys.exit(1)

    # CSV einlesen: 10 Kopfzeilen überspringen, Encoding Windows-1252
    df = pd.read_csv(dateipfad, encoding="cp1252", skiprows=10, header=0, sep=",")

    # Spalten benennen
    df.columns = ["_leer", "Nr", "Datum", "Zeit", "_leer2",
                  "Durchfluss", "Differenzdruck", "_leer3", "_leer4", "Temperatur"]

    # Komma als Dezimalzeichen ersetzen und in Zahlen umwandeln
    for col in ["Durchfluss", "Differenzdruck", "Temperatur"]:
        df[col] = df[col].astype(str).str.replace(",", ".").str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Zeitstempel zusammenbauen
    df["Zeitstempel"] = pd.to_datetime(df["Datum"] + " " + df["Zeit"],
                                        format="%d/%m/%y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["Zeitstempel"])

    # --- Diagramm (3 Kurven, 3 Y-Achsen wie Vorlage) ---
    fig, ax1 = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor("#d0d0d0")
    ax1.set_facecolor("#d0d0d0")

    # Y-Achse 1: Durchfluss (links)
    l1, = ax1.plot(df["Zeitstempel"], df["Durchfluss"],
                   color="#1a1aff", linewidth=0.8, label="Durchfluss")
    ax1.set_ylabel("Durchfluss [l/h]", color="#1a1aff", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="#1a1aff")
    ax1.set_ylim(df["Durchfluss"].min() - 100, df["Durchfluss"].max() + 200)

    # Y-Achse 2: Differenzdruck (links, zweite Achse)
    ax2 = ax1.twinx()
    ax2.spines["left"].set_position(("outward", 60))
    ax2.yaxis.set_label_position("left")
    ax2.yaxis.set_ticks_position("left")
    l2, = ax2.plot(df["Zeitstempel"], df["Differenzdruck"],
                   color="#00bcd4", linewidth=0.8, label="Differenzdruck")
    ax2.set_ylabel("Differenzdruck [kPa]", color="#00bcd4", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="#00bcd4")

    # Y-Achse 3: Temperatur (rechts)
    ax3 = ax1.twinx()
    l3, = ax3.plot(df["Zeitstempel"], df["Temperatur"],
                   color="#ff9800", linewidth=0.8, label="Temp. (DpF-T2)")
    ax3.set_ylabel("Temperatur [°C]", color="#ff9800", fontsize=10)
    ax3.tick_params(axis="y", labelcolor="#ff9800")
    ax3.set_ylim(df["Temperatur"].min() - 0.2, df["Temperatur"].max() + 0.2)

    # X-Achse Zeitformat
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 15)))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)

    # Legende
    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9,
               framealpha=0.8)

    ax1.grid(True, linestyle="-", color="white", alpha=0.4, linewidth=0.5)
    plt.title("Aufzeichnen 06", fontsize=12, pad=10)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_csv(DATEIPFAD)