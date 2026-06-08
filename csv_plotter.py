import pandas as pd
import matplotlib.pyplot as plt
import sys
import os


def plot_csv(dateipfad: str):
    """
    Liest eine CSV-Datei ein und plottet die ersten zwei Spalten als Diagramm.

    :param dateipfad: Pfad zur CSV-Datei
    """
    # --- Datei einlesen ---
    if not os.path.exists(dateipfad):
        print(f"Fehler: Datei '{dateipfad}' wurde nicht gefunden.")
        sys.exit(1)

    try:
        # Automatische Erkennung von Trennzeichen (Komma oder Semikolon)
        df = pd.read_csv(dateipfad, sep=None, engine="python")
    except Exception as e:
        print(f"Fehler beim Einlesen der CSV-Datei: {e}")
        sys.exit(1)

    if df.shape[1] < 2:
        print("Fehler: Die CSV-Datei muss mindestens zwei Spalten enthalten.")
        sys.exit(1)

    # Erste zwei Spalten auswählen
    spalte_x = df.columns[0]
    spalte_y = df.columns[1]

    print(f"Geladene Spalten: X='{spalte_x}', Y='{spalte_y}'")
    print(f"Anzahl Zeilen: {len(df)}")

    # --- Diagramm erstellen ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Versuche numerische Y-Achse; sonst Balkendiagramm
    if pd.api.types.is_numeric_dtype(df[spalte_y]):
        ax.plot(df[spalte_x], df[spalte_y], marker="o", linewidth=2,
                markersize=5, color="#2563EB", label=spalte_y)
        ax.fill_between(df.index, df[spalte_y], alpha=0.1, color="#2563EB")
        diagramm_typ = "Liniendiagramm"
    else:
        ax.bar(df[spalte_x], df[spalte_y], color="#2563EB")
        diagramm_typ = "Balkendiagramm"

    # Beschriftungen
    ax.set_xlabel(spalte_x, fontsize=12)
    ax.set_ylabel(spalte_y, fontsize=12)
    ax.set_title(f"{diagramm_typ}: {spalte_x} vs. {spalte_y}", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Dateipfad per Kommandozeile oder als Eingabe
    if len(sys.argv) > 1:
        pfad = sys.argv[1]
    else:
        pfad = input("Bitte den Pfad zur CSV-Datei eingeben: ").strip()

    plot_csv(pfad)
