import numpy as np
import matplotlib.pyplot as plt


def berechne_coflow_profile(T_h_in, T_h_out, T_c_in, T_c_out, punkte=100):
    """
    Berechnet die physikalisch exakten, exponentiellen Temperaturverläufe
    für einen Gleichstrom-Wärmetauscher (Co-Flow) über die normierte Fläche x [0, 1].
    """
    # Im Gleichstrom gilt: Maximale Differenz am Eintritt, minimale am Austritt
    dT_max = T_h_in - T_c_in  # Links (Eintritt beider Fluide bei x = 0)
    dT_min = T_h_out - T_c_out  # Rechts (Austritt beider Fluide bei x = 1)

    if dT_max <= 0 or dT_min <= 0:
        raise ValueError("Thermodynamischer Fehler: Unmögliche Temperaturverhältnisse für Gleichstrom.")

    x = np.linspace(0, 1, punkte)

    # Berechnung der Logarithmic Mean Temperature Difference (LMTD) für Gleichstrom
    if np.isclose(dT_max, dT_min):
        lmtd = dT_max
        T_h = np.full_like(x, T_h_in)
        T_c = np.full_like(x, T_c_in)
    else:
        lmtd = (dT_max - dT_min) / np.log(dT_max / dT_min)

        dT_h = T_h_in - T_h_out
        dT_c = T_c_out - T_c_in

        # Abfallfaktor kappa für die exponentielle Abnahme der Temperaturdifferenz entlang des Wärmetauschers
        Abfallfaktor_kappa = np.log(dT_max / dT_min)

        # Lokaler Abfall der treibenden Temperaturdifferenz: dT(x) = T_h(x) - T_c(x)
        dT_x = dT_max * np.exp(-Abfallfaktor_kappa * x)

        gesamt_delta = dT_h + dT_c

        T_h = T_h_in - (dT_h / gesamt_delta) * (dT_max - dT_x)
        T_c = T_c_in + (dT_c / gesamt_delta) * (dT_max - dT_x)

    return x, T_h, T_c, lmtd, dT_max, dT_min


# =============================================================================
# EINSTELLBARE PARAMETER (Betriebspunkte für CO-FLOW)
# =============================================================================
# Temperaturen in °C
T_warm_ein = 70.0
T_warm_aus = 70.0
T_kalt_ein = 25.3
T_kalt_aus = 62.5

# Strömungs- und Stoffdaten (Beispielhaft für die heiße Seite)
volumenstrom_luft = 4086  # in dm³/min
dichte_luft = 1.17  # in kg/m³ (Luft bei ~25°C)
cp_luft = 1006.7  # in J/(kg*K) (spezifische Wärmekapazität)
# =============================================================================

try:
    # 1. Thermodynamische Profilberechnung
    x, T_hot, T_cold, lmtd, dT_max, dT_min = berechne_coflow_profile(T_warm_ein, T_warm_aus, T_kalt_ein, T_kalt_aus)

    # 2. Berechnung von Massenstrom, Wärmestrom und thermischem Leitwert G
    massenstrom_luft = (volumenstrom_luft / 60000.0) * dichte_luft  # Umrechnung dm³/min -> kg/s
    Q_punkt = massenstrom_luft * cp_luft * (T_kalt_aus - T_kalt_ein)  # Wärmeleistung in W
    G_wert = Q_punkt / lmtd  # Leitwert in W/K

    # Konsolenausgabe
    print(f"--- Berechnungsergebnisse (Co-Flow) ---")
    print(f"ΔT_max (Eintritt Links):  {dT_max:6.2f} K")
    print(f"ΔT_min (Austritt Rechts): {dT_min:6.2f} K")
    print(f"LMTD (ΔT_ln):             {lmtd:6.2f} K")
    print(f"Massenstrom (Luft):       {massenstrom_luft:6.2f} kg/s")
    print(f"Wärmestrom (Q_punkt):     {Q_punkt:6.2f} W")
    print(f"Thermischer Leitwert G:   {G_wert:6.2f} W/K")

    # Graphische Darstellung
    plt.figure(figsize=(10, 6))

    plt.plot(x, T_hot, 'r-', linewidth=2.5, label='Heißer Kreis (Fluid)')
    plt.plot(x, T_cold, 'b-', linewidth=2.5, label='Kalter Kreis (Gas)')

    # Pfeile für Strömungsrichtung
    plt.annotate('', xy=(0.55, T_hot[55]), xytext=(0.45, T_hot[45]),
                 arrowprops=dict(arrowstyle="->", color='red', lw=2))
    plt.annotate('', xy=(0.55, T_cold[55]), xytext=(0.45, T_cold[45]),
                 arrowprops=dict(arrowstyle="->", color='blue', lw=2))

    # Dynamische, aber horizontal exakt gleiche Höhe für alle Texte
    y_text_pos = (T_warm_aus + T_kalt_aus) / 2

    # Hilfslinien und Beschriftungen
    plt.vlines([0, 1], ymin=min(T_kalt_ein, T_warm_aus) - 5, ymax=max(T_warm_ein, T_kalt_aus) + 5, colors='gray',
               linestyles='dashed', alpha=0.5)

    # Trennung von LaTeX und f-String-Variablen zur Vermeidung von SyntaxErrors
    plt.text(0.02, y_text_pos, r'$\Delta T_{\max}$ = ' + f'{dT_max:.1f} K', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.7), va='center')
    plt.text(0.40, y_text_pos,
             r'$\Delta T_{\ln}$ = ' + f'{lmtd:.1f} K\n' +
             r'$\dot{V}$ = ' + f'{volumenstrom_luft:.1f} dm³/min\n' +
             r'$\dot{Q}$ = ' + f'{Q_punkt:.1f} W\n' +
             r'$G$ = ' + f'{G_wert:.1f} W/K',
             fontsize=10, bbox=dict(facecolor='white', alpha=0.7), va='center', ha='left')
    plt.text(0.9, y_text_pos, r'$\Delta T_{\min}$ = ' + f'{dT_min:.1f} K', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.7), va='center')

    # Titel und Achsen
    plt.title(f'Exponentielles Temperaturprofil im Gleichstrom (Co-Flow)\nLeitwert $G$ = {G_wert:.2f} W/K', fontsize=12,
              fontweight='bold')
    plt.xlabel('Normierte Position / Fläche des Wärmetauschers', fontsize=10)
    plt.ylabel('Temperatur [$^\\circ$C]', fontsize=10)
    plt.xticks([0, 1], ['Gemeinsamer Eintritt\n(ΔT_max)', 'Gemeinsamer Austritt\n(ΔT_min)'])
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    plt.xlim(-0.05, 1.05)
    plt.ylim(min(T_kalt_ein, T_warm_aus) - 5, max(T_warm_ein, T_kalt_aus) + 5)

    plt.tight_layout()
    plt.show()

except ValueError as e:
    print(f"Fehler: {e}")