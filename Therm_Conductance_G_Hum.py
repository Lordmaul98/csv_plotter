import numpy as np
import matplotlib.pyplot as plt


def berechne_feuchte_luft_eigenschaften(T_C, p_Pa, phi=0.0, x=None):
    """
    Berechnet Dichte, absolute Feuchte und spezifische Enthalpie von feuchter Luft.
    """
    R_d = 287.05  # J/(kg*K) Spezifische Gaskonstante für trockene Luft
    R_v = 461.5  # J/(kg*K) Spezifische Gaskonstante für Wasserdampf
    T_K = T_C + 273.15

    # Magnusklimagleichung für Sättigungsdampfdruck über Wasser
    p_sat = 611.2 * np.exp((17.62 * T_C) / (243.12 + T_C))

    if x is None:
        p_v = phi * p_sat
        x = (R_d / R_v) * p_v / (p_Pa - p_v)
    else:
        p_v = (x * p_Pa) / (R_d / R_v + x)

    # Dichte der feuchten Luft (kg feuchte Luft / m³ Gesamtvolumen)
    dichte = (p_Pa / (R_d * T_K)) * (1 + x) / (1 + (R_v / R_d) * x)

    # Spezifische Enthalpie in J/kg_trockene_luft
    c_pd = 1006.0
    c_pv = 1860.0
    r_0 = 2501000.0
    h = c_pd * T_C + x * (r_0 + c_pv * T_C)

    return dichte, x, h


def berechne_counterflow_profile_luft(
    T_h_in, T_h_out, T_c_in, T_c_out, cp_h_eff, cp_c_eff, punkte=100
):
    """
    Berechnet die exponentiellen Temperaturverläufe für einen Gegenstrom-Wärmetauscher (Counter-Flow).
    x = 0 ist der Eintritt des heißen Fluids und der Austritt des kalten Fluids.
    x = 1 ist der Austritt des heißen Fluids und der Eintritt des kalten Fluids.
    """
    # Im Gegenstrom liegen die Temperaturdifferenzen an den jeweiligen Enden:
    dT_links = T_h_in - T_c_out  # bei x = 0
    dT_rechts = T_h_out - T_c_in  # bei x = 1

    if dT_links <= 0 or dT_rechts <= 0:
        raise ValueError(
            "Thermodynamischer Fehler: Unmögliche Temperaturverhältnisse für Gegenstrom."
        )

    x_nach = np.linspace(0, 1, punkte)

    # Berechnung der LMTD für Gegenstrom
    if np.isclose(dT_links, dT_rechts):
        lmtd = dT_links
        # Linearer Verlauf bei exakt gleichen Kapazitätsströmen
        T_h = T_h_in - (T_h_in - T_h_out) * x_nach
        T_c = T_c_out - (T_c_out - T_c_in) * x_nach
    else:
        lmtd = (dT_links - dT_rechts) / np.log(dT_links / dT_rechts)

        # Kapazitätsstromverhältnis R und NTU bestimmen
        R = (cp_h_eff * (T_h_in - T_h_out)) / (cp_c_eff * (T_c_out - T_c_in))

        # Lokale Temperaturverläufe über analytische Gegenstrom-Gleichung
        # dT(x) = T_h(x) - T_c(x)
        # Für Gegenstrom verhält sich der Verlauf exponentiell mit der Fläche:
        exp_term = np.exp(-np.log(dT_links / dT_rechts) * x_nach)

        T_h = T_h_out + (T_h_in - T_h_out) * (1 - exp_term) / (
            1 - (dT_rechts / dT_links)
        )
        T_c = T_c_in + (T_c_out - T_c_in) * (1 - exp_term) / (
            1 - (dT_rechts / dT_links)
        )

    return x_nach, T_h, T_c, lmtd, dT_links, dT_rechts


def berechne_kv_aus_stellung(stellung_prozent, kv_max):
    """Berechnet den Kv-Wert einer gleichprozentigen DN50 Drosselklappe"""
    if stellung_prozent <= 0:
        return 0.0
    x = stellung_prozent / 100.0
    kv = kv_max * np.exp(3.0 * (x - 1.0))
    return max(0.0, min(kv, kv_max))


# =============================================================================
# NEUE PARAMETER (GEGENSTROM & LUFT-LUFT)
# =============================================================================
p_ein_h = 101325
p_aus_h = 101100

# Neue Temperaturen aus Ihrer Vorgabe
T_feucht_ein, T_feucht_aus = 74.8, 66.0  # Heiße/feuchte Seite
T_trocken_ein, T_trocken_aus = 65.8, 67.3  # Kalte/trockene Seite

# Eintrittsvolumenstrom & Feuchten
volumenstrom_h_ein = 150.0  # m³/h
phi_warm_ein = 0.30
phi_warm_aus = 0.95

phi_kalt_ein, phi_kalt_aus = 0.40, 0.20
p_ein_c = 101325

# Bypass-Ventile (DN50)
kv_max = 90.0
stellung_wt = 100.0  # 100% durch den Wärmetauscher strömend (kein Bypass)
# =============================================================================

try:
    # 1. Kv- & Massenstromaufteilung berechnen
    kv_wt = berechne_kv_aus_stellung(stellung_wt, kv_max)
    kv_bypass = berechne_kv_aus_stellung(100.0 - stellung_wt, kv_max)
    massenstrom_faktor_wt = (
        kv_wt / (kv_wt + kv_bypass) if (kv_wt + kv_bypass) > 0 else 0.5
    )

    # 2. Zustandsgrößen der feuchten Luft bestimmen
    rho_h_in, x_h_in, h_h_in = berechne_feuchte_luft_eigenschaften(
        T_feucht_ein, p_ein_h, phi=phi_warm_ein
    )
    rho_h_out, x_h_out, h_h_out = berechne_feuchte_luft_eigenschaften(
        T_feucht_aus, p_aus_h, phi=phi_warm_aus
    )

    _, _, h_c_in = berechne_feuchte_luft_eigenschaften(
        T_trocken_ein, p_ein_c, phi=phi_kalt_ein
    )
    _, _, h_c_out = berechne_feuchte_luft_eigenschaften(
        T_trocken_aus, p_ein_c, phi=phi_kalt_aus
    )

    # Massenströme (unter Berücksichtigung des Bypasses)
    massenstrom_h_gesamt_stack = (volumenstrom_h_ein / 3600.0) * rho_h_in
    massenstrom_h_gesamt = massenstrom_h_gesamt_stack * massenstrom_faktor_wt
    massenstrom_h_trocken = massenstrom_h_gesamt / (1 + x_h_in)

    # 3. Thermische Leistung & effektive Wärmekapazitäten
    Q_punkt = massenstrom_h_trocken * (h_h_in - h_h_out)
    cp_h_eff = (h_h_in - h_h_out) / (T_feucht_ein - T_feucht_aus)
    cp_c_eff = (h_c_out - h_c_in) / (T_trocken_aus - T_trocken_ein)

    # 4. Profilberechnung im GEGENSTROM
    x, T_hot, T_cold, lmtd, dT_links, dT_rechts = (
        berechne_counterflow_profile_luft(
            T_feucht_ein,
            T_feucht_aus,
            T_trocken_ein,
            T_trocken_aus,
            cp_h_eff,
            cp_c_eff,
        )
    )
    G_wert = Q_punkt / lmtd

    print(f"--- Ergebnisse (Gegenstrom-Betrieb) ---")
    print(f"Massenstrom-Anteil zu WT:  {massenstrom_faktor_wt*100:.1f} %")
    print(f"ΔT (Links bei x=0):        {dT_links:6.2f} K")
    print(f"ΔT (Rechts bei x=1):       {dT_rechts:6.2f} K")
    print(f"LMTD (ΔT_ln):             {lmtd:6.2f} K")
    print(f"Wärmestrom (Q_punkt):     {Q_punkt:6.1f} W")
    print(f"Thermischer Leitwert G:    {G_wert:6.2f} W/K")

    # Graphische Darstellung im Gegenstrom
    plt.figure(figsize=(10, 6))

    plt.plot(x, T_hot, "r-", linewidth=2.5, label="Feuchter Kreis (Heiß)")
    plt.plot(x, T_cold, "b-", linewidth=2.5, label="Trockener Kreis (Kalt)")

    # Pfeile für Strömungsrichtungen (GEGENSTROM!)
    # Heißer Kreis strömt von links (x=0) nach rechts (x=1)
    plt.annotate(
        "",
        xy=(0.55, T_hot[55]),
        xytext=(0.45, T_hot[45]),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )
    # Kalter Kreis strömt von rechts (x=1) nach links (x=0)
    plt.annotate(
        "",
        xy=(0.45, T_cold[45]),
        xytext=(0.55, T_cold[55]),
        arrowprops=dict(arrowstyle="->", color="blue", lw=2),
    )

    # Textbox-Höhe dynamisch mitteln
    y_text_pos = (T_feucht_aus + T_trocken_aus) / 2

    plt.vlines(
        [0, 1],
        ymin=min(T_trocken_ein, T_feucht_aus) - 3,
        ymax=max(T_feucht_ein, T_trocken_aus) + 3,
        colors="gray",
        linestyles="dashed",
        alpha=0.5,
    )

    # Linke Seite (x=0) ist Eintritt Heiß / Austritt Kalt
    plt.text(
        0.02,
        y_text_pos,
        r"$\Delta T_{\mathrm{links}}$ = " + f"{dT_links:.1f} K",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7),
        va="center",
    )

    # Mittlere Infobox
    plt.text(
        0.35,
        y_text_pos,
        r"$\Delta T_{\ln}$ = " + f"{lmtd:.1f} K\n"
        r"$\dot{V}_{\mathrm{ein}}$ = " + f"{volumenstrom_h_ein:.1f} m³/h\n"
        r"$\dot{Q}$ = " + f"{Q_punkt:.1f} W\n"
        r"$G$ = " + f"{G_wert:.1f} W/K",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7),
        va="center",
        ha="left",
    )

    # Rechte Seite (x=1) ist Austritt Heiß / Eintritt Kalt
    plt.text(
        0.82,
        y_text_pos,
        r"$\Delta T_{\mathrm{rechts}}$ = " + f"{dT_rechts:.1f} K",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7),
        va="center",
    )

    plt.title(
        f"Gegenstrom-Temperaturprofil (Counter-Flow)\nThermischer Leitwert G = {G_wert:.2f} W/K",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("Normierte Position / Fläche des Wärmetauschers", fontsize=10)
    plt.ylabel("Temperatur [$^\\circ$C]", fontsize=10)
    plt.xticks(
        [0, 1],
        [
            "Eintritt Feucht (heiß)\nAustritt Trocken (warm)",
            "Austritt Feucht (warm)\nEintritt Trocken (kalt)",
        ],
    )
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.xlim(-0.05, 1.05)
    plt.ylim(
        min(T_trocken_ein, T_feucht_aus) - 3,
        max(T_feucht_ein, T_trocken_aus) + 3,
    )

    plt.tight_layout()
    plt.show()

except ValueError as e:
    print(f"Fehler: {e}")