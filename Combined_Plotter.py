import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import os
import glob

# ================================================================
# KONFIGURATION – hier anpassen
# ================================================================

CSV_DATEI  = r"J:\ECS\02_Projekte\01_Projektentwicklung\2024-01_ÖgP_GenSYS\05_Daten\KW_Mischventil\2026_06_03_Daten\Tst1_0_100_Auf_06.CSV"
DAT_ORDNER = r"J:\ECS\02_Projekte\01_Projektentwicklung\2024-01_ÖgP_GenSYS\05_Daten\KW_Mischventil\2026_06_03_Daten\AufzeichnungSystem"

# Zeitbereich (Lokalzeit) – None = automatisch aus Daten
X_START = "2026-06-03 09:30:00"
X_ENDE  = "2026-06-03 12:00:00"

# Zeitversatz DAT-Segmente in Minuten (positiv = nach vorne, negativ = nach hinten)
DAT_OFFSET_MIN = 1.7
# Höhe der Mittelwert-Blöcke im Diagramm (0 = direkt über höchster Linie)
BLOCK_OFFSET = 0.4 #0.05 = 5% der Y-Achsenhöhe darüber
# Individuelle Y-Offsets pro Segment (überschreibt BLOCK_OFFSET für dieses Segment)
# Index 0 = Segment 1, Index 1 = Segment 2, usw. – None = BLOCK_OFFSET verwenden
BLOCK_OFFSET_SEG = {
    0:  None,    #0:  0.7,   # Segment 1 bei 70% der Y-Achse
    1:  None,
    2:  None,
    3:  None,
    4:  None,
    5:  0.52,
    6:  None,
    7:  None,
    8:  None,
    9:  None,
    10: None,
    11: None,
    12: None,
}
# ================================================================
# KURVENFARBEN
# ================================================================
FARBE_DURCHFLUSS = "#1565C0"
FARBE_DP_CSV     = "#00838F"
FARBE_TEMP_CSV   = "#E65100"
FARBE_TC20       = "#AD1457"
FARBE_TC21       = "#6A1B9A"
FARBE_P_I        = "#1B5E20"
FARBE_P_O        = "#F57F17"
FARBE_DP_DAT     = "#BF360C"
FARBE_KV         = "#880E4F"
FARBE_FR         = "#2E7D32"

# ================================================================
# KV-BERECHNUNG  kv = Q [m³/h] / sqrt(|Δp| [bar] · ρ_rel)
# ================================================================

def dichte_wasser(T):
    return 999.842 - 0.0624 * T - 0.00366 * T**2

def berechne_kv(Q_lh, dp_bar, T_celsius):
    Q   = np.asarray(Q_lh, dtype=float) / 1000.0        # l/h → m³/h
    rho = dichte_wasser(np.asarray(T_celsius, dtype=float))  # kg/m³
    dp  = np.abs(np.asarray(dp_bar, dtype=float))        # bar
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(dp > 0.001, Q * np.sqrt(rho / (dp * 1000.0)), np.nan)

# ================================================================

def lade_csv(pfad):
    print(f"  Lade CSV: {os.path.basename(pfad)}")
    df = pd.read_csv(pfad, encoding="cp1252", skiprows=10, header=0, sep=",")
    df.columns = ["_a","Nr","Datum","Zeit","_b","Durchfluss","Differenzdruck","_c","_d","Temperatur"]
    for c in ["Durchfluss","Differenzdruck","Temperatur"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",",".").str.strip(), errors="coerce")
    df = df.assign(
        Differenzdruck_bar = df["Differenzdruck"] / 100.0,
        Zeitstempel = pd.to_datetime(df["Datum"] + " " + df["Zeit"],
                                     format="%d/%m/%y %H:%M:%S", errors="coerce")
    )
    df = df.dropna(subset=["Zeitstempel"]).sort_values("Zeitstempel").reset_index(drop=True)
    print(f"    → {len(df)} Zeilen | {df['Zeitstempel'].min()} – {df['Zeitstempel'].max()}")
    return df


def lade_dat_segmente(ordner):
    COLS = ["Date","TC20_T.HF.i.CL.AI [°C]","TC21_T.HF.o.CL.AI [°C]",
            "p_HF_i_CL_AI [bara]","p_HF_o_CL_AI [bara]","Fr_Mix_CL_AI [%]"]
    dateien = sorted(glob.glob(os.path.join(ordner, "*.dat")))
    if not dateien:
        print(f"  WARNUNG: Keine .dat-Dateien in '{ordner}'!"); return []
    print(f"  {len(dateien)} DAT-Datei(en) gefunden:")
    segmente = []
    for pfad in dateien:
        print(f"    Lade: {os.path.basename(pfad)}")
        try:
            alle = pd.read_csv(pfad, encoding="cp1252", skiprows=4, sep="\t", nrows=0).columns.tolist()
            vorh = [c for c in COLS if c in alle]
            df   = pd.read_csv(pfad, encoding="cp1252", skiprows=4, sep="\t", usecols=vorh)
            df   = df.assign(
                Date      = pd.to_datetime(df["Date"], errors="coerce") + pd.Timedelta(minutes=DAT_OFFSET_MIN),
                dp_HF_bar = np.abs(df["p_HF_o_CL_AI [bara]"] - df["p_HF_i_CL_AI [bara]"])
            )
            df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
            print(f"      → {len(df)} Zeilen | {df['Date'].min()} – {df['Date'].max()}")
            segmente.append(df)
        except Exception as e:
            print(f"      FEHLER: {e}")
    segmente.sort(key=lambda d: d["Date"].iloc[0])
    return segmente


def plot_kombiniert(csv_pfad, dat_ordner):
    print("\n=== Daten laden ===")
    df_csv   = lade_csv(csv_pfad)
    segmente = lade_dat_segmente(dat_ordner)

    x0 = pd.Timestamp(X_START) if X_START else df_csv["Zeitstempel"].min()
    x1 = pd.Timestamp(X_ENDE)  if X_ENDE  else df_csv["Zeitstempel"].max()

    segs = [s for s in segmente if s["Date"].iloc[0] <= x1 and s["Date"].iloc[-1] >= x0]
    print(f"  {len(segs)} von {len(segmente)} Segmenten im Anzeigebereich")

    # ── Kv pro Segment punktweise berechnen ──────────────────────────────────
    print("\n=== Kv-Wert pro Segment ===")
    # ── Temperaturstatistik und Kv-Fehleranalyse ─────────────────────────────
    T_csv = df_csv["Temperatur"].dropna()
    T_min = T_csv.min()
    T_max = T_csv.max()
    T_mean = T_csv.mean()
    T_start = df_csv["Temperatur"].dropna().iloc[0]

    print("\n=== Temperaturstatistik (Temp. CSV) ===")
    print(f"  Start:       {T_start:.2f} °C")
    print(f"  Minimum:     {T_min:.2f} °C")
    print(f"  Maximum:     {T_max:.2f} °C")
    print(f"  Durchschnitt:{T_mean:.2f} °C")

    # Kv-Fehler durch Temperaturannahme: kv(T_min/max) vs kv(T_mean)
    # Testpunkt: mittlerer Durchfluss und mittlerer Differenzdruck
    Q_test = df_csv["Durchfluss"].mean()
    dp_test = np.nanmean([s["dp_HF_bar"].mean() for s in segs]) if segs else 3.8

    kv_ref = berechne_kv(Q_test, dp_test, T_mean)
    kv_Tmin = berechne_kv(Q_test, dp_test, T_min)
    kv_Tmax = berechne_kv(Q_test, dp_test, T_max)

    err_min = abs(kv_Tmin - kv_ref) / kv_ref * 100
    err_max = abs(kv_Tmax - kv_ref) / kv_ref * 100

    print(f"\n=== Kv-Fehler durch Temperaturannahme (Q={Q_test:.0f} l/h, dp={dp_test:.3f} bar) ===")
    print(f"  kv bei T_mean ({T_mean:.2f}°C): {kv_ref:.4f} m³/h  ← Referenz")
    print(f"  kv bei T_min  ({T_min:.2f}°C): {kv_Tmin:.4f} m³/h  → Fehler: {err_min:.4f}%")
    print(f"  kv bei T_max  ({T_max:.2f}°C): {kv_Tmax:.4f} m³/h  → Fehler: {err_max:.4f}%")
    print(f"  ► Maximaler Fehler durch Temperaturannahme: {max(err_min, err_max):.4f}%")

    kv_mittelwerte = []
    for i, seg in enumerate(segs):
        mask   = ((df_csv["Zeitstempel"] >= seg["Date"].iloc[0]) &
                  (df_csv["Zeitstempel"] <= seg["Date"].iloc[-1]))
        df_seg = df_csv.loc[mask].copy()

        if len(df_seg) >= 2:
            t_csv    = df_seg["Zeitstempel"].astype(np.int64).values
            t_dat    = seg["Date"].astype(np.int64).values
            Q_interp = np.interp(t_dat, t_csv, df_seg["Durchfluss"].values)
            T_interp = np.interp(t_dat, t_csv, df_seg["Temperatur"].values)
        else:
            Q_interp = np.full(len(seg), df_csv["Durchfluss"].mean())
            T_interp = np.full(len(seg), df_csv["Temperatur"].mean())

        kv = berechne_kv(Q_interp, seg["dp_HF_bar"].values, T_interp)
        segs[i] = seg.assign(Kv=kv)
        kv_mittelwerte.append((i, np.nanmean(kv)))
        print(f"  Segment {i+1:2d} "
              f"({seg['Date'].iloc[0].strftime('%H:%M')}–{seg['Date'].iloc[-1].strftime('%H:%M')}): "
              f"Q̄={Q_interp.mean():.0f} l/h | dp̄={seg['dp_HF_bar'].mean():.3f} bar | "
              f"T̄={T_interp.mean():.1f}°C | kv̄={np.nanmean(kv):.4f} m³/h")

    # ── Diagramm aufbauen ─────────────────────────────────────────────────────
    print("\n=== Diagramm erstellen ===")
    fig, ax1 = plt.subplots(figsize=(20, 7))
    fig.patch.set_facecolor("#d0d0d0")
    ax1.set_facecolor("#d0d0d0")

    # Y1 links: Durchfluss [l/h]
    ax1.set_ylabel("Durchfluss [l/h]", color=FARBE_DURCHFLUSS, fontsize=10)
    ax1.tick_params(axis="y", labelcolor=FARBE_DURCHFLUSS)
    l_Q, = ax1.plot(df_csv["Zeitstempel"], df_csv["Durchfluss"],
                    color=FARBE_DURCHFLUSS, lw=0.9, label="Durchfluss [l/h]", zorder=3)

    # Y2 links versetzt: Druck [bar / bara]
    ax2 = ax1.twinx()
    ax2.spines["left"].set_position(("outward", 72))
    ax2.yaxis.set_label_position("left")
    ax2.yaxis.set_ticks_position("left")
    ax2.set_ylabel("Druck [bar / bara]", color=FARBE_P_I, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=FARBE_P_I)

    l_dp_csv, = ax2.plot(df_csv["Zeitstempel"], df_csv["Differenzdruck_bar"],
                         color=FARBE_DP_CSV, lw=0.9, label="Differenzdruck CSV [bar]", zorder=3)
    l_pi = l_po = l_dp_dat = None
    for seg in segs:
        if "p_HF_i_CL_AI [bara]" in seg.columns:
            li, = ax2.plot(seg["Date"], seg["p_HF_i_CL_AI [bara]"],
                           color=FARBE_P_I, lw=1.0, ls="--", zorder=4,
                           label="p_HF_i Eingang [bara]" if l_pi is None else "_")
            if l_pi is None: l_pi = li
        if "p_HF_o_CL_AI [bara]" in seg.columns:
            lo, = ax2.plot(seg["Date"], seg["p_HF_o_CL_AI [bara]"],
                           color=FARBE_P_O, lw=1.0, ls="--", zorder=4,
                           label="p_HF_o Ausgang [bara]" if l_po is None else "_")
            if l_po is None: l_po = lo
        if "dp_HF_bar" in seg.columns:
            ld, = ax2.plot(seg["Date"], seg["dp_HF_bar"],
                           color=FARBE_DP_DAT, lw=1.0, ls=":", zorder=4,
                           label="|Δp| HF [bar]" if l_dp_dat is None else "_")
            if l_dp_dat is None: l_dp_dat = ld

    # Nulllinien ausrichten
    y1lo = df_csv["Durchfluss"].min(); y1hi = df_csv["Durchfluss"].max(); y1r = y1hi - y1lo
    ax1.set_ylim(y1lo - y1r * 0.05, y1hi + y1r * 0.08)
    y2v = list(df_csv["Differenzdruck_bar"].dropna())
    for s in segs:
        for c in ["p_HF_i_CL_AI [bara]","p_HF_o_CL_AI [bara]","dp_HF_bar"]:
            if c in s.columns: y2v.extend(s[c].dropna())
    y2lo, y2hi = min(y2v), max(y2v); y2r = y2hi - y2lo
    if y1r > 0 and y2r > 0:
        sc = y1r / y2r
        ax2.set_ylim(y2lo - y1lo / sc, y2lo - y1lo / sc + (y1hi + y1r * 0.08 - y1lo) / sc)

    # Y3 – Temperatur ausgeblendet (wird nur für Kv-Berechnung verwendet)
    l_Tcsv = l_tc20 = l_tc21 = None

    # Y4 rechts außen: Kv [m³/h]
    ax4 = ax1.twinx()
    ax4.spines["right"].set_position(("outward", 60))
    ax4.set_ylabel("Kv-Wert [m³/h]", color=FARBE_KV, fontsize=10)
    ax4.tick_params(axis="y", labelcolor=FARBE_KV)
    l_kv = None
    for seg in segs:
        if "Kv" in seg.columns:
            lk, = ax4.plot(seg["Date"], seg["Kv"],
                           color=FARBE_KV, lw=1.4, ls="-", zorder=5,
                           label="Kv-Wert [m³/h]" if l_kv is None else "_")
            if l_kv is None: l_kv = lk

        # Gemeinsamer Mittelwert-Block pro Segment
        y_bottom, y_top = ax1.get_ylim()
        y_range = y_top - y_bottom

        for i, seg in enumerate(segs):
            mask = ((df_csv["Zeitstempel"] >= seg["Date"].iloc[0]) &
                    (df_csv["Zeitstempel"] <= seg["Date"].iloc[-1]))
            Q_seg = df_csv.loc[mask, "Durchfluss"]
            Q_mean = Q_seg.mean() if len(Q_seg) > 0 else float("nan")
            dp_mean = seg["dp_HF_bar"].mean() if "dp_HF_bar" in seg.columns else float("nan")
            _, kv_mean = kv_mittelwerte[i] if i < len(kv_mittelwerte) else (i, float("nan"))
            fr_mean = seg["Fr_Mix_CL_AI [%]"].mean() if "Fr_Mix_CL_AI [%]" in seg.columns else float("nan")

            eintraege = []
            if not np.isnan(Q_mean):  eintraege.append((f"Q̄={Q_mean:.0f} l/h", FARBE_DURCHFLUSS))
            if not np.isnan(dp_mean): eintraege.append((f"dp̄={dp_mean:.3f} bar", FARBE_DP_DAT))
            if not np.isnan(kv_mean): eintraege.append((f"kv̄={kv_mean:.3f} m³/h", FARBE_KV))
            if not np.isnan(fr_mean): eintraege.append((f"Fr̄={fr_mean:.1f}%", FARBE_FR))

            # Offset bestimmen: individuell falls gesetzt, sonst global
            seg_offset = BLOCK_OFFSET_SEG.get(i, None)
            offset_wert = seg_offset if seg_offset is not None else BLOCK_OFFSET

            # Position: 0.0 = unterer Rand, 1.0 = oberer Rand
            y_anker_seg = y_bottom + y_range * np.clip(offset_wert, 0.0, 0.98)

            t_mid = seg["Date"].iloc[0] + (seg["Date"].iloc[-1] - seg["Date"].iloc[0]) / 2

            for j, (label, farbe) in enumerate(eintraege):
                ax1.text(t_mid, y_anker_seg + (len(eintraege) - j - 1) * y_range * 0.036,
                         label, ha="center", va="bottom", fontsize=6.5,
                         color=farbe, fontweight="normal", zorder=7,
                         )
    # Y5 rechts weiter außen: Öffnungswinkel [%]
    ax5 = ax1.twinx()
    ax5.spines["right"].set_position(("outward", 125))
    ax5.set_ylabel("Schlie0ungsswinkel [%]", color=FARBE_FR, fontsize=10)
    ax5.tick_params(axis="y", labelcolor=FARBE_FR)
    ax5.set_ylim(-5, 110)
    lfr = None
    for seg in segs:
        if "Fr_Mix_CL_AI [%]" in seg.columns:
            lf, = ax5.plot(seg["Date"], seg["Fr_Mix_CL_AI [%]"],
                           color=FARBE_FR, lw=1.2, ls="-.", zorder=5,
                           label="Fr_Mix_CL [%]" if lfr is None else "_")
            if lfr is None: lfr = lf

    # X-Achse
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 15)))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
    ax1.set_xlabel("Uhrzeit (Lokalzeit)", fontsize=10)
    ax1.set_xlim(x0, x1)

    # Segment-Markierungen
    _, y1hi_plot = ax1.get_ylim()
    for i, seg in enumerate(segs):
        ax1.axvspan(seg["Date"].iloc[0], seg["Date"].iloc[-1],
                    alpha=0.10, color="#ffffff", zorder=1)
        t_mid = seg["Date"].iloc[0] + (seg["Date"].iloc[-1] - seg["Date"].iloc[0]) / 2
        ax1.annotate(f"{i+1}", xy=(t_mid, y1hi_plot * 0.98),
                     ha="center", va="top", fontsize=7, color="#333333", zorder=5)

    # Legende
    legend_lines = []
    for l in [l_Q, l_dp_csv, l_pi, l_po, l_dp_dat, l_kv, lfr]:
        if l is not None: legend_lines.append(l)
    legend_lines.append(mpatches.Patch(facecolor="white", alpha=0.5, edgecolor="#555",
                                        label=f"ASI-Segmente (1–{len(segs)}, gestrichelt)"))
    ax1.legend(legend_lines, [l.get_label() for l in legend_lines],
               loc="lower center", bbox_to_anchor=(0.5, -0.16),
               ncol=5, fontsize=8.5, framealpha=0.88)

    ax1.grid(True, ls="-", color="white", alpha=0.35, lw=0.5)
    plt.title(f"Berechnung Kv-Wert Mischventil | {len(segs)} Segmente | 1,5 Umdrehungen",
              fontsize=11, pad=10)
    plt.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.12)


    # ── Kv über Öffnungswinkel ────────────────────────────────────────────────
    fig2, ax_kv = plt.subplots(figsize=(10, 6))
    fig2.patch.set_facecolor("#d0d0d0")
    ax_kv.set_facecolor("#d0d0d0")

    # Mittelwerte der ersten 10 Segmente sammeln
    punkte = []
    for i, seg in enumerate(segs[:10]):
        if "Kv" in seg.columns and "Fr_Mix_CL_AI [%]" in seg.columns:
            fr_m = 100 - seg["Fr_Mix_CL_AI [%]"].mean()
            kv_m = np.nanmean(seg["Kv"].values)
            if not np.isnan(fr_m) and not np.isnan(kv_m):
                punkte.append((fr_m, kv_m))

    if len(punkte) >= 2:
        punkte.sort(key=lambda x: x[0])
        fr_punkte = np.array([p[0] for p in punkte])
        kv_punkte = np.array([p[1] for p in punkte])

        # Glatte Kurve
        fr_fein = np.linspace(fr_punkte.min(), fr_punkte.max(), 500)
        kv_fein = np.interp(fr_fein, fr_punkte, kv_punkte)
        ax_kv.plot(fr_fein, kv_fein,
                   color="#1565C0", lw=2.0, zorder=6, label="Interpolation (Seg. 1–10)")

        # Punkte auf 10%-Raster
        fr_raster = np.arange(0, 101, 10)
        kv_raster = np.interp(fr_raster, fr_punkte, kv_punkte)
        ax_kv.scatter(fr_raster, kv_raster,
                      s=80, color="#1565C0", zorder=7,
                      edgecolors="white", linewidths=0.8)
        for fr_r, kv_r in zip(fr_raster, kv_raster):
            ax_kv.annotate(f"{kv_r:.3f}",
                           xy=(fr_r, kv_r),
                           xytext=(0, 8), textcoords="offset points",
                           ha="center", fontsize=7, color="#1565C0", zorder=8)

    ax_kv.set_xlim(-2, 102)
    ax_kv.set_xticks(np.arange(0, 101, 10))
    ax_kv.set_xlabel("Öffnungswinkel [%] (0=geschlossen, 100=offen)", fontsize=11)
    ax_kv.set_ylabel("Kv-Wert [m³/h]", color="#1565C0", fontsize=11)
    ax_kv.tick_params(axis="y", labelcolor="#1565C0")
    ax_kv.grid(True, ls="-", color="white", alpha=0.5, lw=0.5)
    ax_kv.set_title("Kv-Kennlinie Mischventil (Seg. 1–10)", fontsize=12, pad=10)
    ax_kv.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_kombiniert(CSV_DATEI, DAT_ORDNER)