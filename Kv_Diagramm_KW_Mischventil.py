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

TXT_DATEI  = r"J:\ECS\02_Projekte\01_Projektentwicklung\2024-01_ÖgP_GenSYS\05_Daten\KW_Mischventil\2026_06_11_Daten\Tst1_STAD50_U05.txt"
DAT_ORDNER = r"J:\ECS\02_Projekte\01_Projektentwicklung\2024-01_ÖgP_GenSYS\05_Daten\KW_Mischventil\2026_06_11_Daten"

# Zeitbereich Gesamtdiagramm – None = automatisch
X_START = "2026-06-11 10:06:00"
X_ENDE  = "2026-06-11 12:10:00"

# Zeitversatz DAT in Minuten (0 = keine Korrektur, DAT ist bereits Lokalzeit)
DAT_OFFSET_MIN = 2.1
# Zeitversatz der Segmente in Minuten (verschiebt alle Segmentzeitgrenzen)
SEG_OFFSET_MIN = 2.1
# ================================================================
# SEGMENTE – manuelle Zeitbereiche für Kv-Berechnung
# Format: ("Name", "Start", "Ende")
# ================================================================
SEGMENTE = [
    ("Seg 1",  "2026-06-11 10:07:00", "2026-06-11 10:16:00"),
    ("Seg 2",  "2026-06-11 10:18:00", "2026-06-11 10:26:00"),
    ("Seg 3",  "2026-06-11 10:28:00", "2026-06-11 10:36:00"),
    ("Seg 4",  "2026-06-11 10:38:00", "2026-06-11 10:46:00"),
    ("Seg 5",  "2026-06-11 10:48:00", "2026-06-11 10:57:00"),
    ("Seg 6",  "2026-06-11 10:59:00", "2026-06-11 11:08:00"),
    ("Seg 7",  "2026-06-11 11:09:00", "2026-06-11 11:18:00"),
    ("Seg 8",  "2026-06-11 11:20:00", "2026-06-11 11:29:00"),
    ("Seg 9",  "2026-06-11 11:31:00", "2026-06-11 11:39:00"),
    ("Seg 10", "2026-06-11 11:41:00", "2026-06-11 11:50:00"),
    ("Seg 11", "2026-06-11 11:51:00", "2026-06-11 12:00:00"),
]

# Höhe der Mittelwert-Blöcke (0.0=unten, 1.0=oben)
BLOCK_OFFSET = 0.5
BLOCK_OFFSET_SEG = {
    0: None, 1: None, 2: None, 3: None, 4: None,
    5: None, 6: None, 7: None, 8: None, 9: None, 10: None,
}

# ================================================================
# KURVENFARBEN
# ================================================================
FARBE_DURCHFLUSS = "#1565C0"
FARBE_DP_CSV     = "#00838F"
FARBE_TEMP_CSV   = "#E65100"
FARBE_P_I        = "#1B5E20"
FARBE_P_O        = "#F57F17"
FARBE_DP_DAT     = "#BF360C"
FARBE_KV         = "#880E4F"
FARBE_FR         = "#2E7D32"

# ================================================================
# KV-BERECHNUNG  kv = Q [l/h] * sqrt(rho_rel / (dp [bar] * 1000))
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

def lade_txt(pfad):
    print(f"  Lade TXT: {os.path.basename(pfad)}")
    df = pd.read_csv(pfad, encoding="utf-16", skiprows=12, sep="\t", header=0)
    df.columns = ["Nr", "Datum", "Zeit", "Durchfluss", "Temperatur"]
    for c in ["Durchfluss", "Temperatur"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.assign(
        Differenzdruck_bar = np.nan,  # TXT hat keinen Differenzdruck
        Zeitstempel = pd.to_datetime(df["Datum"] + " " + df["Zeit"],
                                     format="%d/%m/%y %H:%M:%S", errors="coerce")
    )
    df = df.dropna(subset=["Zeitstempel"]).sort_values("Zeitstempel").reset_index(drop=True)
    print(f"    → {len(df)} Zeilen | {df['Zeitstempel'].min()} – {df['Zeitstempel'].max()}")
    return df


def lade_dat(ordner):
    COLS = ["Date", "TC20_T.HF.i.CL.AI [°C]", "TC21_T.HF.o.CL.AI [°C]",
            "p_HF_i_CL_AI [bara]", "p_HF_o_CL_AI [bara]", "Fr_Mix_CL_AI [%]"]
    dateien = sorted(glob.glob(os.path.join(ordner, "*.dat")) +
                     glob.glob(os.path.join(ordner, "*.DAT")))
    if not dateien:
        print(f"  WARNUNG: Keine .dat-Dateien in '{ordner}'!"); return pd.DataFrame()
    print(f"  {len(dateien)} DAT-Datei(en) gefunden:")
    teile = []
    for pfad in dateien:
        print(f"    Lade: {os.path.basename(pfad)}")
        try:
            alle = pd.read_csv(pfad, encoding="cp1252", skiprows=4,
                               sep="\t", nrows=0).columns.tolist()
            vorh = [c for c in COLS if c in alle]
            df   = pd.read_csv(pfad, encoding="cp1252", skiprows=4,
                               sep="\t", usecols=vorh)
            df   = df.assign(
                Date      = pd.to_datetime(df["Date"], errors="coerce")
                            + pd.Timedelta(minutes=DAT_OFFSET_MIN),
                dp_HF_bar = np.abs(df["p_HF_o_CL_AI [bara]"] - df["p_HF_i_CL_AI [bara]"])
            )
            df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
            print(f"      → {len(df)} Zeilen | {df['Date'].min()} – {df['Date'].max()}")
            teile.append(df)
        except Exception as e:
            print(f"      FEHLER: {e}")
    if not teile:
        return pd.DataFrame()
    gesamt = pd.concat(teile).sort_values("Date").reset_index(drop=True)
    return gesamt


def plot_kombiniert(txt_pfad, dat_ordner):
    print("\n=== Daten laden ===")
    df_csv = lade_txt(txt_pfad)
    df_dat = lade_dat(dat_ordner)

    x0 = pd.Timestamp(X_START) if X_START else df_csv["Zeitstempel"].min()
    x1 = pd.Timestamp(X_ENDE)  if X_ENDE  else df_csv["Zeitstempel"].max()

    # Segmente als DataFrames aus DAT ausschneiden
    segs = []
    for name, t_start, t_end in SEGMENTE:
        s0 = pd.Timestamp(t_start) + pd.Timedelta(minutes=SEG_OFFSET_MIN)
        s1 = pd.Timestamp(t_end) + pd.Timedelta(minutes=SEG_OFFSET_MIN)
        if not df_dat.empty:
            seg = df_dat[(df_dat["Date"] >= s0) & (df_dat["Date"] <= s1)].copy()
        else:
            seg = pd.DataFrame()
        segs.append((name, s0, s1, seg))

    # ── Temperaturstatistik ───────────────────────────────────────────────────
    T_csv   = df_csv["Temperatur"].dropna()
    T_min, T_max, T_mean = T_csv.min(), T_csv.max(), T_csv.mean()
    T_start = T_csv.iloc[0]
    print("\n=== Temperaturstatistik ===")
    print(f"  Start: {T_start:.2f}°C | Min: {T_min:.2f}°C | Max: {T_max:.2f}°C | Mittel: {T_mean:.2f}°C")

    Q_test  = df_csv["Durchfluss"].mean()
    dp_test = np.nanmean([seg["dp_HF_bar"].mean() for _, _, _, seg in segs if len(seg) > 0]) if segs else 3.8
    kv_ref  = berechne_kv(Q_test, dp_test, T_mean)
    kv_Tmin = berechne_kv(Q_test, dp_test, T_min)
    kv_Tmax = berechne_kv(Q_test, dp_test, T_max)
    err_max = max(abs(kv_Tmin - kv_ref), abs(kv_Tmax - kv_ref)) / kv_ref * 100
    print(f"  Maximaler Kv-Fehler durch Temperaturannahme: {err_max:.4f}%")

    # ── Kv pro Segment berechnen ──────────────────────────────────────────────
    print("\n=== Kv-Wert pro Segment ===")
    kv_mittelwerte = []
    segs_mit_kv = []
    for i, (name, s0, s1, seg) in enumerate(segs):
        if len(seg) < 2:
            print(f"  {name}: keine DAT-Daten im Zeitbereich")
            kv_mittelwerte.append((i, float("nan")))
            segs_mit_kv.append((name, s0, s1, seg))
            continue

        mask     = ((df_csv["Zeitstempel"] >= s0) & (df_csv["Zeitstempel"] <= s1))
        df_seg   = df_csv.loc[mask].copy()
        if len(df_seg) >= 2:
            t_csv    = df_seg["Zeitstempel"].astype(np.int64).values
            t_dat    = seg["Date"].astype(np.int64).values
            Q_interp = np.interp(t_dat, t_csv, df_seg["Durchfluss"].values)
            T_interp = np.interp(t_dat, t_csv, df_seg["Temperatur"].values)
        else:
            Q_interp = np.full(len(seg), df_csv["Durchfluss"].mean())
            T_interp = np.full(len(seg), T_mean)

        kv  = berechne_kv(Q_interp, seg["dp_HF_bar"].values, T_interp)
        seg = seg.assign(Kv=kv)
        kv_mittelwerte.append((i, np.nanmean(kv)))
        segs_mit_kv.append((name, s0, s1, seg))

        fr_mean = (100 - seg["Fr_Mix_CL_AI [%]"].mean()) if "Fr_Mix_CL_AI [%]" in seg.columns else float("nan")
        print(f"  {name} ({s0.strftime('%H:%M')}–{s1.strftime('%H:%M')}): "
              f"Q̄={Q_interp.mean():.0f} l/h | dp̄={seg['dp_HF_bar'].mean():.3f} bar | "
              f"T̄={T_interp.mean():.1f}°C | Öffnung={fr_mean:.1f}% | kv̄={np.nanmean(kv):.4f} m³/h")

    segs = segs_mit_kv

    # ── Diagramm aufbauen ─────────────────────────────────────────────────────
    print("\n=== Diagramm erstellen ===")
    fig, ax1 = plt.subplots(figsize=(22, 7))
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

    l_dp_csv = None
    l_pi = l_po = l_dp_dat = None
    if not df_dat.empty:
        if "p_HF_i_CL_AI [bara]" in df_dat.columns:
            l_pi, = ax2.plot(df_dat["Date"], df_dat["p_HF_i_CL_AI [bara]"],
                             color=FARBE_P_I, lw=1.0, ls="--", zorder=4,
                             label="p_HF_i Eingang [bara]")
        if "p_HF_o_CL_AI [bara]" in df_dat.columns:
            l_po, = ax2.plot(df_dat["Date"], df_dat["p_HF_o_CL_AI [bara]"],
                             color=FARBE_P_O, lw=1.0, ls="--", zorder=4,
                             label="p_HF_o Ausgang [bara]")
        if "dp_HF_bar" in df_dat.columns:
            l_dp_dat, = ax2.plot(df_dat["Date"], df_dat["dp_HF_bar"],
                                 color=FARBE_DP_DAT, lw=1.0, ls=":", zorder=4,
                                 label="|Δp| HF [bar]")

    # Nulllinien ausrichten
    y1lo = df_csv["Durchfluss"].min(); y1hi = df_csv["Durchfluss"].max(); y1r = y1hi - y1lo
    ax1.set_ylim(y1lo - y1r * 0.05, y1hi + y1r * 0.35)
    if not df_dat.empty:
        y2v = []
        for _, _, _, seg in segs:
            for c in ["p_HF_i_CL_AI [bara]","p_HF_o_CL_AI [bara]","dp_HF_bar"]:
                if c in seg.columns: y2v.extend(seg[c].dropna())
        if y2v:
            y2lo, y2hi = min(y2v), max(y2v); y2r = y2hi - y2lo
            if y1r > 0 and y2r > 0:
                sc = y1r / y2r
                ax2.set_ylim(y2lo - y1lo / sc, y2lo - y1lo / sc + (y1hi + y1r * 0.35 - y1lo) / sc)

    # Y4 rechts außen: Kv [m³/h]
    ax4 = ax1.twinx()
    ax4.spines["right"].set_position(("outward", 60))
    ax4.set_ylabel("Kv-Wert [m³/h]", color=FARBE_KV, fontsize=10)
    ax4.tick_params(axis="y", labelcolor=FARBE_KV)
    l_kv = None
    for _, _, _, seg in segs:
        if "Kv" in seg.columns:
            lk, = ax4.plot(seg["Date"], seg["Kv"],
                           color=FARBE_KV, lw=1.4, ls="-", zorder=5,
                           label="Kv-Wert [m³/h]" if l_kv is None else "_")
            if l_kv is None: l_kv = lk

    # Mittelwert-Blöcke
    y_bottom, y_top = ax1.get_ylim()
    y_range = y_top - y_bottom
    for i, (name, s0, s1, seg) in enumerate(segs):
        mask    = ((df_csv["Zeitstempel"] >= s0) & (df_csv["Zeitstempel"] <= s1))
        Q_seg   = df_csv.loc[mask, "Durchfluss"]
        Q_mean  = Q_seg.mean() if len(Q_seg) > 0 else float("nan")
        dp_mean = seg["dp_HF_bar"].mean() if "dp_HF_bar" in seg.columns and len(seg) > 0 else float("nan")
        _, kv_mean = kv_mittelwerte[i] if i < len(kv_mittelwerte) else (i, float("nan"))
        fr_mean = (100 - seg["Fr_Mix_CL_AI [%]"].mean()) if "Fr_Mix_CL_AI [%]" in seg.columns and len(seg) > 0 else float("nan")

        eintraege = []
        if not np.isnan(Q_mean):  eintraege.append((f"Q̄={Q_mean:.0f} l/h",     FARBE_DURCHFLUSS))
        if not np.isnan(dp_mean): eintraege.append((f"dp̄={dp_mean:.3f} bar",   FARBE_DP_DAT))
        if not np.isnan(kv_mean): eintraege.append((f"kv̄={kv_mean:.3f} m³/h", FARBE_KV))
        if not np.isnan(fr_mean): eintraege.append((f"Fr̄={fr_mean:.1f}%",      FARBE_FR))

        seg_offset  = BLOCK_OFFSET_SEG.get(i, None)
        offset_wert = seg_offset if seg_offset is not None else BLOCK_OFFSET
        y_anker_seg = y_bottom + y_range * np.clip(offset_wert, 0.0, 0.98)
        t_mid = s0 + (s1 - s0) / 2

        for j, (label, farbe) in enumerate(eintraege):
            ax1.text(t_mid, y_anker_seg + (len(eintraege) - j - 1) * y_range * 0.036,
                     label, ha="center", va="bottom", fontsize=6.5,
                     color=farbe, fontweight="normal", zorder=7)

    # Y5 rechts weiter außen: Öffnungswinkel [%]
    ax5 = ax1.twinx()
    ax5.spines["right"].set_position(("outward", 125))
    ax5.set_ylabel("Öffnungswinkel [%]", color=FARBE_FR, fontsize=10)
    ax5.tick_params(axis="y", labelcolor=FARBE_FR)
    ax5.set_ylim(-5, 110)
    lfr = None
    if not df_dat.empty and "Fr_Mix_CL_AI [%]" in df_dat.columns:
        lfr, = ax5.plot(df_dat["Date"], 100 - df_dat["Fr_Mix_CL_AI [%]"],
                        color=FARBE_FR, lw=1.2, ls="-.", zorder=5,
                        label="Fr_Mix_CL [%]")

    # X-Achse
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 15)))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
    ax1.set_xlabel("Uhrzeit (Lokalzeit)", fontsize=10)
    ax1.set_xlim(x0, x1)

    # Segment-Markierungen
    _, y1hi_plot = ax1.get_ylim()
    for i, (name, s0, s1, seg) in enumerate(segs):
        ax1.axvspan(s0, s1, alpha=0.10, color="#ffffff", zorder=1)
        t_mid = s0 + (s1 - s0) / 2
        ax1.annotate(f"{i+1}", xy=(t_mid, y1hi_plot * 0.98),
                     ha="center", va="top", fontsize=7, color="#333333", zorder=5)

    # Legende
    legend_lines = [l_Q]
    for l in [l_dp_csv, l_pi, l_po, l_dp_dat, l_kv, lfr]:
        if l is not None: legend_lines.append(l)
    legend_lines.append(mpatches.Patch(facecolor="white", alpha=0.5, edgecolor="#555",
                                        label=f"Segmente (1–{len(segs)})"))
    ax1.legend(legend_lines, [l.get_label() for l in legend_lines],
               loc="lower center", bbox_to_anchor=(0.5, -0.16),
               ncol=5, fontsize=8.5, framealpha=0.88)

    ax1.grid(True, ls="-", color="white", alpha=0.35, lw=0.5)
    plt.title(f"Berechnung Kv-Wert Mischventil | {len(segs)} Segmente | 0,5 Umdrehungen",
              fontsize=11, pad=10)
    plt.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.12)

    # ── Kv-Kennlinie ─────────────────────────────────────────────────────────
    fig2, ax_kv = plt.subplots(figsize=(10, 6))
    fig2.patch.set_facecolor("#d0d0d0")
    ax_kv.set_facecolor("#d0d0d0")

    punkte = []
    for i, (name, s0, s1, seg) in enumerate(segs[:10]):
        if "Kv" in seg.columns and "Fr_Mix_CL_AI [%]" in seg.columns and len(seg) > 0:
            fr_m = 100 - seg["Fr_Mix_CL_AI [%]"].mean()
            kv_m = np.nanmean(seg["Kv"].values)
            if not np.isnan(fr_m) and not np.isnan(kv_m):
                punkte.append((fr_m, kv_m, i+1))

    if len(punkte) >= 2:
        punkte.sort(key=lambda x: x[0])
        fr_p = np.array([p[0] for p in punkte])
        kv_p = np.array([p[1] for p in punkte])

        fr_fein = np.linspace(fr_p.min(), fr_p.max(), 500)
        kv_fein = np.interp(fr_fein, fr_p, kv_p)
        ax_kv.plot(fr_fein, kv_fein, color="#1565C0", lw=2.0, zorder=6, label="Interpolation (Seg. 1–10)")

        fr_raster = np.arange(0, 101, 10)
        kv_raster = np.interp(fr_raster, fr_p, kv_p)
        ax_kv.scatter(fr_raster, kv_raster, s=80, color="#1565C0", zorder=7,
                      edgecolors="white", linewidths=0.8)
        for fr_r, kv_r in zip(fr_raster, kv_raster):
            ax_kv.annotate(f"{kv_r:.3f}", xy=(fr_r, kv_r),
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
    plot_kombiniert(TXT_DATEI, DAT_ORDNER)