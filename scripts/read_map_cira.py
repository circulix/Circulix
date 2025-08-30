# scripts/read_map_cira.py
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw/cira")
OUT_PATH = Path("data/processed/combined_cira.csv")

TARGET_COLUMNS = [
    "Timestamp",
    "Vibration_RMS", "Pressure_Fluct", "Temperature", "Motor_Current",
    "Flow_Rate", "Cavitation_Margin", "Viscosity", "Leakage_Rate",
    "Cold_Flag",
    "Label"
]

def estimate_viscosity_cP(temp_c):
    pts = np.array([[-10, 3.3], [0, 1.79], [20, 1.0], [40, 0.65]])
    x = pts[:,0]; y = pts[:,1]
    return np.interp(temp_c, x, y)

def rms(series, window=50):
    return series.rolling(window=window, min_periods=1).apply(
        lambda x: np.sqrt(np.mean(np.square(x))), raw=True
    )

def load_and_map(csv_path):
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}

    ts_col = next((cols[k] for k in ["timestamp","time","date_time"] if k in cols), None)
    pres_col = next((cols[k] for k in ["x_pres.pv","x_pres.sv","pressure","outlet_pressure"] if k in cols), None)
    temp_col = next((cols[k] for k in ["temperature","temp","x_temp.sv","ambient_temp","env_temp"] if k in cols), None)
    curr_col = next((cols[k] for k in ["motor_current","current","i_motor","motor_i"] if k in cols), None)
    vib_raw = next((cols[k] for k in ["x_acr_pmp.pv","vibration","accel","vib","x_acr_pmp"] if k in cols), None)

    out = pd.DataFrame()
    out["Timestamp"] = df[ts_col] if ts_col else np.arange(len(df))
    out["Vibration_RMS"] = rms(df[vib_raw]) if vib_raw else np.nan
    if pres_col:
        out["Pressure_Fluct"] = df[pres_col] - df[pres_col].rolling(100, min_periods=1).mean()
    else:
        out["Pressure_Fluct"] = np.nan

    if temp_col:
        temp = pd.to_numeric(df[temp_col], errors="coerce")
        out["Temperature"] = temp
    else:
        out["Temperature"] = 10.0

    out["Motor_Current"] = pd.to_numeric(df[curr_col], errors="coerce") if curr_col else np.nan
    out["Flow_Rate"] = np.nan
    out["Cavitation_Margin"] = np.nan
    out["Leakage_Rate"] = np.nan
    out["Cold_Flag"] = (out["Temperature"] <= 0).astype(int)
    out["Viscosity"] = estimate_viscosity_cP(out["Temperature"])
    out["Label"] = "Normal"

    for col in TARGET_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[TARGET_COLUMNS]
    out = out.dropna(how="all")
    return out

def main():
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        print("هیچ CSVی در data/raw/cira پیدا نشد.")
        return
    frames = [load_and_map(p) for p in files]
    combined = pd.concat(frames, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"ساخته شد: {OUT_PATH} (rows={len(combined)})")

if __name__ == "__main__":
    main()
