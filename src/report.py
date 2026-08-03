"""
v2 model (Compound + TyreLife + LapNumber, RandomForest, trained on the
DELTA to each circuit's baseline pace) evaluated on Miami after training on
Jeddah + Bahrain, plus a simple pit-strategy simulation on that same test
circuit. Generates a self-contained HTML report (no server, no external JS)
styled like the portfolio (bone background, ink text, terracotta accent,
IBM Plex Mono) with an animated track showing the simulated strategies
racing against each other.
"""

import base64
import io

import fastf1
import matplotlib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache("data/cache")

BONE = "#F0EEE9"
INK = "#151412"
ACCENT = "#C1663D"
LINE = "rgba(21,20,18,0.16)"
COMPOUND_COLOR = {"SOFT": "#C1663D", "MEDIUM": "#B8433F", "HARD": "#151412"}
STRATEGY_COLOR = ["#C1663D", "#B8433F", "#151412", "#5A6B8C", "#3F7A4E"]

MAE_BASELINE_V1 = 3.49  # compound-mean, train=Jeddah+Bahrain, test=Miami (baselinec2.py)


def load_race(year, gp):
    """Returns a race's clean laps plus its baseline (median lap time).
    Different circuits have very different lap-time paces just from track
    layout (Bahrain ~98s, Miami ~93s) — without subtracting that baseline
    before training, the model confuses "we're on a different circuit" with
    "the tyres changed", and the error doesn't drop no matter how many
    features you add."""
    session = fastf1.get_session(year, gp, "R")
    session.load()
    laps = session.laps.pick_quicklaps().copy()
    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
    cols = ["Compound", "TyreLife", "LapNumber", "LapTimeSeconds"]
    df = laps[cols].dropna()
    baseline = df["LapTimeSeconds"].median()
    df["Delta"] = df["LapTimeSeconds"] - baseline
    return df, baseline


def features(df, compound_cols):
    """One-hot encode Compound aligned to the columns seen in training, so
    test data never produces a column the model doesn't know."""
    dummies = pd.get_dummies(df["Compound"]).reindex(columns=compound_cols, fill_value=0)
    return pd.concat([df[["TyreLife", "LapNumber"]].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=BONE)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def style_ax(ax):
    ax.set_facecolor(BONE)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(INK)
        ax.spines[spine].set_alpha(0.3)
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


print("Loading races...")
jeddah_df, jeddah_baseline = load_race(2023, "Jeddah")
bahrain_df, bahrain_baseline = load_race(2023, "Bahrain")
test_df, miami_baseline = load_race(2023, "Miami")
train_df = pd.concat([jeddah_df, bahrain_df], ignore_index=True)

compound_cols = sorted(train_df["Compound"].unique())
X_train, y_train = features(train_df, compound_cols), train_df["Delta"]
X_test, y_test = features(test_df, compound_cols), test_df["LapTimeSeconds"]

model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=0)
model.fit(X_train, y_train)
pred_delta = model.predict(X_test)
# The simulation assumes the test circuit's baseline pace is known ahead of
# time (e.g. from practice/qualifying) — the model only has to get the
# degradation right, not the absolute pace.
pred = pred_delta + miami_baseline
mae_v2 = mean_absolute_error(y_test, pred)
print(f"MAE v1 (compound-mean, absolute lap time): {MAE_BASELINE_V1:.2f}s")
print(f"MAE v2 (Compound+TyreLife+LapNumber on delta to baseline): {mae_v2:.2f}s")

# ── Chart 1: tyre degradation by compound (mid-race) ──
fig1, ax1 = plt.subplots(figsize=(6.4, 4))
style_ax(ax1)
tyre_life_range = list(range(1, 31))
for compound in compound_cols:
    rows = pd.DataFrame({"TyreLife": tyre_life_range, "LapNumber": [25] * len(tyre_life_range)})
    rows_feat = features(pd.concat([rows, pd.DataFrame({"Compound": [compound] * len(tyre_life_range)})], axis=1), compound_cols)
    preds = model.predict(rows_feat)
    ax1.plot(tyre_life_range, preds, label=compound, color=COMPOUND_COLOR.get(compound, INK), linewidth=2)
ax1.set_xlabel("Laps on the same tyre set")
ax1.set_ylabel("Seconds relative to circuit baseline pace")
ax1.legend(frameon=False, labelcolor=INK, fontsize=9)
degradation_img = fig_to_base64(fig1)

# ── Chart 2: predicted vs. actual on Miami ──
fig2, ax2 = plt.subplots(figsize=(6.4, 4))
style_ax(ax2)
ax2.scatter(y_test, pred, alpha=0.4, s=14, color=ACCENT, edgecolors="none")
lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
ax2.plot(lims, lims, color=INK, linewidth=1, alpha=0.4, linestyle="--")
ax2.set_xlabel("Actual lap time (s)")
ax2.set_ylabel("Predicted lap time (s)")
prediction_img = fig_to_base64(fig2)

# ── Pit strategy simulation on Miami (57 laps) ──
TOTAL_LAPS = int(test_df["LapNumber"].max())

strategies = {
    "One stop — Medium → Hard": [("MEDIUM", 20), ("HARD", TOTAL_LAPS - 20)],
    "One stop — Hard → Medium": [("HARD", 30), ("MEDIUM", TOTAL_LAPS - 30)],
    "Two stops — Medium → Hard → Medium": [("MEDIUM", 18), ("HARD", 20), ("MEDIUM", TOTAL_LAPS - 38)],
    "Two stops — Soft → Medium → Hard": [("SOFT", 14), ("MEDIUM", 20), ("HARD", TOTAL_LAPS - 34)],
}


def simulate(stints):
    rows = []
    lap_number = 1
    for compound, n_laps in stints:
        for tyre_life in range(1, n_laps + 1):
            rows.append({"Compound": compound, "TyreLife": tyre_life, "LapNumber": lap_number})
            lap_number += 1
    df = pd.DataFrame(rows)
    pred_deltas = model.predict(features(df, compound_cols))
    return pred_deltas.sum() + miami_baseline * len(rows)


results = {name: simulate(stints) for name, stints in strategies.items()}
best = min(results, key=results.get)
fastest_time = results[best]

# Animation duration scales with total race time: the fastest strategy
# completes one lap of the track in BASE_DURATION seconds, the others take
# proportionally longer — so the finishing order and the gap on screen
# directly reflect the simulated time difference.
BASE_DURATION = 6.0
car_rows = []
for i, (name, total) in enumerate(results.items()):
    color = STRATEGY_COLOR[i % len(STRATEGY_COLOR)]
    dur = BASE_DURATION * (total / fastest_time)
    car_rows.append((name, total, color, dur))

table_rows = "\n".join(
    f"""<tr{' class="best"' if name == best else ''}>
        <td><span class="swatch" style="background:{color}"></span>{name}{' <span class="chip">BEST</span>' if name == best else ''}</td>
        <td class="num">{results[name] / 60:.1f} min</td>
        <td class="num">{results[name] - results[best]:+.1f}s</td>
    </tr>"""
    for name, total, color, dur in car_rows
)

cars_svg = "\n".join(
    f"""<circle r="7" fill="{color}" stroke="{BONE}" stroke-width="1.5">
        <animateMotion dur="{dur:.2f}s" repeatCount="indefinite" rotate="auto">
          <mpath href="#track" />
        </animateMotion>
      </circle>"""
    for name, total, color, dur in car_rows
)

legend_html = "\n".join(
    f"""<div class="legend-item">
        <span class="swatch" style="background:{color}"></span>
        <span>{name}</span>
      </div>"""
    for name, total, color, dur in car_rows
)

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>F1 Strategy Agent — model report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0;
    background: {BONE}; color: {INK};
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    letter-spacing: -0.011em;
  }}
  .wrap {{ max-width: 1360px; margin: 0 auto; padding: 40px 32px 96px; }}
  h1 {{ font-size: 28px; font-weight: 600; letter-spacing: -0.02em; margin: 0 0 6px; }}
  .sub {{ color: rgba(21,20,18,0.6); font-size: 13px; margin: 0 0 32px; max-width: 74ch; }}
  h2 {{
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.14em;
    color: rgba(21,20,18,0.68); margin: 40px 0 14px; border-bottom: 1px solid {LINE};
    padding-bottom: 10px;
  }}
  .metric-row {{ display: flex; gap: 14px; flex-wrap: wrap; }}
  .metric {{
    border: 1px solid {LINE}; background: #fff; padding: 16px 18px; flex: 1; min-width: 220px;
  }}
  .metric .label {{ font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em; color: rgba(21,20,18,0.6); }}
  .metric .value {{ font-size: 26px; font-weight: 500; margin-top: 8px; }}
  .metric.win .value {{ color: {ACCENT}; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 860px) {{ .charts {{ grid-template-columns: 1fr; }} }}
  img {{ width: 100%; border: 1px solid {LINE}; background: #fff; display: block; }}
  .track-card {{ border: 1px solid {LINE}; background: #fff; padding: 20px; }}
  .track-svg {{ width: 100%; height: auto; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 16px; margin-top: 14px; font-size: 12px; }}
  .legend-item {{ display: flex; align-items: center; gap: 8px; }}
  .swatch {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex-shrink: 0; margin-right: 8px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: #fff; border: 1px solid {LINE}; margin-top: 16px; }}
  th, td {{ text-align: left; padding: 10px 14px; border-bottom: 1px solid {LINE}; }}
  th {{ font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: rgba(21,20,18,0.6); }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr.best {{ background: rgba(193,102,61,0.07); font-weight: 600; }}
  .chip {{
    display: inline-block; font-size: 9px; background: {ACCENT}; color: {BONE};
    padding: 2px 6px; border-radius: 2px; margin-left: 6px; letter-spacing: 0.08em;
  }}
  footer {{ margin-top: 56px; font-size: 11px; color: rgba(21,20,18,0.5); }}
</style>
</head>
<body>
<div class="wrap">
  <h1>F1 Strategy Agent</h1>
  <p class="sub">Lap-time model + pit-stop strategy simulation · trained on Jeddah + Bahrain 2023, evaluated on Miami 2023</p>

  <h2>Model performance</h2>
  <div class="metric-row">
    <div class="metric">
      <div class="label">v1 — mean lap time by compound</div>
      <div class="value">{MAE_BASELINE_V1:.2f}s MAE</div>
    </div>
    <div class="metric win">
      <div class="label">v2 — Compound + TyreLife + LapNumber, delta to circuit baseline</div>
      <div class="value">{mae_v2:.2f}s MAE</div>
    </div>
  </div>

  <h2>Pit strategy simulation — Miami 2023 ({TOTAL_LAPS} laps)</h2>
  <p class="sub">Assumes the circuit's baseline pace is known ahead of time (e.g. from practice/qualifying — here taken from the real race: {miami_baseline:.2f}s). The model only predicts how far each lap drifts from that pace, based on compound and tyre wear. Each car laps the track once, at a speed proportional to its simulated total race time — first one home wins.</p>

  <div class="track-card">
    <svg class="track-svg" viewBox="0 0 900 280" xmlns="http://www.w3.org/2000/svg">
      <path id="track" d="M120,40 L780,40 A100,100 0 0 1 780,240 L120,240 A100,100 0 0 1 120,40 Z"
            fill="none" stroke="{INK}" stroke-opacity="0.16" stroke-width="26" stroke-linejoin="round" />
      <path d="M120,40 L780,40 A100,100 0 0 1 780,240 L120,240 A100,100 0 0 1 120,40 Z"
            fill="none" stroke="{BONE}" stroke-width="2" stroke-dasharray="6 6" />
      <line x1="120" y1="27" x2="120" y2="53" stroke="{INK}" stroke-width="4" />
      {cars_svg}
    </svg>
    <div class="legend">
      {legend_html}
    </div>
  </div>

  <table>
    <thead><tr><th>Strategy</th><th>Estimated total time</th><th>Gap to best</th></tr></thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>

  <h2>Charts</h2>
  <div class="charts">
    <div>
      <p class="sub" style="margin-bottom:8px;">Tyre degradation by compound</p>
      <img src="data:image/png;base64,{degradation_img}" alt="Degradation by compound" />
    </div>
    <div>
      <p class="sub" style="margin-bottom:8px;">Predicted vs. actual — Miami 2023</p>
      <img src="data:image/png;base64,{prediction_img}" alt="Predicted vs actual" />
    </div>
  </div>

  <footer>Generated by src/report.py · F1-Strategy-Agent</footer>
</div>
</body>
</html>
"""

with open("report.html", "w") as f:
    f.write(html)

print("Saved to report.html")
