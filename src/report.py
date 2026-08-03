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

# Distance-marker ticks along both straights — purely decorative, gives the
# track a "blueprint" feel instead of a bare rounded rectangle.
ticks_svg = "\n".join(
    f'<line x1="{x}" y1="33" x2="{x}" y2="47" stroke="{INK}" stroke-opacity="0.25" stroke-width="2" />\n'
    f'<line x1="{x}" y1="233" x2="{x}" y2="247" stroke="{INK}" stroke-opacity="0.25" stroke-width="2" />'
    for x in range(160, 780, 60)
)

# Callouts (dot + leader line + label) in the same language as an
# engineering exploded-view diagram: TURN 1 / TURN 2 / START-FINISH.
callouts_svg = f"""
  <circle cx="850.7" cy="69.3" r="3" fill="{ACCENT}" />
  <line x1="850.7" y1="69.3" x2="900" y2="10" stroke="{ACCENT}" stroke-width="1.5" />
  <text x="855" y="4" font-family="'IBM Plex Mono', monospace" font-size="13" fill="{INK}" letter-spacing="1">TURN 1</text>

  <circle cx="49.3" cy="210.7" r="3" fill="{ACCENT}" />
  <line x1="49.3" y1="210.7" x2="10" y2="300" stroke="{ACCENT}" stroke-width="1.5" />
  <text x="15" y="304" font-family="'IBM Plex Mono', monospace" font-size="13" fill="{INK}" letter-spacing="1">TURN 2</text>

  <circle cx="120" cy="40" r="3" fill="{INK}" />
  <line x1="120" y1="40" x2="120" y2="-25" stroke="{INK}" stroke-width="1.5" />
  <text x="120" y="-33" font-family="'IBM Plex Mono', monospace" font-size="13" fill="{INK}" letter-spacing="1" text-anchor="middle">START / FINISH</text>
"""

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>F1 Strategy Agent — model report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; }}
  body {{
    background: {BONE}; color: {INK};
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    letter-spacing: -0.011em;
  }}

  /* ── Top ticker — same device as a classified-doc header strip ── */
  .ticker {{
    display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap;
    background: {INK}; color: {BONE}; padding: 8px 24px;
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.14em;
  }}
  .ticker span {{ opacity: 0.85; }}

  .page {{ display: flex; max-width: 1420px; margin: 0 auto; }}

  /* ── Left rail — rotated label, same move as a spec-sheet side band ── */
  .rail {{
    writing-mode: vertical-rl; transform: rotate(180deg);
    background: {ACCENT}; color: {BONE};
    flex-shrink: 0; width: 44px;
    display: flex; align-items: center; justify-content: center; gap: 10px;
    font-size: 12px; letter-spacing: 0.22em; text-transform: uppercase; font-weight: 600;
  }}

  .wrap {{ flex: 1; min-width: 0; padding: 28px 32px 96px; }}

  /* ── Header meta bar — mirrors the "P1 INDUSTRIAL ROBOT / SERIAL NO." row ── */
  .meta-bar {{
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;
    border-bottom: 2px solid {INK}; padding-bottom: 10px; margin-bottom: 28px;
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.14em; color: rgba(21,20,18,0.6);
  }}

  h1 {{ font-size: 34px; font-weight: 700; letter-spacing: -0.02em; margin: 0 0 8px; }}
  .sub {{ color: rgba(21,20,18,0.62); font-size: 13px; margin: 0 0 32px; max-width: 78ch; line-height: 1.5; }}
  h2 {{
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.16em; font-weight: 600;
    color: {INK}; margin: 44px 0 14px; border-bottom: 2px solid {INK};
    padding-bottom: 10px;
  }}

  .metric-row {{ display: flex; gap: 0; flex-wrap: wrap; border: 2px solid {INK}; border-right: none; }}
  .metric {{
    border-right: 2px solid {INK}; background: #fff; padding: 16px 18px; flex: 1; min-width: 240px;
  }}
  .metric .label {{ font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: rgba(21,20,18,0.6); }}
  .metric .value {{ font-size: 30px; font-weight: 700; margin-top: 10px; font-variant-numeric: tabular-nums; }}
  .metric.win {{ background: {INK}; }}
  .metric.win .label {{ color: rgba(240,238,233,0.65); }}
  .metric.win .value {{ color: {ACCENT}; }}

  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 860px) {{ .charts {{ grid-template-columns: 1fr; }} .page {{ flex-direction: column; }} .rail {{ writing-mode: horizontal-tb; transform: none; width: auto; height: 32px; }} }}
  .chart-label {{
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em; color: rgba(21,20,18,0.6);
    margin: 0 0 8px; border-top: 2px solid {INK}; padding-top: 8px;
  }}
  img {{ width: 100%; border: 2px solid {INK}; background: #fff; display: block; }}

  /* ── Corner-bracket frame — the viewfinder-corner trick from the ref shots ── */
  .frame {{ position: relative; border: 2px solid {INK}; background: #fff; padding: 24px; }}
  .cnr {{ position: absolute; width: 16px; height: 16px; pointer-events: none; }}
  .cnr-tl {{ top: -2px; left: -2px; border-top: 3px solid {ACCENT}; border-left: 3px solid {ACCENT}; }}
  .cnr-tr {{ top: -2px; right: -2px; border-top: 3px solid {ACCENT}; border-right: 3px solid {ACCENT}; }}
  .cnr-bl {{ bottom: -2px; left: -2px; border-bottom: 3px solid {ACCENT}; border-left: 3px solid {ACCENT}; }}
  .cnr-br {{ bottom: -2px; right: -2px; border-bottom: 3px solid {ACCENT}; border-right: 3px solid {ACCENT}; }}

  .track-svg {{ width: 100%; height: auto; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 18px; margin-top: 18px; padding-top: 14px; border-top: 1px solid {LINE}; font-size: 12px; }}
  .legend-item {{ display: flex; align-items: center; gap: 8px; }}
  .swatch {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex-shrink: 0; margin-right: 8px; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: #fff; border: 2px solid {INK}; margin-top: 16px; }}
  th {{
    text-align: left; padding: 10px 14px; background: {INK}; color: {BONE};
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 500;
  }}
  td {{ text-align: left; padding: 10px 14px; border-bottom: 1px solid {LINE}; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr.best {{ background: rgba(193,102,61,0.08); font-weight: 600; }}
  .chip {{
    display: inline-block; font-size: 9px; background: {ACCENT}; color: {BONE};
    padding: 2px 6px; margin-left: 6px; letter-spacing: 0.08em;
  }}
  footer {{
    margin-top: 56px; font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em;
    color: rgba(21,20,18,0.5); border-top: 2px solid {INK}; padding-top: 12px;
  }}
</style>
</head>
<body>
<div class="ticker">
  <span>TRAINED — JEDDAH + BAHRAIN GP 2023</span>
  <span>F1 STRATEGY AGENT — LAP-TIME MODEL</span>
  <span>&gt;&gt;&gt; EVALUATED — MIAMI GP 2023</span>
</div>

<div class="page">
  <div class="rail">MODEL REPORT</div>

  <div class="wrap">
    <div class="meta-bar">
      <span>F1-Strategy-Agent / src/report.py</span>
      <span>Source: FastF1 · 2023 season</span>
    </div>

    <h1>Lap-Time Model &amp; Pit Strategy Simulation</h1>
    <p class="sub">A RandomForest model predicts lap time from tyre compound, tyre wear and lap number — trained on two Grand Prix, evaluated on a third it never saw. The result feeds a pit-stop strategy simulation on the test circuit.</p>

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

    <div class="frame">
      <span class="cnr cnr-tl"></span><span class="cnr cnr-tr"></span>
      <span class="cnr cnr-bl"></span><span class="cnr cnr-br"></span>
      <svg class="track-svg" viewBox="-30 -50 980 400" xmlns="http://www.w3.org/2000/svg">
        <path id="track" d="M120,40 L780,40 A100,100 0 0 1 780,240 L120,240 A100,100 0 0 1 120,40 Z"
              fill="none" stroke="{INK}" stroke-opacity="0.14" stroke-width="26" stroke-linejoin="round" />
        <path d="M120,40 L780,40 A100,100 0 0 1 780,240 L120,240 A100,100 0 0 1 120,40 Z"
              fill="none" stroke="{BONE}" stroke-width="2" stroke-dasharray="6 6" />
        {ticks_svg}
        <line x1="120" y1="27" x2="120" y2="53" stroke="{INK}" stroke-width="4" />
        {callouts_svg}
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
        <p class="chart-label">Tyre degradation by compound</p>
        <img src="data:image/png;base64,{degradation_img}" alt="Degradation by compound" />
      </div>
      <div>
        <p class="chart-label">Predicted vs. actual — Miami 2023</p>
        <img src="data:image/png;base64,{prediction_img}" alt="Predicted vs actual" />
      </div>
    </div>

    <footer>Generated by src/report.py &middot; F1-Strategy-Agent</footer>
  </div>
</div>
</body>
</html>
"""

with open("report.html", "w") as f:
    f.write(html)

print("Saved to report.html")
