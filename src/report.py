"""
v2 model (Compound + TyreLife + LapNumber, RandomForest, trained on the
DELTA to each circuit's baseline pace) evaluated on Suzuka (Japanese GP)
after training on Jeddah + Bahrain, plus a simple pit-strategy simulation
on that same test circuit. Generates a self-contained HTML report (no
server, no external JS) styled like the portfolio (bone background, ink
text, terracotta accent, IBM Plex Mono) with an animated track showing the
simulated strategies racing against each other.

The track SVG is a real Suzuka silhouette, not a generic rounded blob:
Suzuka is the only circuit on the calendar that crosses over itself (the
famous figure-8, via the bridge between the Esses and the back straight),
so the shape is generated as a lemniscate (figure-8 curve) with the two
lobes scaled unevenly to match Suzuka's proportions — small tight loop for
the Esses, one big sweeping loop for Degner/Hairpin/Spoon/130R — instead
of hand-drawn bezier points, which is what kept "getting worse" (drifting
into an arbitrary blob) under manual editing. See build_track_path() below.
"""

import matplotlib.pyplot as plt
import base64
import io
import math

import fastf1
import matplotlib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

matplotlib.use("Agg")

fastf1.Cache.enable_cache("data/cache")

BONE = "#F0EEE9"
INK = "#151412"
ACCENT = "#C1663D"
LINE = "rgba(21,20,18,0.16)"
COMPOUND_COLOR = {"SOFT": "#C1663D", "MEDIUM": "#B8433F", "HARD": "#151412"}
STRATEGY_COLOR = ["#C1663D", "#B8433F", "#151412", "#5A6B8C", "#3F7A4E"]

# compound-mean, train=Jeddah+Bahrain, test=Miami (baselinec2.py)
MAE_BASELINE_V1 = 3.49


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
    dummies = pd.get_dummies(df["Compound"]).reindex(
        columns=compound_cols, fill_value=0)
    return pd.concat([df[["TyreLife", "LapNumber"]].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150,
                bbox_inches="tight", facecolor=BONE)
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


def build_track_path():
    """Real Suzuka silhouette, generated (not hand-drawn) from a lemniscate
    (figure-8 curve, x = cos(t)/(1+sin²t), y = sin(t)cos(t)/(1+sin²t)) — the
    only curve family that crosses itself exactly once, which is what makes
    a hand-tuned bezier path so easy to wreck by editing (it either stops
    crossing itself, or crosses in an ugly kink) and so easy to get right
    generated from a formula instead.

    The two lobes are scaled unevenly (small tight lobe vs. one big sweeping
    lobe) to match Suzuka's real proportions — the small lobe stands in for
    the Esses, the big lobe for Degner/Hairpin/Spoon Curve/130R — then a few
    corner-like features are layered on top: a wiggle through the small
    lobe's top arc (Esses), a tightened flat spot at the big lobe's far tip
    (Hairpin), and one straightened edge (back straight / start-finish).

    Returns (path_d, landmarks) where landmarks has the (x, y) points used
    to place the callouts and ticks, so they track the shape instead of
    being independently hand-placed coordinates that drift out of sync
    with it (which is exactly what happened before).
    """
    def smoothstep(t):
        t = max(0.0, min(1.0, t))
        return t * t * (3 - 2 * t)

    n = 200
    cx, cy = 480, 320
    raw = []
    for i in range(n):
        t = 2 * math.pi * i / n
        denom = 1 + math.sin(t) ** 2
        x = math.cos(t) / denom
        y = math.sin(t) * math.cos(t) / denom
        # Blend factor: 0 deep in the small lobe, 1 deep in the big lobe,
        # smoothstep'd over a band around the crossing so the transition is
        # gradual — a hard switch here is what produced an ugly notch right
        # at the bridge in the first pass of this generator.
        b = smoothstep((x + 0.12) / 0.24)
        sx = 190 + b * (330 - 190)
        sy = 145 + b * (235 - 145)
        shift_x = (1 - b) * 35
        shift_y = (1 - b) * -50
        X = cx + x * sx + shift_x
        Y = cy - y * sy + shift_y
        raw.append([X, Y, t, b])

    for i, (X, Y, t, b) in enumerate(raw):
        if b < 0.15:  # Esses wiggle, small lobe's top arc only
            wig = 14 * math.sin(t * 5.5)
            raw[i][0] += wig * math.cos(t + math.pi / 2) * 0.6
            raw[i][1] += wig * math.sin(t + math.pi / 2) * 0.6
        if b > 0.85 and 2.7 < (t % (2 * math.pi)) < 3.6:  # Hairpin tip
            raw[i][0] -= 26

    straight_idx = [i for i, (X, Y, t, b) in enumerate(raw)
                     if b > 0.9 and 0.15 < (t % (2 * math.pi)) < 0.85]
    x0, y0 = raw[straight_idx[0]][0], raw[straight_idx[0]][1]
    x1, y1 = raw[straight_idx[-1]][0], raw[straight_idx[-1]][1]
    for k, i in enumerate(straight_idx):
        f = k / (len(straight_idx) - 1)
        raw[i][0] = x0 + (x1 - x0) * f
        raw[i][1] = y0 + (y1 - y0) * f

    points = [(p[0], p[1]) for p in raw]

    d = f"M {points[0][0]:.1f},{points[0][1]:.1f} "
    for i in range(n):
        p0, p1, p2, p3 = (points[(i - 1) % n], points[i % n],
                          points[(i + 1) % n], points[(i + 2) % n])
        c1x, c1y = p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6
        c2x, c2y = p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6
        d += f"C {c1x:.1f},{c1y:.1f} {c2x:.1f},{c2y:.1f} {p2[0]:.1f},{p2[1]:.1f} "
    d += "Z"

    crossing = min(raw, key=lambda r: abs(r[3] - 0.5))
    hairpin = max((r for r in raw if r[3] > 0.85), key=lambda r: r[1])
    esses_top = min((r for r in raw if r[3] < 0.1), key=lambda r: r[1])
    landmarks = {
        "straight_start": (x0, y0),
        "straight_end": (x1, y1),
        "crossing": (crossing[0], crossing[1]),
        "hairpin": (hairpin[0], hairpin[1]),
        "esses_top": (esses_top[0], esses_top[1]),
    }
    return d, landmarks


TRACK_PATH_D, TRACK_LANDMARKS = build_track_path()


print("Loading races...")
jeddah_df, jeddah_baseline = load_race(2023, "Jeddah")
bahrain_df, bahrain_baseline = load_race(2023, "Bahrain")
test_df, test_baseline = load_race(2023, "Japan")
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
pred = pred_delta + test_baseline
mae_v2 = mean_absolute_error(y_test, pred)
print(f"MAE v1 (compound-mean, absolute lap time): {MAE_BASELINE_V1:.2f}s")
print(
    f"MAE v2 (Compound+TyreLife+LapNumber on delta to baseline): {mae_v2:.2f}s")

# ── Chart 1: tyre degradation by compound (mid-race) ──
fig1, ax1 = plt.subplots(figsize=(6.4, 4))
style_ax(ax1)
tyre_life_range = list(range(1, 31))
for compound in compound_cols:
    rows = pd.DataFrame({"TyreLife": tyre_life_range,
                        "LapNumber": [25] * len(tyre_life_range)})
    rows_feat = features(pd.concat([rows, pd.DataFrame(
        {"Compound": [compound] * len(tyre_life_range)})], axis=1), compound_cols)
    preds = model.predict(rows_feat)
    ax1.plot(tyre_life_range, preds, label=compound,
             color=COMPOUND_COLOR.get(compound, INK), linewidth=2)
ax1.set_xlabel("Laps on the same tyre set")
ax1.set_ylabel("Seconds relative to circuit baseline pace")
ax1.legend(frameon=False, labelcolor=INK, fontsize=9)
degradation_img = fig_to_base64(fig1)

# ── Chart 2: predicted vs. actual on Suzuka ──
fig2, ax2 = plt.subplots(figsize=(6.4, 4))
style_ax(ax2)
ax2.scatter(y_test, pred, alpha=0.4, s=14, color=ACCENT, edgecolors="none")
lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
ax2.plot(lims, lims, color=INK, linewidth=1, alpha=0.4, linestyle="--")
ax2.set_xlabel("Actual lap time (s)")
ax2.set_ylabel("Predicted lap time (s)")
prediction_img = fig_to_base64(fig2)

# ── Pit strategy simulation on Suzuka (53 laps) ──
TOTAL_LAPS = int(test_df["LapNumber"].max())

strategies = {
    "One stop — Medium → Hard": [("MEDIUM", 20), ("HARD", TOTAL_LAPS - 20)],
    "One stop — Hard → Medium": [("HARD", 30), ("MEDIUM", TOTAL_LAPS - 30)],
    "Two stops — Medium → Hard → Medium": [("MEDIUM", 18), ("HARD", 20), ("MEDIUM", TOTAL_LAPS - 38)],
    "Two stops — Soft → Medium → Hard": [("SOFT", 14), ("MEDIUM", 20), ("HARD", TOTAL_LAPS - 34)],
    "Two stops - Soft → Hard → Soft": [("SOFT", 12), ("HARD", 30), ("SOFT", TOTAL_LAPS - 42)],
}


def simulate(stints):
    rows = []
    lap_number = 1
    for compound, n_laps in stints:
        for tyre_life in range(1, n_laps + 1):
            rows.append(
                {"Compound": compound, "TyreLife": tyre_life, "LapNumber": lap_number})
            lap_number += 1
    df = pd.DataFrame(rows)
    pred_deltas = model.predict(features(df, compound_cols))
    return pred_deltas.sum() + test_baseline * len(rows)


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
    f"""<circle r="9" fill="{color}" stroke="{BONE}" stroke-width="2">
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

# Distance-marker ticks along the back straight — purely decorative, gives
# the track a "blueprint" feel. Placed along the actual straight segment of
# the generated path (TRACK_LANDMARKS), not independent fixed coordinates,
# so they can't drift out of sync with the track shape the way the old
# hand-placed ticks (tied to a since-deleted rounded-rectangle path) did.
_sx0, _sy0 = TRACK_LANDMARKS["straight_start"]
_sx1, _sy1 = TRACK_LANDMARKS["straight_end"]
_slen = math.hypot(_sx1 - _sx0, _sy1 - _sy0)
_perp = (-(_sy1 - _sy0) / _slen, (_sx1 - _sx0) / _slen)
ticks_svg = "\n".join(
    f'<line x1="{_sx0 + (_sx1 - _sx0) * f + _perp[0] * o1:.1f}" y1="{_sy0 + (_sy1 - _sy0) * f + _perp[1] * o1:.1f}" '
    f'x2="{_sx0 + (_sx1 - _sx0) * f + _perp[0] * o2:.1f}" y2="{_sy0 + (_sy1 - _sy0) * f + _perp[1] * o2:.1f}" '
    f'stroke="{INK}" stroke-opacity="0.25" stroke-width="2" />'
    for f in [i / 8 for i in range(1, 8)]
    for o1, o2 in [(-20, -34)]
)

# Callouts (dot + leader line + label), tied to TRACK_LANDMARKS so they
# always point at the actual generated shape — ESSES on the small lobe,
# HAIRPIN at the big lobe's tightened tip, START/FINISH at the straight's
# midpoint, BRIDGE at the one point the track crosses itself (the feature
# that makes Suzuka Suzuka).
_ex, _ey = TRACK_LANDMARKS["esses_top"]
_hx, _hy = TRACK_LANDMARKS["hairpin"]
_cx, _cy = TRACK_LANDMARKS["crossing"]
_mx, _my = (_sx0 + _sx1) / 2, (_sy0 + _sy1) / 2
callouts_svg = f"""
  <circle cx="{_ex:.1f}" cy="{_ey:.1f}" r="3" fill="{ACCENT}" />
  <line x1="{_ex:.1f}" y1="{_ey:.1f}" x2="{_ex - 60:.1f}" y2="{_ey - 40:.1f}" stroke="{ACCENT}" stroke-width="1.5" />
  <text x="{_ex - 130:.1f}" y="{_ey - 44:.1f}" font-family="'IBM Plex Mono', monospace" font-size="13" fill="{INK}" letter-spacing="1">ESSES</text>

  <circle cx="{_hx:.1f}" cy="{_hy:.1f}" r="3" fill="{ACCENT}" />
  <line x1="{_hx:.1f}" y1="{_hy:.1f}" x2="{_hx + 40:.1f}" y2="{_hy + 45:.1f}" stroke="{ACCENT}" stroke-width="1.5" />
  <text x="{_hx - 20:.1f}" y="{_hy + 62:.1f}" font-family="'IBM Plex Mono', monospace" font-size="13" fill="{INK}" letter-spacing="1">HAIRPIN</text>

  <circle cx="{_cx:.1f}" cy="{_cy:.1f}" r="3" fill="{INK}" />
  <line x1="{_cx:.1f}" y1="{_cy:.1f}" x2="{_cx - 70:.1f}" y2="{_cy + 10:.1f}" stroke="{INK}" stroke-width="1.5" />
  <text x="{_cx - 175:.1f}" y="{_cy + 14:.1f}" font-family="'IBM Plex Mono', monospace" font-size="13" fill="{INK}" letter-spacing="1">BRIDGE — TRACK CROSSES ITSELF</text>

  <circle cx="{_mx:.1f}" cy="{_my:.1f}" r="3" fill="{INK}" />
  <line x1="{_mx:.1f}" y1="{_my:.1f}" x2="{_mx:.1f}" y2="{_my - 65:.1f}" stroke="{INK}" stroke-width="1.5" />
  <text x="{_mx:.1f}" y="{_my - 73:.1f}" font-family="'IBM Plex Mono', monospace" font-size="13" fill="{INK}" letter-spacing="1" text-anchor="middle">START / FINISH</text>
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
  <span>&gt;&gt;&gt; EVALUATED — JAPANESE GP 2023</span>
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

    <h2>Pit strategy simulation — Suzuka 2023 ({TOTAL_LAPS} laps)</h2>
    <p class="sub">Assumes the circuit's baseline pace is known ahead of time (e.g. from practice/qualifying — here taken from the real race: {test_baseline:.2f}s). The model only predicts how far each lap drifts from that pace, based on compound and tyre wear. Each car laps the track once, at a speed proportional to its simulated total race time — first one home wins.</p>

    <div class="frame">
      <span class="cnr cnr-tl"></span><span class="cnr cnr-tr"></span>
      <span class="cnr cnr-bl"></span><span class="cnr cnr-br"></span>
      <svg class="track-svg" viewBox="0 0 900 600" xmlns="http://www.w3.org/2000/svg">
        <path id="track" d="{TRACK_PATH_D}"
              fill="none" stroke="{INK}" stroke-opacity="0.14" stroke-width="26" stroke-linecap="round" stroke-linejoin="round" />
        <path d="{TRACK_PATH_D}"
              fill="none" stroke="{BONE}" stroke-width="2" stroke-dasharray="6 6" />
        {ticks_svg}
        <line x1="{_mx - _perp[0] * 26:.1f}" y1="{_my - _perp[1] * 26:.1f}" x2="{_mx + _perp[0] * 26:.1f}" y2="{_my + _perp[1] * 26:.1f}" stroke="{INK}" stroke-width="4" />
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
        <p class="chart-label">Predicted vs. actual — Suzuka 2023</p>
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
