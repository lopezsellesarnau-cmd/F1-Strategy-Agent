"""
v2 model (Compound + TyreLife + LapNumber, RandomForest, trained on the
DELTA to each circuit's baseline pace) evaluated on Miami after training on
Jeddah + Bahrain, plus a simple pit-strategy simulation on that same test
circuit. Generates a self-contained HTML report (no server, no external JS)
styled like the portfolio (bone background, ink text, terracotta accent,
IBM Plex Mono) with an animated track showing the simulated strategies
racing against each other.

The track SVG is a real Miami International Autodrome silhouette, not a
generic rounded blob: sampled from the actual public track map (see
MIAMI_RAW_POINTS below) via build_track_path(), the same real-trace method
used for this project's earlier Suzuka report — a hand-drawn bezier path is
what caused that one to keep "getting worse" under manual editing, so this
one is generated from real geometry instead, never hand-tuned.
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

# "Industrial spec sheet" chrome (2 sept 2026) — tried two restyles after
# the original (warm Scandinavian, then black/white + Dross chrome) and
# neither stuck; reverted the chrome to the first version wholesale. The
# track itself moved on independently: first replaced the original's
# placeholder rounded-rectangle with a real Suzuka trace, then swapped that
# for Miami (this file) — same real-trace method, different circuit, for a
# new video without touching the model or the chrome.
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


# Miami International Autodrome's real centerline, sampled from the
# public-domain Wikimedia Commons track map ("Formula1 Circuit Miami Hard
# Rock Stadium.svg") the same way as the earlier Suzuka trace: 201 points,
# evenly spaced by arc length, extracted via the SVG's own
# getPointAtLength()/getScreenCTM() so the file's transforms are already
# baked in — not hand-drawn, not approximated. Miami doesn't cross itself
# like Suzuka does; its real signature is the tight, stop-start infield
# section around the stadium (the hairpin-heavy loop on one side) against
# the long DRS back straight on the other — that contrast is what the
# callouts below are chosen to show.
MIAMI_RAW_POINTS = [
    (388.9, 100), (399.87, 106.41), (410.83, 112.83), (421.8, 119.24), (432.77, 125.64),
    (443.75, 132.04), (454.73, 138.42), (465.67, 144.89), (475.19, 153.22), (475.67, 164.87),
    (466.5, 173.45), (455.81, 180.29), (448.38, 190.44), (446.39, 202.92), (445.96, 215.62),
    (443.99, 228.13), (437.49, 238.94), (427.79, 247.09), (416.52, 252.91), (404.51, 257.05),
    (392.14, 259.9), (379.48, 260.73), (366.78, 260.31), (354.15, 259.15), (341.91, 255.77),
    (330.25, 250.76), (319.18, 244.52), (308.14, 238.25), (297.09, 231.98), (286.04, 225.7),
    (275, 219.43), (263.95, 213.15), (252.9, 206.88), (241.86, 200.6), (230.81, 194.33),
    (218.9, 190.02), (206.3, 188.68), (193.78, 190.6), (182.1, 195.46), (171.79, 202.88),
    (160.78, 209.15), (148.28, 210.89), (135.73, 209.19), (124.58, 203.3), (115.19, 194.75),
    (105.85, 186.2), (94.28, 181.1), (81.68, 179.8), (69.07, 181.14), (56.85, 184.58),
    (45.61, 190.43), (36.32, 199.03), (29.45, 209.64), (27.4, 222.11), (30.88, 234.18),
    (39.53, 243.26), (52, 244.5), (64.51, 242.39), (77.2, 241.93), (89.83, 243.1),
    (101.97, 246.78), (114.14, 250.43), (126.52, 253.28), (139.04, 255.4), (151.71, 256.11),
    (164.41, 256.01), (177.12, 255.84), (189.82, 255.64), (202.52, 255.42), (215.22, 255.2),
    (227.93, 254.97), (240.63, 254.73), (253.33, 254.5), (266.03, 254.27), (278.73, 254.1),
    (291.41, 254.9), (303.9, 257.18), (315.98, 261.06), (327.59, 266.21), (339.18, 271.42),
    (350.86, 276.42), (362.83, 280.63), (375.03, 284.18), (387.49, 286.63), (400.14, 287.8),
    (412.84, 287.79), (425.53, 287.31), (438.21, 286.52), (450.86, 285.39), (463.48, 283.87),
    (476.01, 281.83), (488.41, 279.07), (500.5, 275.18), (512.48, 270.95), (524.46, 266.74),
    (536.45, 262.54), (548.44, 258.33), (560.42, 254.11), (572.39, 249.86), (584.35, 245.56),
    (596.28, 241.2), (608.16, 236.71), (619.92, 231.89), (631.49, 226.65), (642.98, 221.24),
    (654.42, 215.71), (665.81, 210.08), (677.13, 204.31), (688.32, 198.3), (698.77, 191.13),
    (699.53, 179.73), (689.52, 172.05), (678.53, 165.75), (670.13, 156.36), (668.25, 144.01),
    (673.55, 132.69), (684.82, 127.59), (697.53, 127.56), (710.23, 127.3), (722.28, 123.66),
    (730.75, 114.44), (736.86, 103.31), (736.48, 92.88), (730.83, 82.92), (734.35, 70.71),
    (737.39, 58.38), (736.55, 46.27), (724.27, 44.39), (711.57, 43.96), (698.88, 43.51),
    (686.18, 43.06), (673.48, 42.6), (660.79, 42.14), (648.09, 41.67), (635.4, 41.21),
    (622.7, 40.74), (610.01, 40.27), (597.31, 39.81), (584.62, 39.34), (571.92, 38.88),
    (559.22, 38.42), (546.53, 37.97), (533.83, 37.52), (521.14, 37.08), (508.44, 36.72),
    (495.74, 36.32), (483.04, 35.88), (470.35, 35.42), (457.65, 34.96), (444.96, 34.49),
    (432.26, 34.02), (419.57, 33.54), (406.87, 33.06), (394.18, 32.58), (381.48, 32.09),
    (368.79, 31.6), (356.09, 31.11), (343.4, 30.62), (330.7, 30.13), (318.01, 29.63),
    (305.32, 29.13), (292.62, 28.63), (279.93, 28.13), (267.23, 27.62), (254.54, 27.11),
    (241.85, 26.6), (229.15, 26.09), (216.46, 25.57), (203.77, 25.05), (191.07, 24.52),
    (178.38, 23.99), (165.69, 23.44), (153, 22.89), (140.3, 22.32), (127.62, 21.7),
    (115.38, 23.95), (113.47, 35.05), (122.54, 43.88), (132.91, 51.22), (143.77, 57.81),
    (155.41, 62.78), (168.01, 64.23), (180.62, 63.04), (192.25, 58.06), (203.24, 51.69),
    (214.74, 46.3), (226.73, 42.12), (239.17, 39.64), (251.85, 38.88), (264.55, 39.06),
    (277.16, 40.46), (289.37, 43.93), (301.04, 48.93), (312.18, 55.02), (323.16, 61.42),
    (334.12, 67.84), (345.08, 74.26), (356.04, 80.69), (367, 87.12), (377.95, 93.56),
    (388.9, 100),
]

# Landmark points, by index into MIAMI_RAW_POINTS — picked the same way as
# Suzuka's: by eye against the real trace, for the features a viewer
# actually recognizes (the stadium infield, the back straight's DRS zone,
# the fastest corner, a clean straight for start/finish).
MIAMI_LANDMARK_IDX = {"STADIUM SECTION": 53, "DRS ZONE": 174, "TURN 17": 125, "START / FINISH": 0}


def build_track_path(raw_points=MIAMI_RAW_POINTS, landmark_idx=MIAMI_LANDMARK_IDX, vb_w=900, vb_h=600, margin=34):
    """Places a real circuit trace (raw_points, defaulting to Miami) inside
    a vb_w×vb_h viewBox, self-correcting scale and offset until the combined
    footprint
    of the track *and* its callout labels sits centered with `margin`
    clearance on every side.

    A single scale+center computed from the track alone isn't enough: the
    callout leader lines and text extend past the track's own bounding box
    by different amounts per label ("START / FINISH" is a lot wider than
    "ESSES"), so centering only the track left both text clipped off the
    canvas edge and the whole composition reading as off-center — exactly
    the two bugs reported against the previous version. Fixed point loop:
    measure the combined bbox, rescale/recenter to fit it, repeat — a
    handful of iterations converges because the callout offsets scale down
    together with the track once the loop shrinks things to fit.
    """
    def place(scale, ox, oy):
        pts = [(ox + x * scale, oy + y * scale) for x, y in raw_points]
        n = len(pts)
        ccx = sum(p[0] for p in pts) / n
        ccy = sum(p[1] for p in pts) / n
        callouts = {}
        for label, idx in landmark_idx.items():
            px, py = pts[idx]
            dx, dy = px - ccx, py - ccy
            dl = math.hypot(dx, dy) or 1
            dx, dy = dx / dl, dy / dl
            lx, ly = px + dx * 34, py + dy * 34
            text_len = 34 + len(label) * 3.2
            tx = px + dx * (34 + text_len * 0.55)
            ty = py + dy * (34 + text_len * 0.55)
            anchor = "end" if dx < -0.15 else ("start" if dx > 0.15 else "middle")
            text_w = len(label) * 7.2
            if anchor == "end":
                tbox = (tx - text_w, ty - 12, tx, ty + 4)
            elif anchor == "start":
                tbox = (tx, ty - 12, tx + text_w, ty + 4)
            else:
                tbox = (tx - text_w / 2, ty - 12, tx + text_w / 2, ty + 4)
            callouts[label] = {"point": (px, py), "leader": (lx, ly),
                                "text": (tx, ty), "anchor": anchor, "tbox": tbox}
        return pts, callouts

    def combined_bbox(pts, callouts):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        for c in callouts.values():
            tb = c["tbox"]
            x0, x1 = min(x0, tb[0]), max(x1, tb[2])
            y0, y1 = min(y0, tb[1]), max(y1, tb[3])
        return x0, x1, y0, y1

    xs0 = [p[0] for p in raw_points]
    ys0 = [p[1] for p in raw_points]
    w0, h0 = max(xs0) - min(xs0), max(ys0) - min(ys0)
    scale = min((vb_w - 2 * margin) / w0, (vb_h - 2 * margin) / h0) * 0.72
    ox, oy = 0.0, 0.0
    for _ in range(8):
        pts, callouts = place(scale, ox, oy)
        x0, x1, y0, y1 = combined_bbox(pts, callouts)
        bw, bh = x1 - x0, y1 - y0
        fit_scale = min((vb_w - 2 * margin) / bw, (vb_h - 2 * margin) / bh)
        bcx, bcy = (x0 + x1) / 2, (y0 + y1) / 2
        ox = vb_w / 2 - (bcx - ox) * fit_scale
        oy = vb_h / 2 - (bcy - oy) * fit_scale
        scale *= fit_scale

    pts, callouts = place(scale, ox, oy)

    n = len(pts)
    d = f"M {pts[0][0]:.1f},{pts[0][1]:.1f} "
    for i in range(n):
        p0, p1, p2, p3 = pts[(i - 1) % n], pts[i % n], pts[(i + 1) % n], pts[(i + 2) % n]
        c1x, c1y = p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6
        c2x, c2y = p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6
        d += f"C {c1x:.1f},{c1y:.1f} {c2x:.1f},{c2y:.1f} {p2[0]:.1f},{p2[1]:.1f} "
    d += "Z"

    return d, callouts, scale, ox, oy


TRACK_PATH_D, TRACK_CALLOUTS, TRACK_SCALE, TRACK_OX, TRACK_OY = build_track_path()


def _placed(idx):
    """A raw Miami track point, mapped through the same scale/offset the fit
    loop converged on — for anything (like the tick marks) that needs a
    point *near* a landmark rather than exactly on it."""
    x, y = MIAMI_RAW_POINTS[idx % len(MIAMI_RAW_POINTS)]
    return TRACK_OX + x * TRACK_SCALE, TRACK_OY + y * TRACK_SCALE


print("Loading races...")
jeddah_df, jeddah_baseline = load_race(2023, "Jeddah")
bahrain_df, bahrain_baseline = load_race(2023, "Bahrain")
test_df, test_baseline = load_race(2023, "Miami")
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

# Distance-marker ticks either side of the start/finish point — purely
# decorative, gives the track a "blueprint" feel. Perpendicular to the
# *local* tangent there (a few raw points either side of the landmark
# index, not the direction toward some other landmark far along the
# track — using ESSES for that direction was what made the first pass of
# these ticks sit at the wrong angle to the actual road) and offset
# outward from the stroke so they never sit on top of it.
_sx, _sy = TRACK_CALLOUTS["START / FINISH"]["point"]
_sf_idx = MIAMI_LANDMARK_IDX["START / FINISH"]
_tx0, _ty0 = _placed(_sf_idx - 4)
_tx1, _ty1 = _placed(_sf_idx + 4)
_dxs, _dys = _tx1 - _tx0, _ty1 - _ty0
_slen = math.hypot(_dxs, _dys) or 1
_dxs, _dys = _dxs / _slen, _dys / _slen
_perp = (-_dys, _dxs)
# Sign the outward direction away from the track's own centroid (same rule
# the callouts use), so the ticks land on the empty side of the straight.
_ccx = sum(p[0] for p in [TRACK_CALLOUTS[k]["point"] for k in TRACK_CALLOUTS]) / len(TRACK_CALLOUTS)
_ccy = sum(p[1] for p in [TRACK_CALLOUTS[k]["point"] for k in TRACK_CALLOUTS]) / len(TRACK_CALLOUTS)
if (_sx - _ccx) * _perp[0] + (_sy - _ccy) * _perp[1] < 0:
    _perp = (-_perp[0], -_perp[1])
# Dropped the perpendicular blueprint-style tick marks that used to sit
# here — one thin black trace reads as more minimal without them; the
# geometry (_sx/_sy/_perp) stays, since the start/finish marker line below
# still needs it.
ticks_svg = ""

# Callouts (dot + leader line + label) — every point, leader endpoint, text
# position and text-anchor comes straight out of TRACK_CALLOUTS, computed by
# build_track_path()'s fit loop so the label can never land on the road or
# clip off the canvas (the two bugs in the previous version) regardless of
# how the track ends up scaled.
_callout_parts = []
for _label, _c in TRACK_CALLOUTS.items():
    _px, _py = _c["point"]
    _lx, _ly = _c["leader"]
    _tx, _ty = _c["text"]
    _callout_parts.append(f"""
  <circle cx="{_px:.1f}" cy="{_py:.1f}" r="3" fill="{ACCENT if _label != "BRIDGE" else INK}" />
  <line x1="{_px:.1f}" y1="{_py:.1f}" x2="{_lx:.1f}" y2="{_ly:.1f}" stroke="{ACCENT if _label != "BRIDGE" else INK}" stroke-width="1.5" />
  <text x="{_tx:.1f}" y="{_ty:.1f}" font-family="'IBM Plex Mono', monospace" font-size="13" fill="{INK}" letter-spacing="1" text-anchor="{_c["anchor"]}">{_label}</text>""")
callouts_svg = "\n".join(_callout_parts)

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
    font-size: 12px; letter-spacing: 0.22em; text-transform: uppercase; font-weight: 500;
  }}

  .wrap {{ flex: 1; min-width: 0; padding: 28px 32px 96px; }}

  /* ── Header meta bar — mirrors the "P1 INDUSTRIAL ROBOT / SERIAL NO." row ── */
  .meta-bar {{
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;
    border-bottom: 1px solid {INK}; padding-bottom: 10px; margin-bottom: 28px;
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.14em; color: rgba(21,20,18,0.6);
  }}

  h1 {{ font-size: 34px; font-weight: 500; letter-spacing: -0.015em; margin: 0 0 8px; }}
  .sub {{ color: rgba(21,20,18,0.62); font-size: 13px; margin: 0 0 32px; max-width: 78ch; line-height: 1.5; }}
  h2 {{
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.16em; font-weight: 500;
    color: {INK}; margin: 44px 0 14px; border-bottom: 1px solid {INK};
    padding-bottom: 10px;
  }}

  .metric-row {{ display: flex; gap: 0; flex-wrap: wrap; border: 1px solid {INK}; border-right: none; }}
  .metric {{
    border-right: 1px solid {INK}; background: #fff; padding: 16px 18px; flex: 1; min-width: 240px;
  }}
  .metric .label {{ font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: rgba(21,20,18,0.6); }}
  .metric .value {{ font-size: 28px; font-weight: 500; margin-top: 10px; font-variant-numeric: tabular-nums; }}
  .metric.win {{ background: {INK}; }}
  .metric.win .label {{ color: rgba(240,238,233,0.65); }}
  .metric.win .value {{ color: {ACCENT}; }}

  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 860px) {{ .charts {{ grid-template-columns: 1fr; }} .page {{ flex-direction: column; }} .rail {{ writing-mode: horizontal-tb; transform: none; width: auto; height: 32px; }} }}
  .chart-label {{
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em; color: rgba(21,20,18,0.6);
    margin: 0 0 8px; border-top: 1px solid {INK}; padding-top: 8px;
  }}
  img {{ width: 100%; border: 1px solid {INK}; background: #fff; display: block; }}

  /* ── Corner-bracket frame — the viewfinder-corner trick from the ref shots ── */
  .frame {{ position: relative; border: 1px solid {INK}; background: #fff; padding: 24px; }}
  .cnr {{ position: absolute; width: 16px; height: 16px; pointer-events: none; }}
  .cnr-tl {{ top: -2px; left: -2px; border-top: 2px solid {ACCENT}; border-left: 2px solid {ACCENT}; }}
  .cnr-tr {{ top: -2px; right: -2px; border-top: 2px solid {ACCENT}; border-right: 2px solid {ACCENT}; }}
  .cnr-bl {{ bottom: -2px; left: -2px; border-bottom: 2px solid {ACCENT}; border-left: 2px solid {ACCENT}; }}
  .cnr-br {{ bottom: -2px; right: -2px; border-bottom: 2px solid {ACCENT}; border-right: 2px solid {ACCENT}; }}

  .track-svg {{ width: 100%; height: auto; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 18px; margin-top: 18px; padding-top: 14px; border-top: 1px solid {LINE}; font-size: 12px; }}
  .legend-item {{ display: flex; align-items: center; gap: 8px; }}
  .swatch {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex-shrink: 0; margin-right: 8px; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: #fff; border: 1px solid {INK}; margin-top: 16px; }}
  th {{
    text-align: left; padding: 10px 14px; background: {INK}; color: {BONE};
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 500;
  }}
  td {{ text-align: left; padding: 10px 14px; border-bottom: 1px solid {LINE}; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr.best {{ background: rgba(193,102,61,0.08); font-weight: 500; }}
  .chip {{
    display: inline-block; font-size: 9px; background: {ACCENT}; color: {BONE};
    padding: 2px 6px; margin-left: 6px; letter-spacing: 0.08em;
  }}
  footer {{
    margin-top: 56px; font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em;
    color: rgba(21,20,18,0.5); border-top: 1px solid {INK}; padding-top: 12px;
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
    <p class="sub">Assumes the circuit's baseline pace is known ahead of time (e.g. from practice/qualifying — here taken from the real race: {test_baseline:.2f}s). The model only predicts how far each lap drifts from that pace, based on compound and tyre wear. Each car laps the track once, at a speed proportional to its simulated total race time — first one home wins.</p>

    <div class="frame">
      <span class="cnr cnr-tl"></span><span class="cnr cnr-tr"></span>
      <span class="cnr cnr-bl"></span><span class="cnr cnr-br"></span>
      <svg class="track-svg" viewBox="0 0 900 600" xmlns="http://www.w3.org/2000/svg">
        <path id="track" d="{TRACK_PATH_D}"
              fill="none" stroke="{INK}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />
        {ticks_svg}
        <line x1="{_sx - _perp[0] * 26:.1f}" y1="{_sy - _perp[1] * 26:.1f}" x2="{_sx + _perp[0] * 26:.1f}" y2="{_sy + _perp[1] * 26:.1f}" stroke="{INK}" stroke-width="4" />
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
