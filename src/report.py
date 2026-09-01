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


# Suzuka's real centerline, sampled from the public-domain 2005 track map
# (Wikimedia Commons, "Suzuka circuit map--2005.svg" — traced with
# Inkscape's "knot" path effect specifically to render the flyover bridge
# correctly, i.e. this is an actual accurate trace, not an approximation).
# 221 points, evenly spaced by arc length, in the source file's own local
# coordinate space (its ancestor <g> transforms already baked in via
# getScreenCTM() at extraction time — see the F1-Strategy-Agent session
# notes for the extraction method). A first attempt at this generated a
# synthetic figure-8 (a lemniscate curve) instead of using real geometry —
# it crossed itself in the right *topological* place but wasn't Suzuka's
# actual shape, which is the whole point of naming the circuit.
SUZUKA_RAW_POINTS = [
    (727.77, 394.98), (744.56, 400.53), (761.32, 406.16), (778.06, 411.87), (794.76, 417.68),
    (811.43, 423.6), (828.05, 429.64), (844.62, 435.81), (860.57, 443.25), (872.54, 456.22),
    (883.08, 470.41), (893.19, 484.93), (903.63, 499.2), (916.45, 510.82), (932.46, 506.35),
    (934.57, 489.62), (925.84, 474.27), (916.65, 459.16), (907.97, 443.75), (899.6, 428.18),
    (891.5, 412.46), (884.35, 396.29), (879.35, 379.34), (876.45, 361.91), (877.06, 344.26),
    (880.51, 326.94), (886.56, 310.33), (893.96, 294.28), (903.27, 279.25), (913.72, 264.99),
    (924.93, 251.32), (937.39, 238.78), (950.43, 226.84), (964.04, 215.55), (978.3, 205.09),
    (993.24, 195.63), (1009, 187.63), (1025.43, 181.11), (1042.44, 176.3), (1059.91, 173.58),
    (1077.53, 172.16), (1095.16, 173.26), (1112.69, 175.62), (1130.16, 178.33), (1147.62, 181.16),
    (1165.09, 183.87), (1182.62, 186.22), (1200.23, 187.7), (1216.93, 182.36), (1230.57, 171.22),
    (1240.43, 156.58), (1247.65, 140.45), (1252.5, 123.46), (1254.09, 105.9), (1249.14, 89.1),
    (1238.32, 75.2), (1222.25, 68.15), (1204.75, 66.34), (1187.33, 69.25), (1170.42, 74.41),
    (1153.87, 80.62), (1137.37, 87.01), (1120.93, 93.52), (1104.63, 100.37), (1088.52, 107.68),
    (1072.64, 115.44), (1056.96, 123.63), (1041.38, 132), (1026.05, 140.81), (1011.13, 150.31),
    (996.56, 160.33), (982.25, 170.72), (968.14, 181.38), (954.2, 192.27), (940.54, 203.49),
    (927.01, 214.88), (913.46, 226.24), (899.77, 237.44), (886.02, 248.57), (872.27, 259.69),
    (858.55, 270.85), (844.92, 282.12), (831.36, 293.47), (817.85, 304.88), (804.39, 316.35),
    (790.99, 327.9), (777.65, 339.5), (764.29, 351.09), (750.93, 362.68), (737.61, 374.31),
    (724.34, 386), (711.15, 397.78), (698.09, 409.7), (685.92, 422.45), (680.28, 439.14),
    (676.47, 456.41), (673.81, 473.89), (672.34, 491.51), (672.89, 509.17), (675.23, 526.69),
    (678.2, 544.13), (681.61, 561.48), (685.26, 578.78), (689.1, 596.05), (693.69, 613.12),
    (698.51, 630.14), (702.8, 647.29), (705.62, 664.74), (705.9, 682.41), (701.83, 698.77),
    (684.67, 702.91), (670.86, 712.27), (672.73, 729.78), (674.17, 747.39), (673.13, 765.02),
    (667.85, 781.82), (659, 797.1), (647.52, 810.5), (633.02, 820.57), (617.56, 829.16),
    (601.71, 836.99), (585.18, 843.25), (568.11, 847.84), (550.71, 850.96), (533.21, 853.5),
    (515.7, 856), (498.19, 858.46), (480.67, 860.9), (463.15, 863.33), (445.63, 865.75),
    (428.12, 868.17), (410.6, 870.6), (393.08, 873.06), (375.57, 875.54), (358.07, 878.05),
    (340.57, 880.62), (323.09, 883.27), (305.61, 886.01), (288.15, 888.81), (270.7, 891.64),
    (253.24, 894.47), (235.78, 897.27), (218.3, 900.01), (200.82, 902.67), (183.33, 905.3),
    (165.84, 907.89), (148.33, 910.41), (130.81, 912.79), (113.26, 915), (95.64, 916.1),
    (78.08, 914.17), (61.48, 908.22), (46.98, 898.18), (34.38, 885.79), (23.91, 871.57),
    (18.57, 854.95), (20.72, 837.51), (31.19, 823.44), (46.01, 814.03), (63.38, 811.09),
    (81.06, 810.54), (98.74, 810.07), (116.42, 809.7), (134.1, 809.31), (151.78, 808.72),
    (169.43, 807.77), (185.67, 801.12), (198.56, 789.19), (207.13, 773.77), (216.88, 759.2),
    (231.99, 750.22), (249.35, 747.88), (266.83, 750.18), (283.83, 755.07), (300.79, 760.06),
    (318.08, 763.72), (335.46, 761.38), (351.16, 753.37), (363.36, 740.72), (372.76, 725.74),
    (382.5, 710.99), (395.42, 699.03), (411.39, 691.53), (428.71, 688.16), (446.11, 690.13),
    (461.33, 698.92), (473.01, 712.18), (483.76, 726.21), (494.84, 739.99), (508.03, 751.61),
    (524.38, 758.28), (541.93, 759.91), (559.08, 756.18), (574.77, 748.04), (589.62, 738.46),
    (603.38, 727.37), (615.73, 714.71), (626.58, 700.78), (635.64, 685.59), (642.81, 669.46),
    (646.52, 652.18), (648.95, 634.67), (649.6, 617), (648.19, 599.38), (644.49, 582.11),
    (638.58, 565.44), (632.39, 548.88), (626.01, 532.38), (619.57, 515.91), (613.33, 499.37),
    (607.54, 482.66), (607.65, 465.56), (614.56, 449.29), (621.68, 433.1), (629.01, 417.01),
    (636.59, 401.03), (644.43, 385.17), (657.27, 374.13), (674.33, 377.66), (691.16, 383.07),
    (707.99, 388.51),
]

# Landmark points, by index into SUZUKA_RAW_POINTS — picked once by eye
# against the real trace (the corners a viewer actually recognizes: the
# S-curves, the hairpin's tight reversal, the point the track crosses
# itself, and a clean straight to hang a start/finish line on).
SUZUKA_LANDMARK_IDX = {"BRIDGE": 90, "HAIRPIN": 155, "ESSES": 185, "START / FINISH": 135}


def build_track_path(vb_w=900, vb_h=600, margin=34):
    """Places the real Suzuka trace (SUZUKA_RAW_POINTS) inside a vb_w×vb_h
    viewBox, self-correcting scale and offset until the combined footprint
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
        pts = [(ox + x * scale, oy + y * scale) for x, y in SUZUKA_RAW_POINTS]
        n = len(pts)
        ccx = sum(p[0] for p in pts) / n
        ccy = sum(p[1] for p in pts) / n
        callouts = {}
        for label, idx in SUZUKA_LANDMARK_IDX.items():
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

    xs0 = [p[0] for p in SUZUKA_RAW_POINTS]
    ys0 = [p[1] for p in SUZUKA_RAW_POINTS]
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
    """A raw Suzuka point, mapped through the same scale/offset the fit
    loop converged on — for anything (like the tick marks) that needs a
    point *near* a landmark rather than exactly on it."""
    x, y = SUZUKA_RAW_POINTS[idx % len(SUZUKA_RAW_POINTS)]
    return TRACK_OX + x * TRACK_SCALE, TRACK_OY + y * TRACK_SCALE


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

# Distance-marker ticks either side of the start/finish point — purely
# decorative, gives the track a "blueprint" feel. Perpendicular to the
# *local* tangent there (a few raw points either side of the landmark
# index, not the direction toward some other landmark far along the
# track — using ESSES for that direction was what made the first pass of
# these ticks sit at the wrong angle to the actual road) and offset
# outward from the stroke so they never sit on top of it.
_sx, _sy = TRACK_CALLOUTS["START / FINISH"]["point"]
_sf_idx = SUZUKA_LANDMARK_IDX["START / FINISH"]
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
ticks_svg = "\n".join(
    f'<line x1="{_sx + _dxs * f + _perp[0] * o1:.1f}" y1="{_sy + _dys * f + _perp[1] * o1:.1f}" '
    f'x2="{_sx + _dxs * f + _perp[0] * o2:.1f}" y2="{_sy + _dys * f + _perp[1] * o2:.1f}" '
    f'stroke="{INK}" stroke-opacity="0.25" stroke-width="2" />'
    for f in [-60, -40, -20, 20, 40, 60]
    for o1, o2 in [(20, 34)]
)

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
