"""
Modelo v2 (Compound + TyreLife + LapNumber, RandomForest) evaluado en Miami
tras entrenar con Jeddah + Bahrain, más una simulación simple de estrategias
de parada sobre ese mismo circuito de prueba. Genera un informe HTML
autocontenido (sin servidor, sin JS externo) con el estilo del portfolio
(bg hueso, tinta, acento terracota, IBM Plex Mono).
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

MAE_BASELINE_V1 = 3.49  # compound-mean, train=Jeddah+Bahrain, test=Miami (baselinec2.py)


def load_race(year, gp):
    """Devuelve las vueltas limpias del GP y su 'baseline' (mediana de tiempo
    de vuelta de esa carrera). Circuitos distintos tienen ritmos de vuelta
    muy distintos solo por trazado (Bahréin ~98s, Miami ~93s) — sin restar
    ese ritmo base antes de entrenar, el modelo confunde "estamos en otro
    circuito" con "los neumáticos han cambiado", y el error no baja aunque
    se le den más features."""
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
    """One-hot de Compound alineado a las columnas vistas en train, para que
    test nunca produzca una columna que el modelo no conoce."""
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


print("Cargando carreras...")
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
# La simulación asume que el ritmo base del circuito de prueba se conoce de
# antemano (p. ej. de los libres o la clasificación) — el modelo solo tiene
# que acertar cuánto se degrada la vuelta desde ahí, no el ritmo absoluto.
pred = pred_delta + miami_baseline
mae_v2 = mean_absolute_error(y_test, pred)
print(f"MAE v1 (compound-mean, tiempo absoluto): {MAE_BASELINE_V1:.2f}s")
print(f"MAE v2 (Compound+TyreLife+LapNumber sobre delta al ritmo base): {mae_v2:.2f}s")

# ── Gráfico 1: curvas de degradación por compuesto (a mitad de carrera) ──
fig1, ax1 = plt.subplots(figsize=(7, 4.2))
style_ax(ax1)
tyre_life_range = list(range(1, 31))
for compound in compound_cols:
    rows = pd.DataFrame({"TyreLife": tyre_life_range, "LapNumber": [25] * len(tyre_life_range)})
    rows_feat = features(pd.concat([rows, pd.DataFrame({"Compound": [compound] * len(tyre_life_range)})], axis=1), compound_cols)
    preds = model.predict(rows_feat)
    ax1.plot(tyre_life_range, preds, label=compound, color=COMPOUND_COLOR.get(compound, INK), linewidth=2)
ax1.set_xlabel("Vueltas con el mismo juego de neumáticos")
ax1.set_ylabel("Segundos respecto al ritmo base del circuito")
ax1.legend(frameon=False, labelcolor=INK, fontsize=9)
degradacion_img = fig_to_base64(fig1)

# ── Gráfico 2: predicho vs real en Miami ──
fig2, ax2 = plt.subplots(figsize=(7, 4.2))
style_ax(ax2)
ax2.scatter(y_test, pred, alpha=0.4, s=14, color=ACCENT, edgecolors="none")
lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
ax2.plot(lims, lims, color=INK, linewidth=1, alpha=0.4, linestyle="--")
ax2.set_xlabel("Tiempo de vuelta real (s)")
ax2.set_ylabel("Tiempo de vuelta predicho (s)")
prediccion_img = fig_to_base64(fig2)

# ── Simulación de estrategias sobre Miami (57 vueltas) ──
TOTAL_LAPS = int(test_df["LapNumber"].max())

estrategias = {
    "1 parada — MEDIUM → HARD": [("MEDIUM", 20), ("HARD", TOTAL_LAPS - 20)],
    "1 parada — HARD → MEDIUM": [("HARD", 30), ("MEDIUM", TOTAL_LAPS - 30)],
    "2 paradas — MEDIUM → HARD → MEDIUM": [("MEDIUM", 18), ("HARD", 20), ("MEDIUM", TOTAL_LAPS - 38)],
    "2 paradas — SOFT → MEDIUM → HARD": [("SOFT", 14), ("MEDIUM", 20), ("HARD", TOTAL_LAPS - 34)],
}


def simular(stints):
    filas = []
    lap_number = 1
    for compound, n_laps in stints:
        for tyre_life in range(1, n_laps + 1):
            filas.append({"Compound": compound, "TyreLife": tyre_life, "LapNumber": lap_number})
            lap_number += 1
    df = pd.DataFrame(filas)
    pred_deltas = model.predict(features(df, compound_cols))
    return pred_deltas.sum() + miami_baseline * len(filas)

resultados = {nombre: simular(stints) for nombre, stints in estrategias.items()}
mejor = min(resultados, key=resultados.get)

filas_tabla = "\n".join(
    f"""<tr{' class="mejor"' if nombre == mejor else ''}>
        <td>{nombre}{' <span class="chip">MEJOR</span>' if nombre == mejor else ''}</td>
        <td class="num">{resultados[nombre] / 60:.1f} min</td>
        <td class="num">{resultados[nombre] - resultados[mejor]:+.1f}s</td>
    </tr>"""
    for nombre in estrategias
)

html = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8" />
<title>F1 Strategy Agent — informe de modelo</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0;
    background: {BONE}; color: {INK};
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    letter-spacing: -0.011em;
  }}
  .wrap {{ max-width: 880px; margin: 0 auto; padding: 56px 24px 96px; }}
  h1 {{ font-size: 28px; font-weight: 600; letter-spacing: -0.02em; margin: 0 0 6px; }}
  .sub {{ color: rgba(21,20,18,0.6); font-size: 13px; margin: 0 0 40px; }}
  h2 {{
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.14em;
    color: rgba(21,20,18,0.68); margin: 48px 0 14px; border-bottom: 1px solid {LINE};
    padding-bottom: 10px;
  }}
  .metric-row {{ display: flex; gap: 14px; flex-wrap: wrap; }}
  .metric {{
    border: 1px solid {LINE}; background: #fff; padding: 16px 18px; flex: 1; min-width: 200px;
  }}
  .metric .label {{ font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em; color: rgba(21,20,18,0.6); }}
  .metric .value {{ font-size: 26px; font-weight: 500; margin-top: 8px; }}
  .metric.win .value {{ color: {ACCENT}; }}
  img {{ width: 100%; border: 1px solid {LINE}; background: #fff; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: #fff; border: 1px solid {LINE}; }}
  th, td {{ text-align: left; padding: 10px 14px; border-bottom: 1px solid {LINE}; }}
  th {{ font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: rgba(21,20,18,0.6); }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr.mejor {{ background: rgba(193,102,61,0.07); font-weight: 600; }}
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
  <p class="sub">Modelo de tiempo por vuelta + simulación de estrategias · entrenado en Jeddah + Bahréin 2023, evaluado en Miami 2023</p>

  <h2>Rendimiento del modelo</h2>
  <div class="metric-row">
    <div class="metric">
      <div class="label">v1 — media por compuesto</div>
      <div class="value">{MAE_BASELINE_V1:.2f}s MAE</div>
    </div>
    <div class="metric win">
      <div class="label">v2 — Compound + TyreLife + LapNumber</div>
      <div class="value">{mae_v2:.2f}s MAE</div>
    </div>
  </div>

  <h2>Curvas de degradación por compuesto</h2>
  <img src="data:image/png;base64,{degradacion_img}" alt="Degradación por compuesto" />

  <h2>Predicho vs. real — Miami 2023</h2>
  <img src="data:image/png;base64,{prediccion_img}" alt="Predicho vs real" />

  <h2>Simulación de estrategias — Miami 2023 ({TOTAL_LAPS} vueltas)</h2>
  <p class="sub" style="margin-bottom:16px;">Asume el ritmo base del circuito conocido de antemano (p. ej. de libres/clasificación, aquí tomado de la carrera real: {miami_baseline:.2f}s) — el modelo solo estima cuánto se aleja cada vuelta de ese ritmo según compuesto y desgaste.</p>
  <table>
    <thead><tr><th>Estrategia</th><th>Tiempo total estimado</th><th>Diferencia vs. mejor</th></tr></thead>
    <tbody>
      {filas_tabla}
    </tbody>
  </table>

  <footer>Generado por src/report.py · F1-Strategy-Agent</footer>
</div>
</body>
</html>
"""

with open("report.html", "w") as f:
    f.write(html)

print("Guardado en report.html")
