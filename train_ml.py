"""fpl-ml — model v0.2 (ML, day one).

Framing: wiersz = (zawodnik, mecz w sezonie t). Features znane PRZED meczem:
agregaty zawodnika z sezonu t-1 (pkt/90, minuty, xG/90, xA/90, starty),
cena, pozycja, dom/wyjazd, sila rywala i wlasna, nr kolejki.
Target: punkty FPL w tym meczu. Model: HistGradientBoosting (sklearn),
natywna obsluga NaN => cold start (nowi w PL) obslugiwany przez model.
Walidacja: time split — trening na parze 23/24->24/25, test na 24/25->25/26.
"""
import unicodedata
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pulp
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

POS_CODE = {"GK": 0, "GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
FEATS = ["prev_p90", "prev_mins", "prev_xg90", "prev_xa90", "prev_start_share",
         "price", "pos_code", "was_home", "opp_strength", "own_strength", "gw"]


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def season_agg(gw_file):
    """Agregaty zawodnika z calego sezonu (features na sezon nastepny)."""
    d = pd.read_csv(gw_file)
    g = d.groupby(d["name"].map(norm)).agg(
        pts=("total_points", "sum"), mins=("minutes", "sum"),
        xg=("expected_goals", "sum"), xa=("expected_assists", "sum"),
        starts=("starts", "sum"), games=("total_points", "size")).reset_index()
    p90 = g["mins"].clip(lower=720) / 90.0
    return pd.DataFrame({
        "key": g["name"], "prev_p90": g["pts"] / p90, "prev_mins": g["mins"],
        "prev_xg90": g["xg"] / p90, "prev_xa90": g["xa"] / p90,
        "prev_start_share": g["starts"] / g["games"], "prev_pts_total": g["pts"]})


def build_rows(gw_file, teams_file, prev_agg):
    """Wiersze treningowe sezonu t z features z sezonu t-1."""
    d = pd.read_csv(gw_file)
    t = pd.read_csv(teams_file)[["id", "name", "strength_overall_home",
                                 "strength_overall_away"]]
    d["key"] = d["name"].map(norm)
    d = d.merge(t.add_prefix("opp_"), left_on="opponent_team", right_on="opp_id")
    d = d.merge(t.add_prefix("own_"), left_on="team", right_on="own_name")
    d["opp_strength"] = np.where(d["was_home"], d["opp_strength_overall_away"],
                                 d["opp_strength_overall_home"])
    d["own_strength"] = np.where(d["was_home"], d["own_strength_overall_home"],
                                 d["own_strength_overall_away"])
    d = d.merge(prev_agg, on="key", how="left")  # NaN = brak historii (cold start)
    d["price"] = d["value"] / 10.0
    d["pos_code"] = d["position"].map(POS_CODE)
    d["was_home"] = d["was_home"].astype(int)
    d["gw"] = d["round"]
    return d


def heuristic_v01(row):
    if np.isnan(row["prev_pts_total"]):
        return 1.0
    games = row["prev_mins"] / 90.0
    return (row["prev_pts_total"] / max(games, 8)) * min(row["prev_mins"] / 2700, 1)


def main():
    agg23 = season_agg("gw_2023-24.csv")
    agg24 = season_agg("gw_2024-25.csv")
    agg25 = season_agg("gw_2025-26.csv")

    train = build_rows("gw_2024-25.csv", "teams_2024-25.csv", agg23)
    test = build_rows("gw_2025-26.csv", "teams_2025-26.csv", agg24)

    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                          max_depth=6, random_state=42)
    model.fit(train[FEATS], train["total_points"])

    # --- ewaluacja na odlozonym sezonie 25/26 ---
    pred = model.predict(test[FEATS])
    mae_ml = mean_absolute_error(test["total_points"], pred)
    naive = (test["prev_pts_total"].fillna(38) / 38.0)
    mae_naive = mean_absolute_error(test["total_points"], naive)
    heur = test.apply(heuristic_v01, axis=1)
    mae_heur = mean_absolute_error(test["total_points"], heur)
    print(f"MAE na sezonie 25/26 (holdout, {len(test)} meczo-wierszy):")
    print(f"  naive (pkt_zeszly/38):  {mae_naive:.3f}")
    print(f"  heurystyka v0.1:        {mae_heur:.3f}")
    print(f"  ML v0.2 (HGB):          {mae_ml:.3f}")

    # --- retrain na wszystkim i predykcja GW1 26/27 ---
    full = pd.concat([train, test], ignore_index=True)
    model.fit(full[FEATS], full["total_points"])

    cur = pd.read_csv("players_2627.csv")
    teams = pd.read_csv("teams_2627.csv")
    fx = pd.read_csv("fixtures_2627.csv")
    fx1 = fx[fx["event"] == 1]

    cur["key"] = (cur["first_name"] + " " + cur["second_name"]).map(norm)
    cur["pos"] = cur["element_type"].map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"})
    cur = cur.merge(teams[["id", "short_name", "strength_overall_home",
                           "strength_overall_away"]],
                    left_on="team", right_on="id", suffixes=("", "_t"))

    home = fx1[["team_h", "team_a"]].rename(columns={"team_h": "team", "team_a": "opp"})
    home["was_home"] = 1
    away = fx1[["team_a", "team_h"]].rename(columns={"team_a": "team", "team_h": "opp"})
    away["was_home"] = 0
    gw1 = pd.concat([home, away])
    cur = cur.merge(gw1, on="team", how="left")
    st = teams.set_index("id")
    cur["opp_strength"] = np.where(cur["was_home"] == 1,
                                   st.loc[cur["opp"], "strength_overall_away"].values,
                                   st.loc[cur["opp"], "strength_overall_home"].values)
    cur["own_strength"] = np.where(cur["was_home"] == 1,
                                   cur["strength_overall_home"],
                                   cur["strength_overall_away"])
    cur = cur.merge(agg25, on="key", how="left")
    cur["price"] = cur["now_cost"] / 10.0
    cur["pos_code"] = cur["pos"].map(POS_CODE)
    cur["gw"] = 1

    cur["pred"] = model.predict(cur[FEATS]).clip(min=0)
    chance = cur["chance_of_playing_next_round"]
    cur.loc[cur["status"].isin(["i", "s", "u", "n"]), "pred"] = 0.0
    doubt = cur["status"].eq("d")
    cur.loc[doubt, "pred"] *= chance[doubt].fillna(50) / 100.0

    # --- optymalizacja (jak v0.1) ---
    df = cur.reset_index(drop=True)
    idx = df.index
    x = pulp.LpVariable.dicts("squad", idx, cat="Binary")
    y = pulp.LpVariable.dicts("xi", idx, cat="Binary")
    m = pulp.LpProblem("fpl_gw1_ml", pulp.LpMaximize)
    m += pulp.lpSum(y[i] * df.pred[i] for i in idx) + \
         0.1 * pulp.lpSum((x[i] - y[i]) * df.pred[i] for i in idx)
    m += pulp.lpSum(x[i] for i in idx) == 15
    m += pulp.lpSum(y[i] for i in idx) == 11
    m += pulp.lpSum(x[i] * df.now_cost[i] for i in idx) <= 1000
    for i in idx:
        m += y[i] <= x[i]
    for p, q in {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}.items():
        m += pulp.lpSum(x[i] for i in idx if df.pos[i] == p) == q
    for p, (lo, hi) in {"GK": (1, 1), "DEF": (3, 5),
                        "MID": (2, 5), "FWD": (1, 3)}.items():
        m += pulp.lpSum(y[i] for i in idx if df.pos[i] == p) >= lo
        m += pulp.lpSum(y[i] for i in idx if df.pos[i] == p) <= hi
    for t_ in df["short_name"].unique():
        m += pulp.lpSum(x[i] for i in idx if df.short_name[i] == t_) <= 3
    m.solve(pulp.PULP_CBC_CMD(msg=0))
    assert pulp.LpStatus[m.status] == "Optimal"

    df["in_squad"] = [int(x[i].value()) for i in idx]
    df["in_xi"] = [int(y[i].value()) for i in idx]
    sq = df[df.in_squad == 1].copy()
    sq["role"] = sq["in_xi"].map({1: "XI", 0: "bench"})
    cap = sq.loc[sq[sq.in_xi == 1]["pred"].idxmax(), "web_name"]
    sq.loc[sq.web_name == cap, "role"] = "XI (C)"
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    sq = sq.sort_values(["in_xi", "pos", "pred"], ascending=[False, True, False],
                        key=lambda c: c.map(order) if c.name == "pos" else c)

    print(f"\n=== ML v0.2 | GW1 | koszt {sq.now_cost.sum()/10:.1f}m ===")
    for _, r in sq.iterrows():
        print(f"{r.role:8} {r.pos:3} {r.web_name:20} {r.short_name}  "
              f"{r.now_cost/10:.1f}m  pred {r.pred:.2f}")
    xi_pts = sq[sq.in_xi == 1].pred.sum() + sq[sq.role == 'XI (C)'].pred.iloc[0]
    print(f"\nKapitan: {cap} | pred XI (z kapitanem): {xi_pts:.1f} pkt")

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open("predictions_gw1_ml.csv", "w") as f:
        f.write(f"# generated_utc: {stamp} | model v0.2 ML | "
                f"holdout MAE: ml={mae_ml:.3f} heur={mae_heur:.3f} "
                f"naive={mae_naive:.3f}\n")
        sq[["web_name", "short_name", "pos", "now_cost", "pred",
            "role"]].to_csv(f, index=False)
    print(f"Zapisano predictions_gw1_ml.csv ({stamp} UTC)")


if __name__ == "__main__":
    main()
