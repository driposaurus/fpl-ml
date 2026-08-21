"""fpl-ml — shared feature engineering + squad optimizer (v0.3).

Features per (player, match) row, all known BEFORE kickoff:
- prev-season aggregates: pts/90, minutes, xG/90, xA/90, start share
- in-season rolling form (shifted, no leakage): pts r3/r5/r10, minutes r3/r5,
  xG r5, xA r5, starts r5
- team form from actual goals (rolling 5, shifted): own/opp attack & defence
- static API strengths, price, position, home/away, gameweek number
"""
import unicodedata

import numpy as np
import pandas as pd
import pulp

POS_CODE = {"GK": 0, "GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
POS_NAME = {0: "GK", 1: "DEF", 2: "MID", 3: "FWD"}

FEATS = [
    "prev_p90", "prev_mins", "prev_xg90", "prev_xa90", "prev_start_share",
    "pts_r3", "pts_r5", "pts_r10", "mins_r3", "mins_r5",
    "xg_r5", "xa_r5", "starts_r5",
    "own_att_r5", "own_def_r5", "opp_att_r5", "opp_def_r5",
    "own_strength", "opp_strength", "price", "pos_code", "was_home", "gw",
]


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def season_agg(gw_file):
    d = pd.read_csv(gw_file)
    for c in ("expected_goals", "expected_assists", "starts"):
        if c not in d:
            d[c] = np.nan
    g = d.groupby(d["name"].map(norm)).agg(
        pts=("total_points", "sum"), mins=("minutes", "sum"),
        xg=("expected_goals", "sum"), xa=("expected_assists", "sum"),
        starts=("starts", "sum"), games=("total_points", "size")).reset_index()
    p90 = g["mins"].clip(lower=720) / 90.0
    return pd.DataFrame({
        "key": g["name"], "prev_p90": g["pts"] / p90, "prev_mins": g["mins"],
        "prev_xg90": g["xg"] / p90, "prev_xa90": g["xa"] / p90,
        "prev_start_share": g["starts"] / g["games"],
        "prev_pts_total": g["pts"]})


def build_season(gw_path, teams_path, prev_agg=None, extra_rows=None):
    """Feature table for one season. extra_rows: future fixtures to score
    (same columns, NaN target) appended before rolling computation."""
    d = pd.read_csv(gw_path) if gw_path else pd.DataFrame()
    if extra_rows is not None:
        d = pd.concat([d, extra_rows], ignore_index=True)
    for c in ("expected_goals", "expected_assists", "starts"):
        if c not in d:
            d[c] = np.nan

    t = pd.read_csv(teams_path)[["id", "name", "strength_overall_home",
                                 "strength_overall_away"]]
    name2id = dict(zip(t["name"], t["id"]))

    d["key"] = d["name"].map(norm)
    d["team_id"] = d["team"].map(name2id)
    d["pos_code"] = d["position"].map(POS_CODE)
    d["was_home"] = d["was_home"].astype(bool)

    st = t.set_index("id")
    d["own_strength"] = np.where(
        d["was_home"], st.loc[d["team_id"], "strength_overall_home"].values,
        st.loc[d["team_id"], "strength_overall_away"].values)
    d["opp_strength"] = np.where(
        d["was_home"], st.loc[d["opponent_team"], "strength_overall_away"].values,
        st.loc[d["opponent_team"], "strength_overall_home"].values)

    # team attack/defence from actual goals, rolling 5 shifted (no leakage)
    d["gf"] = np.where(d["was_home"], d["team_h_score"], d["team_a_score"])
    d["ga"] = np.where(d["was_home"], d["team_a_score"], d["team_h_score"])
    ft = (d.drop_duplicates(["team_id", "fixture"])
            [["team_id", "fixture", "kickoff_time", "gf", "ga"]]
            .sort_values(["team_id", "kickoff_time"]))
    for src, dst in (("gf", "att_r5"), ("ga", "def_r5")):
        ft[dst] = ft.groupby("team_id")[src].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    own = ft[["team_id", "fixture", "att_r5", "def_r5"]].rename(
        columns={"att_r5": "own_att_r5", "def_r5": "own_def_r5"})
    opp = ft[["team_id", "fixture", "att_r5", "def_r5"]].rename(
        columns={"team_id": "opponent_team",
                 "att_r5": "opp_att_r5", "def_r5": "opp_def_r5"})
    d = d.merge(own, on=["team_id", "fixture"], how="left")
    d = d.merge(opp, on=["opponent_team", "fixture"], how="left")

    # player rolling form, shifted
    d = d.sort_values(["key", "kickoff_time"]).reset_index(drop=True)
    g = d.groupby("key")
    roll = {("total_points", 3): "pts_r3", ("total_points", 5): "pts_r5",
            ("total_points", 10): "pts_r10", ("minutes", 3): "mins_r3",
            ("minutes", 5): "mins_r5", ("expected_goals", 5): "xg_r5",
            ("expected_assists", 5): "xa_r5", ("starts", 5): "starts_r5"}
    for (col, w), dst in roll.items():
        d[dst] = g[col].transform(
            lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())

    if prev_agg is not None:
        d = d.merge(prev_agg, on="key", how="left")
    else:
        for c in ("prev_p90", "prev_mins", "prev_xg90", "prev_xa90",
                  "prev_start_share", "prev_pts_total"):
            d[c] = np.nan

    d["price"] = d["value"] / 10.0
    d["was_home"] = d["was_home"].astype(int)
    d["gw"] = d["round"]
    d["played"] = d["minutes"].fillna(0) > 0
    d["played60"] = d["minutes"].fillna(0) >= 60
    return d


def pick_squad(players, pred_col="pred", budget=100.0, bench_weight=0.1):
    """LP: 15-man squad, XI, captain IN the objective (doubled points).
    players: DataFrame with columns [pred_col, price, pos, team_key]."""
    df = players.reset_index(drop=True)
    idx = df.index
    x = pulp.LpVariable.dicts("sq", idx, cat="Binary")
    y = pulp.LpVariable.dicts("xi", idx, cat="Binary")
    c = pulp.LpVariable.dicts("cap", idx, cat="Binary")
    m = pulp.LpProblem("fpl", pulp.LpMaximize)
    m += (pulp.lpSum(y[i] * df[pred_col][i] for i in idx)
          + pulp.lpSum(c[i] * df[pred_col][i] for i in idx)
          + bench_weight * pulp.lpSum((x[i] - y[i]) * df[pred_col][i]
                                      for i in idx))
    m += pulp.lpSum(x[i] for i in idx) == 15
    m += pulp.lpSum(y[i] for i in idx) == 11
    m += pulp.lpSum(c[i] for i in idx) == 1
    m += pulp.lpSum(x[i] * df["price"][i] for i in idx) <= budget
    for i in idx:
        m += y[i] <= x[i]
        m += c[i] <= y[i]
    quota = {"GK": (2, 1, 1), "DEF": (5, 3, 5),
             "MID": (5, 2, 5), "FWD": (3, 1, 3)}
    for p, (sq, lo, hi) in quota.items():
        sel = [i for i in idx if df["pos"][i] == p]
        m += pulp.lpSum(x[i] for i in sel) == sq
        m += pulp.lpSum(y[i] for i in sel) >= lo
        m += pulp.lpSum(y[i] for i in sel) <= hi
    for t in df["team_key"].unique():
        m += pulp.lpSum(x[i] for i in idx if df["team_key"][i] == t) <= 3
    m.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))
    df["in_squad"] = [int(x[i].value()) for i in idx]
    df["in_xi"] = [int(y[i].value()) for i in idx]
    df["is_cap"] = [int(c[i].value()) for i in idx]
    return df[df["in_squad"] == 1].copy()
