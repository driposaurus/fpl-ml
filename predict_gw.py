"""fpl-ml v0.3 — weekly prediction for the upcoming 2026/27 gameweek.

Usage: python predict_gw.py [--gw N] [--model hgb|two]
Reads local CSVs (run fetch_data.sh first). Trains on all history
(23/24, 24/25, 25/26 + played 26/27 GWs), predicts the target GW,
zeroes injured/suspended players, picks the squad via LP (captain in
objective), writes predictions/gw{N}.csv with a UTC timestamp.
"""
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)

from fpl_features import FEATS, POS_NAME, build_season, norm, pick_squad, \
    season_agg

ELEM = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def load_optional(path):
    try:
        d = pd.read_csv(path)
        return path if len(d) and "name" in d.columns else None
    except Exception:
        return None


def hgb():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                         max_depth=6, random_state=42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, default=None)
    ap.add_argument("--model", choices=["hgb", "two"], default="hgb",
                    help="hgb won the 25/26 season backtest (default)")
    args = ap.parse_args()

    players = pd.read_csv("players_2627.csv")
    teams = pd.read_csv("teams_2627.csv")
    fx = pd.read_csv("fixtures_2627.csv")

    gw = args.gw
    if gw is None:
        now = datetime.now(timezone.utc).isoformat()
        fut = fx[(~fx["finished"].astype(bool)) & (fx["kickoff_time"] > now)]
        gw = int(fut["event"].min())
    fx_t = fx[fx["event"] == gw]
    id2name = dict(zip(teams["id"], teams["name"]))
    id2short = dict(zip(teams["id"], teams["short_name"]))

    # synthetic rows for the target GW (both sides of every fixture)
    rows = []
    for _, f in fx_t.iterrows():
        for home in (True, False):
            tid = f["team_h"] if home else f["team_a"]
            opp = f["team_a"] if home else f["team_h"]
            p = players[players["team"] == tid]
            rows.append(pd.DataFrame({
                "name": p["first_name"] + " " + p["second_name"],
                "position": p["element_type"].map(ELEM),
                "team": id2name[tid], "opponent_team": opp,
                "was_home": home, "round": gw, "fixture": f["id"],
                "value": p["now_cost"], "kickoff_time": f["kickoff_time"],
                "minutes": np.nan, "total_points": np.nan,
                "team_h_score": np.nan, "team_a_score": np.nan,
                "expected_goals": np.nan, "expected_assists": np.nan,
                "starts": np.nan}))
    targets = pd.concat(rows, ignore_index=True)

    cur_path = load_optional("gw_2026-27.csv")
    s26 = build_season(cur_path, "teams_2627.csv",
                       season_agg("gw_2025-26.csv"), extra_rows=targets)
    played_max = s26.loc[s26["total_points"].notna(), "gw"].max()
    print(f"GW{gw} | in-season data through GW"
          f"{int(played_max) if pd.notna(played_max) else 0} "
          f"| model: {args.model}")

    s23 = build_season("gw_2023-24.csv", "teams_2023-24.csv", None)
    s24 = build_season("gw_2024-25.csv", "teams_2024-25.csv",
                       season_agg("gw_2023-24.csv"))
    s25 = build_season("gw_2025-26.csv", "teams_2025-26.csv",
                       season_agg("gw_2024-25.csv"))
    train = pd.concat([s23, s24, s25,
                       s26[s26["total_points"].notna()
                           & (s26["gw"] < gw)]], ignore_index=True)
    te = s26[(s26["gw"] == gw) & (s26["total_points"].isna())].copy()

    if args.model == "hgb":
        m = hgb()
        m.fit(train[FEATS], train["total_points"])
        te["pred"] = m.predict(te[FEATS]).clip(min=0)
    else:
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=42)
        clf.fit(train[FEATS], train["played60"])
        reg = hgb()
        pl = train[train["played"]]
        reg.fit(pl[FEATS], pl["total_points"])
        te["pred"] = (clf.predict_proba(te[FEATS])[:, 1]
                      * reg.predict(te[FEATS])).clip(min=0)

    # injuries / suspensions from current API state
    pr = players[["first_name", "second_name", "web_name", "status",
                  "chance_of_playing_next_round"]].copy()
    pr["key"] = (pr["first_name"] + " " + pr["second_name"]).map(norm)
    te = te.merge(pr[["key", "web_name", "status",
                      "chance_of_playing_next_round"]], on="key", how="left")
    te.loc[te["status"].isin(["i", "s", "u", "n"]), "pred"] = 0.0
    doubt = te["status"].eq("d")
    te.loc[doubt, "pred"] *= \
        te.loc[doubt, "chance_of_playing_next_round"].fillna(50) / 100.0

    agg = (te.groupby("key")
             .agg(pred=("pred", "sum"), price=("price", "first"),
                  pos_code=("pos_code", "first"), team_key=("team_id", "first"),
                  web_name=("web_name", "first")).reset_index())
    agg["pos"] = agg["pos_code"].map(POS_NAME)
    agg["club"] = agg["team_key"].map(id2short)

    sel = pick_squad(agg)
    sel["role"] = np.where(sel["is_cap"] == 1, "XI (C)",
                           np.where(sel["in_xi"] == 1, "XI", "bench"))
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    sel = sel.sort_values(["in_xi", "pos", "pred"],
                          ascending=[False, True, False],
                          key=lambda c: c.map(order) if c.name == "pos" else c)

    print(f"\n=== GW{gw} squad | cost {sel.price.sum():.1f}m ===")
    for _, r in sel.iterrows():
        print(f"{r.role:8} {r.pos:3} {r.web_name:20} {r.club}  "
              f"{r.price:.1f}m  pred {r.pred:.2f}")
    xi = sel[sel["in_xi"] == 1]
    print(f"\npred XI (with captain): "
          f"{xi.pred.sum() + sel.loc[sel.is_cap == 1, 'pred'].iloc[0]:.1f}")

    print(f"\nTop 20 by predicted points (transfer shortlist):")
    for _, r in agg.nlargest(20, "pred").iterrows():
        print(f"  {r.pos:3} {r.web_name:20} {r.club} {r.price:5.1f}m  "
              f"{r.pred:.2f}")

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fn = f"predictions/gw{gw}.csv"
    with open(fn, "w") as f:
        f.write(f"# generated_utc: {stamp} | v0.3 {args.model} | "
                f"in-season data through GW"
                f"{int(played_max) if pd.notna(played_max) else 0}\n")
        sel[["web_name", "club", "pos", "price", "pred",
             "role"]].to_csv(f, index=False)
    print(f"\nSaved {fn} ({stamp} UTC)")


if __name__ == "__main__":
    main()
