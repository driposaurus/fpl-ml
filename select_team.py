"""fpl-ml — model v0.1 (GW1 baseline).

Scoring: punkty/90 z sezonu 25/26 x prawdopodobienstwo grania (z minut),
filtr kontuzji/zawieszen ze statusu API. Sklad: LP (PuLP) pod pelne
ograniczenia FPL (budzet 100.0, 2GK/5DEF/5MID/3FWD, max 3/klub, formacja XI).
"""
import unicodedata
from datetime import datetime, timezone

import pandas as pd
import pulp

POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
SQUAD_QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
BUDGET = 1000  # w dziesiatych (100.0m)
DEFAULT_PRED = 1.0  # cold start: nowi w PL / beniaminkowie


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def main():
    cur = pd.read_csv("players_2627.csv")
    teams = pd.read_csv("teams_2627.csv")[["id", "short_name"]]
    last = pd.read_csv("cleaned_2526.csv")

    cur = cur[["id", "first_name", "second_name", "web_name", "team",
               "element_type", "now_cost", "status",
               "chance_of_playing_next_round"]].copy()
    cur["pos"] = cur["element_type"].map(POS)
    cur = cur.merge(teams, left_on="team", right_on="id", suffixes=("", "_t"))
    cur["key"] = (cur["first_name"] + " " + cur["second_name"]).map(norm)

    last["key"] = (last["first_name"] + " " + last["second_name"]).map(norm)
    last = (last.sort_values("minutes", ascending=False)
                .drop_duplicates("key")[["key", "total_points", "minutes"]])

    df = cur.merge(last, on="key", how="left")

    # --- scoring v0.1 ---
    games = df["minutes"].fillna(0) / 90.0
    p90 = df["total_points"].fillna(0) / games.clip(lower=8)  # shrinkage dla malych minut
    start_prob = (df["minutes"].fillna(0) / 2700.0).clip(upper=1.0)
    df["pred"] = (p90 * start_prob).where(df["total_points"].notna(), DEFAULT_PRED)

    # kontuzje / zawieszenia / niedostepni
    chance = df["chance_of_playing_next_round"]
    df.loc[df["status"].isin(["i", "s", "u", "n"]), "pred"] = 0.0
    doubt = df["status"].eq("d")
    df.loc[doubt, "pred"] = df.loc[doubt, "pred"] * chance[doubt].fillna(50) / 100.0

    # --- optymalizacja ---
    df = df.reset_index(drop=True)
    idx = df.index
    x = pulp.LpVariable.dicts("squad", idx, cat="Binary")
    y = pulp.LpVariable.dicts("xi", idx, cat="Binary")

    m = pulp.LpProblem("fpl_gw1", pulp.LpMaximize)
    m += pulp.lpSum(y[i] * df.pred[i] for i in idx) + \
         0.1 * pulp.lpSum((x[i] - y[i]) * df.pred[i] for i in idx)

    m += pulp.lpSum(x[i] for i in idx) == 15
    m += pulp.lpSum(y[i] for i in idx) == 11
    m += pulp.lpSum(x[i] * df.now_cost[i] for i in idx) <= BUDGET
    for i in idx:
        m += y[i] <= x[i]
    for p, q in SQUAD_QUOTA.items():
        m += pulp.lpSum(x[i] for i in idx if df.pos[i] == p) == q
        m += pulp.lpSum(y[i] for i in idx if df.pos[i] == p) >= XI_MIN[p]
        m += pulp.lpSum(y[i] for i in idx if df.pos[i] == p) <= XI_MAX[p]
    for t in df["short_name"].unique():
        m += pulp.lpSum(x[i] for i in idx if df.short_name[i] == t) <= 3

    m.solve(pulp.PULP_CBC_CMD(msg=0))
    assert pulp.LpStatus[m.status] == "Optimal"

    df["in_squad"] = [int(x[i].value()) for i in idx]
    df["in_xi"] = [int(y[i].value()) for i in idx]
    sq = df[df.in_squad == 1].copy()
    sq["role"] = sq["in_xi"].map({1: "XI", 0: "bench"})
    cap = sq.loc[sq[sq.in_xi == 1]["pred"].idxmax(), "web_name"]
    sq.loc[sq.web_name == cap, "role"] = "XI (C)"

    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    sq = sq.sort_values(["in_xi", "pos", "pred"],
                        ascending=[False, True, False],
                        key=lambda c: c.map(order) if c.name == "pos" else c)

    print(f"\n=== fpl-ml GW1 | koszt {sq.now_cost.sum()/10:.1f}m / 100.0m ===")
    for _, r in sq.iterrows():
        print(f"{r.role:8} {r.pos:3} {r.web_name:20} {r.short_name}  "
              f"{r.now_cost/10:.1f}m  pred {r.pred:.2f}")
    xi_pts = sq[sq.in_xi == 1].pred.sum() + sq[sq.role == 'XI (C)'].pred.iloc[0]
    print(f"\nKapitan: {cap} | pred XI (z kapitanem): {xi_pts:.1f} pkt")

    out = sq[["web_name", "short_name", "pos", "now_cost", "pred", "role"]]
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fn = "predictions_gw1.csv"
    with open(fn, "w") as f:
        f.write(f"# generated_utc: {stamp} | model v0.1 | GW1 2026/27\n")
        out.to_csv(f, index=False)
    print(f"\nZapisano {fn} ({stamp} UTC)")


if __name__ == "__main__":
    main()
