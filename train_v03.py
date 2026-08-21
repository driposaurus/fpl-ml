"""fpl-ml v0.3 — walk-forward evaluation + season backtest.

Ladder: naive -> Ridge -> gradient boosting -> two-stage P(60+) x E(pts|played).
Walk-forward: retrain every 4 GWs on everything strictly before the chunk.
Metrics: MAE, mean per-GW Spearman, mean per-GW precision@10.
Backtest: fresh LP squad each GW (oracle transfers), actual points, captain x2.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fpl_features import FEATS, POS_NAME, build_season, pick_squad, season_agg


def hgb():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                         max_depth=6, random_state=42)


def main():
    s23 = build_season("gw_2023-24.csv", "teams_2023-24.csv", None)
    s24 = build_season("gw_2024-25.csv", "teams_2024-25.csv",
                       season_agg("gw_2023-24.csv"))
    s25 = build_season("gw_2025-26.csv", "teams_2025-26.csv",
                       season_agg("gw_2024-25.csv"))
    hist = pd.concat([s23, s24], ignore_index=True)

    s25 = s25.sort_index()
    for col in ("p_ridge", "p_hgb", "p_two"):
        s25[col] = np.nan

    for g0 in range(1, 39, 4):
        chunk = (s25["gw"] >= g0) & (s25["gw"] < g0 + 4)
        if not chunk.any():
            continue
        tr = pd.concat([hist, s25[s25["gw"] < g0]], ignore_index=True)
        te = s25[chunk]

        ridge = make_pipeline(SimpleImputer(strategy="median"),
                              StandardScaler(), Ridge(alpha=1.0))
        ridge.fit(tr[FEATS], tr["total_points"])
        s25.loc[chunk, "p_ridge"] = ridge.predict(te[FEATS])

        m1 = hgb()
        m1.fit(tr[FEATS], tr["total_points"])
        s25.loc[chunk, "p_hgb"] = m1.predict(te[FEATS])

        clf = HistGradientBoostingClassifier(max_iter=200, random_state=42)
        clf.fit(tr[FEATS], tr["played60"])
        reg = hgb()
        pl = tr[tr["played"]]
        reg.fit(pl[FEATS], pl["total_points"])
        s25.loc[chunk, "p_two"] = (clf.predict_proba(te[FEATS])[:, 1]
                                   * reg.predict(te[FEATS]))

    s25["p_naive"] = s25["pts_r5"].fillna(
        s25["prev_pts_total"] / 38).fillna(1.0)

    def metrics(col):
        y, p = s25["total_points"], s25[col]
        mae = mean_absolute_error(y, p)
        sp, prec = [], []
        for _, g in s25.groupby("gw"):
            if g[col].nunique() > 1:
                sp.append(spearmanr(g[col], g["total_points"]).statistic)
            top_p = set(g.nlargest(10, col)["key"])
            top_a = set(g.nlargest(10, "total_points")["key"])
            prec.append(len(top_p & top_a) / 10)
        return mae, np.nanmean(sp), np.mean(prec)

    print(f"{'model':<28}{'MAE':>7}{'Spearman':>10}{'prec@10':>9}")
    for name, col in [("naive (rolling 5 / prev)", "p_naive"),
                      ("Ridge", "p_ridge"),
                      ("gradient boosting", "p_hgb"),
                      ("two-stage P(60+) x E(pts)", "p_two")]:
        mae, sp, pr = metrics(col)
        print(f"{name:<28}{mae:>7.3f}{sp:>10.3f}{pr:>9.3f}")

    # --- season backtest: fresh LP squad each GW, actual pts, captain x2 ---
    def backtest(col):
        total = 0.0
        for gw, g in s25.groupby("gw"):
            agg = (g.groupby("key")
                     .agg(pred=(col, "sum"), actual=("total_points", "sum"),
                          price=("price", "first"), pos_code=("pos_code", "first"),
                          team_key=("team_id", "first")).reset_index())
            agg["pos"] = agg["pos_code"].map(POS_NAME)
            sel = pick_squad(agg, pred_col="pred")
            xi = sel[sel["in_xi"] == 1]
            total += xi["actual"].sum() + sel.loc[sel["is_cap"] == 1,
                                                  "actual"].sum()
        return total

    for name, col in [("naive", "p_naive"), ("gradient boosting", "p_hgb"),
                      ("two-stage", "p_two")]:
        print(f"backtest 25/26 (oracle transfers, no auto-subs) "
              f"{name}: {backtest(col):.0f} pts")


if __name__ == "__main__":
    main()
