#!/bin/bash
# Fetch current + historical FPL data (vaastav archive).
# Exact GW1 state preserved in data/data_snapshot_gw1.zip.
BASE=https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data
curl -s -o players_2627.csv  $BASE/2026-27/players_raw.csv
curl -s -o teams_2627.csv    $BASE/2026-27/teams.csv
curl -s -o fixtures_2627.csv $BASE/2026-27/fixtures.csv
curl -s -o cleaned_2526.csv  $BASE/2025-26/cleaned_players.csv
# in-season gameweek data (404 until vaastav publishes GW1 - handled gracefully)
curl -s -o gw_2026-27.csv    $BASE/2026-27/gws/merged_gw.csv
for s in 2023-24 2024-25 2025-26; do
  curl -s -o gw_$s.csv    $BASE/$s/gws/merged_gw.csv
  curl -s -o teams_$s.csv $BASE/$s/teams.csv
done
echo "done"
