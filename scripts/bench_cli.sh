#!/bin/sh
# End-to-end throughput of the self-contained CLI (PNG decode + CDCH + linear model),
# by thread count. Usage: scripts/bench_cli.sh PAIRS_DIR [MODEL_JSON] > reports/perf/cli_throughput.csv
set -eu
PAIRS=${1:?pairs directory}
MODEL=${2:-models/cdch_A_logistic.json}
BIN=target/release/cdch
FILES=$(find "$PAIRS" -name '*.png' ! -name '*_lr.png' | sort)
echo "threads,repeat,images,seconds,img_per_s"
for t in 1 2 4 8 $(getconf _NPROCESSORS_ONLN); do
  for rep in 0 1 2 3; do
    # shellcheck disable=SC2086
    line=$($BIN score --model "$MODEL" --threads "$t" --timing $FILES 2>&1 >/dev/null | tail -n 1)
    n=$(echo "$line" | awk '{print $1}'); s=$(echo "$line" | awk '{print $4}'); r=$(echo "$line" | sed 's/.*(\([0-9]*\) img\/s).*/\1/')
    [ "$rep" -gt 0 ] && echo "$t,$rep,$n,$s,$r"   # repeat 0 warms the file cache
  done
done
