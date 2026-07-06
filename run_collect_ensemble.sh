#!/usr/bin/env bash
# Archive today's live 50-member NWP ensemble forecast for EMOS training.
# Run ONCE per day at the SAME time as the live day-ahead submission so the
# archived lead times match production. The archive can only be built going
# forward — Open-Meteo does not serve past ensemble runs.
#
# Writes one Parquet per day to data/nwp_archive/<date>.parquet.
# Logs go to logs/collect_ensemble_<timestamp>.log and .../collect_ensemble_latest.log.
#
# Crontab example (submission runs 10:00 CET = 09:00 UTC):
#   0 9 * * * /path/to/solar-forecast/run_collect_ensemble.sh >> /path/to/cron.log 2>&1

set -euo pipefail
cd "$(dirname "$0")"

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_PATH="$LOG_DIR/collect_ensemble_$TIMESTAMP.log"
LATEST_LOG="$LOG_DIR/collect_ensemble_latest.log"

echo "Writing log to $LOG_PATH"
.venv/bin/python -u -m solar_forecast.collect_ensemble_forecasts "$@" 2>&1 | tee "$LOG_PATH"
EXIT_CODE=${PIPESTATUS[0]}

cp "$LOG_PATH" "$LATEST_LOG"
exit "$EXIT_CODE"
