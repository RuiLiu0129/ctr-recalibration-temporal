#!/bin/bash
cd ~/workspace/goals/goal/avazu-temporal-calibration
for i in $(seq 1 160); do
  if grep -q "TIERC CHAIN COMPLETE" tierc_chain.log 2>/dev/null; then
    echo "CHAIN_COMPLETE"; tail -35 tierc_chain.log; echo "---DONE MARKERS---"
    for f in tierc_*.done; do echo "$f: $(cat $f)"; done; exit 0
  fi
  if ! pgrep -f "run_tierc" > /dev/null; then
    echo "CHAIN_PROCESS_GONE without COMPLETE"; tail -35 tierc_chain.log; echo "---DONE MARKERS---"
    for f in tierc_*.done; do echo "$f: $(cat $f)"; done; exit 2
  fi
  sleep 240
done
echo "WATCH_TIMEOUT"; tail -5 tierc_chain.log; exit 3
