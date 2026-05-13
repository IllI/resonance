#!/bin/bash
pkill -9 -f python3 2>/dev/null
sleep 5
echo "processes cleared"
nohup python3 -u ~/teleport_sim_tpu.py \
  --N_haar 5000 --K_vq 2000 \
  --out ~/teleport_results.json \
  > ~/teleport_sim.log 2>&1 &
echo "PID:$!"
