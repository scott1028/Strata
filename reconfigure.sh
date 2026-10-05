#!/usr/bin/bash

./setup.sh \
  --setup \
  --yes \
  --host 0.0.0.0 \
  --port 19090 \
  --family qwen \
  --model IQ3_S \
  --context 250000 \
  --kv int8 \
  --vision yes \
  --experimental-speed-projection off \
  --no-browser \
  --gguf-dir /media/T500_Data/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF \

