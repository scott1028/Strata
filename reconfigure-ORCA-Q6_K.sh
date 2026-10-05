#!/usr/bin/bash

.venv/bin/python tools/reconfigure_orca.py \
  --quant Q6_K \
  --host 0.0.0.0 \
  --port 19090 \
  --context 256000 \
  --kv int8 \
  --kv-streaming off \
  --vision yes \
  --vram-reserve-mib 3072 \
  --resident-budget-gib 70 \
  --pool-workers 13 \
  --no-browser \
  --data-dir /media/T710_Data_ntfs/Strata-data \
  --gguf-dir /media/T710_Data_ntfs/models/orcarouter \
  --api-key 0908705265
