#!/usr/bin/bash

.venv/bin/python tools/reconfigure_orca.py \
  --host 0.0.0.0 \
  --port 19090 \
  --context 256000 \
  --kv int8 \
  --kv-streaming off \
  --vision yes \
  --no-browser \
  --data-dir /media/T710_Data_ntfs/Strata-data \
  --gguf-dir /media/T500_Data2_ntfs/models/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF \
  --api-key 0908705265
