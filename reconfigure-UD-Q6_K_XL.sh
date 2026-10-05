#!/usr/bin/bash

./setup.sh \
  --setup \
  --yes \
  --host 0.0.0.0 \
  --port 19090 \
  --family unsloth \
  --model UD-Q6_K_XL \
  --context 250000 \
  --kv int8 \
  --vision yes \
  --experimental-speed-projection off \
  --no-browser \
  --data-dir /media/T710_Data_ntfs/Strata-data \
  --gguf-dir /media/T710_Data_ntfs/models/unsloth/Qwen3.8-Flash-Next-GGUF \
  --api-key 0908705265

