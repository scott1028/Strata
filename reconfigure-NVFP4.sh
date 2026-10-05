#!/usr/bin/bash

./setup.sh \
  --setup \
  --yes \
  --host 0.0.0.0 \
  --port 19090 \
  --family nvfp4 \
  --model NVFP4 \
  --context 256000 \
  --kv int8 \
  --kv-streaming off \
  --vision yes \
  --experimental-speed-projection off \
  --no-browser \
  --data-dir /media/T710_Data_ntfs/Strata-data \
  --api-key 0908705265

