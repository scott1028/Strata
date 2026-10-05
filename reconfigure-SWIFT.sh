#!/usr/bin/bash

./setup.sh \
  --setup \
  --yes \
  --host 0.0.0.0 \
  --port 19090 \
  --family swift \
  --model IQ3_S \
  --context 250000 \
  --kv int8 \
  --vision yes \
  --experimental-speed-projection off \
  --no-browser \
  --data-dir /media/T710_Data_ntfs/Strata-data \
  --api-key 0908705265

