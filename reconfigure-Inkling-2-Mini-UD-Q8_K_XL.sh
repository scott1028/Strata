#!/usr/bin/bash

./setup.sh \
  --setup \
  --yes \
  --host 0.0.0.0 \
  --port 19090 \
  --family inkling-2-mini \
  --model UD-Q8_K_XL \
  --context 256000 \
  --vision yes \
  --drafter off \
  --no-browser \
  --data-dir /media/T710_Data_ntfs/Strata-data \
  --gguf-dir /media/T710_Data_ntfs/models/unsloth/Inkling-2-Mini-GGUF \
  --api-key 0908705265

