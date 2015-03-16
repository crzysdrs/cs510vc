#!/bin/sh
mkdir -p out
./tracking.py annex/Walking.mkv --max_size=10000 --mask_shadow --output out/walk.avi
./tracking.py annex/Walking.mkv --max_size=10000 --mask_shadow --output out/walk_debug.avi --debug
./tracking.py annex/Subway.mkv --max_size=6400 --output out/subway.avi
./tracking.py annex/Subway.mkv --max_size=6400 --output out/subway_debug.avi --debug
