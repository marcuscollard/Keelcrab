#!/bin/bash

# Directory containing videos
VIDEO_DIR="."
FPS_EXTRACT=3

# Ensure jq is installed for JSON (Ubuntu: sudo apt install jq)
command -v ffmpeg >/dev/null 2>&1 || { echo >&2 "ffmpeg is not installed. Aborting."; exit 1; }
command -v jq >/dev/null 2>&1 || { echo >&2 "jq is not installed. Install it with 'sudo apt install jq'. Aborting."; exit 1; }

for f in "$VIDEO_DIR"/*.mp4; do
  [ -e "$f" ] || continue  # Skip if no .mp4 files exist
  name=$(basename "${f%.mp4}")
  out_dir="${VIDEO_DIR}/${name}/frames"
  meta_file="${VIDEO_DIR}/${name}/metadata.json"

  if [ -d "$out_dir" ]; then
    echo "Skipping $f — already processed."
    continue
  fi

  mkdir -p "$out_dir"

  # Get metadata using ffprobe
  duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$f")
  fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate \
      -of default=noprint_wrappers=1:nokey=1 "$f" | awk -F/ '{printf "%.2f", $1/$2}')
  width=$(ffprobe -v error -select_streams v:0 -show_entries stream=width \
      -of default=noprint_wrappers=1:nokey=1 "$f")
  height=$(ffprobe -v error -select_streams v:0 -show_entries stream=height \
      -of default=noprint_wrappers=1:nokey=1 "$f")

  # Extract frames at specified FPS
  ffmpeg -i "$f" -vf fps=$FPS_EXTRACT "$out_dir/frame_%05d.jpg"

  # Count extracted frames
  frame_count=$(ls "$out_dir" | grep -c '\.jpg')

  # Write metadata.json
  jq -n \
    --arg video_name "$(basename "$f")" \
    --arg resolution "${width}x${height}" \
    --arg duration "$duration" \
    --arg original_fps "$fps" \
    --arg extracted_fps "$FPS_EXTRACT" \
    --arg frames "$frame_count" \
    --arg quality "" \
    '{
      video_name: $video_name,
      resolution: $resolution,
      duration_seconds: ($duration | tonumber),
      original_fps: ($original_fps | tonumber),
      extracted_fps: ($extracted_fps | tonumber),
      extracted_frames: ($frames | tonumber),
      quality: $quality
    }' > "$meta_file"

  echo "Processed $f → $frame_count frames with metadata saved."
done
