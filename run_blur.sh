#!/bin/bash

LOCKFILE="blur_lock"
MAX_PASSES=50

echo "🧹 Cleaning up old processes..."
pkill -f 'python3 preprocess/dof_blur.py'

echo "🧼 Removing old lock file (if any)..."
rm -f "$LOCKFILE"

for ((i=1;i<=MAX_PASSES;i++)); do
  echo "▶ Starting video processing..."
  echo "🔁 Pass $i of $MAX_PASSES..."

  if [ -f "$LOCKFILE" ]; then
    echo "🚫 Already running. Exiting."
    sleep 10
    continue
  fi

  touch "$LOCKFILE"
  python3 preprocess/dof_blur.py
  status=$?
  rm -f "$LOCKFILE"

  if [ $status -eq 0 ]; then
    echo "✅ All videos completed!"
    break
  else
    echo "🔁 Restarting after 10s..."
    sleep 10
  fi
done
