#!/bin/sh
set -e

# Copy pre-downloaded default models if the volume is missing them
if [ -d "/app/default_voices" ] && [ -d "/data/voices" ]; then
    echo "Initializing voice models volume with pre-downloaded templates..."
    # Copy only if file does not exist in destination (-n)
    cp -n /app/default_voices/* /data/voices/ 2>/dev/null || true
fi

# Execute the main container command
exec "$@"
