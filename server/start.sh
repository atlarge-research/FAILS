#!/bin/bash

# Activate virtual environment if you're using one
# source venv/bin/activate  # Uncomment if using virtualenv


cd "$(dirname "$0")"


PID_FILE="gunicorn.pid"

# Check if already running
if [ -f "$PID_FILE" ]; then
    pid=$(cat "$PID_FILE")
    if ps -p $pid > /dev/null 2>&1; then
        echo "Server is already running with PID: $pid"
        exit 1
    else
        rm "$PID_FILE"
    fi
fi

mkdir -p logs

# Start Gunicorn with specific log files
python -m gunicorn --bind 0.0.0.0:5000 wsgi:app \
    --workers 4 \
    --timeout 120 \
    --access-logfile logs/access.log \
    --error-logfile logs/error.log \
    --reload \
    # --daemon \
    --pid "$PID_FILE"

echo "Server started. PID file: $PID_FILE"
echo "Logs available in:"
echo "  - Access log: logs/access.log"
echo "  - Error log: logs/error.log"
