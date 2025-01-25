#!/bin/bash

# Ensure we're in the server directory
cd "$(dirname "$0")"

# Define PID file location
PID_FILE="gunicorn.pid"

# Check if PID file exists
if [ -f "$PID_FILE" ]; then
    pid=$(cat "$PID_FILE")
    
    # Try graceful shutdown first
    if kill -TERM $pid > /dev/null 2>&1; then
        echo "Stopping Gunicorn gracefully..."
        # Wait for up to 10 seconds for process to stop
        for i in {1..10}; do
            if ! ps -p $pid > /dev/null 2>&1; then
                break
            fi
            sleep 1
        done
    fi
    
    # If process still running, force kill
    if ps -p $pid > /dev/null 2>&1; then
        echo "Force stopping Gunicorn..."
        kill -9 $pid
    fi
    
    # Remove PID file
    rm "$PID_FILE"
    echo "Server stopped"
else
    echo "No PID file found. Server might not be running."
    # Cleanup any stray processes just in case
    pkill gunicorn
fi 