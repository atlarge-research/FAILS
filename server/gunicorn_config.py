import multiprocessing
import os

# Gunicorn configuration
bind = "0.0.0.0:5000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 2

# Static file serving
static_folder = os.path.join(os.path.dirname(__file__), '..', 'client', 'build')

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Worker configurations
max_requests = 1000
max_requests_jitter = 50
graceful_timeout = 30

# Add static files configuration
def when_ready(server):
    if not os.path.exists(static_folder):
        raise RuntimeError(f"Static folder not found: {static_folder}")