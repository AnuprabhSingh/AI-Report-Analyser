#!/bin/bash
# Production startup script for Medical Interpreter

set -e

echo "🚀 Starting Medical Interpreter Application..."

# Check if running in production
if [ "$FLASK_ENV" = "production" ]; then
    echo "📦 Production mode detected"
    
    # Use gunicorn for production
    if command -v gunicorn &> /dev/null; then
        echo "🦄 Starting with Gunicorn..."
        exec gunicorn --bind 0.0.0.0:${PORT:-5000} \
                     --workers ${WORKERS:-2} \
                     --timeout 120 \
                     --access-logfile - \
                     --error-logfile - \
                     "src.api:app"
    else
        echo "⚠️  Gunicorn not found, falling back to Flask development server"
        echo "⚠️  For production, install gunicorn: pip install gunicorn"
        exec python -m flask --app src/api.py run --host=0.0.0.0 --port=${PORT:-5000}
    fi
else
    echo "🔧 Development mode"
    exec python -m flask --app src/api.py run --host=0.0.0.0 --port=${PORT:-5000} --debug
fi
