#!/bin/sh
set -e

echo "Aplicando migrations..."
alembic upgrade head

echo "Subindo a API..."
# --forwarded-allow-ips="*": atrás do proxy do Render, sem isto o uvicorn ignora
# o X-Forwarded-For (só confia em 127.0.0.1) e o rate limit vira um balde único
# para o serviço inteiro -- um visitante derruba todos. O custo é que um cliente
# pode forjar o X-Forwarded-For e escapar do limite; pior ainda é o balde único.
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --forwarded-allow-ips="*"
