#!/usr/bin/env bash
# Start the image and prove the service in it works, without calling a model.
#
#   scripts/container_smoke_test.sh support-bot:ci
#
# Runs in CI after `docker build`, and on any machine with Docker and curl.
# The API key passed in is a placeholder: every check below is answered by the
# service itself, before any model would be called, so the test costs nothing
# and needs no secret.

set -euo pipefail

IMAGE="${1:?usage: $0 IMAGE}"
NAME="support-bot-smoke-$$"
PORT="${PORT:-18000}"
BASE="http://127.0.0.1:${PORT}"
failures=0

cleanup() {
    if [ "$failures" -ne 0 ] || [ "${KEEP_LOGS:-0}" = "1" ]; then
        echo "--- container logs ---"
        docker logs "$NAME" 2>&1 | tail -50 || true
    fi
    docker rm -f "$NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

check() {  # check DESCRIPTION COMMAND...
    local description="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        echo "ok    $description"
    else
        echo "FAIL  $description"
        failures=$((failures + 1))
    fi
}

status_of() { curl -s -o /dev/null -w '%{http_code}' "$@"; }

docker run -d --name "$NAME" -p "${PORT}:8000" \
    -e ANTHROPIC_API_KEY=sk-ant-placeholder-for-the-smoke-test "$IMAGE" >/dev/null

echo "waiting for the service to answer /health ..."
for _ in $(seq 1 60); do
    if [ "$(status_of "$BASE/health")" = "200" ]; then break; fi
    sleep 1
done

check "/health answers 200" test "$(status_of "$BASE/health")" = "200"
check "/health reports the version" bash -c "curl -s '$BASE/health' | grep -q '\"version\"'"
check "/ready answers 200 (key set, index baked into the image)" \
    test "$(status_of "$BASE/ready")" = "200"
check "/ready reports the index loaded" \
    bash -c "curl -s '$BASE/ready' | grep -q '\"index\":true'"
check "/ redirects to /docs" \
    bash -c "curl -s -o /dev/null -w '%{http_code} %{redirect_url}' '$BASE/' | grep -q '^307 .*/docs$'"
check "/docs is served" test "$(status_of "$BASE/docs")" = "200"
check "an empty message is refused with 422, before any model call" \
    test "$(status_of -X POST "$BASE/chat" -H 'content-type: application/json' -d '{"message": ""}')" = "422"
check "an unknown conversation is refused with 404, before any model call" \
    test "$(status_of -X POST "$BASE/chat" -H 'content-type: application/json' \
        -d '{"message": "hi", "session_id": "00000000000000000000000000000000"}')" = "404"
check "every response carries an X-Request-ID" \
    bash -c "curl -s -D - -o /dev/null '$BASE/health' | grep -qi '^x-request-id: [0-9a-f]\{32\}'"
check "the service does not run as root" \
    bash -c "[ \"\$(docker exec '$NAME' id -u)\" != 0 ]"
check "no tests, secrets or local state were copied into the image" \
    docker exec "$NAME" sh -c 'cd /app && [ ! -e tests ] && [ ! -e .env ] && [ ! -e .git ] && [ ! -e .venv ]'
check "the embedding model and index were built into the image" \
    docker exec "$NAME" sh -c '[ -d /app/.models ] && [ -d /app/.index ]'

echo "waiting for Docker's own health check ..."
health="starting"
for _ in $(seq 1 45); do
    health="$(docker inspect --format '{{.State.Health.Status}}' "$NAME")"
    if [ "$health" != "starting" ]; then break; fi
    sleep 1
done
check "Docker reports the container healthy" test "$health" = "healthy"
check "no Python traceback in the logs" bash -c "! docker logs '$NAME' 2>&1 | grep -q Traceback"

size="$(docker image inspect --format '{{.Size}}' "$IMAGE")"
echo "image size: $((size / 1024 / 1024)) MB"

if [ "$failures" -ne 0 ]; then
    echo "$failures check(s) failed"
    exit 1
fi
echo "all checks passed"
