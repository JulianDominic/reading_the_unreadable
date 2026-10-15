#!/bin/bash
# Run nginx and Cantaloupe side by side; stop the container when either exits.
set -euo pipefail

nginx -g "daemon off;" &
java ${JAVA_OPTS} -Dcantaloupe.config=/opt/cantaloupe/cantaloupe.properties -jar /opt/cantaloupe/cantaloupe.jar &

trap 'kill -TERM $(jobs -p) 2>/dev/null; wait' TERM INT
wait -n
status=$?
kill -TERM $(jobs -p) 2>/dev/null || true
wait
exit $status
