#!/usr/bin/env bash
set -euo pipefail

DATASTORE=${CACHEURL:="https://epyc.astro.washington.edu/~mjuric/mpsky-data"}
MAX_LOADED_NIGHTS=7
MAX_ONDISK_NIGHTS=3

exec mpsky serve \
	--cache-datastore "$DATASTORE" \
	--max-loaded-nights "$MAX_LOADED_NIGHTS" \
	--max-ondisk-nights "$MAX_ONDISK_NIGHTS" \
	${@}
