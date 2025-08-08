#!/usr/bin/env bash
set -euo pipefail

PID=
CACHEFN=
CACHEDIR=${CACHEDIR:=/tmp}
CACHEURL=${CACHEURL:="https://epyc.astro.washington.edu/~mjuric/mpsky-data/caches"}
CHECK_INTERVAL=${CHECK_INTERVAL:=120}

# make sure we clean up mpsky serve if we're interrupted
trap 'echo "cleaning up [pid=$PID]"; kill $PID 2>/dev/null; exit' INT

# fetch the latest cache file for a given MJD
latest_available_cache()
{
	local MJD=$1
	curl -s "$CACHEURL/" | grep -oP 'href="\K[^"]+' | grep -vE '^\.\.?$' | sort -r | grep -E "eph\\.$MJD\\..*\\.bin" | head -n 1
}

while :
do
	# wait until we check for newer files, unless we're here for the first time
	[[ -z $PID ]] || sleep $CHECK_INTERVAL;

	# Compute the MJD of the current (local time) observing night in La Serena
	OBSDATE=$(TZ="America/Santiago" date --date='12 hours ago' +"%Y/%m/%d")
	MJD=$(echo "$(date --date="$OBSDATE" +%s) / 86400.0 + 2440587.5 - 2400000.5" | bc -l | cut -f 1 -d .)


	# fetch the most recent file for that MJD
	NEW_CACHEFN=$(latest_available_cache $MJD)

	echo "[$(date)] OBSDATE=$OBSDATE MJD=$MJD NEW_CACHEFN=$NEW_CACHEFN CACHEFN=$CACHEFN"

	# if the file has changed, download and restart the server
	if [[ "$CACHEFN" != "$NEW_CACHEFN" ]]; then
		## download the cache file
		echo "    downloading $CACHEURL/$NEW_CACHEFN"
		curl -sS "$CACHEURL/$NEW_CACHEFN" -o "$CACHEDIR/$NEW_CACHEFN" || { continue; }
		echo "    downloaded $CACHEDIR/$NEW_CACHEFN"

		## kill current server process
		[[ -n $PID ]] && { echo "    killing $PID"; kill "$PID" && wait "$PID" 2>/dev/null || true; }

		## serve the new cache
		echo "    starting mpsky..."
		mpsky serve "$CACHEDIR/$NEW_CACHEFN" ${@} &
		PID=$!
		echo "    mpsky serve started, pid=$PID"

		## delete the old cache
		[[ ! -z "$CACHEFN" ]] && { echo "    deleting $CACHEDIR/$CACHEFN"; rm -f "$CACHEDIR/$CACHEFN"; }
		CACHEFN="$NEW_CACHEFN"
	fi

	# test that the process is alive
	kill -0 "$PID" 2>/dev/null || { echo "error: process $PID not alive; exiting."; exit -1; }
done

#mpsky serve /caches/eph.60792.2025-04-27.bin ${@}
