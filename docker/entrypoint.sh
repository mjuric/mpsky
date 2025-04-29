#!/usr/bin/env bash
set -euxo pipefail

PID=
CACHEFN=

# make sure we clean up mpsky serve if we're interrupted
trap 'kill $PID 2>/dev/null; exit' INT

# fetch the latest cache file for a given MJD
CACHEURL="https://epyc.astro.washington.edu/~mjuric/mpsky-caches"
latest_available_cache()
{
	local MJD=$1
	curl -s "$CACHEURL/" | grep -oP 'href="\K[^"]+' | grep -vE '^\.\.?$' | sort | grep -E "eph\\.$MJD\\..*\\.bin"
}

while :
do
	# wait until we check for newer files, unless we're here for the first time
	[[ -z $PID ]] || sleep 10;

	# Compute the MJD of the current (local time) observing night in La Serena
	OBSDATE=$(TZ="America/Santiago" date --date='12 hours ago' +"%Y/%m/%d")
	MJD=$(echo "$(date --date="$OBSDATE" +%s) / 86400.0 + 2440587.5 - 2400000.5" | bc -l | cut -f 1 -d .)

	echo $OBSDATE $MJD
	exit

	# fetch the most recent file for that MJD
	NEW_CACHEFN=$(latest_available_cache $MJD)

	# if the file has changed, download and restart the server
	if [[ "$CACHEFN" != "$NEW_CACHEFN" ]]; then
		## download the cache file
		curl "$CACHEURL/$NEW_CACHEFN" -o "/caches/$NEW_CACHEFN" || { continue; }

		## kill current server process
		[[ ! -z $PID ]] && { kill $PID; wait $PID; }

		## serve the new cache
		mpsky serve "/caches/$NEW_CACHEFN" &
		PID=$!

		## delete the old cache
		[[ ! -z "$CACHEFN" ]] && rm -f "/caches/$CACHEFN"
		CACHEFN="$NEW_CACHEFN"
	fi

	# test that the process is alive
	kill -0 $PID || { echo "error: process $PID not alive; exiting."; exit -1; }
done

#mpsky serve /caches/eph.60792.2025-04-27.bin ${@}
