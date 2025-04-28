#!/usr/bin/env bash
set -euxo pipefail

#
# Check if we have the caches
#
mpsky serve /caches/eph.60792.2025-04-27.bin ${@}
