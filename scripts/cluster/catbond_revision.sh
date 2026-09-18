#!/bin/bash
# Reuse the tested production inventory with a cat-bond-specific paired pilot
# and separate output/manifest roots. Never submit the old FHCF comparison here.
export EF_CAMPAIGN=catbond
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/earths_future.sh" "$@"
