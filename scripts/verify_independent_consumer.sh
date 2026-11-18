#!/usr/bin/env bash
# Prove that Ferris can be consumed through its public Cargo API alone.
#
# 1. Clone the Ferris repository (default: this checkout) at a revision (default: HEAD)
#    into a temporary directory. Uncommitted changes are deliberately NOT included.
# 2. Copy examples/independent-consumer elsewhere and point its Git dependency at that
#    clone and revision.
# 3. Build and run it. It must print INDEPENDENT_CONSUMER_OK.
#
# Usage: scripts/verify_independent_consumer.sh [REPO_URL_OR_PATH] [REV]
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
repo="${1:-$here}"
rev="${2:-HEAD}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

git clone --quiet "$repo" "$work/ferris"
sha="$(git -C "$work/ferris" rev-parse "$rev")"
echo "Ferris clone: $repo @ $sha"

cp -R "$here/examples/independent-consumer" "$work/consumer"
manifest="$work/consumer/Cargo.toml"
# Replace the declared dependency line with the local clone pinned to the exact commit.
sed -i.bak -E "s#^ferris-cdch = .*#ferris-cdch = { git = \"file://$work/ferris\", rev = \"$sha\" }#" "$manifest"
grep '^ferris-cdch' "$manifest"

cd "$work/consumer"
CARGO_TARGET_DIR="$work/target" cargo run --quiet --release | tee "$work/out.txt"
grep -q INDEPENDENT_CONSUMER_OK "$work/out.txt"
echo "independent consumer verified against $sha"
