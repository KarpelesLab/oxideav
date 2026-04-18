#!/bin/bash
# Clone every OxideAV/oxideav-* repo into crates/ so the workspace is
# fully self-contained — no crates.io resolution needed for any oxideav-*
# dep, local development and CI both work from a single `cargo build`.
#
# Usage:
#   ./scripts/clone-siblings.sh
#
# Behaviour:
#   * Enumerates OxideAV repos via the `gh` CLI.
#   * Skips repos that are not Rust crates (`.github`, `oxideav.github.io`,
#     `oxideav-dot-github`, `demo-repository`) and the workspace repo itself
#     (`oxideav-workspace`).
#   * For each remaining repo: if `crates/<name>` does not exist, clones
#     it; otherwise leaves it alone (doesn't fetch, doesn't touch local
#     work).
#
# CI-safe: uses HTTPS clones via `gh repo clone`, which works with the
# default `GITHUB_TOKEN` in GitHub Actions.

set -euo pipefail

cd "$(dirname "$0")/.."
repo_root="$(pwd)"
crates_dir="$repo_root/crates"

# Repos that are in the OxideAV org but are not Rust crates we need here.
# Extend this list if the org grows in other directions.
SKIP=(
    ".github"
    "demo-repository"
    "oxideav-dot-github"
    "oxideav-workspace"
    "oxideav.github.io"
)

is_skipped() {
    local name="$1"
    for s in "${SKIP[@]}"; do
        [ "$name" = "$s" ] && return 0
    done
    return 1
}

echo "Enumerating OxideAV/oxideav-* repos..."
names="$(gh repo list OxideAV --limit 200 --json name --jq '.[] | .name' | sort)"
if [ -z "$names" ]; then
    echo "error: gh repo list returned nothing — are you logged in (gh auth status)?" >&2
    exit 1
fi

mkdir -p "$crates_dir"

cloned=0
skipped=0
present=0

while IFS= read -r name; do
    [ -z "$name" ] && continue
    if is_skipped "$name"; then
        skipped=$((skipped + 1))
        continue
    fi
    # Only clone oxideav-* repos. The org may pick up unrelated names over
    # time; the prefix check keeps those out.
    case "$name" in
        oxideav-*) ;;
        *)
            skipped=$((skipped + 1))
            continue
            ;;
    esac
    target="$crates_dir/$name"
    if [ -e "$target/Cargo.toml" ]; then
        present=$((present + 1))
        continue
    fi
    if [ -e "$target" ]; then
        echo "warning: $target exists but has no Cargo.toml — moving aside to $target.bak" >&2
        mv "$target" "$target.bak"
    fi
    echo "clone: OxideAV/$name -> crates/$name"
    gh repo clone "OxideAV/$name" "$target" -- --quiet
    cloned=$((cloned + 1))
done <<< "$names"

echo ""
echo "Summary: $cloned cloned, $present already present, $skipped skipped."
