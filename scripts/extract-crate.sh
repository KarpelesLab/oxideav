#!/bin/bash
# Extract one workspace crate into a standalone OxideAV/<name> repo.
#
# Usage: scripts/extract-crate.sh <crate-name> "<github description>"
#
# Assumes:
#   - /home/magicaltux/projects/oxideav/oxideav   = this monorepo
#   - /home/magicaltux/projects/oxideav/<name>    = target sibling
#   - /tmp/sibling-ci/{ci,release-plz}.yml + release-plz.toml exist
#   - /tmp/branch-protection.json exists
#
# Idempotent: re-running on a crate that's already out just builds + pushes
# any local-only edits. Does NOT modify the monorepo's workspace members
# or Cargo.toml — that's a separate cleanup pass after all extractions.

set -euo pipefail

NAME="$1"
DESC="$2"
PARENT="/home/magicaltux/projects/oxideav"
MONOREPO="$PARENT/oxideav"
SIBLING="$PARENT/$NAME"
SKEL="/tmp/sibling-ci"

cd "$MONOREPO"

if [ ! -d "crates/$NAME" ]; then
    echo "crates/$NAME missing — already extracted?"
    exit 0
fi

if [ ! -d "$SIBLING" ]; then
    echo "=== $NAME: subtree split + create repo + push + clone ==="
    git subtree split --prefix="crates/$NAME" -b "extract-$NAME" 2>&1 | tail -1
    gh repo create "OxideAV/$NAME" --public --description "$DESC" 2>&1 | tail -1 || true
    git push "git@github.com:OxideAV/$NAME.git" "extract-$NAME:master" 2>&1 | tail -1
    cd "$PARENT"
    git clone "git@github.com:OxideAV/$NAME.git"
    cd "$MONOREPO"
    git branch -D "extract-$NAME" 2>&1 | tail -1 || true
fi

cd "$SIBLING"

# Rewrite Cargo.toml to standalone form.
sed -i \
  -e 's|^version\.workspace = true$|version = "0.0.3"|' \
  -e 's|^edition\.workspace = true$|edition = "2021"|' \
  -e 's|^rust-version\.workspace = true$|rust-version = "1.80"|' \
  -e 's|^license\.workspace = true$|license = "MIT"|' \
  -e "s|^repository\\.workspace = true\$|repository = \"https://github.com/OxideAV/$NAME\"|" \
  -e 's|^authors\.workspace = true$|authors = ["Mark Karpeles"]|' \
  -e "s|^homepage\\.workspace = true\$|homepage = \"https://github.com/OxideAV/$NAME\"|" \
  -e 's|thiserror = { workspace = true }|thiserror = "1"|' \
  -e 's|clap = { workspace = true.*|clap = { version = "4", features = ["derive"] }|' \
  Cargo.toml

# oxideav-* workspace deps → "0.0"
sed -i -E 's|(oxideav-[a-z0-9-]+) = \{ workspace = true \}|\1 = "0.0"|g' Cargo.toml

# Infra: LICENSE, .gitignore, CI, release-plz.
[ -f LICENSE ] || cp "$MONOREPO/LICENSE" LICENSE
[ -f .gitignore ] || echo "/target" > .gitignore
mkdir -p .github/workflows
cp "$SKEL/ci.yml" .github/workflows/ci.yml
cp "$SKEL/release-plz.yml" .github/workflows/release-plz.yml
[ -f release-plz.toml ] || cp "$SKEL/release-plz.toml" release-plz.toml

# README: repoint any remaining monorepo URL + update install version hint.
if [ -f README.md ]; then
    sed -i \
      -e 's|KarpelesLab/oxideav|OxideAV/oxideav-workspace|g' \
      -e 's|https://github.com/OxideAV/oxideav-workspace/blob/master/LICENSE|LICENSE|g' \
      -e "s|^$NAME = \"0\\.0\\.[0-9]\"$|$NAME = \"0.0\"|" \
      README.md
fi

# Build sanity-check (best-effort — deferred deps may not be published
# yet, so don't gate the extraction on standalone build success).
cargo build 2>&1 | tail -2 || true

# Commit any local-only changes.
git add -A
if ! git diff --cached --quiet; then
    git -c user.email=magicaltux@gmail.com -c user.name="Mark Karpeles" \
      commit -m "extract: make crate standalone (pin deps, add CI + release-plz + LICENSE)" \
      2>&1 | tail -1
    git push origin master 2>&1 | tail -1
fi

# Branch protection (idempotent).
gh api --method PUT "/repos/OxideAV/$NAME/branches/master/protection" \
    --input /tmp/branch-protection.json > /dev/null 2>&1 || true

echo "=== $NAME: done ==="
