#!/usr/bin/env bash
# Quality gate for gtcrn-ladspa-ort.
#
# Modes:
#   --ci    blocking gate (rustfmt, clippy, tests, deny, machete, lizard, jscpd)
#   --full  --ci + slower informational checks (typos, tarpaulin, geiger, scc)
#   --fix   autofix what's autofixable (fmt, clippy --fix)
#
# Exit non-zero on any blocking failure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly ROOT_DIR
readonly CCN_THRESHOLD=25
readonly JSCPD_MIN_LINES=50
readonly JSCPD_MIN_TOKENS=100
readonly JSCPD_THRESHOLD=5

cd "${ROOT_DIR}"

mode="${1:---ci}"

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$1"; }
ok()   { printf '\033[1;32m✓ %s\033[0m\n' "$1"; }
warn() { printf '\033[1;33m! %s\033[0m\n' "$1"; }

run_blocking() {
    local name="$1"; shift
    step "${name}"
    if "$@"; then
        ok "${name}"
    else
        printf '\033[1;31m✗ %s failed\033[0m\n' "${name}" >&2
        exit 1
    fi
}

run_informational() {
    local name="$1"; shift
    step "${name} (informational)"
    if "$@"; then
        ok "${name}"
    else
        warn "${name} reported issues — informational, not blocking"
    fi
}

case "${mode}" in
    --fix)
        step "rustfmt --fix"
        cargo fmt
        step "clippy --fix"
        cargo clippy --all-targets --features dynamic --no-default-features --fix --allow-dirty --allow-staged \
            -- -D warnings || warn "clippy --fix had unfixable warnings"
        ok "autofix done"
        exit 0
        ;;
    --ci|--full)
        ;;
    *)
        printf 'Usage: %s [--ci|--full|--fix]\n' "$0" >&2
        exit 2
        ;;
esac

# ── Blocking gate ──────────────────────────────────────────────────
run_blocking "rustfmt"       cargo fmt --check
run_blocking "clippy"        cargo clippy --all-targets --features dynamic --no-default-features -- -D warnings
run_blocking "cargo test"    cargo test --features dynamic --no-default-features

if command -v cargo-deny >/dev/null 2>&1; then
    run_blocking "cargo deny" cargo deny check
else
    warn "cargo-deny not installed — skipping (cargo install cargo-deny)"
fi

if command -v cargo-machete >/dev/null 2>&1; then
    run_blocking "cargo machete" cargo machete
else
    warn "cargo-machete not installed — skipping (cargo install cargo-machete)"
fi

if command -v lizard >/dev/null 2>&1; then
    run_blocking "lizard CCN ≤ ${CCN_THRESHOLD}" \
        lizard src/ -l rust -C "${CCN_THRESHOLD}" -w
else
    warn "lizard not installed — skipping (pipx install lizard)"
fi

if command -v jscpd >/dev/null 2>&1; then
    run_blocking "jscpd duplicates" \
        jscpd \
            --min-lines "${JSCPD_MIN_LINES}" \
            --min-tokens "${JSCPD_MIN_TOKENS}" \
            --threshold "${JSCPD_THRESHOLD}" \
            --reporters console \
            --silent \
            src/
else
    warn "jscpd not installed — skipping (npm install -g jscpd)"
fi

if [[ "${mode}" == "--ci" ]]; then
    ok "All blocking checks passed"
    exit 0
fi

# ── Informational (--full only) ────────────────────────────────────
if command -v typos >/dev/null 2>&1; then
    run_informational "typos" typos src/
fi

if command -v cargo-tarpaulin >/dev/null 2>&1; then
    run_informational "tarpaulin coverage" cargo tarpaulin --lib --skip-clean
fi

if command -v cargo-outdated >/dev/null 2>&1; then
    run_informational "cargo outdated" cargo outdated --exit-code 1
fi

if command -v cargo-geiger >/dev/null 2>&1; then
    run_informational "cargo geiger (unsafe count)" cargo geiger --quiet
fi

if command -v scc >/dev/null 2>&1; then
    run_informational "scc complexity by file" scc src/ --by-file --sort complexity
fi

ok "Full check finished"
