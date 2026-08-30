#!/bin/sh
# Installs the latest (or a pinned) pico release for the current OS/arch.
#
#   curl -fsSL https://raw.githubusercontent.com/reno/pico-code/main/install.sh | sh
#
# Env vars:
#   VERSION      release tag to install, e.g. "v0.3.0" (default: latest)
#   INSTALL_DIR  where to place the binary (default: /usr/local/bin, falling
#                back to ~/.local/bin if that isn't writable)
#   BIN_NAME     command name to install as (default: pico). Use this when a
#                different pico — commonly nano's Pine-compatibility alias at
#                /usr/bin/pico — is already on your PATH and you want to keep
#                both, e.g. BIN_NAME=pico-code.
set -eu

repo="reno/pico-code"

os=$(uname -s)
case "$os" in
	Linux) goos=linux ;;
	Darwin) goos=darwin ;;
	*)
		echo "error: unsupported OS '$os' (pico only ships linux/darwin builds)" >&2
		exit 1
		;;
esac

arch=$(uname -m)
case "$arch" in
	x86_64 | amd64) goarch=amd64 ;;
	arm64 | aarch64) goarch=arm64 ;;
	*)
		echo "error: unsupported architecture '$arch'" >&2
		exit 1
		;;
esac

version=${VERSION:-}
if [ -z "$version" ]; then
	# -f is deliberately absent: on a non-2xx we want the status code and body
	# to tell the user *why* there is no tag, not an empty string.
	response=$(curl -sSL -w '\n%{http_code}' "https://api.github.com/repos/$repo/releases/latest" 2>/dev/null || true)
	status=$(printf '%s\n' "$response" | tail -1)
	version=$(printf '%s\n' "$response" | grep '"tag_name"' | head -1 | cut -d '"' -f 4)
	if [ -z "$version" ]; then
		case "$status" in
			404)
				echo "error: $repo has no published releases yet, so there is no binary to download" >&2
				echo "       build from source instead: go install github.com/$repo/cmd/pico@latest" >&2
				;;
			403 | 429)
				echo "error: GitHub API rate limit hit while looking up the latest release" >&2
				echo "       retry later, or pin a known tag: VERSION=v0.2.0 sh install.sh" >&2
				;;
			000 | "")
				echo "error: could not reach the GitHub API — check your network connection" >&2
				;;
			*)
				echo "error: could not resolve the latest release tag (GitHub API returned HTTP $status)" >&2
				;;
		esac
		exit 1
	fi
fi

archive="pico_${goos}_${goarch}.tar.gz"
base_url="https://github.com/$repo/releases/download/$version"
tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

echo "Downloading pico $version for $goos/$goarch..."
curl -fsSL "$base_url/$archive" -o "$tmp_dir/$archive"
curl -fsSL "$base_url/checksums.txt" -o "$tmp_dir/checksums.txt"

(
	cd "$tmp_dir"
	if command -v sha256sum >/dev/null 2>&1; then
		grep " $archive\$" checksums.txt | sha256sum -c -
	elif command -v shasum >/dev/null 2>&1; then
		grep " $archive\$" checksums.txt | shasum -a 256 -c -
	else
		echo "warning: no sha256sum/shasum found, skipping checksum verification" >&2
	fi
)

tar -xzf "$tmp_dir/$archive" -C "$tmp_dir" pico

bin_name=${BIN_NAME:-pico}

existing_pico=$(command -v "$bin_name" 2>/dev/null || true)

install_dir=${INSTALL_DIR:-/usr/local/bin}
if [ ! -w "$install_dir" ] 2>/dev/null; then
	install_dir="$HOME/.local/bin"
	mkdir -p "$install_dir"
fi

install -m 755 "$tmp_dir/pico" "$install_dir/$bin_name"
echo "Installed $("$install_dir/$bin_name" --version) to $install_dir/$bin_name"

if [ -n "$existing_pico" ] && [ "$existing_pico" != "$install_dir/$bin_name" ]; then
	echo "note: another '$bin_name' was already on your PATH at $existing_pico (often nano's Pine-compatibility alias) — $install_dir/$bin_name takes priority only if $install_dir comes first in PATH; re-run with BIN_NAME=pico-code to install under a different name and keep both"
fi

case ":$PATH:" in
	*":$install_dir:"*) ;;
	*) echo "note: $install_dir is not on your PATH — add it to use '$bin_name' directly" ;;
esac
