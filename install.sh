#!/usr/bin/env bash
# Link the three commands into a directory on $PATH (default: ~/.local/bin).
#
#   ./install.sh                 # -> ~/.local/bin
#   ./install.sh ~/bin           # -> ~/bin
set -euo pipefail

here="$(dirname "$(readlink -f "$0")")"
target="${1:-$HOME/.local/bin}"
mkdir -p "$target"

for name in audiosr audiosr-play audiosr-sink; do
  chmod +x "$here/$name"
  ln -sfn "$here/$name" "$target/$name"
  printf 'linked %s -> %s\n' "$target/$name" "$here/$name"
done

case ":$PATH:" in
  *":$target:"*) ;;
  *) printf '\nnote: %s is not on $PATH\n' "$target" ;;
esac

printf '\ntry: audiosr --check\n'
