#!/bin/sh
set -eu

mode="${1:-}"
case "$mode" in
  forward)
    exec /usr/local/bin/hepemshow-forward "$@"
    ;;
  backward)
    exec /usr/local/bin/hepemshow-backward "$@"
    ;;
  *)
    echo "usage: hepemshow-native {forward|backward} [options]" >&2
    exit 2
    ;;
esac
