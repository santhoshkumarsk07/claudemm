#!/usr/bin/env bash
# Nightly journal backup: a consistent SQLite snapshot, compressed, optionally encrypted
# (BACKUP_PASSPHRASE_FILE) and copied off the server (BACKUP_RCLONE_REMOTE). Keeps 30 days.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=data/backups
mkdir -p "$DEST"
STAMP=$(date +%F)
for db in data/arena.db data/demo.db; do
  [ -f "$db" ] || continue
  out="$DEST/$(basename "$db" .db)-$STAMP.db"
  sqlite3 "$db" ".backup '$out'"
  gzip -f "$out"
  if [ -n "${BACKUP_PASSPHRASE_FILE:-}" ]; then
    gpg --batch --yes --pinentry-mode loopback --passphrase-file "$BACKUP_PASSPHRASE_FILE" \
        --symmetric --cipher-algo AES256 -o "$out.gz.gpg" "$out.gz"
    rm -f "$out.gz"
  fi
done
find "$DEST" -type f -mtime +30 -delete
if [ -n "${BACKUP_RCLONE_REMOTE:-}" ]; then
  rclone copy "$DEST" "$BACKUP_RCLONE_REMOTE" --max-age 2d
fi
echo "Backup done: $(ls "$DEST" | grep -c "$STAMP") file(s) for $STAMP"
