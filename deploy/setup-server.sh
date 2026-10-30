#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 24.04 server (AWS Lightsail Mumbai, Oracle Cloud, DigitalOcean...).
#   sudo git clone <your repo URL> /opt/arena
#   sudo bash /opt/arena/deploy/setup-server.sh
# Safe to run again: every step is idempotent.
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "Run with sudo."; exit 1; }
APP_DIR=/opt/arena
APP_USER=arena
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ADMIN_USER="${SUDO_USER:-}"
[ "$SRC_DIR" = "$APP_DIR" ] || { echo "Clone the repo to $APP_DIR first (the service files expect that path)."; exit 1; }

echo "== Packages"
apt-get update -q
DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
  python3 python3-venv sqlite3 gnupg chrony ufw fail2ban unattended-upgrades git
python3 -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ is required"'

echo "== Clock: IST, synced (order timing and the 15:15 square-off depend on it)"
timedatectl set-timezone Asia/Kolkata
systemctl enable --now chrony

echo "== Automatic security updates"
cat > /etc/apt/apt.conf.d/20auto-upgrades <<'CONF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
CONF

echo "== Firewall: SSH in, everything else closed (the panel is never exposed)"
ufw default deny incoming
ufw default allow outgoing
ufw allow OpenSSH
ufw --force enable
systemctl enable --now fail2ban

echo "== SSH: key login only"
if [ -n "$ADMIN_USER" ] && [ -s "/home/$ADMIN_USER/.ssh/authorized_keys" ]; then
  cat > /etc/ssh/sshd_config.d/10-arena-hardening.conf <<'CONF'
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin no
CONF
  systemctl reload ssh
else
  echo "   SKIPPED: no SSH key found for '${ADMIN_USER:-?}'. Add one, then run this script again."
fi

echo "== App user, virtualenv (hash-pinned dependencies), private files"
id "$APP_USER" >/dev/null 2>&1 || useradd --system --create-home --shell /usr/sbin/nologin "$APP_USER"
cd "$APP_DIR"
mkdir -p data/cache reports
[ -f .env ] || cp .env.example .env
chown -R "$APP_USER:$APP_USER" "$APP_DIR"
chmod 600 .env
chmod 700 data
sudo -u "$APP_USER" python3 -m venv .venv
sudo -u "$APP_USER" .venv/bin/pip install -q --upgrade pip
sudo -u "$APP_USER" .venv/bin/pip install -q --require-hashes -r requirements.lock
chmod +x deploy/backup.sh

echo "== Self-test"
sudo -u "$APP_USER" .venv/bin/python -m unittest discover -q tests

echo "== Services"
install -m 644 deploy/arena.service deploy/arena-backup.service deploy/arena-backup.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now arena-backup.timer

cat <<'NEXT'

Setup complete. Next:
  1. Add your keys:      sudo -u arena nano /opt/arena/.env
  2. Check everything:   cd /opt/arena && sudo -u arena .venv/bin/python -m arena check
  3. Start 24x7:         sudo systemctl enable --now arena
  4. Watch the log:      journalctl -u arena -f
  5. Open the panel from your computer through an SSH tunnel (see deploy/README.md).
NEXT
