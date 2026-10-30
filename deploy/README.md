# Run the Arena 24×7 on a server

A small always-on Linux server in India keeps the agent running through market hours without your computer. The control panel is **never exposed to the internet**: you reach it through an SSH tunnel.

## 1. Pick a server

| Option | Approx. cost (check current pricing) | Notes |
|---|---|---|
| **AWS Lightsail, Mumbai**, 2 GB RAM (recommended) | ~US$10–12 / month | Fixed price, static IP free while attached, snapshots, reliable |
| Oracle Cloud *Always Free*, Mumbai/Hyderabad | Free | Free ARM capacity is often unavailable in Indian regions, and idle free instances can be reclaimed. Fine for paper and shadow trading. |
| DigitalOcean, Bangalore, 2 GB | ~US$12 / month | Reserved IP free while assigned |

Choose **Ubuntu 24.04 LTS**. 2 vCPU / 2 GB RAM / 40 GB disk is plenty.

## 2. Create it (Lightsail example)

1. Lightsail console → **Create instance** → region **Mumbai (ap-south-1)** → *Linux/Unix* → *OS only* → **Ubuntu 24.04 LTS** → the 2 GB plan → Create.
2. **Networking** tab → **Create static IP** and attach it to the instance. Brokers may ask you to whitelist this IP for API trading.
3. Same tab, **IPv4 firewall**: keep only **SSH (22)**; delete HTTP (80). Optionally restrict SSH to your own IP.
4. Connect with SSH (Lightsail's browser terminal or your own key) and add your personal SSH public key to `~/.ssh/authorized_keys`.

## 3. Install

```bash
sudo git clone https://github.com/santhoshkumarsk07/claudemm.git /opt/arena
sudo bash /opt/arena/deploy/setup-server.sh
```

For a **private** repository, clone with a read-only fine-grained GitHub token or a deploy key.

The script:
- installs Python, SQLite, time sync and security tools;
- sets the clock to IST, with automatic time sync;
- turns on automatic security updates;
- configures a firewall that only allows SSH in, plus `fail2ban`;
- switches SSH to key-only login, once it finds your key;
- creates a locked-down `arena` user;
- installs hash-pinned dependencies;
- runs the test suite;
- registers the service and the nightly backup timer.

It is safe to run again.

## 4. Configure and start

```bash
sudo -u arena nano /opt/arena/.env          # ANTHROPIC_API_KEY=..., and Upstox keys if you use that feed
cd /opt/arena && sudo -u arena .venv/bin/python -m arena check
sudo systemctl enable --now arena
journalctl -u arena -f                      # live log; Ctrl+C to stop watching
```

The service runs `python -m arena gui --autostart`: it resumes the active run, waits for market hours, restarts itself after a crash or reboot, and restarts trading if it stops with an error. A Stop pressed in the panel is respected until you press Start again.

## 5. Open the control panel from your computer or phone

The panel listens on `127.0.0.1:8787` on the server only. Open a tunnel:

```bash
ssh -N -L 8787:127.0.0.1:8787 ubuntu@YOUR_STATIC_IP
```

Then browse to **http://127.0.0.1:8787** on your own machine. On a phone, SSH apps such as Termius support the same port forwarding.

**Even safer (optional):** install [Tailscale](https://tailscale.com) (free for personal use) on the server and your devices, SSH over the Tailscale address, then remove port 22 from the Lightsail firewall entirely. The server then has no open public ports at all.

## 6. Daily Upstox login (only if `feed = "upstox"`)

Upstox access tokens expire every day. Each trading morning, before 09:15:

```bash
cd /opt/arena && sudo -u arena .venv/bin/python -m arena upstox-login
sudo systemctl restart arena
```

## 7. Updates

```bash
cd /opt/arena
sudo -u arena git pull
sudo -u arena .venv/bin/pip install --require-hashes -r requirements.lock
sudo systemctl restart arena
```

## 8. Backups and the audit trail

- Every day at 16:30 IST, `deploy/backup.sh` writes a consistent snapshot of the journal to `/opt/arena/data/backups/` and keeps 30 days.
- **Encrypt and copy off the server** by creating `/opt/arena/data/backup.env` (owner `arena`, mode 600):
  ```
  BACKUP_PASSPHRASE_FILE=/opt/arena/data/backup.pass   # a file holding a long random passphrase
  BACKUP_RCLONE_REMOTE=gdrive:arena-backups            # optional: install rclone and configure a remote
  RCLONE_CONFIG=/opt/arena/data/rclone.conf
  ```
  Keep a copy of the passphrase somewhere other than the server, or the backups can't be restored.
- Restore an encrypted backup with:
  `gpg -d arena-YYYY-MM-DD.db.gz.gpg | gunzip > arena.db`
- Run `python -m arena audit` weekly. It verifies the tamper-evident event log and prints a **head hash**; note it down outside the server (for example in a message to yourself). If the log is ever edited or events are deleted, verification fails or the head hash stops matching what you noted.
- Lightsail **automatic snapshots** (a small extra cost) add whole-server recovery.

## Oracle Cloud notes

Oracle's Ubuntu images ship with their own iptables rules. If `ufw` and those rules conflict, SSH stays allowed but check `sudo iptables -L` after setup. The VCN security list only needs port 22 open.
