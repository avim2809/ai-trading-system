#!/usr/bin/env bash
# Launch IB Gateway headlessly via IBC, auto-logging in from ~/.ibc/config.ini.
# Usage: ./scripts/start_ibgateway.sh [paper|live]
#
# Directly exec'ing $INSTALL_DIR/ibgateway (the raw install4j launcher) only
# opens the GUI login dialog — it never reads config.ini. IBC is what
# performs the headless auto-login, so this drives IBC's own launcher
# (ibcstart.sh) under a virtual X display instead.
set -euo pipefail

MODE="${1:-paper}"
INSTALL_DIR="${IBKR_INSTALL_DIR:-/opt/ibgateway}"
IBC_DIR="${IBC_INSTALL_DIR:-/opt/ibc}"
IBC_INI="${IBC_INI:-$HOME/.ibc/config.ini}"
SETTINGS_PATH="${IBKR_SETTINGS_PATH:-$HOME/Jts}"
DISPLAY_NUM="${IBC_DISPLAY:-:99}"
LOG_DIR="$HOME/.ibc/logs"

if [ ! -d "$INSTALL_DIR" ]; then
    echo "IB Gateway not found at $INSTALL_DIR"
    echo "Run: ./setup.sh --components live"
    exit 1
fi

if [ ! -x "$IBC_DIR/scripts/ibcstart.sh" ]; then
    echo "IBC not found at $IBC_DIR"
    echo "Run: ./setup.sh --components live"
    exit 1
fi

if [ ! -f "$IBC_INI" ]; then
    echo "IBC config not found at $IBC_INI"
    exit 1
fi

if grep -q "YOUR_IBKR_USERNAME\|YOUR_IBKR_PASSWORD" "$IBC_INI"; then
    echo "IBC config at $IBC_INI still has placeholder credentials — edit it first."
    exit 1
fi

# IBC expects the classic "<settings>/ibgateway/<majorVersion>/jars" layout;
# the standalone installer setup.sh uses lays files flat under $INSTALL_DIR
# instead. Bridge the two with a one-time symlink rather than touching the
# real install.
MAJOR_VERSION=$(grep -oE 'majorVersion" value="[0-9]+"' "$INSTALL_DIR/.install4j/i4jparams.conf" | grep -oE '[0-9]+')
if [ -z "$MAJOR_VERSION" ]; then
    echo "Could not detect IB Gateway major version from $INSTALL_DIR/.install4j/i4jparams.conf"
    exit 1
fi
mkdir -p "$SETTINGS_PATH/ibgateway"
if [ ! -e "$SETTINGS_PATH/ibgateway/$MAJOR_VERSION" ]; then
    ln -s "$INSTALL_DIR" "$SETTINGS_PATH/ibgateway/$MAJOR_VERSION"
fi

mkdir -p "$LOG_DIR"

# A virtual X display is required even in headless mode — IBC drives the
# real Gateway Swing UI, it just does so without a physical monitor attached.
if ! pgrep -f "Xvfb $DISPLAY_NUM" >/dev/null 2>&1; then
    Xvfb "$DISPLAY_NUM" -screen 0 1024x768x24 -ac &
    sleep 2
fi

echo "Starting IB Gateway ($MODE) via IBC — logs: $LOG_DIR/ibgateway.log"
DISPLAY="$DISPLAY_NUM" exec "$IBC_DIR/scripts/ibcstart.sh" "$MAJOR_VERSION" -g \
    --tws-path="$SETTINGS_PATH" \
    --ibc-path="$IBC_DIR" \
    --ibc-ini="$IBC_INI" \
    --mode="$MODE" \
    --on2fatimeout=exit \
    > "$LOG_DIR/ibgateway.log" 2>&1
