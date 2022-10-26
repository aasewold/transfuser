#!/bin/sh

if [ -z "$PUID" ]; then
    echo "Warning: PUID should be set" >&2
    PUID=0
fi

if [ -z "$PGID" ]; then
    echo "Warning: PGID should be set" >&2
    PGID=0
fi

if ! getent passwd $PUID > /dev/null; then
    echo "Creating password entry for $PUID" >&2
    groupadd -g $PGID user
    useradd -u $PUID -g $PGID -m user
fi

if [ ! -z "$TZ" ]; then
    echo "Setting timezone to $TZ" >&2
    ln -snf "/usr/share/zoneinfo/$TZ" /etc/localtime
    echo "$TZ" > /etc/timezone
    dpkg-reconfigure -f noninteractive tzdata
fi

if [ "$PUID" -gt 0 ]; then
    echo "PS1='\[\e]0;\u@carla-container: \w\a\]${debian_chroot:+($debian_chroot)}\u@carla-container:\w\$ '" >> /home/user/.bashrc
    exec runuser -u user -- "$@"
else
    echo "PS1='\[\e]0;\u@carla-container: \w\a\]${debian_chroot:+($debian_chroot)}\u@carla-container:\w# '" >> /root/.bashrc
    exec "$@"
fi
