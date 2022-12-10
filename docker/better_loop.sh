#!/usr/bin/bash


if [ -d "$1" ]; then
    MODELS="$1"
else
    echo "Usage: $0 <models> [target.json]"
    exit 1
fi

if [ ! -z "$2" ]; then
    TARGET="$2"
else
    TARGET=$MODELS/results/$(date +%Y-%m-%d_%H-%M-%S).json
fi

echo "Models: $MODELS"
echo "Target: $TARGET"

echo "Correct?"
read || exit 1

if [ ! -f "$TARGET" ]; then
    echo "Creating target file"
    mkdir -p "$(dirname "$TARGET")"
    touch "$TARGET"
fi

_continue() {
    done=$(jq \
        '._checkpoint.progress[0] >= ._checkpoint.progress[1]' \
        "$TARGET"
    )
    if [ "$done" = "true" ]; then
        return 1
    else
        return 0
    fi
}

while _continue
do
    ./docker/eval.sh \
        --carla \
        longest6 \
        /"$MODELS" \
        "$TARGET" \
        && break;
    n=$((n+1))
done
