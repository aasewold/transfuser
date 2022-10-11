#!/bin/sh

if [ -z "$VIRTUAL_ENV" ]; then
    echo "Please activate your virtual environment first."
    exit 1
fi

pip install wheel
pip install -r requirements.txt
