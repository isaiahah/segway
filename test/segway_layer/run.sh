#!/usr/bin/env bash

#Test segway-layer command

set -o nounset -o pipefail -o errexit

if [ $# != 0 ]; then
    echo usage: "$0"
    exit 2
fi

testdir="$(mktemp -dp . "test-$(date +%Y%m%d).XXXXXX")"
echo >&2 "entering directory $testdir"
cd "$testdir"

segway-layer -m ../mnemonics -b segway.layered.bb < ../segway.bed > segway.layered.bed

cd ..

../compare_directory.py ../segway_layer/touchstone ../segway_layer/${testdir#"./"}
