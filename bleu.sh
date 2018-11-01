#!/usr/bin/env bash
./multeval.sh eval --refs ./data/mscoco/test.target.txt \
                   --hyps-baseline ./data/mscoco/trimed_source_target.txt \
                   --meteor.language en