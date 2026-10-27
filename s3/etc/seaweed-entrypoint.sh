#!/bin/sh
# starts SeaweedFS (S3 gateway) and creates arbitrary BUCKETS.

set -Cue -o pipefail

ready_file="/tmp/buckets.ready"

_shell() { echo "$*" | weed shell -master=localhost:9333 -filer=localhost:8888 2>/dev/null ; }

/entrypoint.sh server -s3 -s3.port=8333 -s3.config=/project/credentials.json &
server_pid=$!

for bucket in ${BUCKETS} ; do
  until _shell "s3.bucket.list" | grep -qw "${bucket}" ; do
    _shell "s3.bucket.create -name ${bucket}" > /dev/null || true
    sleep 1
  done
  echo "bucket-aws: bucket ${bucket} ready"
done

touch "${ready_file}"
wait "${server_pid}"
