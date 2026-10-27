## S3 compatible servers


````bash
docker compose up -d

cat >>~/.aws/config << EOF
[profile play]
endpoint_url = http://localhost:8333
EOF

cat >>~/.aws/credentials << EOF
[play]
aws_access_key_id = demo
aws_secret_access_key= demo-secret
EOF

aws --profile play s3 ls

aws --profile play cp ./README.md s3://playground/
````
