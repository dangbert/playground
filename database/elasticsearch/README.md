# Elastic Search
A playground for experimenting with / learning elastic search.


## Usage
````bash
cp .env.sample .env

docker compose up -d

# generate enrollment token for kibana:
docker exec -it es-elastic-1 /usr/share/elasticsearch/bin/elasticsearch-create-enrollment-token -s kibana

# watch logs (kibana will print verification code later)
docker compose logs -f
````


Kibana setup:
1. Visit kibana here http://localhost:5601/ (enter enrollment token, and watch kibana's logs to see the verification code).
2. Login with user `elastic`, password (see `ELASTIC_PASSWORD` in .env file).

Misc tricks:
````bash
# reset elastic user's password (if desired)
#   (update ELASTIC_PASSWORD in .env file afterwards)
docker exec -it es-elastic-1 /usr/share/elasticsearch/bin/elasticsearch-reset-password -u elastic

# get terminal inside elastic container
docker exec -it es-elastic-1 bash
````

## References
* https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html
  * [tutorial's docker-compose.yml](https://github.com/elastic/elasticsearch/blob/8.12/docs/reference/setup/install/docker/docker-compose.yml)
* https://www.elastic.co/guide/en/kibana/current/docker.html