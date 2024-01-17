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
3. Go to "Mangement > Dev Tools" to start making queries (e.g. run `GET _cluster/health`)
  * [example queries from tutorial here](https://github.com/LisaHJung/Part-1-Intro-to-Elasticsearch-and-Kibana/tree/main#getting-information-about-cluster-and-nodes)

Misc tricks:
````bash
# reset elastic user's password (if desired)
#   (update ELASTIC_PASSWORD in .env file afterwards)
docker exec -it es01 /usr/share/elasticsearch/bin/elasticsearch-reset-password -u elastic

# get terminal inside elastic container
docker exec -it es01 bash
````

## References
* Docker setup:
  * https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html
    * [tutorial's docker-compose.yml](https://github.com/elastic/elasticsearch/blob/8.12/docs/reference/setup/install/docker/docker-compose.yml)
    * [more minimal docker-compose.yml](https://github.com/LisaHJung/Part-1-Intro-to-Elasticsearch-and-Kibana/blob/main/docker-compose.yml)
  * https://www.elastic.co/guide/en/kibana/current/docker.html

* [playlist: Official Beginners Crash Course](https://www.youtube.com/playlist?list=PL_mJOmq4zsHZYAyK606y7wjQtC0aoE6Es)
  * [companion repo](https://github.com/LisaHJung/Beginners-Crash-Course-to-Elastic-Stack-Series-Table-of-Contents)