# Elastic Search
A playground for experimenting with / learning elastic search.


## Usage
````bash
cp .env.sample .env

docker compose up -d

# get terminal inside elastic container
docker exec -it es-elastic-1 bash
````

Now you can visit kibana here http://localhost:5601/

## References
* https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html
  * [tutorial's docker-compose.yml](https://github.com/elastic/elasticsearch/blob/8.12/docs/reference/setup/install/docker/docker-compose.yml)
* https://www.elastic.co/guide/en/kibana/current/docker.html