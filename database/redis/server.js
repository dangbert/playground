const express = require('express');
const axios = require('axios');
const cors = require('cors');
const Redis = require('redis');

console.log('creating redis client...');
const redisClient = Redis.createClient(); // using default URL
console.log('done!');

const app = express();
app.use(express.urlencoded({ extended: true }));
app.use(cors());

const DEFAULT_EXPIRATION = 30 * 60; // 30 min

app.get('/photos/', async (req, res) => {
  const albumId = req.query.albumId;

  const photos = await redisClient.get(
    `photos?albumId=${albumId}`,
    async (error, photos) => {
      if (error) console.error(error);

      if (photos != null) {
        console.log('cache hit');
        return res.json(JSON.parse(photos));
      }

      console.log('cache miss');
      const { data } = await axios.get(
        `https://jsonplaceholder.typicode.com/photos/`,
        { params: { albumId } }
      );

      console.log('cached photos data to redis');
      redisClient.setex(
        `photos?albumId=${albumId}`,
        DEFAULT_EXPIRATION,
        JSON.stringify(data)
      );
      return res.json(data);
    }
  );
});

// start
app.listen(3000, () => {
  console.log('\nServer listening on port 3000...\n');
});
