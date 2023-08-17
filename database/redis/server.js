const express = require('express');
const axios = require('axios');
const cors = require('cors');

const app = express();

const DEFAULT_EXPIRATION = 30 * 60; // 30 min

app.use(cors());

app.get('/photos/', async (req, res) => {
  const albumId = req.query.albumId;
  const { data } = await axios.get(
    `https://jsonplaceholder.typicode.com/photos/`,
    { params: { albumId } }
  );
  res.json(data);
});

// start
app.listen(3000, () => {
  console.log('Server listening on port 3000.');
});
