require('dotenv').config();
const express = require('express');
const app = express();
const jwt = require('jsonwebtoken');

app.use(express.json());

// you would really store these in a DB:
let refreshTokens = [];

app.post('/login', (req, res) => {
  const username = req.body.username;
  // TODO: we would authenticate user first here

  // now assuming user has been authenticated:
  const user = { name: username };
  const accessToken = generateAccessToken(user);
  // expiration of refresh tokens will be handled manually
  const refreshToken = jwt.sign(user, process.env.REFRESH_TOKEN_SECRET);
  refreshTokens.push(refreshToken);
  res.json({ accessToken, refreshToken });
});

app.delete('/logout', (req, res) => {
  const oldCount = refreshTokens.length;
  refreshTokens = refreshTokens.filter(token => token !== req.body.token);
  console.log(`deleted ${lldCount - refreshTokens} refresh tokens`);
  res.sendStatus(204);
});

app.post('/token', (req, res) => {
  const refreshToken = req.body.token;
  if (refreshToken == null) return res.sendStatus(401);

  if (!refreshTokens.includes(refreshToken)) return res.sendStatus(403);

  jwt.verify(refreshToken, process.env.REFRESH_TOKEN_SECRET, (err, user) => {
    if (err) return res.sendStatus(403);
    // user object contains additional info on expiration so pull out user.name
    const accessToken = generateAccessToken({ name: user.name });
    res.json({ accessToken });
  });

  // check if refresh token is known
  //res.json({ accessToken, refreshToken });
});

// really quick expiration for testing purposes
// (better to use 10-15 min)...
function generateAccessToken(user) {
  return jwt.sign(user, process.env.ACCESS_TOKEN_SECRET, { expiresIn: '40s' });
}

app.listen(4001);
