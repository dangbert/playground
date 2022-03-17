require('dotenv').config();
const express = require('express');
const app = express();
const jwt = require('jsonwebtoken');


const posts = [
  {username: 'Kyle', title: 'post1'},
  {username: 'Jim', title: 'post2'},
];

app.use(express.json());

// runs authenticateToken as preprocessor
app.get('/posts', authenticateToken, (req, res) => {
  console.log('looking for posts from user: ');
  console.log(req.user.name);
  res.json(posts.filter(p => p.username == req.user.name));
});

app.post('/login', (req, res) => {
  const username = req.body.username;
  // TODO: we would authenticate user first here

  // now assuming user has been authenticated:
  const user = { name: username };
  const accessToken = jwt.sign(user, process.env.ACCESS_TOKEN_SECRET);
  res.json({ accessToken });
});


// middleware, get user if provided token is valid
function authenticateToken(req, res, next) {
  const authHeader = req.headers['authorization'];
  // get token portion of:
  // Bearer TOKEN
  const token = authHeader && authHeader.split(' ')[1];
  if (token == null) {
    return res.sendStatus(401);
  }

  jwt.verify(token, process.env.ACCESS_TOKEN_SECRET, (err, user) => {
    // token is not valid (or expired)
    if (err) return err.sendStatus(403);

    req.user = user;
    next();
  });

}

app.listen(3001);
