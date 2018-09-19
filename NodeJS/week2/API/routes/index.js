var express = require('express');
var router = express.Router();

router.use('/signin', require('./login/signin.js'));
router.use('/signup', require('./login/signup.js'));

module.exports = router;