const express = require('express');
const router = express.Router();

const crypto = require('crypto-promise');
const pool = require('../../module/pool.js');

//POST localhost:3000/signin
router.post('/', async function (req, res) {

    // FIXME const로 변경
    // const {id, pw} = req.body -> 간결하게 표현 가능
    let id = req.body.id;
    let pw = req.body.pw;

    if (!id || !pw) { // 클라가 아이디나 비번 값은 안줬을 때
        return res.status(400).send({
            message : "Null Value"
        });
    }

    // FIXME const로 변경
    let checkQuery = 'SELECT user_idx, id, pw, salt FROM user WHERE id = ?';
    let checkResult = await pool.queryParamCnt_Arr(checkQuery, [id]);

    if (!checkResult) {
        return res.status(500).send({
            message : "Internal Server Error"
        });
    } else if (checkResult.length === 0) {
        return res.status(400).send({
            message : "Login Failed"
        });
    }

    let hashed = await crypto.pbkdf2(pw, checkResult[0].salt, 100000, 32, 'sha512'); // 값이 하나여도 배열로 넘어온다
    console.log(hashed.toString)
    if (hashed.toString('base64') === checkResult[0].pw) { // 암호화한 값과 디비 비번이 같을 때
        return res.status(201).send({
            message : "Success Login"
        });
    } else { // 암호화한 값과 디비 비번이 다를 때
        console.log('Wrong pw');

        return res.status(400).send({
            message : "Login Failed"
        });
    }

});

module.exports = router;
