const express = require('express');
const router = express.Router();

const crypto = require('crypto-promise');
const pool = require('../../module/pool.js');

//POST localhost:3000/signup
router.post('/', async function (req, res) {

    let id = req.body.id;
    let pw = req.body.pw;

    if (!id || !pw) { // 클라가 아이디나 비번 값은 안줬을 때
        res.status(400).send({ 
            message : "Null Value"
        });
    } else { // 아이디 중복 확인
        let checkQuery = 'SELECT * FROM user WHERE id = ?'; // 동적으로 들어가는 것
        let checkResult = await pool.queryParamCnt_Arr(checkQuery, [id]);

        if (!checkResult) { // 서버가 결과 값을 주지 않음
            res.status(500).send({ 
                message : "Internal Server Error"
            });
        } else if (checkResult.length === 1) { // 중복 아이디 존재
            res.status(400).send({
                message : "Already Exists"
            });
        } else { // 잘되었을 때
            let salt = await crypto.randomBytes(32);
            let hashed = await crypto.pbkdf2(pw, salt.toString('base64'), 100000, 32, 'sha512');

            let insertQurey = 'INSERT INTO user (id, pw, salt) VALUES (?, ?, ?)';
            let insertResult = await pool.queryParamCnt_Arr(insertQurey, [id, hashed.toString('base64'), salt.toString('base64')]);
            console.log("insertResult : ", insertResult);

            if (!insertResult) {
                res.status(500).send({
                    message : "Internal Server Error"
                });
            } else {
                console.log("insertResult: ", insertResult);
                res.status(201).send({
                    message : "Success to Register"
                });
            }
        }
    }
    
});

module.exports = router;