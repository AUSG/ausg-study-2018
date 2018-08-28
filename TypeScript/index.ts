import express, { Request, Response } from 'express'

const app = express()

app.get('/', (req: Request, res: Response) => {
  res.json({
    message: 'hello, world',
  })
})

app.listen(4000)
