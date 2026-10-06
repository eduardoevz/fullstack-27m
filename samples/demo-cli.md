# Transcripción de la demo CLI (`python src/cli.py`)

Salida real del modelo `best.pt` (paso 15.000), T=0.7, sin selección. Lo que va tras `--- completando ---` y el código escrito por el usuario es lo que genera el modelo. Es una transcripción, no un video.

```text
fullstack-27m (paso 15000) | lenguaje: tsx | T=0.7 | 200 tokens
Escribe el inicio de tu codigo y termina con una linea que solo tenga un punto (.)
  :lang ts|tsx|js|jsx|py   cambia el lenguaje      :temp 0.7    temperatura (0 = determinista)
  :tokens 200              largo de la continuacion  :quit        salir

>>> ... ... 
--- completando ---
import express from "express";
const app = express();
const server = app.listen(3000, () => {});

console.log(`Server listening on port ${server.address().port}`);

app.use(express.static(__dirname + '/public'));
app.use(express.static(__dirname + '/../'));

console.log(`Serving the application on port ${server.address().port}`);

app.use('/', express.static(__dirname + '/../'));

const server = express();
server.use(express.static(__dirname + '/../'));
server.use(express.static(__dirname + '/../'));

server.listen(3000);

const port = process.env.PORT ||
--- fin ---

>>> lenguaje: tsx | T=0.7 | 200 tokens
>>> ... ... ... ... 
--- completando ---
import { useState } from "react";

export default function Counter() {
  const [count, setCount] = useState(0);
  const [countOrEmpty, setCountOrEmpty] = useState(0);

  return (
    <>
      <div>
        <span>
          Count: {countOrEmpty}
          <span>
            Count: {countOrEmpty}
          </span>
        </span>
      </div>
    </>
  );
}

--- fin ---

>>> lenguaje: py | T=0.7 | 200 tokens
>>> ... ... ... ... 
--- completando ---
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def index(request):
    response = make_response(json.dumps({"data": "{"+str(request.url)+"}}"), mimetype="application/json")
    return response

@app.get("/test")
def test(request):
    response = make_response(json.dumps({"data": "{"+str(request.url)+"}}"), mimetype="application/json")
    return response

--- fin ---

>>> ```
