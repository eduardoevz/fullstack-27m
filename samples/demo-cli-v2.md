# Transcripción de la demo v2 con relleno en el cursor (`demo_v2.bat`)

Salida real del modelo final de la Fase 11 (`hito-0018310.pt`, paso 18.310), T=0,7, penalización por repetición 1,1, 200 tokens, **una sola generación por
bloque y sin selección**. Es una transcripción, no un video. Los bloques con `<CURSOR>` se rellenan en ese hueco (formato FIM); el último no lo tiene y el modelo
continúa el texto. En cada bloque, lo que sale entre `--- completando ---` y `--- fin ---` es el código del usuario más lo que escribió el modelo.

```text
fullstack-27m (paso 18310) | lenguaje: tsx | T=0.7 | 200 tokens | rep=1.1
Escribe el inicio de tu codigo y termina con una linea que solo tenga un punto (.)
Con <CURSOR> en el texto, el modelo rellena ese hueco (modelos v2).
  :lang ts|tsx|js|jsx|py   cambia el lenguaje      :temp 0.7    temperatura (0 = determinista)
  :tokens 200              largo de la continuacion  :rep 1.1     penalizacion por repeticion
  :quit                    salir

>>> lenguaje: ts | T=0.7 | 200 tokens | rep=1.1
>>> ... ... ... 
--- completando ---
export function sumar(a: number, b: number): number {
  return b + 1;
}
--- fin ---

>>> ... ... ... ... ... 
--- completando ---
export async function GET(request: Request) {
  const data = await request.json();
  const query = `
    query {
      user: getUser(_id: ${data.id}) {
        ${data.username}
      }
    }
  `;

  return Response.json({ ok: true });
}
--- fin ---

>>> lenguaje: py | T=0.7 | 200 tokens | rep=1.1
>>> ... ... ... ... 
--- completando ---
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    """
    Get the item.
    """
    return {"item_id": item_id}


@app.get("/items/{item_id}")
async def read_item(item_id: int, user: User = Depends(get_current_user)):
    """
    Get a specific item.
    """
    item = await get_item_from_id(item_id, user=user)
    return {"item": item}


@app.get("/items/{item_id}/{item_id}")
async def read_item(item_id: int, user: User = Depends(get_current_user)):
    """
    Get the item.
    """
    return {"item": item_id}


@app.get("/items/{item_id}")
async def read_item(item_id: int, user: User = Depends(get_current
    return {"item": item}
--- fin ---

>>> ... ... 
--- completando ---
import express from "express";
const app = express();

app.use(express.static(__dirname + "/public"));

const port = process.env.PORT || 3000;

app.use(express.static(__dirname + "/public"));

app.get("/", (req: express.Request, res: express.Response) => {
    res.sendFile(__dirname + "/public/index.html");
});

app.listen(port, function () {
    console.log(`Example app listening on port ${port}`);
});
--- fin ---

>>>
```

## Cómo leerla

- **Los dos primeros rellenos cierran bien** y respetan el sufijo, pero la calidad es la de un modelo de 20 M de parámetros: `return b + 1;` no es una suma, y la
  consulta inventada en el segundo bloque no se usa en ninguna parte.
- **El relleno de Python no cierra:** no encuentra el punto de parada, agota los 200 tokens repitiendo endpoints de FastAPI y el sufijo (`return {"item": item}`) queda
  pegado a una línea cortada. Es la debilidad conocida del v2 (en la evaluación de 200 casos solo el 35,5 % de los rellenos parsea, frente al 44,5 % de v1).
- **El último bloque** (continuación sin cursor) produce un servidor Express coherente, sin marcadores FIM, gracias al veto de la Fase 11.
