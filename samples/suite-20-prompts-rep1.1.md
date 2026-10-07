# Suite de 20 prompts del dominio

Modelo: `v1-best.pt` (paso 15000). temperature=0.8, top_k=40, top_p=0.95, repetition_penalty=1.1, repetition_window=64, no_repeat_ngram_size=0, min_p=0.0, hasta 200 tokens, semilla base 0, 5 muestra(s) por prompt. Sin selección: es lo primero que salió.

**Resumen:** sintaxis estricta 63% (IC95 52%-74%) · recortada 68% (IC95 55%-80%) · en tema 69% (IC95 54%-82%) (prompts=20, muestras=100).

## react-hook-counter — Componente React con useState

estricta: sí · recortada: sí · en tema: sí

```tsx
import { useState } from 'react';

export default function Counter() {
  const [count, setCount] = useState(0);
  return (
«« continuación del modelo »»
    <>
      <div className="row">
        <div className="col-md-12">
          <h3>{count}</h3>
        </div>
        <div className="col-md-6">
          <label>
            <input
              id="counter"
              placeholder="Counter"
              value={count}
              className="text"
            />
          </label>
          <div className="col-md-12">
            <button
              id="counter"
              onClick={() => setCount(count - 1)}
            >
              Increment
            </button>
          </div>
        </div>
      </div>
    </>
  );
}

```

## react-effect-fetch — Componente React con useEffect y fetch

estricta: no · recortada: no · en tema: sí

```tsx
import { useEffect, useState } from 'react';

interface User {
  id: number;
  name: string;
}

export function UserList() {
  const [users, setUsers] = useState<User[]>([]);

  useEffect(() => {
«« continuación del modelo »»
    if (!users.length) {
      setUsers([]);

    } else {
      setUsers([]);
    }
  }, [users]);

  return (
    <>
      <span className="UserList">
        <a
          href={`/users/${username}`}
          target="_blank"
          rel="noopener noreferrer"
          style={{ width: '5px', marginTop: '10px' }}
        >
          {name}
        </a>
      </span>

      <div className="UserList-container">
        {users.map(user => (
          <UserList
            user={user}
            id={user.id}
            setUsers={setUsers}
            onUsersChange={setUsers}
          />
        ))}
      </div>
    </
```

## react-context — Context provider de React

estricta: no · recortada: no · en tema: sí

```tsx
import { createContext, useContext, useState } from 'react';

const ThemeContext = createContext<{ dark: boolean; toggle: () => void } | null>(null);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
«« continuación del modelo »»
    const { dark, toggle } = useContext(ThemeContext);

    return (
        <ThemeProvider theme={...themeContext}>
            <ThemeProvider theme={theme} />
            <div className="radium">
                {children}
            </div>
        </ThemeProvider>
    );
}

```

## react-form — Formulario controlado en React

estricta: no · recortada: no · en tema: sí

```jsx
import React, { useState } from 'react';

function LoginForm({ onSubmit }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = (e) => {
«« continuación del modelo »»
    if (e.target === 'submit') {
      e.preventDefault();

      const data = e.target.value;
      form.resetForm(data);

      e.preventDefault();

      onSubmit(data);
    } else if (e.target === 'password') {
      values[email] = { ...form, password: '' };
    }
  }

  return (
    <div className="login-form">
      {
        values.map((value, index) => {
          return (
            <div key={index} className="login-form-control">
              <input
                type="text"
                ref={(input) => {
                  this.input = input;
                  this.props.onSubmit();
                }}
                value={value}
              />
            </div>
          )
        })
      }
    </div
```

## react-custom-hook — Hook personalizado useDebounce

estricta: no · recortada: no · en tema: sí

```ts
import { useEffect, useState } from 'react';

export function useDebounce<T>(value: T, delay = 300): T {
«« continuación del modelo »»
  const [value, setValue] = useState(null);

  return useRef<{}>(
    // @ts-ignore - if we don't have a ref, then it will still work out with an unfocus
    useRef();

  const [value, setValue] = useState(value);

  useEffect(() => {
    setValue(value);
  }, []);

  return useRef() as T | undefined;
}

```

## next-app-page — Pagina del App Router de Next.js

estricta: no · recortada: no · en tema: sí

```tsx
import Link from 'next/link';

export const metadata = { title: 'Blog' };

export default async function BlogPage() {
«« continuación del modelo »»
  const page = new Link({ title: 'Blog', link: '/blog' });

  return (
    <>
      <div className="container">
        <ul>
          {page.map(item => <Link key={item.id} to={`/` + item.id}>{item.title}</Link>)}
        </ul>
        <h2 className="text-center">Blog {item.blog}</h2>
      </div>

      <Link to={`${metadata.url}/${metadata.link}`}>
        <a className="link link" href={`${metadata.link}/`} target="_blank" rel="noopener noreferrer">
          <span className="text-right">&#fff;</span>
        </a>
```

## next-route-handler — Route handler de Next.js (App Router)

estricta: sí · recortada: sí · en tema: sí

```ts
import { NextResponse } from 'next/server';

export async function GET(request: Request) {
«« continuación del modelo »»
  let { code, params } = request.params;

  const options: NextResponse = {
    name: 'test',
    code: params['code'],
    params: params,
  };

  const request = new NextResponse({ code, params });

  // create a response with the next response as well and return the full page
  return request.buildResult(options);
}

```

## next-server-action — Server action de Next.js

estricta: sí · recortada: sí · en tema: sí

```ts
'use server';

import { revalidatePath } from 'next/cache';

export async function createPost(formData: FormData) {
«« continuación del modelo »»
    const req = await createRequest(
        'POST',
        formData.name
    );

    const res = await request(req);

    const data: any[] = [];

    return {
        'success': true,
        'error': 'missing post',
        'data': res
    };
}

```

## nest-controller — Controlador NestJS

estricta: sí · recortada: sí · en tema: no

```ts
import { Controller, Get, Post, Body, Param } from '@nestjs/common';
import { CatsService } from './cats.service';

@Controller('cats')
export class CatsController {
  constructor(private readonly catsService: CatsService) {}

«« continuación del modelo »»}

```

## nest-service — Servicio NestJS inyectable

estricta: sí · recortada: sí · en tema: sí

```ts
import { Injectable, NotFoundException } from '@nestjs/common';

@Injectable()
export class UsersService {
  private users: { id: number; name: string }[] = [];

«« continuación del modelo »»constructor(private router: Router) {
  }

  public getUsers(): Promise<any>[] {
    return this.users.get();
  }

  public create(): Promise<any> {
    return Promise.resolve().then((result) => result);
  }
}

```

## nest-dto — DTO con class-validator

estricta: sí · recortada: sí · en tema: no

```ts
import { IsEmail, IsString, MinLength } from 'class-validator';

export class CreateUserDto {
«« continuación del modelo »»
    constructor(email: IsEmail, password: MinLength);

    getEmailWithPassword(password: Password): string | null;

    hasPassword(password: Password): boolean {
        return this.getUsernameWithPassword(password) !== null;
    }

    getUserById(id: number): UserDto {
        const user: UserDto = {
            email: id,
            password: password
        };

        return this.getEmailWithPassword(user) as UserDto;
    }

    getUsername(email: string): string | null {
        const user: UserDto = {
            username: email,
            email: email
        };

        return this.getUsernameWithPassword(user) as UserDto;
    }

}
```

## express-middleware-ts — Middleware de autenticacion de Express en TS

estricta: sí · recortada: sí · en tema: sí

```ts
import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';

export function authMiddleware(req: Request, res: Response, next: NextFunction) {
«« continuación del modelo »»
    if (!jwt.valid(req, res)) {
        res.status(401);
        return next(new Error('Invalid OAuth token'));
    }

    next(null, jwt);
}
```

## express-router — Router de Express con CRUD

estricta: no · recortada: no · en tema: sí

```js
const express = require('express');
const router = express.Router();

router.get('/', async (req, res) => {
«« continuación del modelo »»
    let name = '';
    if (req.query && req.query.name) {
        name = `${req.query.name}.${req.query.name}`;
    } else if (req.query) {
        name = `${req.query.name}.${req.query.name}`;
    }

    const authRouter = new Router({
        save: async (req, res) => {
            res.sendStatus(200);
            res.status(200).end();
        }
    });

    router.post('/', async (req, res) => {

        let name = `${req.query.name}.${req.body.name}`;

        if (req.body && req.body.username) {
            name = `${req.body.username}.${req
```

## express-server — Servidor Express minimo

estricta: no · recortada: sí · en tema: no

```js
const express = require('express');
const app = express();

app.use(express.json());

«« continuación del modelo »»app.use(express.urlencoded());
app.use(express.methodOverride());

/////////////////////////////////////////////////////////////////////////
// App
//////////////////////////////////////////////////////////////////////////

app.use(express.static(__dirname + '/client'));
app.
```

## node-error-handler — Manejador de errores de Express

estricta: sí · recortada: sí · en tema: sí

```js
function errorHandler(err, req, res, next) {
«« continuación del modelo »»
  var err = new Error('Not Found');

  err.status = 404;

  next(err);
}

function getNextHandler(req, res, next) {

  res.status(404);

  // res.send({error: 'Server not found'});

  // res.send({error: 'Server not found'});

  next();

}

function createErrorHandler(err) {
  return next(err);
}

function getErrorHandler(err) {
  return next(err);
}

module.exports = {};

```

## fastapi-endpoint — Endpoint FastAPI con modelo Pydantic

estricta: sí · recortada: sí · en tema: sí

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str
    price: float


@app.post('/items')
«« continuación del modelo »»async def get():
    return Response(await app.send_json(
        '{}/items.json'.format(self.name),
        body={'price': self.price},
        status=200
    ), content_type='application/json')

```

## fastapi-depends — Dependencia con Depends en FastAPI

estricta: sí · recortada: sí · en tema: no

```python
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

app = FastAPI()


def get_db():
«« continuación del modelo »»
    db = Session()

    if 'mysql' in request.args:
        mysql = request.args['mysql']

    if 'mysql' in request.args:
        mysql = request.args['mysql']

    return db

```

## flask-route — Rutas de Flask con JSON

estricta: sí · recortada: sí · en tema: no

```python
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/api/tasks', methods=['GET', 'POST'])
def tasks():
«« continuación del modelo »»
    # Run the app
    app.run(host='0.0.0.0', port=5010)

```

## sqlalchemy-model — Modelo SQLAlchemy

estricta: no · recortada: no · en tema: sí

```python
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

from .database import Base


class User(Base):
    __tablename__ = 'users'

«« continuación del modelo »»
class Role(Base):
    __tablename__ = 'roles'

    id = Column(Integer, primary_key=True)
    username = Column(String(255), unique=True, nullable=False)
    role_id = Column(Integer, ForeignKey('roles.id'))
    role = relationship(Role, primaryjoin='and_(User.role_id == Role.id, User.name == Role.name) == 'user')

    @classmethod
    def create(cls, name, role_id):
        if cls.name in cls.query.all():
            return cls.query.filter_by(
                user=cls.user,
                role_id=role_id
            ).first()
        else:
            return None

```

## pytest-api — Test de API con pytest

estricta: sí · recortada: sí · en tema: no

```python
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_read_items():
«« continuación del modelo »»
    with pytest.raises(ValueError) as exc_info:
        client.read_items('/foo/bar')

```
