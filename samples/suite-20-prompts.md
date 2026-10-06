# Suite de 20 prompts del dominio

Modelo: `best.pt` (paso 15000). T=0.7, top-k 40, top-p 0.95, hasta 200 tokens, semilla base 0. Sin selección: es lo primero que salió.

**Resumen:** sintaxis estricta 60% · recortada 75% · en tema 60% (n=20).

## react-hook-counter — Componente React con useState

estricta: sí · recortada: sí · en tema: no

```tsx
import { useState } from 'react';

export default function Counter() {
  const [count, setCount] = useState(0);
  return (
«« continuación del modelo »»
    <>
      <div>
        <span>Counter</span>
        <span>Count</span>
        <span>Count</span>
        <span>Count</span>
        <span>Count</span>
        <span>Count</span>
      </div>

      <div>
        <span>Count</span>
        <span>Count</span>
        <span>Count</span>
        <span>Count</span>
      </div>
    </>
  );
}

```

## react-effect-fetch — Componente React con useEffect y fetch

estricta: no · recortada: no · en tema: no

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
      return [];
    }

    return users.map((user) => ({
      id: user.id,
      name: user.name,
    }));
  }, [users]);

  return (
    <div className="user-list">
      <div className="user-list-header">
        <div className="user-list-header-container">
          <img src="/img/users/avatar.png" alt="User" />
          <span>{user.name}</span>
        </div>
      </div>
      <div className="user-list-header-container">
        <div className="user-list-header-container-container">
          <img src="/img/users/avatar.png" alt="User" />
          <
```

## react-context — Context provider de React

estricta: no · recortada: no · en tema: sí

```tsx
import { createContext, useContext, useState } from 'react';

const ThemeContext = createContext<{ dark: boolean; toggle: () => void } | null>(null);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
«« continuación del modelo »»
    const { dark, toggle, toggle, toggleIsDisabled } = useContext<ThemeContext | undefined>();

    const theme = useContext(ThemeContext);

    const [dark, toggle, toggleIsDisabled] = useState<boolean>();
    const [toggleIsDisabled, toggleIsDisabled] = useState<boolean>();
    const [toggleIsDisabled, toggleIsDisabled] = useState<boolean>();

    const [toggleIsDisabled, toggleIsDisabled] = useState<boolean>();
    const [toggleIsDisabled, toggleIsDisabled] = useState<boolean>();
    const [toggleIsDisabled, toggleIsDisabled] = useState<boolean>();

    const [toggleIsDisabled, toggleIsDisabled] = useState<boolean>();

    const theme = useContext(ThemeContext);

    const [theme, toggleIsDisabled] = useState<boolean
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
    if (e.target.name === 'password') {
      e.preventDefault();
      onSubmit();
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="email"
        value={email}
        onChange={handleSubmit(email)}
        name="email"
        ref="email"
      />

      <input
        type="password"
        value={password}
        onChange={handleChange('password', '')}
        name="password"
        ref="password"
      />

      <input
        type="password"
        value={password}
        onChange={handleChange('password', '')}
        name="password"
        ref="password"
      />

      <input
        type="submit"
        value="Submit"
        ref="submit"
      />

      
```

## react-custom-hook — Hook personalizado useDebounce

estricta: sí · recortada: sí · en tema: no

```ts
import { useEffect, useState } from 'react';

export function useDebounce<T>(value: T, delay = 300): T {
«« continuación del modelo »»
    if (value > 0) {
        return value;
    }

    return useRef().current;
}

export function useRef<T>(value: T, delay = 300): T {
    if (!value) {
        return value;
    }

    return useRef().current;
}

```

## next-app-page — Pagina del App Router de Next.js

estricta: sí · recortada: sí · en tema: sí

```tsx
import Link from 'next/link';

export const metadata = { title: 'Blog' };

export default async function BlogPage() {
«« continuación del modelo »»
  const page = await Link.create({ title: 'Blog' });
  const html = await page.read('/');
  await html.find().eq(1).html(page);
  await html.find().eq(2).html(page);
}

```

## next-route-handler — Route handler de Next.js (App Router)

estricta: sí · recortada: sí · en tema: sí

```ts
import { NextResponse } from 'next/server';

export async function GET(request: Request) {
«« continuación del modelo »»
  let { code, params } = request.params;

  let { response } = await request.get(`${nextResponse.status}`);

  let codeCode = response.status;

  let errorCode = response.status || response.statusCode;

  let responseError = response.status ? response.status : response.error;

  return {
    status: responseStatus,
    code: responseCode,
    errorCode: errorCode,
    response: response,
    errorCode: errorCode,
  };
}

export async function getNextResponse(response: NextResponse) {
  const { response } = response.status;

  return {
    status: responseStatus,
    code: responseCode,
    response: response,
    response: response,
    errorCode: responseErrorCode,
    response: responseError,
  };
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
        '/api/v1/post/:id/posts/:id',
        {
            path: '/',
            headers: {
                'Authorization': 'Bearer ' + token
            }
        }
    );

    let post = req.post;
    if (!post) {
        throw new Error('Missing post');
    }

    const reqBody = post.query;
    if (!reqBody || !reqBody.id) {
        throw new Error('Missing post id');
    }

    try {
        const postId = await post.get('id');
        return new Post(postId, reqBody.id);
    } catch (e) {
        throw new Error('Missing post id');
    }
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

«« continuación del modelo »»
  constructor() {
    this.users = [];
  }

  getUsers() {
    return this.users;
  }

}

@Injectable()
export class UsersService {
  constructor() {
    this.users = [];
  }

  getUsers() {
    return this.users;
  }

}

@Injectable()
export class UserService {
  constructor() {
    this.users = new { id: 1, name: 'hello', email: 'hello@example.com' };
  }

}

```

## nest-dto — DTO con class-validator

estricta: sí · recortada: sí · en tema: no

```ts
import { IsEmail, IsString, MinLength } from 'class-validator';

export class CreateUserDto {
«« continuación del modelo »»
    constructor(email: IsEmail, password: MinLength, user: IsEmail, type: string) {

        this.email = email;

        this.password = password;

        this.user = user;
    }

    public validate(email: IsEmail): boolean {
        return this.email && this.email.isValid();
    }
}
```

## express-middleware-ts — Middleware de autenticacion de Express en TS

estricta: no · recortada: no · en tema: sí

```ts
import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';

export function authMiddleware(req: Request, res: Response, next: NextFunction) {
«« continuación del modelo »»
    if (!jwt.valid(req.session.username)) {
        res.status(401);
        res.send({ error: 'Invalid username' });
        return;
    }

    const jwtExpire = jwt.secret();

    // Request to the API
    let uri = req.protocol + '//' + req.get('host') + '/' + req.get('port');
    console.log('authMiddleware: ' + uri);

    // Validate token
    if (jwt.valid(req.body)) {
        jwtExpire.verify(req.body, jwtExpire).then(() => {
            res.status(401);
            res.send({ error: 'Invalid token' });
        }).catch((err) => {
            if (err.status === 401) {
                
```

## express-router — Router de Express con CRUD

estricta: no · recortada: no · en tema: sí

```js
const express = require('express');
const router = express.Router();

router.get('/', async (req, res) => {
«« continuación del modelo »»
    let user = await getUser(req.query.username);
    let userName = user.username.split('/').pop();
    if(!user) {
        return res.status(404).send('Please specify a username');
    }

    const token = await getToken(req.query.token);
    const user = await getUser(req.query.username);

    if(user.username.split('/').length === 1) {
        return res.status(404).send('Please specify a username');
    }

    const token2 = await getToken(req.query.token);
    const user = await getUser(req.query.username);

    if(!user.username.split('/').length === 1) {
        return res.status(404).send('
```

## express-server — Servidor Express minimo

estricta: sí · recortada: sí · en tema: no

```js
const express = require('express');
const app = express();

app.use(express.json());

«« continuación del modelo »»app.use(express.urlencoded());
app.use(express.methodOverride());

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
```

## node-error-handler — Manejador de errores de Express

estricta: no · recortada: sí · en tema: sí

```js
function errorHandler(err, req, res, next) {
«« continuación del modelo »»
  var err = err || new Error('Not Found');
  err.status = 404;
  next(err);
}

function getNextHandler(req, res, next) {
  var err = new Error('Not Found');
  err.status = 404;
  next(err);
}

function getNext(req, res, next) {
  var err = new Error('Not Found');
  err.status = 404;
  next(err);
}

function getNextHandler(req, res, next) {
  var err = new Error('Not Found');
  err.status = 404;
  next(err);
}

function getNextHandler(req, res, next) {
  var err = new Error('Not Found');
  err.status = 404;
  next(err);
}

function
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
    return Response(await app.send('/items'))


@app.post('/items/<int:id>')
async def get(id):
    return Response(await app.send('/items/' + id))


@app.post('/items/<int:id>')
async def put(id):
    return Response(await app.send('/items/' + id))

```

## fastapi-depends — Dependencia con Depends en FastAPI

estricta: no · recortada: sí · en tema: no

```python
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

app = FastAPI()


def get_db():
«« continuación del modelo »»
    db = Session()
    db.session = session
    db.session.add(db)

    return db


def get_db_():

    db = Session()
    db.session.add(db)

    return db


def create_index(app, **kwargs):

    db = Session()
    db.session.add(db)

    return db


def create_net(app, **kwargs):

    # Create index
    db = Session()
    db.session.add(db)

    return db


def create_net(app, **kwargs):

    db = Session()
    db.session.add(db)

    return db


def create_net_adapter(app, **kwargs):

    db = Session()
    db.session.add(db)

    return db


def
```

## flask-route — Rutas de Flask con JSON

estricta: no · recortada: sí · en tema: no

```python
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/api/tasks', methods=['GET', 'POST'])
def tasks():
«« continuación del modelo »»
    # Run the app
    app.run(host='0.0.0.0', port=5010, debug=True)


@app.route('/api/tasks/<task_id>', methods=['GET', 'POST'])
def tasks(task_id):

    # Run the app
    app.run(host='0.0.0.0', port=5010, debug=True)


@app.route('/api/tasks/<task_id>', methods=['GET', 'POST'])
def tasks(task_id):

    # Run the app
    app.run(host='0.0.0.0', port=5010, debug=True)


@app.route('/api/tasks/<task_id>', methods=['GET',
```

## sqlalchemy-model — Modelo SQLAlchemy

estricta: sí · recortada: sí · en tema: sí

```python
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

from .database import Base


class User(Base):
    __tablename__ = 'users'

«« continuación del modelo »»
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(255), unique=True, nullable=False)
    username_name = Column(String(255))
    username = Column(String(255), unique=True, nullable=False)
    password = Column(String(255))
    admin = Column(String(255))
    last_login = Column(DateTime)
    last_login_time = Column(DateTime)
    last_login_timestamp = Column(DateTime)


class User_User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(255), unique=True, nullable=False)
    username = Column(String(255), unique=True, nullable=False)

```

## pytest-api — Test de API con pytest

estricta: sí · recortada: sí · en tema: sí

```python
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_read_items():
«« continuación del modelo »»
    with pytest.raises(ValueError) as exc_info:
        client.read_items()

    assert not client.read_items()

    assert not client.read_items()


def test_read_items_empty():

    with pytest.raises(ValueError) as exc_info:
        client.read_items()

    assert client.read_items()


def test_read_items_empty():

    with pytest.raises(ValueError) as exc_info:
        client.read_items()

    assert client.read_items()


def test_read_items_empty():

    with pytest.raises(ValueError) as exc_info:
        client.read_items()

    assert client.read_items()


def test_read_items():

    with pytest.raises(ValueError) as exc_info:
        client.read_items()

    assert client.read_items()

    
```
