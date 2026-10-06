"""Suite de 20 prompts del dominio Full Stack.

`lang` es la clase de sintaxis (js, jsx, ts, tsx, py); en el enmarcado del modelo jsx usa la etiqueta de js y
tsx la de ts. `expect`: si alguna de esas cadenas aparece en la continuacion, el prompt se cuenta «en tema»
(heuristica simple: no evalua correccion).
"""


def _p(id, lang, description, prompt, expect):
    return {"id": id, "lang": lang, "description": description, "prompt": prompt, "expect": expect}


PROMPTS = [
    _p("react-hook-counter", "tsx", "Componente React con useState",
       "import { useState } from 'react';\n\nexport default function Counter() {\n  const [count, setCount] = useState(0);\n  return (\n",
       ["<button", "setCount", "count"]),
    _p("react-effect-fetch", "tsx", "Componente React con useEffect y fetch",
       "import { useEffect, useState } from 'react';\n\ninterface User {\n  id: number;\n  name: string;\n}\n\nexport function UserList() {\n  const [users, setUsers] = useState<User[]>([]);\n\n  useEffect(() => {\n",
       ["fetch(", "setUsers", "axios"]),
    _p("react-context", "tsx", "Context provider de React",
       "import { createContext, useContext, useState } from 'react';\n\nconst ThemeContext = createContext<{ dark: boolean; toggle: () => void } | null>(null);\n\nexport function ThemeProvider({ children }: { children: React.ReactNode }) {\n",
       ["Provider", "useState", "toggle"]),
    _p("react-form", "jsx", "Formulario controlado en React",
       "import React, { useState } from 'react';\n\nfunction LoginForm({ onSubmit }) {\n  const [email, setEmail] = useState('');\n  const [password, setPassword] = useState('');\n\n  const handleSubmit = (e) => {\n",
       ["preventDefault", "onSubmit", "<form"]),
    _p("react-custom-hook", "ts", "Hook personalizado useDebounce",
       "import { useEffect, useState } from 'react';\n\nexport function useDebounce<T>(value: T, delay = 300): T {\n",
       ["setTimeout", "useEffect", "clearTimeout"]),
    _p("next-app-page", "tsx", "Pagina del App Router de Next.js",
       "import Link from 'next/link';\n\nexport const metadata = { title: 'Blog' };\n\nexport default async function BlogPage() {\n",
       ["await", "fetch(", "<Link", "map("]),
    _p("next-route-handler", "ts", "Route handler de Next.js (App Router)",
       "import { NextResponse } from 'next/server';\n\nexport async function GET(request: Request) {\n",
       ["NextResponse", "json("]),
    _p("next-server-action", "ts", "Server action de Next.js",
       "'use server';\n\nimport { revalidatePath } from 'next/cache';\n\nexport async function createPost(formData: FormData) {\n",
       ["revalidatePath", "formData.get", "await"]),
    _p("nest-controller", "ts", "Controlador NestJS",
       "import { Controller, Get, Post, Body, Param } from '@nestjs/common';\nimport { CatsService } from './cats.service';\n\n@Controller('cats')\nexport class CatsController {\n  constructor(private readonly catsService: CatsService) {}\n\n",
       ["@Get", "@Post", "this.catsService"]),
    _p("nest-service", "ts", "Servicio NestJS inyectable",
       "import { Injectable, NotFoundException } from '@nestjs/common';\n\n@Injectable()\nexport class UsersService {\n  private users: { id: number; name: string }[] = [];\n\n",
       ["find", "NotFoundException", "this.users"]),
    _p("nest-dto", "ts", "DTO con class-validator",
       "import { IsEmail, IsString, MinLength } from 'class-validator';\n\nexport class CreateUserDto {\n",
       ["@IsEmail", "@IsString", "@MinLength"]),
    _p("express-middleware-ts", "ts", "Middleware de autenticacion de Express en TS",
       "import { Request, Response, NextFunction } from 'express';\nimport jwt from 'jsonwebtoken';\n\nexport function authMiddleware(req: Request, res: Response, next: NextFunction) {\n",
       ["next()", "jwt.verify", "401"]),
    _p("express-router", "js", "Router de Express con CRUD",
       "const express = require('express');\nconst router = express.Router();\n\nrouter.get('/', async (req, res) => {\n",
       ["res.json", "res.send", "res.status"]),
    _p("express-server", "js", "Servidor Express minimo",
       "const express = require('express');\nconst app = express();\n\napp.use(express.json());\n\n",
       ["app.listen", "app.get", "app.post"]),
    _p("node-error-handler", "js", "Manejador de errores de Express",
       "function errorHandler(err, req, res, next) {\n",
       ["res.status", "console.error", "next("]),
    _p("fastapi-endpoint", "py", "Endpoint FastAPI con modelo Pydantic",
       "from fastapi import FastAPI, HTTPException\nfrom pydantic import BaseModel\n\napp = FastAPI()\n\n\nclass Item(BaseModel):\n    name: str\n    price: float\n\n\n@app.post('/items')\n",
       ["async def", "def create", "return"]),
    _p("fastapi-depends", "py", "Dependencia con Depends en FastAPI",
       "from fastapi import Depends, FastAPI, HTTPException\nfrom sqlalchemy.orm import Session\n\napp = FastAPI()\n\n\ndef get_db():\n",
       ["yield", "SessionLocal", "finally"]),
    _p("flask-route", "py", "Rutas de Flask con JSON",
       "from flask import Flask, jsonify, request\n\napp = Flask(__name__)\n\n\n@app.route('/api/tasks', methods=['GET', 'POST'])\ndef tasks():\n",
       ["jsonify", "request.", "return"]),
    _p("sqlalchemy-model", "py", "Modelo SQLAlchemy",
       "from sqlalchemy import Column, Integer, String, ForeignKey\nfrom sqlalchemy.orm import relationship\n\nfrom .database import Base\n\n\nclass User(Base):\n    __tablename__ = 'users'\n\n",
       ["Column(", "primary_key", "relationship"]),
    _p("pytest-api", "py", "Test de API con pytest",
       "import pytest\nfrom fastapi.testclient import TestClient\n\nfrom app.main import app\n\nclient = TestClient(app)\n\n\ndef test_read_items():\n",
       ["client.get", "assert", "status_code"]),
]
