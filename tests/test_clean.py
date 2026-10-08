"""Fase 9: reglas de limpieza determinista (clean_reject, strip_license_header, clean_document)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.filters import clean_document, clean_reject, strip_license_header

BODY_PY = "def suma(a, b):\n    return a + b\n"
BODY_TS = "export const suma = (a: number, b: number) => a + b;\n"


def test_dts_rechazado_por_ruta():
    assert clean_reject(BODY_TS, "src/types/index.d.ts", "ts") == "dts"
    assert clean_reject(BODY_TS, "src/index.ts", "ts") is None
    assert clean_reject(BODY_TS, "SRC/Index.D.TS", "ts") == "dts"


def test_plantillas_ejs_y_handlebars():
    assert clean_reject("var s = <%= user.name %>;\n", "a.js", "js") == "template"
    assert clean_reject("{{#each items}}<li>{{this}}</li>{{/each}}\n", "a.js", "js") == "template"
    # JSX y plantillas inline de Angular son codigo legitimo
    assert clean_reject("const a = <div style={{ color: 'red' }} />;\n", "a.tsx", "tsx") is None
    assert clean_reject("template: `<p>{{ name }}</p>`,\n", "a.component.ts", "ts") is None
    # Una comparacion con <% sin cierre no es plantilla
    assert clean_reject("if (a <% 3) {}\n", "a.js", "js") is None


def test_plantillas_no_aplican_a_python():
    assert clean_reject('x = "<%= a %>"\n' + BODY_PY, "a.py", "py") is None


def test_python2():
    assert clean_reject("print 'hola'\n", "a.py", "py") == "python2"
    assert clean_reject("try:\n    pass\nexcept ValueError, e:\n    pass\n", "a.py", "py") == "python2"
    assert clean_reject("raise ValueError, 'x'\n", "a.py", "py") == "python2"
    assert clean_reject("print('hola')\n" + BODY_PY, "a.py", "py") is None
    assert clean_reject("print\n" + BODY_PY, "a.py", "py") is None


def test_python_que_no_parsea():
    assert clean_reject("def f(:\n    pass\n", "a.py", "py") == "py_syntax"
    assert clean_reject(BODY_PY, "a.py", "py") is None


def test_licencia_bloque_c():
    header = "/*\n * Copyright (c) 2019 ACME\n * Licensed under the MIT License.\n * Permission is hereby granted,\n * free of charge, to any person.\n */\n"
    assert strip_license_header(header + BODY_TS) == BODY_TS


def test_licencia_lineas_hash_y_barras():
    py = "# Copyright 2020 Foo\n# Licensed under the Apache License, Version 2.0\n# you may not use this file\n# except in compliance with the License.\n# WITHOUT WARRANTIES\n"
    assert strip_license_header(py + BODY_PY) == BODY_PY
    ts = "// SPDX-License-Identifier: MIT\n// Copyright 2021 X\n// All rights reserved.\n// License text\n// more\n"
    assert strip_license_header(ts + BODY_TS) == BODY_TS


def test_comentario_corto_o_sin_licencia_se_conserva():
    doc = "# Utilidades de suma\n# para el modulo\n# de calculo\n# basico\n# fin\n" + BODY_PY
    assert strip_license_header(doc) == doc
    corto = "# Copyright 2020 Foo\n" + BODY_PY  # menos de 4 lineas: se conserva
    assert strip_license_header(corto) == corto


def test_licencia_conserva_shebang_y_use_client():
    header = "/*\n * Copyright X\n * License MIT\n * Permission granted\n * Warranty none\n */\n"
    assert strip_license_header("'use client'\n" + header + BODY_TS).startswith("'use client'\n")
    assert strip_license_header("#!/usr/bin/env python\n# Copyright 1\n# License 2\n# Warranty 3\n# Permission 4\n" + BODY_PY) == "#!/usr/bin/env python\n" + BODY_PY


def test_clean_document():
    header = "// Copyright 2020 X\n// License MIT\n// Permission granted\n// Warranty none\n"
    out, reason = clean_document(header + BODY_TS, "a.ts", "ts")
    assert (out, reason) == (BODY_TS, None)
    assert clean_document(BODY_TS, "a.d.ts", "ts") == (None, "dts")


def test_python2_print_redirigido():
    assert clean_reject("import sys\nprint >>sys.stderr, 'x'\n", "a.py", "py") == "python2"


B64_LINE = "MIIDeDCCAuGgAwIBAgIJALPHPDcjk979MA0GCSqGSIb3DQEBBQUAMIGFMQswCQYD"


def test_blob_certificado_o_base64():
    cert = 'CERT = b"""-----BEGIN CERTIFICATE-----\n' + "\n".join([B64_LINE] * 10) + '\n-----END CERTIFICATE-----"""\n'
    assert clean_reject(cert + BODY_PY, "a.py", "py") == "blob"
    one_line = "const img = '" + "QUJD" * 80 + "';\n"
    assert clean_reject(one_line + BODY_TS, "a.ts", "ts") == "blob"


def test_pocas_lineas_base64_no_son_blob():
    assert clean_reject("x = '''\n" + "\n".join([B64_LINE] * 3) + "\n'''\n" + BODY_PY, "a.py", "py") is None
    assert clean_reject("import { aVeryLongModuleNameThatIsFine } from './module';\n" + BODY_TS, "a.ts", "ts") is None


def test_sourcemap_embebido():
    code = "'use strict';\nmodule.exports = require('./at');\n//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9u\n"
    assert clean_reject(code, "a.js", "js") == "sourcemap"
    assert clean_reject("//# sourceMappingURL=app.js.map\n" + BODY_TS, "a.js", "js") is None


def test_mojibake():
    assert clean_reject("# ben\ufffd\ufffdtigten Imports \ufffd\n" + BODY_PY, "a.py", "py") == "mojibake"
    assert clean_reject("# una vez \ufffd en un comentario\n" + BODY_PY, "a.py", "py") is None


def test_separadores_de_comentarios_no_son_blob():
    sep = "/" * 250 + "\n" + "=" * 60 + "\n"
    assert clean_reject(sep * 10 + BODY_TS, "a.ts", "ts") is None
    assert clean_reject("# " + "=" * 220 + "\n" + BODY_PY, "a.py", "py") is None


from src.data.filters import strip_metadata_tags  # noqa: E402


def test_strip_metadata_tags_formas_de_starcoder():
    assert strip_metadata_tags("<reponame>u/r<filename>src/a.ts<gh_stars>10-100\n" + BODY_TS) == BODY_TS
    assert strip_metadata_tags("<gh_stars>0\n" + BODY_TS) == BODY_TS
    assert strip_metadata_tags("<filename>pages/x.tsx<gh_stars>1-1\n" + BODY_TS) == BODY_TS
    assert strip_metadata_tags("<reponame>u/r\n" + BODY_TS) == BODY_TS
    assert strip_metadata_tags("<filename>a/b.ts\n" + BODY_TS) == BODY_TS


def test_strip_metadata_tags_no_toca_codigo_normal():
    assert strip_metadata_tags(BODY_TS) == BODY_TS
    jsx = "const a = <filenameX>hola</filenameX>;\n" + BODY_TS
    assert strip_metadata_tags(jsx) == jsx
    assert strip_metadata_tags("\n" + BODY_TS) == "\n" + BODY_TS


def test_etiquetas_dentro_del_archivo_se_rechazan():
    assert clean_reject(BODY_TS + "x = '<reponame>a/b<filename>c'\n", "a.ts", "ts") == "tags"


def test_clean_document_quita_etiquetas_y_luego_licencia():
    header = "// Copyright 2020 X\n// License MIT\n// Permission granted\n// Warranty none\n"
    out, reason = clean_document("<reponame>u/r<filename>a.ts<gh_stars>0\n" + header + BODY_TS, "a.ts", "ts")
    assert (out, reason) == (BODY_TS, None)
