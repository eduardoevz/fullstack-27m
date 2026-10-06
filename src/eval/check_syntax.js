// Valida sintaxis (no tipos ni semantica) con el parser de TypeScript.
// stdin:  [{"id": .., "lang": "ts|tsx|js|jsx", "code": ".."}]
// stdout: [{"id": .., "ok": true|false, "error": ".."}]
// Requiere NODE_PATH apuntando a una carpeta con node_modules/typescript (rama 5.x).
const ts = require('typescript');
const KIND = { ts: ts.ScriptKind.TS, tsx: ts.ScriptKind.TSX, js: ts.ScriptKind.JS, jsx: ts.ScriptKind.JSX };
let raw = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', d => (raw += d));
process.stdin.on('end', () => {
  const out = JSON.parse(raw).map(({ id, lang, code }) => {
    const sf = ts.createSourceFile('f.' + lang, code, ts.ScriptTarget.ES2022, false, KIND[lang]);
    const d = sf.parseDiagnostics || [];
    const error = d.length ? ts.flattenDiagnosticMessageText(d[0].messageText, '\n') + ' @' + d[0].start : '';
    return { id, ok: d.length === 0, error };
  });
  process.stdout.write(JSON.stringify(out));
});
