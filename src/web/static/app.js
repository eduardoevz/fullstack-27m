"use strict";
/* Nexora: cliente del servidor local (src/web/server.py). JS puro, sin dependencias ni CDN. */

const $ = (id) => document.getElementById(id);
const src = $("src"), hl = $("hl"), gutter = $("gutter"), editor = $("editor");
const DEFAULTS = { lang: "tsx", temp: 0.7, tokens: 200, rep: 1.1 };
const LINE_PX = 13 * 1.6;
let CURSOR = "<CURSOR>";
let gen = null;              // {start, end}: tramo del texto que escribio el modelo
let controller = null;       // AbortController de la generacion en curso

/* ------------------------------------------------------------ utilidades */
const fmt = (n, d = 0) => Number(n).toLocaleString("es", { minimumFractionDigits: d, maximumFractionDigits: d });
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function store(key, value) { try { localStorage.setItem("nexora." + key, JSON.stringify(value)); } catch (_) { /* sin almacenamiento */ } }
function recall(key, fallback) {
  try { const v = localStorage.getItem("nexora." + key); return v === null ? fallback : JSON.parse(v); } catch (_) { return fallback; }
}

/* ------------------------------------------------------------ resaltado de sintaxis */
const KW = {
  js: new Set(("const let var function return if else for while do switch case break continue new this class extends super import export from default " +
    "async await try catch finally throw typeof instanceof in of void delete yield static get set interface type enum implements public private " +
    "protected readonly as namespace declare abstract true false null undefined").split(" ")),
  py: new Set(("def class return if elif else for while in not and or is import from as with try except finally raise pass break continue lambda " +
    "yield async await global nonlocal del assert True False None self").split(" ")),
};
const RE = {
  js: /(\/\/[^\n]*|\/\*[\s\S]*?(?:\*\/|$))|("(?:\\.|[^"\\\n])*"?|'(?:\\.|[^'\\\n])*'?|`(?:\\[\s\S]|[^`\\])*`?)|(\b\d[\d_]*(?:\.\d+)?\b)|(<\/?[A-Za-z][\w.-]*)|([A-Za-z_$][\w$]*)/g,
  py: /(#[^\n]*)|("""[\s\S]*?(?:"""|$)|'''[\s\S]*?(?:'''|$)|"(?:\\.|[^"\\\n])*"?|'(?:\\.|[^'\\\n])*'?)|(\b\d[\d_]*(?:\.\d+)?\b)|(@[A-Za-z_][\w.]*)|([A-Za-z_][\w]*)/g,
};

function tokenize(text, family) {
  const kw = KW[family], re = new RegExp(RE[family].source, "g"), out = [];
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m[0] === "") { re.lastIndex++; continue; }
    const a = m.index, b = a + m[0].length;
    let cls = null;
    if (m[1] !== undefined) cls = "c";
    else if (m[2] !== undefined) cls = "s";
    else if (m[3] !== undefined) cls = "n";
    else if (m[4] !== undefined) cls = family === "py" ? "f" : "g";
    else if (m[5] !== undefined) {
      const w = m[5];
      if (kw.has(w)) cls = "k";
      else if (text[b] === "(") cls = "f";
      else if (/^[A-Z]/.test(w)) cls = "t";
    }
    if (cls) out.push([a, b, cls]);
  }
  return out;
}

/* HTML del texto con colores; `range` (opcional) marca el tramo escrito por el modelo. */
function highlightHTML(text, lang, range) {
  const toks = tokenize(text, lang === "py" ? "py" : "js");
  const cuts = new Set([0, text.length]);
  toks.forEach(([a, b]) => { cuts.add(a); cuts.add(b); });
  if (range) { cuts.add(Math.max(0, Math.min(text.length, range.start))); cuts.add(Math.max(0, Math.min(text.length, range.end))); }
  const points = [...cuts].sort((x, y) => x - y);
  const clsAt = new Map(toks.map(([a, b, c]) => [a, [b, c]]));
  let html = "", openGen = false, tokEnd = -1, tokCls = null;
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i], b = points[i + 1];
    if (clsAt.has(a)) { [tokEnd, tokCls] = clsAt.get(a); }
    const inTok = a < tokEnd;
    const inGen = range && a >= range.start && b <= range.end && range.end > range.start;
    if (inGen && !openGen) { html += '<span class="gen">'; openGen = true; }
    if (!inGen && openGen) { html += "</span>"; openGen = false; }
    const piece = esc(text.slice(a, b));
    html += inTok && tokCls ? `<span class="${tokCls}">${piece}</span>` : piece;
  }
  if (openGen) html += "</span>";
  return html + "\n";
}

function render() {
  hl.innerHTML = highlightHTML(src.value, $("lang").value, gen);
  const lines = src.value.split("\n").length;
  gutter.textContent = Array.from({ length: lines }, (_, i) => i + 1).join("\n");
  syncScroll();
}
function syncScroll() {
  hl.scrollTop = src.scrollTop; hl.scrollLeft = src.scrollLeft; gutter.scrollTop = src.scrollTop;
}
function ensureVisible(pos) {
  const line = src.value.slice(0, pos).split("\n").length - 1;
  const top = line * LINE_PX + 14, bottom = top + LINE_PX;
  if (bottom > src.scrollTop + src.clientHeight - LINE_PX) src.scrollTop = bottom - src.clientHeight + 2 * LINE_PX;
  else if (top < src.scrollTop) src.scrollTop = Math.max(0, top - LINE_PX);
  syncScroll();
}

/* ------------------------------------------------------------ ajustes */
function readSettings() {
  return { lang: $("lang").value, temperature: parseFloat($("temp").value), tokens: parseInt($("tokens").value, 10), rep_penalty: parseFloat($("rep").value) };
}
function showSettings() {
  $("o-temp").textContent = fmt($("temp").value, 2);
  $("o-tokens").textContent = fmt($("tokens").value);
  $("o-rep").textContent = fmt($("rep").value, 2);
}
function saveSettings() {
  store("settings", { lang: $("lang").value, temp: +$("temp").value, tokens: +$("tokens").value, rep: +$("rep").value });
}
function applySettings(s) {
  $("lang").value = s.lang; $("temp").value = s.temp; $("tokens").value = s.tokens; $("rep").value = s.rep;
  showSettings();
}

/* ------------------------------------------------------------ generacion (SSE por fetch) */
function setStatus(msg, isErr) {
  const s = $("status");
  s.textContent = msg;
  s.className = isErr ? "err" : "";
}
function setBusy(busy) {
  editor.classList.toggle("busy", busy);
  $("btn-stop").hidden = !busy;
  $("btn-complete").disabled = busy;
  $("btn-fill").disabled = busy;
  src.readOnly = busy;
}

async function* sse(response) {
  const reader = response.body.getReader(), decoder = new TextDecoder("utf-8");
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, i); buf = buf.slice(i + 2);
      let event = "message", data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (data) yield [event, JSON.parse(data)];
    }
  }
}

async function run(mode) {
  if (controller) return;
  const value = src.value;
  let prefix, suffix, text;
  if (mode === "fill") {
    prefix = value.slice(0, src.selectionStart); suffix = value.slice(src.selectionEnd);
    text = prefix + CURSOR + suffix;
  } else {
    prefix = value; suffix = ""; text = value;
  }
  if (!text.replace(CURSOR, "").trim()) { setStatus("Escribe algo de código primero.", true); return; }

  controller = new AbortController();
  setBusy(true); gen = { start: prefix.length, end: prefix.length };
  setStatus(mode === "fill" ? "Rellenando el hueco…" : "Generando…");
  let generated = "", finished = false;
  const mine = readSettings();
  try {
    const resp = await fetch("/api/complete", {
      method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal,
      body: JSON.stringify({ text, lang: mine.lang, temperature: mine.temperature, tokens: mine.tokens, rep_penalty: mine.rep_penalty }),
    });
    if (!resp.ok) {
      let msg = "Error " + resp.status;
      try { msg = (await resp.json()).error || msg; } catch (_) { /* cuerpo no JSON */ }
      throw new Error(msg);
    }
    for await (const [event, data] of sse(resp)) {
      if (event === "token") {
        generated += data.text;
        src.value = prefix + generated + suffix;
        gen = { start: prefix.length, end: prefix.length + generated.length };
        render();
        const caret = prefix.length + generated.length;
        src.setSelectionRange(caret, caret);
        ensureVisible(caret);
      } else if (event === "done") {
        finished = true;
        $("s-tokens").textContent = fmt(data.tokens);
        $("s-toks").textContent = fmt(data.tok_s, 1) + " tok/s";
        $("s-time").textContent = fmt(data.seconds, 1) + " s";
        $("s-mode").textContent = data.fim ? "Relleno" : "Continuar";
        const why = data.stopped ? "terminó por sí mismo" : "llegó al límite de tokens";
        setStatus(`Listo · ${fmt(data.tokens)} tokens · ${why}.` + (data.fim && !data.stopped ? " No encontró dónde parar: es una limitación conocida del modelo." : ""));
      } else if (event === "error") {
        throw new Error(data.error);
      }
    }
    if (!finished) setStatus("La conexión se cerró antes de terminar.", true);
  } catch (e) {
    if (e.name === "AbortError") setStatus("Detenido.");
    else setStatus(e.message === "Failed to fetch" ? "No se pudo conectar con el servidor de Nexora." : e.message, true);
  } finally {
    controller = null; setBusy(false);
    if (!generated) gen = null;
    render(); src.focus();
  }
}

/* ------------------------------------------------------------ ficha del modelo */
function renderCard(info) {
  const c = info.card, root = $("card");
  root.replaceChildren();
  root.append(el("h2", null, c.nombre), el("p", "lead", c.resumen));

  root.append(el("h3", null, "Lo que logra"));
  const wins = el("div", "wins");
  c.logros.forEach((w) => { const d = el("div", "win"); d.append(el("b", null, w.valor), el("span", null, w.texto)); wins.append(d); });
  root.append(wins);

  root.append(el("h3", null, "Lo que no hace (todavía)"));
  const ul = el("ul", "limits");
  c.limitaciones.forEach((t) => ul.append(el("li", null, t)));
  root.append(ul);

  root.append(el("h3", null, "Especificaciones"));
  const table = el("table", "specs"), tb = el("tbody");
  c.especificaciones.forEach(([k, v]) => { const r = el("tr"); r.append(el("th", null, k), el("td", null, v)); tb.append(r); });
  table.append(tb);
  root.append(table);
  root.append(el("p", "src", c.fuentes));
}

/* ------------------------------------------------------------ ejemplos y pestañas */
let EXAMPLES = [];
function loadExample(i) {
  const ex = EXAMPLES[i];
  if (!ex) return;
  const at = ex.texto.indexOf(CURSOR);
  src.value = at >= 0 ? ex.texto.replace(CURSOR, "") : ex.texto;
  $("lang").value = ex.lang; saveSettings();
  gen = null; render(); src.focus();
  const caret = at >= 0 ? at : src.value.length;
  src.setSelectionRange(caret, caret); ensureVisible(caret);
  setStatus(at >= 0 ? "Ejemplo cargado: el cursor está en el hueco. Pulsa Rellenar en el cursor." : "Ejemplo cargado. Pulsa Completar.");
}
function showTab(name) {
  const isCard = name === "card";
  $("view-editor").hidden = isCard; $("view-card").hidden = !isCard;
  $("tab-editor").classList.toggle("active", !isCard); $("tab-card").classList.toggle("active", isCard);
  $("tab-editor").setAttribute("aria-selected", String(!isCard)); $("tab-card").setAttribute("aria-selected", String(isCard));
}

/* ------------------------------------------------------------ arranque */
async function init() {
  applySettings({ ...DEFAULTS, ...recall("settings", {}) });
  render();
  src.addEventListener("input", () => { gen = null; render(); });
  src.addEventListener("scroll", syncScroll);
  src.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.ctrlKey) { e.preventDefault(); run("complete"); }
    else if (e.key === "Enter" && e.altKey) { e.preventDefault(); run("fill"); }
  });
  ["temp", "tokens", "rep"].forEach((id) => $(id).addEventListener("input", () => { showSettings(); saveSettings(); }));
  $("lang").addEventListener("change", () => { saveSettings(); render(); });
  $("btn-complete").addEventListener("click", () => run("complete"));
  $("btn-fill").addEventListener("click", () => run("fill"));
  $("btn-stop").addEventListener("click", () => controller && controller.abort());
  $("btn-reset").addEventListener("click", () => { applySettings(DEFAULTS); saveSettings(); render(); });
  $("example").addEventListener("change", (e) => { if (e.target.value !== "") loadExample(+e.target.value); e.target.value = ""; });
  $("tab-editor").addEventListener("click", () => showTab("editor"));
  $("tab-card").addEventListener("click", () => showTab("card"));
  $("to-card").addEventListener("click", (e) => { e.preventDefault(); showTab("card"); });

  try {
    const info = await (await fetch("/api/info")).json();
    CURSOR = info.cursor || CURSOR;
    $("chip").textContent = `paso ${fmt(info.step)} · ${fmt(info.params / 1e6, 2)} M parámetros`;
    $("chip").classList.add("ok");
    EXAMPLES = info.card.ejemplos;
    EXAMPLES.forEach((ex, i) => { const o = el("option", null, ex.titulo); o.value = i; $("example").append(o); });
    renderCard(info);
    if (!src.value) loadExample(0);
  } catch (_) {
    $("chip").textContent = "sin conexión con el servidor";
    $("chip").classList.add("err");
    setStatus("No se pudo leer la información del modelo. ¿Sigue abierta la ventana de demo_nexora.bat?", true);
  }
}
init();
