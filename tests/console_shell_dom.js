// Headless harness for src/lmds/web/static/index.html — a tiny DOM + browser globals
// good enough to boot the real console script in plain `node` (no jsdom on the hub/CI).
//
// ทำไมต้องมี: บั๊ก "กด site/node ที่ rail แล้วตรงกลางว่าง" อยู่ในลำดับเหตุการณ์ (route มาก่อน
// การ์ด / SSE วาดทับ / ไซต์ที่ยุบไว้) ซึ่งเทสแบบ grep สตริงจับไม่ได้ · ต้องรันโค้ดจริงกับ DOM
// จริง ๆ ที่มี hidden/style/classList แล้วดูว่าอะไร "มองเห็น" — jsdom ไม่มีบนเครื่อง จึงเขียน
// DOM ย่อส่วนที่หน้านี้ใช้จริง (parser + selector + event bubbling) ไว้ตรงนี้แทน
//
// usage: node console_shell_dom.js <index.html> <scenario.js>
//   scenario.js runs inside an async function with the page globals in scope and `H`:
//     H.routes         — [[pattern, handler], …] for fetch; handler(url, opts) → body | {status, body}
//     H.fixture        — quick fleet fixture builder (see defaultRoutes)
//     H.sse(snapshot)  — push one SSE frame through the page's EventSource
//     H.tick(n)        — let promises / timers settle
//     H.go(hash)       — set location.hash and settle (hashchange fires like a browser task)
//     H.visible(el)    — is the element actually painted (hidden attr, style.display, CSS rules)
//     H.assert(cond, message)
"use strict";
const fs = require("fs");
const vm = require("vm");

// ───────────────────────────── HTML parser ─────────────────────────────
const VOID = new Set(["area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"]);
const RAW = new Set(["script", "style", "textarea"]);
const ENT = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " ", hellip: "…", middot: "·", times: "×", rarr: "→", larr: "←" };
function decode(s) {
  return s.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (m, e) => {
    if (e[0] === "#") return String.fromCodePoint(e[1] === "x" || e[1] === "X" ? parseInt(e.slice(2), 16) : parseInt(e.slice(1), 10));
    return e in ENT ? ENT[e] : m;
  });
}
function encode(s) { return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

function parseHTML(html, doc, parent) {
  let i = 0; const n = html.length;
  const stack = [parent];
  const top = () => stack[stack.length - 1];
  while (i < n) {
    const lt = html.indexOf("<", i);
    if (lt < 0) { top().appendChild(doc.createTextNode(decode(html.slice(i)))); break; }
    if (lt > i) top().appendChild(doc.createTextNode(decode(html.slice(i, lt))));
    if (html.startsWith("<!--", lt)) { const e = html.indexOf("-->", lt + 4); i = e < 0 ? n : e + 3; continue; }
    if (html.startsWith("<!", lt) || html.startsWith("<?", lt)) { const e = html.indexOf(">", lt); i = e < 0 ? n : e + 1; continue; }
    if (html[lt + 1] === "/") {
      const e = html.indexOf(">", lt);
      const name = html.slice(lt + 2, e).trim().toLowerCase();
      for (let k = stack.length - 1; k > 0; k--) {
        if (stack[k].tagName.toLowerCase() === name) { stack.length = k; break; }
      }
      i = e + 1; continue;
    }
    // open tag
    let j = lt + 1;
    while (j < n && /[^\s/>]/.test(html[j])) j++;
    const name = html.slice(lt + 1, j).toLowerCase();
    if (!name) { top().appendChild(doc.createTextNode("<")); i = lt + 1; continue; }
    const el = doc.createElement(name);
    let selfClose = false;
    for (;;) {
      while (j < n && /\s/.test(html[j])) j++;
      if (j >= n) break;
      if (html[j] === ">") { j++; break; }
      if (html[j] === "/") { selfClose = true; j++; continue; }
      let k = j;
      while (k < n && /[^\s=/>]/.test(html[k])) k++;
      const attr = html.slice(j, k);
      j = k;
      while (j < n && /\s/.test(html[j])) j++;
      let value = "";
      if (html[j] === "=") {
        j++;
        while (j < n && /\s/.test(html[j])) j++;
        const q = html[j];
        if (q === '"' || q === "'") { const e = html.indexOf(q, j + 1); value = html.slice(j + 1, e < 0 ? n : e); j = e < 0 ? n : e + 1; }
        else { k = j; while (k < n && /[^\s>]/.test(html[k])) k++; value = html.slice(j, k); j = k; }
      }
      if (attr) el.setAttribute(attr, decode(value));
    }
    top().appendChild(el);
    i = j;
    if (RAW.has(name)) {
      const close = html.toLowerCase().indexOf("</" + name, i);
      const raw = html.slice(i, close < 0 ? n : close);
      if (raw) el.appendChild(doc.createTextNode(raw));
      const e = html.indexOf(">", close < 0 ? n : close);
      i = e < 0 ? n : e + 1;
      continue;
    }
    if (!selfClose && !VOID.has(name)) stack.push(el);
  }
}

// ───────────────────────────── selectors ─────────────────────────────
// Right-to-left compound matching: tag · #id · .class · [attr] · [attr=v] · :pseudo · space / > / + / ~
function unescapeIdent(s) {
  return s.replace(/\\([0-9a-f]{1,6}\s?|.)/gi, (m, e) => /^[0-9a-f]/i.test(e) ? String.fromCodePoint(parseInt(e.trim(), 16)) : e);
}
function splitTop(s, sep) {
  const out = []; let depth = 0, q = null, cur = "";
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (q) { cur += c; if (c === "\\") { cur += s[++i] ?? ""; } else if (c === q) q = null; continue; }
    if (c === '"' || c === "'") { q = c; cur += c; continue; }
    if (c === "\\") { cur += c + (s[++i] ?? ""); continue; }
    if (c === "(" || c === "[") depth++;
    if (c === ")" || c === "]") depth--;
    if (depth === 0 && c === sep) { out.push(cur); cur = ""; continue; }
    cur += c;
  }
  out.push(cur);
  return out;
}
function parseCompound(s) {
  const parts = [];
  const re = /(\*)|([a-zA-Z][\w-]*)|#((?:\\.|[\w-])+)|\.((?:\\.|[\w-])+)|\[\s*([\w-]+)\s*(?:([~|^$*]?=)\s*("(?:\\.|[^"])*"|'(?:\\.|[^'])*'|(?:\\.|[^\]\s])+))?\s*\]|:([\w-]+)(?:\((.*?)\))?/y;
  let m; re.lastIndex = 0;
  while ((m = re.exec(s))) {
    if (m[1]) parts.push({ t: "any" });
    else if (m[2]) parts.push({ t: "tag", v: m[2].toLowerCase() });
    else if (m[3]) parts.push({ t: "id", v: unescapeIdent(m[3]) });
    else if (m[4]) parts.push({ t: "class", v: unescapeIdent(m[4]) });
    else if (m[5]) {
      let v = m[7];
      if (v != null) { if (/^["']/.test(v)) v = v.slice(1, -1); v = unescapeIdent(v); }
      parts.push({ t: "attr", name: m[5], op: m[6], v });
    } else if (m[8]) parts.push({ t: "pseudo", v: m[8], arg: m[9] });
    if (re.lastIndex >= s.length) break;
  }
  if (re.lastIndex !== s.length) throw new Error("unsupported selector: " + s);
  return parts;
}
function parseSelector(sel) {
  return splitTop(sel, ",").map(one => {
    const tokens = one.trim().replace(/\s*([>+~])\s*/g, " $1 ").split(/\s+/).filter(Boolean);
    const chain = []; let comb = " ";
    for (const tok of tokens) {
      if (tok === ">" || tok === "+" || tok === "~") { comb = tok; continue; }
      chain.push({ comb, parts: parseCompound(tok) });
      comb = " ";
    }
    return chain;
  });
}
function matchCompound(el, parts) {
  for (const p of parts) {
    switch (p.t) {
      case "any": break;
      case "tag": if (el.tagName.toLowerCase() !== p.v) return false; break;
      case "id": if (el.getAttribute("id") !== p.v) return false; break;
      case "class": if (!el.classList.contains(p.v)) return false; break;
      case "attr": {
        if (!el.hasAttribute(p.name)) return false;
        if (p.op == null) break;
        const a = el.getAttribute(p.name);
        if (p.op === "=" && a !== p.v) return false;
        if (p.op === "^=" && !a.startsWith(p.v)) return false;
        if (p.op === "$=" && !a.endsWith(p.v)) return false;
        if (p.op === "*=" && !a.includes(p.v)) return false;
        if (p.op === "~=" && !a.split(/\s+/).includes(p.v)) return false;
        break;
      }
      case "pseudo":
        if (p.v === "checked") { if (!el.checked) return false; break; }
        if (p.v === "disabled") { if (!el.disabled) return false; break; }
        if (p.v === "not") { if (parseSelector(p.arg).some(chain => matchChain(el, chain))) return false; break; }
        if (p.v === "last-child") { const c = el.parentNode && el.parentNode.children; if (!c || c[c.length - 1] !== el) return false; break; }
        if (p.v === "first-child") { const c = el.parentNode && el.parentNode.children; if (!c || c[0] !== el) return false; break; }
        throw new Error("unsupported pseudo :" + p.v);
    }
  }
  return true;
}
function matchChain(el, chain, idx = chain.length - 1) {
  const step = chain[idx];
  if (!matchCompound(el, step.parts)) return false;
  if (idx === 0) return true;
  const comb = step.comb;
  if (comb === " ") { for (let a = el.parentElement; a; a = a.parentElement) if (matchChain(a, chain, idx - 1)) return true; return false; }
  if (comb === ">") return !!el.parentElement && matchChain(el.parentElement, chain, idx - 1);
  if (comb === "+") { const s = el.previousElementSibling; return !!s && matchChain(s, chain, idx - 1); }
  if (comb === "~") { for (let s = el.previousElementSibling; s; s = s.previousElementSibling) if (matchChain(s, chain, idx - 1)) return true; return false; }
  return false;
}
function matches(el, sel) { return parseSelector(sel).some(chain => matchChain(el, chain)); }

// ───────────────────────────── nodes ─────────────────────────────
class Event {
  constructor(type, init = {}) {
    this.type = type; this.bubbles = init.bubbles !== false; this.cancelable = true;
    this.defaultPrevented = false; this._stop = false; this._stopNow = false;
    Object.assign(this, init);
  }
  preventDefault() { this.defaultPrevented = true; }
  stopPropagation() { this._stop = true; }
  stopImmediatePropagation() { this._stop = true; this._stopNow = true; }
}
class EventTarget {
  constructor() { this._listeners = new Map(); }
  addEventListener(type, fn) { if (!fn) return; (this._listeners.get(type) || this._listeners.set(type, []).get(type)).push(fn); }
  removeEventListener(type, fn) { const l = this._listeners.get(type); if (l) { const i = l.indexOf(fn); if (i >= 0) l.splice(i, 1); } }
  _invoke(ev) {
    ev.currentTarget = this;
    const inline = this["on" + ev.type];
    if (typeof inline === "function") { const r = inline.call(this, ev); if (r === false) ev.preventDefault(); }
    for (const fn of [...(this._listeners.get(ev.type) || [])]) {
      if (ev._stopNow) break;
      typeof fn === "function" ? fn.call(this, ev) : fn.handleEvent(ev);
    }
  }
  dispatchEvent(ev) {
    ev.target = ev.target || this;
    const path = [];
    for (let n = this; n; n = n.parentNode) path.push(n);
    if (path[path.length - 1] instanceof Document) path.push(globalThis.window);
    else if (this !== globalThis.window && !(this instanceof Document)) { /* detached: no bubbling */ }
    for (const n of ev.bubbles ? path : [this]) { if (ev._stop) break; n._invoke(ev); }
    return !ev.defaultPrevented;
  }
}
class Node extends EventTarget {
  constructor(doc) { super(); this.ownerDocument = doc; this.parentNode = null; this.childNodes = []; }
  get parentElement() { return this.parentNode instanceof Element ? this.parentNode : null; }
  get isConnected() { for (let n = this; n; n = n.parentNode) if (n instanceof Document) return true; return false; }
  get firstChild() { return this.childNodes[0] || null; }
  get lastChild() { return this.childNodes[this.childNodes.length - 1] || null; }
  get nextSibling() { const p = this.parentNode; if (!p) return null; return p.childNodes[p.childNodes.indexOf(this) + 1] || null; }
  get previousSibling() { const p = this.parentNode; if (!p) return null; return p.childNodes[p.childNodes.indexOf(this) - 1] || null; }
  get children() { return this.childNodes.filter(c => c instanceof Element); }
  get childElementCount() { return this.children.length; }
  get firstElementChild() { return this.children[0] || null; }
  get lastElementChild() { const c = this.children; return c[c.length - 1] || null; }
  get nextElementSibling() { for (let s = this.nextSibling; s; s = s.nextSibling) if (s instanceof Element) return s; return null; }
  get previousElementSibling() { for (let s = this.previousSibling; s; s = s.previousSibling) if (s instanceof Element) return s; return null; }
  contains(other) { for (let n = other; n; n = n.parentNode) if (n === this) return true; return false; }
  _adopt(node) {
    if (typeof node === "string") node = this.ownerDocument.createTextNode(node);
    if (node instanceof DocumentFragment) { const kids = [...node.childNodes]; kids.forEach(k => node.removeChild(k)); return kids; }
    if (node.parentNode) node.parentNode.removeChild(node);
    return [node];
  }
  appendChild(node) { for (const k of this._adopt(node)) { k.parentNode = this; this.childNodes.push(k); } return node; }
  insertBefore(node, ref) {
    if (!ref) return this.appendChild(node);
    for (const k of this._adopt(node)) { const i = this.childNodes.indexOf(ref); if (i < 0) throw new Error("insertBefore: ref not a child"); k.parentNode = this; this.childNodes.splice(i, 0, k); }
    return node;
  }
  removeChild(node) { const i = this.childNodes.indexOf(node); if (i < 0) throw new Error("removeChild: not a child"); this.childNodes.splice(i, 1); node.parentNode = null; return node; }
  append(...nodes) { nodes.forEach(n => this.appendChild(n)); }
  prepend(...nodes) { const first = this.firstChild; nodes.forEach(n => this.insertBefore(n, first)); }
  replaceChildren(...nodes) { [...this.childNodes].forEach(c => this.removeChild(c)); this.append(...nodes); }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  before(...nodes) { nodes.forEach(n => this.parentNode.insertBefore(n, this)); }
  after(...nodes) { const ref = this.nextSibling; nodes.forEach(n => this.parentNode.insertBefore(n, ref)); }
  replaceWith(...nodes) { const p = this.parentNode; this.before(...nodes); p.removeChild(this); }
  get textContent() { return this.childNodes.map(c => c.textContent).join(""); }
  set textContent(v) { this.replaceChildren(); if (v !== "" && v != null) this.appendChild(this.ownerDocument.createTextNode(String(v))); }
  get innerText() { return this.textContent; }
  set innerText(v) { this.textContent = v; }
  querySelectorAll(sel) { const out = []; const walk = n => { for (const c of n.childNodes) if (c instanceof Element) { if (matches(c, sel)) out.push(c); walk(c); } }; walk(this); return out; }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  getElementsByTagName(t) { return this.querySelectorAll(t); }
  getElementsByClassName(c) { return this.querySelectorAll("." + c); }
}
class TextNode extends Node {
  constructor(doc, text) { super(doc); this.nodeType = 3; this.nodeName = "#text"; this.data = String(text); }
  get textContent() { return this.data; }
  set textContent(v) { this.data = String(v); }
  get nodeValue() { return this.data; }
  cloneNode() { return new TextNode(this.ownerDocument, this.data); }
}
class DocumentFragment extends Node { constructor(doc) { super(doc); this.nodeType = 11; this.nodeName = "#document-fragment"; } }

const BOOL_ATTRS = ["hidden", "disabled", "selected", "readOnly", "required", "multiple", "autofocus"];
class Element extends Node {
  constructor(doc, tag) {
    super(doc);
    this.nodeType = 1; this.tagName = tag.toUpperCase(); this.nodeName = this.tagName;
    this._attrs = new Map(); this._value = undefined; this._checked = undefined;
    this._styleMap = new Map();
    const self = this;
    this.style = new Proxy({}, {
      get(_, k) {
        if (k === "setProperty") return (p, v) => { self._styleMap.set(p, String(v)); };
        if (k === "removeProperty") return p => { const v = self._styleMap.get(p) || ""; self._styleMap.delete(p); return v; };
        if (k === "getPropertyValue") return p => self._styleMap.get(p) || "";
        if (k === "cssText") return [...self._styleMap].map(([p, v]) => `${p}: ${v}`).join("; ");
        if (typeof k !== "string") return undefined;
        return self._styleMap.get(kebab(k)) || "";
      },
      set(_, k, v) {
        if (k === "cssText") { self._styleMap.clear(); String(v).split(";").forEach(d => { const [p, ...r] = d.split(":"); if (p && p.trim()) self._styleMap.set(p.trim(), r.join(":").trim()); }); return true; }
        if (v === "" || v == null) self._styleMap.delete(kebab(k)); else self._styleMap.set(kebab(k), String(v));
        return true;
      },
    });
    this.dataset = new Proxy({}, {
      get: (_, k) => typeof k === "string" ? (self.hasAttribute("data-" + kebab(k)) ? self.getAttribute("data-" + kebab(k)) : undefined) : undefined,
      set: (_, k, v) => { self.setAttribute("data-" + kebab(k), String(v)); return true; },
      has: (_, k) => self.hasAttribute("data-" + kebab(k)),
      deleteProperty: (_, k) => { self.removeAttribute("data-" + kebab(k)); return true; },
      ownKeys: () => [...self._attrs.keys()].filter(a => a.startsWith("data-")).map(a => camel(a.slice(5))),
      getOwnPropertyDescriptor: (_, k) => self.hasAttribute("data-" + kebab(k)) ? { value: self.getAttribute("data-" + kebab(k)), enumerable: true, configurable: true } : undefined,
    });
    this.classList = {
      contains: c => self._classes().includes(c),
      add: (...cs) => { const s = self._classes(); cs.forEach(c => { if (!s.includes(c)) s.push(c); }); self.className = s.join(" "); },
      remove: (...cs) => { self.className = self._classes().filter(c => !cs.includes(c)).join(" "); },
      toggle: (c, force) => { const on = force === undefined ? !self.classList.contains(c) : !!force; on ? self.classList.add(c) : self.classList.remove(c); return on; },
      get length() { return self._classes().length; },
      [Symbol.iterator]: function* () { yield* self._classes(); },
    };
  }
  _classes() { return (this.getAttribute("class") || "").split(/\s+/).filter(Boolean); }
  getAttribute(k) { return this._attrs.has(k.toLowerCase()) ? this._attrs.get(k.toLowerCase()) : null; }
  setAttribute(k, v) { this._attrs.set(k.toLowerCase(), String(v)); }
  removeAttribute(k) { this._attrs.delete(k.toLowerCase()); }
  hasAttribute(k) { return this._attrs.has(k.toLowerCase()); }
  toggleAttribute(k, force) { const on = force === undefined ? !this.hasAttribute(k) : !!force; on ? this.setAttribute(k, "") : this.removeAttribute(k); return on; }
  get attributes() { return [...this._attrs].map(([name, value]) => ({ name, value })); }
  get id() { return this.getAttribute("id") || ""; } set id(v) { this.setAttribute("id", v); }
  get className() { return this.getAttribute("class") || ""; } set className(v) { this.setAttribute("class", v); }
  get title() { return this.getAttribute("title") || ""; } set title(v) { this.setAttribute("title", v); }
  get href() { return this.getAttribute("href") || ""; } set href(v) { this.setAttribute("href", v); }
  get name() { return this.getAttribute("name") || ""; } set name(v) { this.setAttribute("name", v); }
  get type() { return this.getAttribute("type") || (this.tagName === "BUTTON" ? "submit" : "text"); } set type(v) { this.setAttribute("type", v); }
  get placeholder() { return this.getAttribute("placeholder") || ""; } set placeholder(v) { this.setAttribute("placeholder", v); }
  get value() {
    if (this._value !== undefined) return this._value;
    if (this.tagName === "SELECT") { const o = this.querySelectorAll("option"); const s = o.find(x => x.hasAttribute("selected")) || o[0]; return s ? s.value : ""; }
    if (this.tagName === "OPTION") return this.hasAttribute("value") ? this.getAttribute("value") : this.textContent;
    if (this.tagName === "TEXTAREA") return this.textContent;
    return this.getAttribute("value") || "";
  }
  set value(v) { this._value = String(v); }
  get checked() { return this._checked !== undefined ? this._checked : this.hasAttribute("checked"); } set checked(v) { this._checked = !!v; }
  get options() { return this.querySelectorAll("option"); }
  get selectedIndex() { const o = this.options; const v = this.value; return o.findIndex(x => x.value === v); }
  get innerHTML() { return this.childNodes.map(serialize).join(""); }
  set innerHTML(html) { this.replaceChildren(); parseHTML(String(html), this.ownerDocument, this); }
  get outerHTML() { return serialize(this); }
  insertAdjacentHTML(pos, html) {
    const frag = new DocumentFragment(this.ownerDocument); parseHTML(String(html), this.ownerDocument, frag);
    if (pos === "beforeend") this.appendChild(frag); else if (pos === "afterbegin") this.prepend(frag);
    else if (pos === "beforebegin") this.before(frag); else if (pos === "afterend") this.after(frag);
  }
  cloneNode(deep) { const c = new Element(this.ownerDocument, this.tagName); for (const [k, v] of this._attrs) c._attrs.set(k, v); if (deep) this.childNodes.forEach(k => c.appendChild(k.cloneNode(true))); return c; }
  matches(sel) { return matches(this, sel); }
  closest(sel) { for (let n = this; n instanceof Element; n = n.parentElement) if (matches(n, sel)) return n; return null; }
  click() { this.dispatchEvent(new Event("click", { bubbles: true })); }
  focus() { this.ownerDocument.activeElement = this; this.dispatchEvent(new Event("focus", { bubbles: false })); }
  blur() { if (this.ownerDocument.activeElement === this) this.ownerDocument.activeElement = this.ownerDocument.body; }
  select() {}
  requestSubmit() { this.dispatchEvent(new Event("submit", { bubbles: true })); }
  scrollIntoView() { this.ownerDocument._scrolledTo.push(this); }
  getBoundingClientRect() { return { top: 0, left: 0, right: 0, bottom: 0, width: 0, height: 0 }; }
  get offsetHeight() { return 0; } get offsetTop() { return 0; } get clientHeight() { return 0; } get scrollHeight() { return 0; }
  get scrollTop() { return this._scrollTop || 0; } set scrollTop(v) { this._scrollTop = v; }
}
for (const a of BOOL_ATTRS) {
  Object.defineProperty(Element.prototype, a, {
    get() { return this.hasAttribute(a.toLowerCase()); },
    set(v) { v ? this.setAttribute(a.toLowerCase(), "") : this.removeAttribute(a.toLowerCase()); },
  });
}
function kebab(s) { return s.replace(/[A-Z]/g, c => "-" + c.toLowerCase()); }
function camel(s) { return s.replace(/-([a-z])/g, (_, c) => c.toUpperCase()); }
function serialize(n) {
  if (n instanceof TextNode) return RAW.has((n.parentNode && n.parentNode.tagName || "").toLowerCase()) ? n.data : encode(n.data);
  const tag = n.tagName.toLowerCase();
  const attrs = [...n._attrs].map(([k, v]) => v === "" ? ` ${k}` : ` ${k}="${encode(v)}"`).join("");
  if (VOID.has(tag)) return `<${tag}${attrs}>`;
  return `<${tag}${attrs}>${n.childNodes.map(serialize).join("")}</${tag}>`;
}
class Document extends Node {
  constructor() {
    super(null); this.ownerDocument = this; this.nodeType = 9; this.nodeName = "#document";
    this._scrolledTo = []; this.title = "";
    this.documentElement = this.createElement("html"); this.appendChild(this.documentElement);
    this.head = this.createElement("head"); this.body = this.createElement("body");
    this.documentElement.append(this.head, this.body);
    this.activeElement = this.body;
  }
  createElement(tag) { return new Element(this, tag); }
  createElementNS(_, tag) { return new Element(this, tag); }
  createTextNode(t) { return new TextNode(this, t); }
  createDocumentFragment() { return new DocumentFragment(this); }
  getElementById(id) { const walk = n => { for (const c of n.childNodes) if (c instanceof Element) { if (c.getAttribute("id") === id) return c; const r = walk(c); if (r) return r; } return null; }; return walk(this); }
  get scrollingElement() { return this.documentElement; }
}

// ───────────────────────────── browser globals ─────────────────────────────
function installGlobals(doc, H) {
  const g = globalThis;
  // node ≥22 ships its own navigator/localStorage/fetch getters — plain assignment throws or is ignored
  const def = (name, value) => Object.defineProperty(g, name, { value, writable: true, configurable: true, enumerable: true });
  def("window", g);
  def("document", doc);
  def("Event", Event); def("CustomEvent", Event); def("KeyboardEvent", Event); def("MouseEvent", Event);
  def("HTMLElement", Element); def("Element", Element); def("Node", Node);
  // window is an EventTarget too
  def("addEventListener", EventTarget.prototype.addEventListener);
  def("removeEventListener", EventTarget.prototype.removeEventListener);
  def("dispatchEvent", EventTarget.prototype.dispatchEvent);
  def("_invoke", EventTarget.prototype._invoke);
  def("_listeners", new Map());
  def("innerWidth", 1400); def("innerHeight", 900); def("scrollY", 0);
  def("scrollTo", () => {}); def("scrollBy", () => {});
  def("requestAnimationFrame", fn => setTimeout(() => fn(Date.now()), 0));
  def("cancelAnimationFrame", clearTimeout);
  def("getComputedStyle", el => ({ display: H.visible(el) ? "block" : "none", getPropertyValue: p => el.style.getPropertyValue(p) }));
  def("matchMedia", () => ({ matches: false, addEventListener() {}, removeEventListener() {}, addListener() {} }));
  const storage = () => {
    const m = new Map();
    return { getItem: k => m.has(k) ? m.get(k) : null, setItem: (k, v) => m.set(k, String(v)), removeItem: k => m.delete(k),
      clear: () => m.clear(), key: i => [...m.keys()][i] ?? null, get length() { return m.size; } };
  };
  def("localStorage", storage()); def("sessionStorage", storage());
  def("navigator", { userAgent: "node-harness", clipboard: { writeText: async () => {} }, language: "en" });
  def("CSS", { escape: s => String(s).replace(/([\0-\x1f\x7f]|^-?\d)|([^\w-])/g, (m, ctrl, other) => ctrl ? `\\${ctrl.charCodeAt(ctrl.length - 1).toString(16)} ` : "\\" + other) });
  def("alert", m => H.alerts.push(String(m)));
  def("confirm", m => { H.confirms.push(String(m)); return H.confirmAnswer; });
  def("prompt", (m, d) => { H.prompts.push(String(m)); return H.promptAnswer === undefined ? d ?? "" : H.promptAnswer; });
  // location: hash setter fires hashchange as a task (like a browser), not inline
  let hash = "";
  def("location", {
    protocol: "http:", host: "hub.local:8130", hostname: "hub.local", origin: "http://hub.local:8130",
    pathname: "/", search: "",
    get hash() { return hash; },
    set hash(v) {
      v = String(v); if (v && !v.startsWith("#")) v = "#" + v;
      if (v === hash) return;
      hash = v;
      setTimeout(() => g.dispatchEvent(new Event("hashchange", { bubbles: false })), 0);
    },
    get href() { return this.origin + this.pathname + this.search + hash; },
    reload() { H.reloads++; }, assign() {}, replace() {},
  });
  def("history", { replaceState() {}, pushState() {}, back() { H.backs++; }, state: null });
  def("EventSource", class {
    constructor(url) { this.url = url; this.onmessage = null; this.onerror = null; this.closed = false; H.streams.push(this); }
    close() { this.closed = true; }
  });
  function response(status, body) {
    const text = typeof body === "string" ? body : JSON.stringify(body ?? {});
    return { ok: status >= 200 && status < 300, status, headers: { get: () => "application/json" },
      json: async () => JSON.parse(text), text: async () => text,
      body: { getReader: () => ({ read: async () => ({ done: true }), cancel: async () => {} }) } };
  }
  def("fetch", async (url, opts = {}) => {
    url = String(url);
    H.calls.push({ url, method: (opts.method || "GET").toUpperCase(), body: opts.body });
    for (const [pat, handler] of H.routes) {
      const hit = pat instanceof RegExp ? pat.test(url) : (url === pat || url.startsWith(pat + "?"));
      if (!hit) continue;
      let r = typeof handler === "function" ? handler(url, opts) : handler;
      if (r instanceof Promise) r = await r;
      if (r && typeof r === "object" && "status" in r && "body" in r) return response(r.status, r.body);
      return response(200, r);
    }
    return response(404, { detail: `no fake route for ${url}` });
  });
  process.on("unhandledRejection", reason => {
    const ev = new Event("unhandledrejection", { bubbles: false }); ev.reason = reason;
    g.dispatchEvent(ev);
    if (!ev.defaultPrevented) H.errors.push("unhandled: " + String(reason && reason.stack || reason));
  });
}

// ───────────────────────────── harness ─────────────────────────────
function makeHarness(doc) {
  const H = {
    routes: [], calls: [], streams: [], alerts: [], confirms: [], prompts: [], errors: [], reloads: 0, backs: 0,
    confirmAnswer: true, promptAnswer: undefined,
    sleep: ms => new Promise(r => setTimeout(r, ms)),
    async tick(n = 6) { for (let i = 0; i < n; i++) await new Promise(r => setImmediate(r)); await H.sleep(1); for (let i = 0; i < n; i++) await new Promise(r => setImmediate(r)); },
    async go(hash) { globalThis.location.hash = hash; await H.tick(); },
    sse(snapshot) { const s = H.streams[H.streams.length - 1]; if (!s || !s.onmessage) throw new Error("no EventSource open"); s.onmessage({ data: JSON.stringify(snapshot) }); },
    assert(cond, message) { if (!cond) throw new Error("ASSERT: " + message); },
    // The CSS rules the shell relies on for "is this on screen" — keep in sync with index.html.
    visible(el) {
      for (let n = el; n instanceof Element; n = n.parentElement) {
        if (n.hidden) return false;
        if (n.style.display === "none") return false;
        const cl = n.classList;
        if (cl.contains("card") && cl.contains("folded")) return false;
        if (cl.contains("fold-hidden")) return false;
        if (cl.contains("nbody") && cl.contains("collapsed")) return false;
        if (cl.contains("rtree") && !cl.contains("open")) return false;
        if ((cl.contains("nbody") || cl.contains("nclus") || cl.contains("nout")) && n.closest(".machine.ncompact")) return false;
        if (cl.contains("sitehdr") && (n.closest("#nodes.route-node") || n.closest("#nodes.route-site"))) return false;
      }
      return el.isConnected;
    },
    visibleMachines() { return [...doc.querySelectorAll("#nodes .machine")].filter(H.visible).map(m => m.querySelector(".name").firstChild.textContent.trim()); },
    // Fake fleet API from a compact fixture: nodes[{name, site, models:[…], gpu:{…}}], cluster groups optional
    defaultRoutes(fx) {
      const inv = n => ({ name: n.name, reachable: n.reachable !== false, error: n.error || "", age_seconds: 1,
        host: { hostname: n.name, lmds_version: "0.6.0", lmds_commit: "abc1234", gpus: [n.gpu || { name: "NVIDIA GB10", vram_gb: 128, vram_used_gb: 40 }], docker: true, toolkit: true, profile: "spark", arch: "arm64", role: { control_plane: false, engines: ["llamacpp"] } },
        models: n.models || [], summary: { running: (n.models || []).filter(m => m.running).length, total: (n.models || []).length } });
      const hostData = { hostname: "hub", gpus: [], ips: [], docker: true, toolkit: true, profile: "x86", arch: "x86_64", memory_model: "discrete", role: { control_plane: true, engines: [] } };
      return [
        ["/api/auth", () => ({ required: false })],
        ["/api/host", () => hostData],
        ["/api/models", () => ({ models: fx.localModels || [] })],
        ["/api/fleet/summary", () => ({ machines: fx.nodes.length, online: fx.nodes.length, pending: 0, gpus: fx.nodes.length, vram_gb: 128 * fx.nodes.length, models_running: 0, models_healthy: 0, models_total: 0 })],
        ["/api/nodes", () => ({ nodes: fx.nodes.map(n => ({ name: n.name, site: n.site || "", user: "u", host: n.host || "10.0.0.1", port: 22, note: "", local_ip: n.host || "10.0.0.1", stack: true })) })],
        ["/api/cluster", () => fx.cluster || { machines: fx.nodes.map(n => ({ name: n.name, reachable: true, ready: false, has_gpu: true })), groups: [] }],
        ["/api/version", () => ({ version: "0.6.0", commit: "abc1234" })],
        ["/api/provider", () => ({ configured: false, choices: ["openai"], defaults: {} })],
        ["/api/assistant", () => ({ available: false })],
        [/^\/api\/nodes\/[^/]+\/inventory/, url => { const name = decodeURIComponent(url.split("/")[3]); const n = fx.nodes.find(x => x.name === name); return n ? inv(n) : { status: 404, body: { detail: "unknown node" } }; }],
      ];
    },
    snapshot(fx) {
      const nodes = {};
      for (const n of fx.nodes) nodes[n.name] = n.reachable === false ? { error: n.error || "down", age_seconds: 2 } : { age_seconds: 2, data: {
        host: { hostname: n.name, lmds_version: "0.6.0", lmds_commit: "abc1234", gpus: [n.gpu || { name: "NVIDIA GB10", vram_gb: 128, vram_used_gb: 40 }], docker: true, toolkit: true, profile: "spark", arch: "arm64" },
        models: n.models || [], summary: { running: (n.models || []).filter(m => m.running).length, total: (n.models || []).length } } };
      return { nodes };
    },
  };
  return H;
}

async function main() {
  const [htmlPath, scenarioPath] = process.argv.slice(2);
  if (!htmlPath || !scenarioPath) { console.error("usage: node console_shell_dom.js <index.html> <scenario.js>"); process.exit(2); }
  const html = fs.readFileSync(htmlPath, "utf8");
  const doc = new Document();
  const H = makeHarness(doc);
  installGlobals(doc, H);
  globalThis.H = H;
  const bodyStart = html.search(/<body[^>]*>/), bodyEnd = html.lastIndexOf("</body>");
  parseHTML(html.slice(html.indexOf(">", bodyStart) + 1, bodyEnd < 0 ? html.length : bodyEnd), doc, doc.body);
  const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
  // scenario prelude may set H.routes before the page boots
  const scenario = fs.readFileSync(scenarioPath, "utf8");
  const [before, after = ""] = scenario.split(/^\/\/ ---- boot ----$/m);
  const run = (code, name) => vm.runInThisContext(`(async () => {\n${code}\n})()`, { filename: name });
  await run(before, "scenario.pre.js");
  for (const [i, code] of scripts.entries()) vm.runInThisContext(code, { filename: `index.html#script${i}` });
  await H.tick(10);
  await run(after, "scenario.js");
  if (H.errors.length) throw new Error(H.errors.join("\n"));
}
main().then(() => { process.exit(0); }, e => { console.error(e && e.stack || e); process.exit(1); });
