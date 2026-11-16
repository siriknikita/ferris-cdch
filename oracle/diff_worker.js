/* Differential-testing worker for Project Ferris.

   Matrices come from the ARCHIVED, UNMODIFIED pilot oracle (oracle.js), which executes the
   ORIGINAL ip-1.js in a vm sandbox. The worker verifies the SHA-256 of ip-1.js before use.

   Coordinates come from an independent tracing harness that runs the same unmodified ip-1.js
   with an equivalent minimal canvas mock whose getImageData() records (px, py) and whose
   beginPath() (called once per ray, after the radius loop) marks the end of each ray. The
   harness's matrix is also returned so the caller can confirm the instrumentation did not
   perturb the computation (it must equal the oracle matrix bit for bit).

   Protocol (stdin): repeated frames
     u32 LE header length, header JSON {id, width, height, center:[x,y]|null, angle_step, trace},
     u32 LE pixel length, pixel bytes (RGB, row-major).
   Protocol (stdout): one JSON line per frame:
     {id, center, rows, matrix: base64 float64 LE (oracle),
      trace: {ray_lengths:[...], coords: base64 int32 LE (x0,y0,x1,y1,...),
              matrix: base64 float64 LE (tracing harness)} | null,
      js_ns}
*/
'use strict';
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const {createHash} = require('node:crypto');

const DIR = __dirname;
const CANONICAL = '110732b48bc6e8b2d5167eec30f7277245913bed98b9baaa3c65c0c22aab44d8';
const source = fs.readFileSync(path.join(DIR, 'ip-1.js'), 'utf8');
if (createHash('sha256').update(fs.readFileSync(path.join(DIR, 'ip-1.js'))).digest('hex') !== CANONICAL) {
  throw Error('Canonical CDCH source checksum mismatch');
}

// --- 1. archived oracle, extracted exactly as the archived native_worker.js does ---
// Only the checksum-pinned archived adapter is evaluated; no input data reaches new Function().
const ORACLE_SHA256 = '0d45b7d68e76970950c9abc3b5e7bff1800cd92f6959ab972f2ec070243edb72';
if (createHash('sha256').update(fs.readFileSync(path.join(DIR, 'oracle.js'))).digest('hex') !== ORACLE_SHA256) {
  throw Error('Archived oracle.js checksum mismatch');
}
let oracle = fs.readFileSync(path.join(DIR, 'oracle.js'), 'utf8');
oracle = oracle.replace("const input = JSON.parse(fs.readFileSync(0, 'utf8'));", '');
const end = oracle.indexOf('for (const item of input.samples)');
if (end < 0) throw Error('Incompatible archived oracle');
const oracleOne = new Function('require', '__dirname', oracle.slice(0, end) + '\nreturn one;')(require, DIR);

// --- 2. tracing harness: same mock semantics as oracle.js, plus recording hooks ---
function traced(item) {
  const {height: h, width: w, pixels, center, angle_step: step = 1} = item;
  const coords = [];
  const rayLengths = [];
  let current = 0;
  const canvases = {};
  const ctx = {
    drawImage() {}, clearRect() {}, arc() {}, fill() {},
    moveTo() {}, lineTo() {}, stroke() {}, fillRect() {}, fillText() {},
    beginPath() { rayLengths.push(current); current = 0; },
    createLinearGradient() { return {addColorStop() {}}; },
    getImageData(x, y) {
      coords.push(x, y); current += 1;
      const ind = (y * w + x) * 3;
      return {data: [pixels[ind], pixels[ind + 1], pixels[ind + 2], 255]};
    },
  };
  const makeCanvas = () => ({width: w, height: h, value: '', addEventListener() {},
    getContext() { return ctx; }, getBoundingClientRect() { return {left: 0, top: 0}; }});
  const dom = {getElementById(id) { return canvases[id] ?? (canvases[id] = makeCanvas()); },
    createElement() { return makeCanvas(); }};
  const sandbox = {document: dom, Image: class { constructor() { this.width = w; this.height = h; } },
    FileReader: class {}, Math, console: {log() {}}, alert() {}, __result: null};
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox, {filename: 'ip-1.js'});
  vm.runInContext('drawCumulativeHistogramColumns = function(h) { __result = h; };', sandbox);
  canvases.angleStep.value = String(step);
  const [x, y] = center === null || center === undefined ? [w / 2, h / 2] : center;
  sandbox.drawRaysAndGenerateHistograms(x, y, step);
  if (!sandbox.__result) throw Error('Original JS did not produce any histograms');
  return {histograms: sandbox.__result, coords, rayLengths};
}

const f64 = (rows) => Buffer.from(new Float64Array(rows.flat()).buffer).toString('base64');

// --- framed stdin reader ---
let buffer = Buffer.alloc(0);
let pending = null;
process.stdin.on('data', (chunk) => {
  buffer = Buffer.concat([buffer, chunk]);
  for (;;) {
    if (pending === null) {
      if (buffer.length < 4) return;
      const n = buffer.readUInt32LE(0);
      if (buffer.length < 4 + n + 4) return;
      const header = JSON.parse(buffer.subarray(4, 4 + n).toString('utf8'));
      const m = buffer.readUInt32LE(4 + n);
      pending = {header, offset: 8 + n, length: m};
    }
    if (buffer.length < pending.offset + pending.length) return;
    const pixels = Array.from(buffer.subarray(pending.offset, pending.offset + pending.length));
    buffer = buffer.subarray(pending.offset + pending.length);
    const header = pending.header;
    pending = null;
    handle(header, pixels);
  }
});

function handle(header, pixels) {
  try {
    const item = {height: header.height, width: header.width, pixels,
      center: header.center, angle_step: header.angle_step};
    const t0 = process.hrtime.bigint();
    const r = oracleOne(item);
    const t1 = process.hrtime.bigint();
    let trace = null;
    if (header.trace) {
      const t = traced(item);
      trace = {ray_lengths: t.rayLengths,
        coords: Buffer.from(new Int32Array(t.coords).buffer).toString('base64'),
        matrix: f64(t.histograms)};
    }
    process.stdout.write(JSON.stringify({id: header.id, center: r.center, rows: r.histograms.length,
      matrix: f64(r.histograms), trace, js_ns: Number(t1 - t0)}) + '\n');
  } catch (e) {
    process.stdout.write(JSON.stringify({id: header.id, error: String(e)}) + '\n');
  }
}
