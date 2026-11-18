/* Differential-testing worker for the "centre by intensity" feature of ip-1.js.

   Runs the ORIGINAL, UNMODIFIED ip-1.js (SHA-256 verified) in a vm sandbox with the same
   minimal canvas mock semantics as diff_worker.js, then calls its findIntensityCenter(min, max)
   and reports the centre the original code writes into the xCoord / yCoord inputs, or that it
   raised its "no pixels in range" alert.

   Protocol (stdin): repeated frames
     u32 LE header length, header JSON {id, width, height, min, max},
     u32 LE pixel length, pixel bytes (RGB, row-major).
   Protocol (stdout): one JSON line per frame: {id, x, y, alert} where x / y are numbers or null.
*/
'use strict';
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const {createHash} = require('node:crypto');

const DIR = __dirname;
const CANONICAL = '110732b48bc6e8b2d5167eec30f7277245913bed98b9baaa3c65c0c22aab44d8';
const sourceBytes = fs.readFileSync(path.join(DIR, 'ip-1.js'));
if (createHash('sha256').update(sourceBytes).digest('hex') !== CANONICAL) {
  throw Error('Canonical CDCH source checksum mismatch');
}
const source = sourceBytes.toString('utf8');

function centroid({width: w, height: h, min, max}, pixels) {
  const canvases = {};
  const ctx = {
    drawImage() {}, clearRect() {}, arc() {}, fill() {}, beginPath() {},
    moveTo() {}, lineTo() {}, stroke() {}, fillRect() {}, fillText() {},
    createLinearGradient() { return {addColorStop() {}}; },
    getImageData(x, y) {
      const ind = (y * w + x) * 3;
      return {data: [pixels[ind], pixels[ind + 1], pixels[ind + 2], 255]};
    },
  };
  const makeCanvas = () => ({width: w, height: h, value: '', addEventListener() {},
    getContext() { return ctx; }, getBoundingClientRect() { return {left: 0, top: 0}; }});
  const dom = {getElementById(id) { return canvases[id] ?? (canvases[id] = makeCanvas()); },
    createElement() { return makeCanvas(); }};
  let alerted = false;
  const sandbox = {document: dom, Image: class { constructor() { this.width = w; this.height = h; } },
    FileReader: class {}, Math, console: {log() {}}, alert() { alerted = true; }};
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox, {filename: 'ip-1.js'});
  // The original follows a successful search with a full CDCH run; skip only the drawing.
  vm.runInContext('drawCumulativeHistogramColumns = function() {};', sandbox);
  canvases.angleStep.value = '1';
  canvases.xCoord.value = 'unset';
  canvases.yCoord.value = 'unset';
  sandbox.findIntensityCenter(min, max);
  const x = canvases.xCoord.value, y = canvases.yCoord.value;
  return {x: typeof x === 'number' ? x : null, y: typeof y === 'number' ? y : null, alert: alerted};
}

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
    const pixels = buffer.subarray(pending.offset, pending.offset + pending.length);
    buffer = buffer.subarray(pending.offset + pending.length);
    const header = pending.header;
    pending = null;
    try {
      process.stdout.write(JSON.stringify({id: header.id, ...centroid(header, pixels)}) + '\n');
    } catch (e) {
      process.stdout.write(JSON.stringify({id: header.id, error: String(e)}) + '\n');
    }
  }
});
