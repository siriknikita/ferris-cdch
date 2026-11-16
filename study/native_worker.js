/* Calls the original, checksum-verified Muzyka implementation via the pilot oracle.
   Times exclude Node parsing and the output transport. */
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {createInterface} = require('node:readline');
const scriptDir = process.argv[2];
const digest = createHash('sha256').update(fs.readFileSync(path.join(scriptDir, 'ip-1.js'))).digest('hex');
if (digest !== process.argv[3]) throw Error('Canonical CDCH source checksum mismatch');

let oracle = fs.readFileSync(path.join(scriptDir, 'oracle.js'), 'utf8');
oracle = oracle.replace("const input = JSON.parse(fs.readFileSync(0, 'utf8'));", '');
const end = oracle.indexOf('for (const item of input.samples)');
if (end < 0) throw Error('Incompatible archived oracle');
const one = new Function('require', '__dirname', oracle.slice(0, end) + '\nreturn one;')(require, scriptDir);
const ns = () => process.hrtime.bigint();

createInterface({input: process.stdin}).on('line', line => {
  try {
    const start = ns();
    const input = JSON.parse(line);
    const loaded = ns();
    const answer = one(input);
    const executed = ns();
    const values = new Float64Array(answer.histograms.flat());
    const binary = Buffer.from(values.buffer).toString('base64');
    const packed = ns();
    process.stdout.write(JSON.stringify({
      center: answer.center, rows: answer.histograms.length, binary,
      parse_ns: Number(loaded - start), js_ns: Number(executed - loaded), pack_ns: Number(packed - executed),
    }) + '\n');
  } catch (error) {
    process.stdout.write(JSON.stringify({error: String(error)}) + '\n');
  }
});
