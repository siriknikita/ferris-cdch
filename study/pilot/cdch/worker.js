/* Streaming I/O adapter around the archived oracle. Ray and CDF math are unchanged.
   Raw matrices use lossless float64 binary transport. */
const fs = require('fs'), vm = require('vm'), path = require('path');
let oracle = fs.readFileSync(path.join(__dirname,'oracle.js'),'utf8');
oracle = oracle.replace("const input = JSON.parse(fs.readFileSync(0, 'utf8'));",'');
oracle = oracle.substring(0, oracle.indexOf('for (const item of input.samples)'));
oracle += `
require('readline').createInterface({input:process.stdin}).on('line', line => {
 try {
 const item=JSON.parse(line); const r=one(item);
 const a=new Float64Array(r.histograms.flat());
 process.stdout.write(JSON.stringify({center:r.center,rows:r.histograms.length,binary:Buffer.from(a.buffer).toString('base64')})+'\\n');
 } catch(e) {process.stdout.write(JSON.stringify({error:String(e)})+'\\n');}
});`;
new Function('require','__dirname','process','Buffer',oracle)(require,__dirname,process,Buffer);

