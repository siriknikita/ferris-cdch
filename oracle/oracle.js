/* Executes ORIGINAL unmodified ip-1.js, with minimal simulated DOM/canvas API.
   No reimplementation of its ray-generation math is used for parity tests. */
const fs = require('fs');
const vm = require('vm');
const path = require('path');
const source = fs.readFileSync(path.join(__dirname, 'ip-1.js'), 'utf8');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
let results = [];
function one(item) {
  const {height:h, width:w, pixels, center, angle_step:step=1, intensity_range} = item;
  const canvases = {};
  const ctx = {
    drawImage() {}, clearRect() {}, beginPath() {}, arc() {}, fill() {},
    moveTo() {}, lineTo() {}, stroke() {}, fillRect() {}, fillText() {},
    createLinearGradient() {return {addColorStop(){}}},
    getImageData(x, y) {
      const ind = (y*w+x)*3;
      return {data: [pixels[ind],pixels[ind+1],pixels[ind+2],255]};
    }
  };
  const makeCanvas = (id) => ({width:w,height:h, value:'',
    addEventListener(){}, getContext(){return ctx;},
    getBoundingClientRect(){return {left:0,top:0}}});
  const dom = {
    getElementById(id) {return canvases[id]??(canvases[id]=makeCanvas(id));},
    createElement() {return makeCanvas('hidden')}
  };
  const sandbox = {document: dom, Image: class {constructor(){this.width=w;this.height=h;}},
    FileReader: class {}, Math, console:{log(){}}, alert() {}, __result:null};
  vm.createContext(sandbox);
  vm.runInContext(source,sandbox,{filename:'ip-1.js'});
  vm.runInContext('drawCumulativeHistogramColumns = function(h) { __result = h; };', sandbox);
  canvases.angleStep.value = String(step);
  let x,y;
  if (intensity_range) {
    vm.runInContext(`findIntensityCenter(${Number(intensity_range[0])},${Number(intensity_range[1])})`,sandbox);
    x=Number(canvases.xCoord.value); y=Number(canvases.yCoord.value);
  } else {
    [x,y] = center === null || center === undefined ? [w/2,h/2] : center;
    sandbox.drawRaysAndGenerateHistograms(x,y,step);
  }
  if (!sandbox.__result) throw Error('Original JS did not produce any histograms');
  return {center:[x,y], histograms: sandbox.__result};
}
for (const item of input.samples) results.push(one(item));
process.stdout.write(JSON.stringify({results}));

