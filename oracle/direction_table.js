/* Emit the exact IEEE-754 bit patterns V8 produces for the direction expressions in ip-1.js:
     const adjustedAngle = angle - 90;
     const radians = adjustedAngle * (Math.PI / 180);
     Math.cos(radians), Math.sin(radians)
   for every integer angle 0..359. Output: JSON on stdout. */
const bits = (x) => { const b = Buffer.alloc(8); b.writeDoubleLE(x); return b.readBigUInt64LE().toString(16).padStart(16, '0'); };
const rows = [];
for (let angle = 0; angle < 360; angle += 1) {
  const adjustedAngle = angle - 90;
  const radians = adjustedAngle * (Math.PI / 180);
  rows.push({angle, radians: bits(radians), cos: bits(Math.cos(radians)), sin: bits(Math.sin(radians))});
}
process.stdout.write(JSON.stringify({node: process.version, v8: process.versions.v8, platform: process.platform, arch: process.arch, rows}) + '\n');
