const imageUpload = document.getElementById("imageUpload");
const imageCanvas = document.getElementById("imageCanvas");
const rayCanvas = document.getElementById("rayCanvas");
const xCoordInput = document.getElementById("xCoord");
const yCoordInput = document.getElementById("yCoord");
const angleStepInput = document.getElementById("angleStep");
const drawRaysButton = document.getElementById("drawRays");
const centerButton = document.getElementById("centerButton");
const intensityColumnCanvas = document.getElementById("intensityColumnCanvas");
const imageInfo = document.getElementById("imageInfo");
const downloadButton = document.getElementById("download");
const minIntensityInput = document.getElementById("minIntensity");
const maxIntensityInput = document.getElementById("maxIntensity");
const intensityCenterButton = document.getElementById("intensityCenterButton");
const hiddenCanvas = document.createElement("canvas");
let intensityColumnContext;
let context;
let rayContext;
let img = new Image();
const MAX_WIDTH = 500;
const MAX_HEIGHT = 500;
const R_COEF = 0.2125;
const G_COEF = 0.7154;
const B_COEF = 0.0721;

centerButton.addEventListener("click", () => {
  const centerX = imageCanvas.width / 2;
  const centerY = imageCanvas.height / 2;

  xCoordInput.value = centerX;
  yCoordInput.value = centerY;

  drawMarker(centerX, centerY);
  const step = parseInt(angleStepInput.value, 10);

  if (!isNaN(centerX) && !isNaN(centerY) && step > 0 && step <= 360) {
    drawRaysAndGenerateHistograms(centerX, centerY, step);
  } else {
    alert("Будь ласка введіть коректне значення між 1 - 360");
  }
});

imageUpload.addEventListener("change", (event) => {
  context = imageCanvas.getContext("2d");
  const file = event.target.files[0];
  const reader = new FileReader();
  reader.onload = function(e) {
    img.src = e.target.result;
    img.onload = function() {
      let scale = Math.min(MAX_WIDTH / img.width, MAX_HEIGHT / img.height, 1);
      let displayWidth = img.width * scale;
      let displayHeight = img.height * scale;

      imageCanvas.width = displayWidth;
      imageCanvas.height = displayHeight;
      rayCanvas.width = displayWidth;
      rayCanvas.height = displayHeight;

      context.drawImage(img, 0, 0, displayWidth, displayHeight);
      drawMarker(parseInt(xCoordInput.value, 10), parseInt(yCoordInput.value, 10));
      imageInfo.textContent = `Ширина: ${img.naturalWidth}px, Висота: ${img.naturalHeight}px`;
    };
  };
  reader.readAsDataURL(file);
});

imageCanvas.addEventListener("click", (event) => {
  const rect = imageCanvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  xCoordInput.value = x;
  yCoordInput.value = y;
  drawMarker(x, y);
});

xCoordInput.addEventListener("input", updateMarkerFromInput);
yCoordInput.addEventListener("input", updateMarkerFromInput);

function updateMarkerFromInput() {
  const x = parseInt(xCoordInput.value, 10);
  const y = parseInt(yCoordInput.value, 10);
  if (!isNaN(x) && !isNaN(y)) {
    drawMarker(x, y);
  }
}

function drawMarker(x, y) {
  context = imageCanvas.getContext("2d");
  context.drawImage(img, 0, 0, imageCanvas.width, imageCanvas.height);
  context.fillStyle = "red";
  context.beginPath();
  context.arc(x, y, 5, 0, 2 * Math.PI);
  context.fill();
}

function drawRaysAndGenerateHistograms(x, y, step) {
  rayContext = rayCanvas.getContext("2d");
  context = imageCanvas.getContext("2d");
  rayContext.clearRect(0, 0, rayCanvas.width, rayCanvas.height);

  hiddenCanvas.width = imageCanvas.width;
  hiddenCanvas.height = imageCanvas.height;
  const hiddenContext = hiddenCanvas.getContext("2d");
  hiddenContext.drawImage(img, 0, 0, imageCanvas.width, imageCanvas.height);

  rayContext.drawImage(img, 0, 0, rayCanvas.width, rayCanvas.height);
  rayContext.strokeStyle = "red";
  rayContext.lineWidth = 1;

  const rayHistograms = [];
  for (let angle = 0; angle < 360; angle += step) {
    const adjustedAngle = angle - 90;
    const radians = adjustedAngle * (Math.PI / 180);
    const pixels = [];
    let rayHistogram = new Array(256).fill(0);

    for (let r = 0; r < rayCanvas.width; r++) {
      let px = Math.round(x + r * Math.cos(radians));
      let py = Math.round(y + r * Math.sin(radians));
      if (px < 0 || px >= rayCanvas.width || py < 0 || py >= rayCanvas.height) break;

      let pixelData = hiddenContext.getImageData(px, py, 1, 1).data;
      let intensity = Math.round(pixelData[0] * R_COEF + pixelData[1] * G_COEF + pixelData[2] * B_COEF);

      rayHistogram[intensity]++;
      pixels.push(intensity);
    }

    const cumulativeHist = calculateCumulativeHistogram(rayHistogram);

    rayContext.beginPath();
    rayContext.moveTo(x, y);
    rayContext.lineTo(x + rayCanvas.width * Math.cos(radians), y + rayCanvas.height * Math.sin(radians));
    rayContext.stroke();

    rayHistograms.push(cumulativeHist);
  }

  drawCumulativeHistogramColumns(rayHistograms);
}

function calculateCumulativeHistogram(hist) {
  let cumulative = new Array(256).fill(0);
  cumulative[0] = hist[0];
  for (let i = 1; i < hist.length; i++) {
    cumulative[i] = cumulative[i - 1] + hist[i];
  }
  let max = cumulative[255];
  return cumulative.map(value => value / max);
}

function drawCumulativeHistogramColumns(histograms) {
  const intensityColumnContext = intensityColumnCanvas.getContext("2d");
  const leftMargin = 40;
  const canvasWidth = imageCanvas.width - leftMargin;
  const canvasHeight = intensityColumnCanvas.height - 20;

  const columnWidth = 1;

  intensityColumnContext.clearRect(0, 0, intensityColumnCanvas.width, intensityColumnCanvas.height);
  intensityColumnContext.font = "14px Arial";
  intensityColumnContext.fillStyle = "black";

  hiddenCanvas.width = histograms.length;
  hiddenCanvas.height = canvasHeight;
  const hiddenContext = hiddenCanvas.getContext("2d");

  for (let i = 0; i <= 255; i += 50) {
    const y = (canvasHeight - 1) - (i / 255) * (canvasHeight - 1);
    intensityColumnContext.fillText(i, 5, y + 10);
  }

  let step = histograms.length;
  let multiplier = 1;

  if (histograms.length === 90 ) {
    step = 45;
    multiplier = 4;

  } else if (histograms.length === 120 ) {
    step = 60;
    multiplier = 3;

  } else if (histograms.length === 180 ) {
    step = 60;
    multiplier = 2;

  } else if (histograms.length === 360 ) {
    step = 60;
  }

  for (let i = 0; i <= histograms.length; i += step) {
    const x = leftMargin + i;

    if (i === histograms.length) {
      intensityColumnContext.fillText(360, x, canvasHeight + 10);
      break;
    }
    if (i === 0) {
      intensityColumnContext.fillText(0, x, canvasHeight + 10);
    } else {
      intensityColumnContext.fillText(i * multiplier, x, canvasHeight + 10);
    }
  }

  histograms.forEach((histogram, index) => {
    const gradient = intensityColumnContext.createLinearGradient(0, 0, 0, canvasHeight);

    for (let y = 0; y <= 255; y++) {
      const intensityLeft = index > 0 ? histograms[index - 1][y] : histogram[y];
      const intensityRight = index < histograms.length - 1 ? histograms[index + 1][y] : histogram[y];
      const blendedIntensity = Math.round(((histogram[y] + intensityLeft + intensityRight) / 3) * 255);
      const color = `rgb(${blendedIntensity}, ${blendedIntensity}, ${blendedIntensity})`;

      gradient.addColorStop(1 - y / 255, color);
    }

    hiddenContext.fillStyle = gradient;
    hiddenContext.fillRect(index * columnWidth, 0, columnWidth + 1, canvasHeight);

    intensityColumnContext.fillStyle = gradient;
    intensityColumnContext.fillRect(
      leftMargin + index * columnWidth,
      -8,
      columnWidth + 1,
      canvasHeight
    );
  });
}

drawRaysButton.addEventListener("click", () => {
  const x = parseInt(xCoordInput.value, 10);
  const y = parseInt(yCoordInput.value, 10);
  const step = parseInt(angleStepInput.value, 10);

  if (!isNaN(x) && !isNaN(y) && step > 0 && step <= 360) {
    drawRaysAndGenerateHistograms(x, y, step);
  } else {
    alert("Будь ласка введіть коректне значення між 1 - 360");
  }
});

downloadButton.addEventListener("click", () => {
  const imageData = hiddenCanvas.toDataURL("image/png");
  const link = document.createElement("a");
  link.href = imageData;
  link.download = "histogram_image.png";
  link.click();
});

intensityCenterButton.addEventListener("click", () => {
  const minIntensity = parseInt(minIntensityInput.value, 10);
  const maxIntensity = parseInt(maxIntensityInput.value, 10);
  if (isNaN(minIntensity) || isNaN(maxIntensity) || minIntensity > maxIntensity) {
    alert("Будь ласка, введіть коректний діапазон інтенсивності.");
    return;
  }
  findIntensityCenter(minIntensity, maxIntensity);
});

function findIntensityCenter(minIntensity, maxIntensity) {
  context = imageCanvas.getContext("2d");
  hiddenCanvas.width = imageCanvas.width;
  hiddenCanvas.height = imageCanvas.height;
  const hiddenContext = hiddenCanvas.getContext("2d");
  hiddenContext.drawImage(img, 0, 0, imageCanvas.width, imageCanvas.height);

  let sumX = 0, sumY = 0, count = 0;

  console.log('minIntensity', minIntensity)
  console.log('maxIntensity', maxIntensity)
  for (let y = 0; y < imageCanvas.height; y++) {
    for (let x = 0; x < imageCanvas.width; x++) {
      let pixelData = hiddenContext.getImageData(x, y, 1, 1).data;
      let intensity = Math.round(pixelData[0] * R_COEF + pixelData[1] * G_COEF + pixelData[2] * B_COEF);
      console.log('intensity', intensity)
      if (intensity >= minIntensity && intensity <= maxIntensity) {
        sumX += x;
        sumY += y;
        count++;
      }
    }
  }

  console.log('count', count)
  if (count > 0) {
    const centerX = Math.round(sumX / count);
    const centerY = Math.round(sumY / count);
    xCoordInput.value = centerX;
    yCoordInput.value = centerY;
    drawMarker(centerX, centerY);

    const step = parseInt(angleStepInput.value, 10);

    if (!isNaN(centerX) && !isNaN(centerY) && step > 0 && step <= 360) {
      drawRaysAndGenerateHistograms(centerX, centerY, step);
    } else {
      alert("Будь ласка введіть коректне значення між 1 - 360");
    }
  } else {
    alert("Немає пікселів у заданому діапазоні.");
  }
}