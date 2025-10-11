const boardSize = 6;
const numColors = 6;
const maxScoreEstimate = 12;

const config = {
  learningRate: 0.12,
  hiddenUnits: 28,
  batchSize: 20,
  noise: 0.12,
  delayMs: 80,
  burstFactor: 3
};

class MatchNet {
  constructor(inputSize, hiddenUnits, learningRate) {
    this.inputSize = inputSize;
    this.baseHiddenUnits = hiddenUnits;
    this.learningRate = learningRate;
    this.rebuildHiddenStack();
    this.initWeights();
  }

  rebuildHiddenStack() {
    const first = Math.max(8, Math.round(this.baseHiddenUnits));
    const second = Math.max(6, Math.round(first * 0.75));
    const third = Math.max(4, Math.round(first * 0.55));
    this.hiddenStack = [first, second, third];
  }

  initWeights() {
    const layers = this.hiddenStack.length;
    this.weights = new Array(layers);
    this.biases = new Array(layers);

    let prevSize = this.inputSize;
    for (let layer = 0; layer < layers; layer++) {
      const units = this.hiddenStack[layer];
      const scale = Math.sqrt(6 / (prevSize + units));
      const weight = new Float32Array(units * prevSize);
      const bias = new Float32Array(units);
      for (let i = 0; i < weight.length; i++) {
        weight[i] = (Math.random() * 2 - 1) * scale;
      }
      this.weights[layer] = weight;
      this.biases[layer] = bias;
      prevSize = units;
    }

    const lastUnits = this.hiddenStack[this.hiddenStack.length - 1];
    const outScale = Math.sqrt(6 / (lastUnits + 1));
    this.outWeights = new Float32Array(lastUnits);
    for (let i = 0; i < lastUnits; i++) {
      this.outWeights[i] = (Math.random() * 2 - 1) * outScale;
    }
    this.outBias = 0;
  }

  setLearningRate(lr) {
    this.learningRate = lr;
  }

  setHiddenUnits(units) {
    this.baseHiddenUnits = units;
    this.rebuildHiddenStack();
    this.initWeights();
  }

  forward(features) {
    const caches = [];
    let input = features;
    for (let layer = 0; layer < this.hiddenStack.length; layer++) {
      const units = this.hiddenStack[layer];
      const prevSize = input.length;
      const weight = this.weights[layer];
      const bias = this.biases[layer];
      const activation = new Float32Array(units);
      for (let h = 0; h < units; h++) {
        let sum = bias[h];
        const offset = h * prevSize;
        for (let i = 0; i < prevSize; i++) {
          sum += weight[offset + i] * input[i];
        }
        activation[h] = Math.tanh(sum);
      }
      caches.push({ activation, input });
      input = activation;
    }

    let z = this.outBias;
    for (let i = 0; i < this.outWeights.length; i++) {
      z += this.outWeights[i] * input[i];
    }
    const clipped = Math.max(-10, Math.min(10, z));
    const output = 1 / (1 + Math.exp(-clipped));
    return { output, caches, finalActivation: input };
  }

  trainSample(features, target) {
    const { output, caches, finalActivation } = this.forward(features);
    const lr = this.learningRate;
    const error = output - target;
    const dOut = error * output * (1 - output);

    const downstream = new Float32Array(this.outWeights.length);
    for (let i = 0; i < this.outWeights.length; i++) {
      downstream[i] = dOut * this.outWeights[i];
    }

    for (let i = 0; i < this.outWeights.length; i++) {
      this.outWeights[i] -= lr * dOut * finalActivation[i];
    }
    this.outBias -= lr * dOut;

    let nextError = downstream;
    for (let layer = this.hiddenStack.length - 1; layer >= 0; layer--) {
      const { activation, input } = caches[layer];
      const prevInput = layer === 0 ? features : caches[layer - 1].activation;
      const prevSize = prevInput.length;
      const units = this.hiddenStack[layer];
      const weight = this.weights[layer];
      const bias = this.biases[layer];
      const propagated = new Float32Array(prevSize);

      for (let h = 0; h < units; h++) {
        const gradAct = 1 - activation[h] * activation[h];
        const delta = nextError[h] * gradAct;
        const offset = h * prevSize;
        for (let i = 0; i < prevSize; i++) {
          const weightVal = weight[offset + i];
          propagated[i] += delta * weightVal;
          weight[offset + i] -= lr * delta * prevInput[i];
        }
        bias[h] -= lr * delta;
      }

      nextError = propagated;
    }

    return error;
  }
}

const patternGenerators = [
  { key: 'chaos', name: 'Stochastic chaos', generate: patternChaos },
  { key: 'stripes', name: 'Chromatic stripes', generate: patternStripes },
  { key: 'clusters', name: 'Cluster bloom', generate: patternClusters },
  { key: 'diagonals', name: 'Diagonal weave', generate: patternDiagonals },
  { key: 'waves', name: 'Wave interference', generate: patternWaves },
  { key: 'rings', name: 'Radial rings', generate: patternRings },
  { key: 'checker', name: 'Checker storm', generate: patternChecker },
  { key: 'spiral', name: 'Spiral garden', generate: patternSpiral },
  { key: 'bands', name: 'Broken bands', generate: patternBands },
  { key: 'glyphs', name: 'Glyph mosaics', generate: patternGlyphs },
  { key: 'quadrants', name: 'Quadrant shards', generate: patternQuadrants },
  { key: 'cross', name: 'Axial cruciform', generate: patternCross },
  { key: 'constellation', name: 'Star constellations', generate: patternConstellation },
  { key: 'maze', name: 'Lattice mazes', generate: patternMaze },
  { key: 'petals', name: 'Radial petals', generate: patternPetals },
  { key: 'fracture', name: 'Fracture gradients', generate: patternFracture }
];

const patternOrder = patternGenerators.map(g => g.name);

const overlayModes = [
  {
    name: 'xor-fuse',
    apply(base, overlay, size, colors) {
      const out = base.slice();
      for (let i = 0; i < out.length; i++) {
        out[i] = (base[i] + overlay[i]) % colors;
      }
      return out;
    }
  },
  {
    name: 'mask-imprint',
    apply(base, overlay, size, colors) {
      const out = base.slice();
      const counts = new Array(colors).fill(0);
      for (let i = 0; i < overlay.length; i++) counts[overlay[i]]++;
      let pivot = 0;
      for (let c = 1; c < counts.length; c++) {
        if (counts[c] > counts[pivot]) pivot = c;
      }
      for (let i = 0; i < out.length; i++) {
        if (overlay[i] === pivot || Math.random() < 0.15) out[i] = overlay[i];
      }
      return out;
    }
  },
  {
    name: 'interleave-weave',
    apply(base, overlay, size) {
      const out = base.slice();
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          if (((x + y) & 1) === 0) {
            const idx = y * size + x;
            out[idx] = overlay[idx];
          }
        }
      }
      return out;
    }
  },
  {
    name: 'trail-carve',
    apply(base, overlay, size) {
      const out = base.slice();
      let idx = Math.floor(Math.random() * out.length);
      let color = overlay[idx];
      const steps = size * 3 + Math.floor(Math.random() * size);
      for (let step = 0; step < steps; step++) {
        out[idx] = color;
        const options = [];
        if (idx % size !== size - 1) options.push(idx + 1);
        if (idx % size !== 0) options.push(idx - 1);
        if (idx + size < out.length) options.push(idx + size);
        if (idx - size >= 0) options.push(idx - size);
        if (!options.length) break;
        idx = options[Math.floor(Math.random() * options.length)];
        color = overlay[idx];
      }
      return out;
    }
  }
];

let net = createNet();
let running = false;
let timer = null;
let patternCounts = Object.fromEntries(patternOrder.map(n => [n, 0]));
let stats = createStats();
let errorHistory = [];
let avgErrorHistory = [];
let scoreHistoryTeacher = [];
let scoreHistoryModel = [];

function createStats() {
  return {
    boards: 0,
    teacherSum: 0,
    modelSum: 0,
    maeSum: 0,
    weightUpdates: 0,
    transformSet: new Set()
  };
}

function createNet() {
  const inputSize = boardSize * boardSize * numColors + patternOrder.length + 8;
  return new MatchNet(inputSize, config.hiddenUnits, config.learningRate);
}

function resetAll() {
  net = createNet();
  stats = createStats();
  patternCounts = Object.fromEntries(patternOrder.map(n => [n, 0]));
  errorHistory = [];
  avgErrorHistory = [];
  scoreHistoryTeacher = [];
  scoreHistoryModel = [];
  postMessage({ type: 'reset', patternOrder, patternCounts });
}

function encodeBoard(cells, pattern) {
  const size = boardSize;
  const features = new Float32Array(size * size * numColors + patternOrder.length + 8);
  let ptr = 0;
  const colorCounts = new Array(numColors).fill(0);
  for (let i = 0; i < cells.length; i++) {
    const color = cells[i];
    colorCounts[color]++;
    for (let c = 0; c < numColors; c++) {
      features[ptr++] = color === c ? 1 : 0;
    }
  }
  patternOrder.forEach((name, idx) => {
    features[ptr + idx] = pattern.family === name ? 1 : 0;
  });
  ptr += patternOrder.length;

  const total = cells.length;
  let entropy = 0;
  for (let c = 0; c < numColors; c++) {
    if (colorCounts[c] === 0) continue;
    const p = colorCounts[c] / total;
    entropy -= p * Math.log(p);
  }
  const maxEntropy = Math.log(numColors);
  const matches = findMatches(cells, boardSize).size / total;

  features[ptr++] = pattern.noise;
  features[ptr++] = Math.min(1, pattern.transforms.length / 12);
  features[ptr++] = Math.min(1, (pattern.variantId ?? 0) / 16);
  features[ptr++] = pattern.overlay ? 1 : 0;
  features[ptr++] = Math.min(1, (pattern.layerDepth ?? 1) / 6);
  features[ptr++] = maxEntropy ? entropy / maxEntropy : 0;
  features[ptr++] = Math.min(1, matches * 4);
  features[ptr++] = Math.random();
  return features;
}

function patternChaos(size, colors) {
  const cells = new Array(size * size);
  for (let i = 0; i < cells.length; i++) {
    cells[i] = Math.floor(Math.random() * colors);
  }
  return { cells, variant: 'noise field', variantId: 0, layers: ['foundation: chaotic noise'] };
}

function patternStripes(size, colors) {
  const cells = new Array(size * size);
  const orientation = Math.floor(Math.random() * 4);
  const width = 1 + Math.floor(Math.random() * 3);
  const labels = ['horizontal', 'vertical', 'diagonal', 'anti-diagonal'];
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let band;
      if (orientation === 0) band = Math.floor(y / width);
      else if (orientation === 1) band = Math.floor(x / width);
      else if (orientation === 2) band = Math.floor((x + y) / width);
      else band = Math.floor((x - y + size) / width);
      cells[y * size + x] = Math.abs(band) % colors;
    }
  }
  return {
    cells,
    variant: `${labels[orientation]} stripes`,
    variantId: orientation + 1,
    layers: [`foundation: ${labels[orientation]} bands`]
  };
}

function patternClusters(size, colors) {
  const cells = new Array(size * size).fill(-1);
  const seeds = 4 + Math.floor(Math.random() * 4);
  const queue = [];
  for (let s = 0; s < seeds; s++) {
    const idx = Math.floor(Math.random() * cells.length);
    const color = Math.floor(Math.random() * colors);
    cells[idx] = color;
    queue.push(idx);
  }
  const dirs = [1, -1, size, -size];
  while (queue.length) {
    const idx = queue.shift();
    for (const d of dirs) {
      const next = idx + d;
      if (next < 0 || next >= cells.length) continue;
      const x = next % size;
      const y = Math.floor(next / size);
      const px = idx % size;
      const py = Math.floor(idx / size);
      if (Math.abs(px - x) + Math.abs(py - y) !== 1) continue;
      if (cells[next] !== -1) continue;
      if (Math.random() < 0.55) {
        cells[next] = cells[idx];
      } else {
        cells[next] = Math.floor(Math.random() * colors);
      }
      queue.push(next);
    }
  }
  for (let i = 0; i < cells.length; i++) {
    if (cells[i] === -1) cells[i] = Math.floor(Math.random() * colors);
  }
  return { cells, variant: 'organic clusters', variantId: 2, layers: ['foundation: seeded clusters'] };
}

function patternDiagonals(size, colors) {
  const cells = new Array(size * size);
  const paletteRange = Math.max(3, colors - 1);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const diag = (x + y) % paletteRange;
      cells[y * size + x] = diag % colors;
    }
  }
  return { cells, variant: 'diagonal ramp', variantId: 3, layers: ['foundation: diagonal ramp'] };
}

function patternWaves(size, colors) {
  const cells = new Array(size * size);
  const freq = 1 + Math.random() * 2.5;
  const amp = 0.5 + Math.random();
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const value = Math.sin((x / size) * Math.PI * freq) + Math.cos((y / size) * Math.PI * freq * 0.7);
      const norm = (value * amp + 2) / 4;
      cells[y * size + x] = Math.floor(norm * colors) % colors;
    }
  }
  return { cells, variant: 'wave field', variantId: 4, layers: ['foundation: interference waves'] };
}

function patternRings(size, colors) {
  const cells = new Array(size * size);
  const cx = (size - 1) / 2;
  const cy = (size - 1) / 2;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
      cells[y * size + x] = Math.floor(dist) % colors;
    }
  }
  return { cells, variant: 'rings', variantId: 5, layers: ['foundation: concentric rings'] };
}

function patternChecker(size, colors) {
  const cells = new Array(size * size);
  const freq = 1 + Math.floor(Math.random() * 3);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const v = ((x >> freq) ^ (y >> freq)) % colors;
      cells[y * size + x] = v;
    }
  }
  return { cells, variant: 'checker', variantId: 6, layers: ['foundation: checker matrix'] };
}

function patternSpiral(size, colors) {
  const cells = new Array(size * size);
  let minX = 0, minY = 0;
  let maxX = size - 1, maxY = size - 1;
  let color = 0;
  while (minX <= maxX && minY <= maxY) {
    for (let x = minX; x <= maxX; x++) cells[minY * size + x] = color % colors;
    minY++;
    for (let y = minY; y <= maxY; y++) cells[y * size + maxX] = color % colors;
    maxX--;
    if (minY <= maxY) {
      for (let x = maxX; x >= minX; x--) cells[maxY * size + x] = color % colors;
      maxY--;
    }
    if (minX <= maxX) {
      for (let y = maxY; y >= minY; y--) cells[y * size + minX] = color % colors;
      minX++;
    }
    color++;
  }
  return { cells, variant: 'spiral', variantId: 7, layers: ['foundation: spiral sweep'] };
}

function patternBands(size, colors) {
  const cells = new Array(size * size);
  const segments = 3 + Math.floor(Math.random() * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const band = Math.floor(((x / size) + (Math.sin(y / segments) * 0.5)) * segments);
      cells[y * size + x] = ((band % colors) + colors) % colors;
    }
  }
  return { cells, variant: 'bands', variantId: 8, layers: ['foundation: broken bands'] };
}

function patternGlyphs(size, colors) {
  const cells = new Array(size * size).fill(Math.floor(Math.random() * colors));
  const glyphCount = 4 + Math.floor(Math.random() * 4);
  for (let g = 0; g < glyphCount; g++) {
    const gx = Math.floor(Math.random() * size);
    const gy = Math.floor(Math.random() * size);
    const color = Math.floor(Math.random() * colors);
    const radius = 1 + Math.floor(Math.random() * 2);
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const x = gx + dx;
        const y = gy + dy;
        if (x < 0 || x >= size || y < 0 || y >= size) continue;
        if (Math.abs(dx) + Math.abs(dy) <= radius) {
          cells[y * size + x] = color;
        }
      }
    }
  }
  return { cells, variant: 'glyphs', variantId: 9, layers: ['foundation: glyph mosaics'] };
}

function patternQuadrants(size, colors) {
  const cells = new Array(size * size);
  const palette = Array.from({ length: colors }, (_, i) => i);
  for (let i = palette.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [palette[i], palette[j]] = [palette[j], palette[i]];
  }
  const quadColors = [palette[0], palette[1 % palette.length], palette[2 % palette.length], palette[3 % palette.length]];
  const jitter = 0.12 + Math.random() * 0.25;
  const diagonalBias = Math.random() < 0.35;
  const offset = Math.floor(Math.random() * size * 0.5);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let quadrant = (x < size / 2 ? 0 : 1) + (y < size / 2 ? 0 : 2);
      if (diagonalBias && ((x + y + offset) % 2 === 0)) quadrant = (quadrant + 1) % quadColors.length;
      let color = quadColors[quadrant % quadColors.length];
      if (Math.random() < jitter) {
        color = palette[Math.floor(Math.random() * palette.length)];
      }
      cells[y * size + x] = color;
    }
  }
  const variant = diagonalBias ? 'diagonal quadrants' : 'sharp quadrants';
  return { cells, variant, variantId: 10, layers: ['foundation: quadrant shards'] };
}

function patternCross(size, colors) {
  const background = Math.floor(Math.random() * colors);
  let crossColor = Math.floor(Math.random() * colors);
  if (crossColor === background) crossColor = (crossColor + 1) % colors;
  const cells = new Array(size * size).fill(background);
  const center = Math.floor(size / 2);
  const thickness = 1 + Math.floor(Math.random() * 2);
  const addDiagonals = Math.random() < 0.5;
  for (let y = 0; y < size; y++) {
    for (let dx = -thickness; dx <= thickness; dx++) {
      const x = Math.min(size - 1, Math.max(0, center + dx));
      cells[y * size + x] = crossColor;
    }
  }
  for (let x = 0; x < size; x++) {
    for (let dy = -thickness; dy <= thickness; dy++) {
      const y = Math.min(size - 1, Math.max(0, center + dy));
      cells[y * size + x] = crossColor;
    }
  }
  if (addDiagonals) {
    for (let i = 0; i < size; i++) {
      cells[i * size + i] = crossColor;
      cells[i * size + (size - i - 1)] = crossColor;
    }
  }
  const variant = addDiagonals ? 'cross with diagonals' : 'orthogonal cross';
  return { cells, variant, variantId: 11, layers: ['foundation: axial cross'] };
}

function patternConstellation(size, colors) {
  const background = Math.floor(Math.random() * colors);
  const cells = new Array(size * size).fill(background);
  const starCount = size * 2 + Math.floor(Math.random() * size * 3);
  const trails = Math.random() < 0.45;
  for (let s = 0; s < starCount; s++) {
    let idx = Math.floor(Math.random() * cells.length);
    const color = Math.floor(Math.random() * colors);
    cells[idx] = color;
    if (Math.random() < 0.35) {
      const neighbors = [idx + 1, idx - 1, idx + size, idx - size];
      neighbors.forEach(n => {
        if (n >= 0 && n < cells.length && Math.random() < 0.6) cells[n] = color;
      });
    }
    if (trails && Math.random() < 0.2) {
      const length = 2 + Math.floor(Math.random() * 4);
      let current = idx;
      for (let step = 0; step < length; step++) {
        const dir = [1, -1, size, -size][Math.floor(Math.random() * 4)];
        const next = current + dir;
        if (next < 0 || next >= cells.length) break;
        cells[next] = color;
        current = next;
      }
    }
  }
  const variant = trails ? 'clustered constellations' : 'sparse constellations';
  return { cells, variant, variantId: 12, layers: ['foundation: constellation dust'] };
}

function patternMaze(size, colors) {
  const cells = new Array(size * size);
  const baseA = Math.floor(Math.random() * colors);
  let baseB = Math.floor(Math.random() * colors);
  if (baseB === baseA) baseB = (baseA + 2) % colors;
  const accent = (baseA + 1) % colors;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const vertical = x % 2 === 0;
      const horizontal = y % 2 === 0;
      let color = (vertical ^ horizontal) ? baseA : baseB;
      if (Math.random() < 0.15) color = accent;
      cells[y * size + x] = color;
    }
  }
  let idx = Math.floor(Math.random() * cells.length);
  const steps = size * size * 0.6;
  for (let step = 0; step < steps; step++) {
    cells[idx] = accent;
    const dir = [1, -1, size, -size][Math.floor(Math.random() * 4)];
    const next = idx + dir;
    if (next < 0 || next >= cells.length) {
      idx = Math.floor(Math.random() * cells.length);
    } else {
      idx = next;
    }
  }
  return { cells, variant: 'woven maze', variantId: 13, layers: ['foundation: lattice maze'] };
}

function patternPetals(size, colors) {
  const cells = new Array(size * size);
  const cx = (size - 1) / 2;
  const cy = (size - 1) / 2;
  const petals = 4 + Math.floor(Math.random() * 4);
  const twist = 0.4 + Math.random() * 1.1;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const angle = Math.atan2(dy, dx);
      const radius = Math.sqrt(dx * dx + dy * dy);
      const value = Math.sin(angle * petals) + Math.cos(radius * twist);
      const norm = (value + 2) / 4;
      cells[y * size + x] = Math.floor(norm * colors) % colors;
    }
  }
  return { cells, variant: `${petals}-petal bloom`, variantId: 14, layers: ['foundation: radial petals'] };
}

function patternFracture(size, colors) {
  const cells = new Array(size * size);
  const slope = (Math.random() * 2 - 1) * 1.2;
  const bias = (Math.random() - 0.5) * size;
  const base = Math.floor(Math.random() * colors);
  const contrast = (base + 1 + Math.floor(Math.random() * (colors - 1))) % colors;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const plane = x - slope * y + bias;
      let color = plane > 0 ? base : contrast;
      if (Math.random() < 0.2) {
        color = (color + Math.floor(Math.abs(plane)) + colors) % colors;
      }
      cells[y * size + x] = color;
    }
  }
  return { cells, variant: 'fracture gradient', variantId: 15, layers: ['foundation: fracture gradients'] };
}

function rotateBoard(cells, size, times) {
  let current = cells.slice();
  for (let t = 0; t < times; t++) {
    const next = new Array(size * size);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const nx = size - y - 1;
        const ny = x;
        next[ny * size + nx] = current[y * size + x];
      }
    }
    current = next;
  }
  return current;
}

function mirrorBoard(cells, size, axis) {
  const out = new Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let nx = x, ny = y;
      if (axis === 'horizontal') ny = size - y - 1;
      else if (axis === 'vertical') nx = size - x - 1;
      else if (axis === 'diag') { nx = y; ny = x; }
      out[y * size + x] = cells[ny * size + nx];
    }
  }
  return out;
}

function shiftRows(cells, size) {
  const out = cells.slice();
  for (let y = 0; y < size; y++) {
    const offset = Math.floor(Math.random() * size);
    for (let x = 0; x < size; x++) {
      out[y * size + x] = cells[y * size + ((x + offset) % size)];
    }
  }
  return out;
}

function shiftColumns(cells, size) {
  const out = cells.slice();
  for (let x = 0; x < size; x++) {
    const offset = Math.floor(Math.random() * size);
    for (let y = 0; y < size; y++) {
      const srcY = (y + offset) % size;
      out[y * size + x] = cells[srcY * size + x];
    }
  }
  return out;
}

function shearBoard(cells, size) {
  const out = new Array(size * size);
  const shear = (Math.random() * 0.9 + 0.2) * (Math.random() < 0.5 ? 1 : -1);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const shifted = Math.round(x + (y - size / 2) * shear);
      const nx = ((shifted % size) + size) % size;
      out[y * size + x] = cells[y * size + nx];
    }
  }
  return out;
}

function fractureCarve(cells, size, colors) {
  const out = cells.slice();
  const fractures = 2 + Math.floor(Math.random() * 3);
  for (let f = 0; f < fractures; f++) {
    let idx = Math.floor(Math.random() * out.length);
    let color = Math.floor(Math.random() * colors);
    const steps = size + Math.floor(Math.random() * size * 2);
    for (let step = 0; step < steps; step++) {
      out[idx] = color;
      const options = [];
      if (idx % size !== size - 1) options.push(idx + 1);
      if (idx % size !== 0) options.push(idx - 1);
      if (idx + size < out.length) options.push(idx + size);
      if (idx - size >= 0) options.push(idx - size);
      if (!options.length) break;
      idx = options[Math.floor(Math.random() * options.length)];
      if (Math.random() < 0.3) color = Math.floor(Math.random() * colors);
    }
  }
  return out;
}

function smearBoard(cells, size) {
  const out = cells.slice();
  const radius = 1 + Math.floor(Math.random() * 2);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const nx = (x + Math.floor(Math.random() * (radius * 2 + 1)) - radius + size) % size;
      const ny = (y + Math.floor(Math.random() * (radius * 2 + 1)) - radius + size) % size;
      out[y * size + x] = cells[ny * size + nx];
    }
  }
  return out;
}

function jitterBoard(cells, size) {
  const out = cells.slice();
  const swaps = Math.floor(out.length * 0.3);
  for (let i = 0; i < swaps; i++) {
    const a = Math.floor(Math.random() * out.length);
    const neighbors = [];
    if (a % size !== size - 1) neighbors.push(a + 1);
    if (a % size !== 0) neighbors.push(a - 1);
    if (a + size < out.length) neighbors.push(a + size);
    if (a - size >= 0) neighbors.push(a - size);
    if (!neighbors.length) continue;
    const b = neighbors[Math.floor(Math.random() * neighbors.length)];
    const tmp = out[a];
    out[a] = out[b];
    out[b] = tmp;
  }
  return out;
}

function permuteColors(cells, colors) {
  const perm = Array.from({ length: colors }, (_, i) => i).sort(() => Math.random() - 0.5);
  return cells.map(v => perm[v]);
}

function applyNoise(cells, size, colors, intensity) {
  const out = cells.slice();
  let replaced = 0;
  for (let i = 0; i < out.length; i++) {
    if (Math.random() < intensity) {
      out[i] = Math.floor(Math.random() * colors);
      replaced++;
    }
  }
  return { cells: out, noiseApplied: replaced / out.length };
}

function generateBoard() {
  const generatorIndex = Math.floor(Math.random() * patternGenerators.length);
  const generator = patternGenerators[generatorIndex];
  const base = generator.generate(boardSize, numColors);
  let cells = base.cells.slice();
  const layers = Array.isArray(base.layers) && base.layers.length ? base.layers.slice() : [`foundation: ${generator.name}`];
  const transforms = [];
  let variantScore = base.variantId ?? 0;
  let variantLabel = base.variant || generator.name;
  let overlayName = null;

  if (Math.random() < 0.7) {
    let overlayGenIndex = Math.floor(Math.random() * patternGenerators.length);
    if (overlayGenIndex === generatorIndex && patternGenerators.length > 1) {
      overlayGenIndex = (generatorIndex + 1) % patternGenerators.length;
    }
    const overlayGen = patternGenerators[overlayGenIndex];
    const overlay = overlayGen.generate(boardSize, numColors);
    const modeIndex = Math.floor(Math.random() * overlayModes.length);
    const mode = overlayModes[modeIndex];
    cells = mode.apply(cells, overlay.cells, boardSize, numColors);
    overlayName = overlayGen.name;
    const overlayVariant = overlay.variant || overlayGen.name;
    layers.push(`overlay: ${overlayGen.name} via ${mode.name}`);
    variantScore += (overlay.variantId ?? 0) + modeIndex + 1;
    variantLabel = `${variantLabel} + ${overlayVariant}`;
  }

  if (Math.random() < 0.55) {
    cells = shiftRows(cells, boardSize);
    transforms.push('row-shift');
    variantScore += 0.5;
  }
  if (Math.random() < 0.5) {
    cells = shiftColumns(cells, boardSize);
    transforms.push('column-shift');
    variantScore += 0.5;
  }
  if (Math.random() < 0.4) {
    cells = shearBoard(cells, boardSize);
    transforms.push('shear');
    variantScore += 1;
  }
  if (Math.random() < 0.45) {
    cells = jitterBoard(cells, boardSize);
    transforms.push('jitter-swap');
    variantScore += 0.5;
  }

  if (Math.random() < 0.4) {
    cells = fractureCarve(cells, boardSize, numColors);
    layers.push('curriculum: fracture walk');
    variantScore += 1.5;
  }
  if (Math.random() < 0.3) {
    cells = smearBoard(cells, boardSize);
    layers.push('curriculum: smear drift');
    variantScore += 1;
  }

  const rotations = Math.floor(Math.random() * 4);
  if (rotations) {
    cells = rotateBoard(cells, boardSize, rotations);
    transforms.push(`rotate×${rotations}`);
    variantScore += rotations * 0.3;
  }
  if (Math.random() < 0.6) {
    const axis = Math.random() < 0.5 ? 'horizontal' : 'vertical';
    cells = mirrorBoard(cells, boardSize, axis);
    transforms.push(`${axis} flip`);
    variantScore += 0.5;
  }
  if (Math.random() < 0.35) {
    cells = mirrorBoard(cells, boardSize, 'diag');
    transforms.push('transpose');
    variantScore += 0.4;
  }
  if (Math.random() < 0.65) {
    cells = permuteColors(cells, numColors);
    transforms.push('recolour');
    variantScore += 0.4;
  }

  const noiseIntensity = Math.min(0.95, config.noise + Math.random() * 0.12);
  const noiseResult = applyNoise(cells, boardSize, numColors, noiseIntensity);
  cells = noiseResult.cells;
  const noiseApplied = noiseResult.noiseApplied;
  if (noiseApplied > 0.02) {
    layers.push(`noise: ${(noiseApplied * 100).toFixed(1)}%`);
  }
  variantScore += noiseApplied * 8;

  return {
    cells,
    name: overlayName ? `${generator.name} × ${overlayName}` : generator.name,
    family: generator.name,
    key: generator.key,
    transforms,
    layers,
    layerDepth: layers.length,
    overlay: overlayName,
    variantId: variantScore,
    variant: variantLabel,
    noise: noiseApplied
  };
}

function findMatches(cells, size) {
  const matches = new Set();
  // Horizontal
  for (let y = 0; y < size; y++) {
    let runColor = cells[y * size];
    let runStart = 0;
    for (let x = 1; x <= size; x++) {
      const idx = y * size + x;
      const color = x < size ? cells[idx] : null;
      if (color === runColor) continue;
      const runLength = x - runStart;
      if (runColor != null && runLength >= 3) {
        for (let k = runStart; k < x; k++) matches.add(y * size + k);
      }
      runColor = color;
      runStart = x;
    }
  }
  // Vertical
  for (let x = 0; x < size; x++) {
    let runColor = cells[x];
    let runStart = 0;
    for (let y = 1; y <= size; y++) {
      const idx = y * size + x;
      const color = y < size ? cells[idx] : null;
      if (color === runColor) continue;
      const runLength = y - runStart;
      if (runColor != null && runLength >= 3) {
        for (let k = runStart; k < y; k++) matches.add(k * size + x);
      }
      runColor = color;
      runStart = y;
    }
  }
  return matches;
}

function scoreSwap(cells, size, a, b) {
  if (cells[a] === cells[b]) return 0;
  const copy = cells.slice();
  const temp = copy[a];
  copy[a] = copy[b];
  copy[b] = temp;
  const matches = findMatches(copy, size);
  if (!matches.size) return 0;
  let score = matches.size;
  matches.forEach(idx => {
    const color = copy[idx];
    const neighbors = [idx + 1, idx - 1, idx + size, idx - size];
    let bonus = 0;
    neighbors.forEach(nb => {
      if (nb >= 0 && nb < copy.length && copy[nb] === color) bonus += 0.25;
    });
    score += bonus;
  });
  return score;
}

function bestSwap(cells, size) {
  let best = { score: 0, from: null, to: null };
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const idx = y * size + x;
      if (x + 1 < size) {
        const score = scoreSwap(cells, size, idx, idx + 1);
        if (score > best.score) best = { score, from: idx, to: idx + 1 };
      }
      if (y + 1 < size) {
        const score = scoreSwap(cells, size, idx, idx + size);
        if (score > best.score) best = { score, from: idx, to: idx + size };
      }
    }
  }
  return best;
}

function createSample() {
  const pattern = generateBoard();
  const best = bestSwap(pattern.cells, boardSize);
  const features = encodeBoard(pattern.cells, pattern);
  const prediction = net.forward(features).output;
  const targetScore = best.score;
  const target = Math.min(1, targetScore / maxScoreEstimate);
  const absError = Math.abs(prediction - target);
  const modelScore = prediction * maxScoreEstimate;
  return {
    board: { size: boardSize, cells: pattern.cells },
    pattern,
    best,
    target,
    targetScore,
    prediction,
    modelScore,
    absError,
    features
  };
}

function step() {
  const totalSamples = Math.max(1, Math.floor(config.batchSize) * Math.max(1, Math.floor(config.burstFactor)));
  let batchError = 0;
  let teacherSum = 0;
  let modelSum = 0;
  let bestSample = null;
  const localPatternCounts = Object.fromEntries(patternOrder.map(n => [n, 0]));

  for (let i = 0; i < totalSamples; i++) {
    const sample = createSample();
    net.trainSample(sample.features, sample.target);
    batchError += sample.absError;
    teacherSum += sample.targetScore;
    modelSum += sample.modelScore;
    const family = sample.pattern.family;
    patternCounts[family] = (patternCounts[family] || 0) + 1;
    localPatternCounts[family] = (localPatternCounts[family] || 0) + 1;
    stats.boards++;
    stats.teacherSum += sample.targetScore;
    stats.modelSum += sample.modelScore;
    stats.maeSum += sample.absError;
    stats.weightUpdates++;
    sample.pattern.transforms.forEach(t => stats.transformSet.add(t));
    (sample.pattern.layers || []).forEach(l => stats.transformSet.add(l));
    if (
      !bestSample ||
      sample.targetScore > bestSample.targetScore ||
      (sample.targetScore === bestSample.targetScore && sample.absError < bestSample.absError)
    ) {
      bestSample = sample;
    }
  }

  const avgBatchError = batchError / totalSamples;
  errorHistory.push(avgBatchError);
  if (errorHistory.length > 360) errorHistory.shift();
  const runningError = stats.weightUpdates ? stats.maeSum / stats.weightUpdates : 0;
  avgErrorHistory.push(runningError);
  if (avgErrorHistory.length > 360) avgErrorHistory.shift();

  const teacherAvg = teacherSum / totalSamples;
  const modelAvg = modelSum / totalSamples;
  scoreHistoryTeacher.push(teacherAvg);
  scoreHistoryModel.push(modelAvg);
  if (scoreHistoryTeacher.length > 360) {
    scoreHistoryTeacher.shift();
    scoreHistoryModel.shift();
  }

  const diversityFrame = patternOrder.map(name => (localPatternCounts[name] || 0) / totalSamples);

  const confidence = bestSample ? 1 - Math.min(1, bestSample.absError * 2) : 0;

  postMessage({
    type: 'snapshot',
    board: bestSample.board,
    highlight: bestSample.best.from != null ? [bestSample.best.from, bestSample.best.to] : null,
    pattern: {
      name: bestSample.pattern.name,
      family: bestSample.pattern.family,
      variant: bestSample.pattern.variant,
      transforms: bestSample.pattern.transforms,
      layers: bestSample.pattern.layers,
      overlay: bestSample.pattern.overlay,
      layerDepth: bestSample.pattern.layerDepth,
      noise: bestSample.pattern.noise
    },
    bestMove: bestSample.best,
    targetScore: bestSample.targetScore,
    modelScore: bestSample.modelScore,
    batchError: avgBatchError,
    runningError,
    confidence,
    stats: {
      boards: stats.boards,
      teacherAvg: stats.boards ? stats.teacherSum / stats.boards : 0,
      modelAvg: stats.boards ? stats.modelSum / stats.boards : 0,
      mae: runningError,
      transformCount: stats.transformSet.size,
      weightUpdates: stats.weightUpdates
    },
    patternCounts,
    patternOrder,
    errorHistory,
    avgErrorHistory,
    scoreHistoryTeacher,
    scoreHistoryModel,
    diversityFrame
  });
}

function schedule() {
  if (!running) return;
  clearTimeout(timer);
  timer = setTimeout(() => {
    step();
    schedule();
  }, Math.max(0, config.delayMs));
}

onmessage = ev => {
  const msg = ev.data || {};
  if (msg.type === 'start') {
    if (!running) {
      running = true;
      postMessage({ type: 'status', running });
      schedule();
    }
  } else if (msg.type === 'pause') {
    running = false;
    clearTimeout(timer);
    postMessage({ type: 'status', running });
  } else if (msg.type === 'reset') {
    running = false;
    clearTimeout(timer);
    resetAll();
    postMessage({ type: 'status', running });
  } else if (msg.type === 'configure') {
    const cfg = msg.config || {};
    if (cfg.learningRate != null) {
      config.learningRate = cfg.learningRate;
      net.setLearningRate(config.learningRate);
    }
    if (cfg.hiddenUnits != null && cfg.hiddenUnits !== config.hiddenUnits) {
      config.hiddenUnits = Math.max(4, Math.round(cfg.hiddenUnits));
      net.setHiddenUnits(config.hiddenUnits);
    }
    if (cfg.batchSize != null) config.batchSize = Math.max(1, cfg.batchSize);
    if (cfg.noise != null) config.noise = Math.max(0, Math.min(1, cfg.noise));
    if (cfg.delayMs != null) config.delayMs = Math.max(0, cfg.delayMs);
    if (cfg.burstFactor != null) config.burstFactor = Math.max(1, cfg.burstFactor);
  }
};

resetAll();
