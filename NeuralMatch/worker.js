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
    this.hiddenUnits = hiddenUnits;
    this.learningRate = learningRate;
    this.initWeights();
  }

  initWeights() {
    const h = this.hiddenUnits;
    const d = this.inputSize;
    const scale1 = Math.sqrt(6 / (d + h));
    this.w1 = new Float32Array(h * d);
    this.b1 = new Float32Array(h);
    for (let i = 0; i < this.w1.length; i++) {
      this.w1[i] = (Math.random() * 2 - 1) * scale1;
    }
    this.b1.fill(0);
    this.w2 = new Float32Array(h);
    const scale2 = Math.sqrt(6 / (h + 1));
    for (let i = 0; i < h; i++) {
      this.w2[i] = (Math.random() * 2 - 1) * scale2;
    }
    this.b2 = 0;
  }

  setLearningRate(lr) {
    this.learningRate = lr;
  }

  setHiddenUnits(units) {
    this.hiddenUnits = units;
    this.initWeights();
  }

  forward(features) {
    const hidden = new Float32Array(this.hiddenUnits);
    for (let h = 0; h < this.hiddenUnits; h++) {
      let sum = this.b1[h];
      const offset = h * this.inputSize;
      for (let i = 0; i < this.inputSize; i++) {
        sum += this.w1[offset + i] * features[i];
      }
      hidden[h] = Math.tanh(sum);
    }
    let z2 = this.b2;
    for (let h = 0; h < this.hiddenUnits; h++) {
      z2 += this.w2[h] * hidden[h];
    }
    const clipped = Math.max(-8, Math.min(8, z2));
    const output = 1 / (1 + Math.exp(-clipped));
    return { hidden, output };
  }

  trainSample(features, target) {
    const { hidden, output } = this.forward(features);
    const lr = this.learningRate;
    const error = output - target;
    const dOut = error * output * (1 - output);

    const gradHidden = new Float32Array(this.hiddenUnits);
    for (let h = 0; h < this.hiddenUnits; h++) {
      gradHidden[h] = dOut * this.w2[h] * (1 - hidden[h] * hidden[h]);
    }

    for (let h = 0; h < this.hiddenUnits; h++) {
      this.w2[h] -= lr * dOut * hidden[h];
    }
    this.b2 -= lr * dOut;

    for (let h = 0; h < this.hiddenUnits; h++) {
      const offset = h * this.inputSize;
      const grad = gradHidden[h];
      for (let i = 0; i < this.inputSize; i++) {
        this.w1[offset + i] -= lr * grad * features[i];
      }
      this.b1[h] -= lr * grad;
    }
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
  { key: 'glyphs', name: 'Glyph mosaics', generate: patternGlyphs }
];

const patternOrder = patternGenerators.map(g => g.name);

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
  const inputSize = boardSize * boardSize * numColors + patternOrder.length + 4;
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
  const features = new Float32Array(size * size * numColors + patternOrder.length + 4);
  let ptr = 0;
  for (let i = 0; i < cells.length; i++) {
    const color = cells[i];
    for (let c = 0; c < numColors; c++) {
      features[ptr++] = color === c ? 1 : 0;
    }
  }
  patternOrder.forEach((name, idx) => {
    features[ptr + idx] = pattern.name === name ? 1 : 0;
  });
  ptr += patternOrder.length;
  features[ptr++] = pattern.noise;
  features[ptr++] = pattern.transforms.length / 8;
  features[ptr++] = Math.min(1, (pattern.variantId ?? 0) / 6);
  features[ptr++] = Math.random();
  return features;
}

function patternChaos(size, colors) {
  const cells = new Array(size * size);
  for (let i = 0; i < cells.length; i++) {
    cells[i] = Math.floor(Math.random() * colors);
  }
  return { cells, variant: 'noise field', variantId: 0 };
}

function patternStripes(size, colors) {
  const cells = new Array(size * size);
  const orientation = Math.floor(Math.random() * 4);
  const width = 1 + Math.floor(Math.random() * 3);
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
  return { cells, variant: 'striped', variantId: orientation + 1 };
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
  return { cells, variant: 'organic clusters', variantId: 2 };
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
  return { cells, variant: 'diagonal ramp', variantId: 3 };
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
  return { cells, variant: 'wave field', variantId: 4 };
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
  return { cells, variant: 'rings', variantId: 5 };
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
  return { cells, variant: 'checker', variantId: 6 };
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
  return { cells, variant: 'spiral', variantId: 7 };
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
  return { cells, variant: 'bands', variantId: 8 };
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
  return { cells, variant: 'glyphs', variantId: 9 };
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
  const generator = patternGenerators[Math.floor(Math.random() * patternGenerators.length)];
  let { cells, variant, variantId } = generator.generate(boardSize, numColors);
  const transforms = [];

  const rotations = Math.floor(Math.random() * 4);
  if (rotations) {
    cells = rotateBoard(cells, boardSize, rotations);
    transforms.push(`rotate×${rotations}`);
  }
  if (Math.random() < 0.5) {
    const axis = Math.random() < 0.5 ? 'horizontal' : 'vertical';
    cells = mirrorBoard(cells, boardSize, axis);
    transforms.push(`${axis} flip`);
  }
  if (Math.random() < 0.25) {
    cells = mirrorBoard(cells, boardSize, 'diag');
    transforms.push('transpose');
  }
  if (Math.random() < 0.55) {
    cells = shiftRows(cells, boardSize);
    transforms.push('row-shift');
  }
  if (Math.random() < 0.7) {
    cells = permuteColors(cells, numColors);
    transforms.push('recolour');
  }
  const noiseIntensity = Math.min(0.95, config.noise + Math.random() * 0.08);
  const noiseResult = applyNoise(cells, boardSize, numColors, noiseIntensity);
  cells = noiseResult.cells;
  const noiseApplied = noiseResult.noiseApplied;

  return {
    cells,
    name: generator.name,
    key: generator.key,
    transforms,
    variantId,
    variant,
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
    patternCounts[sample.pattern.name] = (patternCounts[sample.pattern.name] || 0) + 1;
    localPatternCounts[sample.pattern.name] = (localPatternCounts[sample.pattern.name] || 0) + 1;
    stats.boards++;
    stats.teacherSum += sample.targetScore;
    stats.modelSum += sample.modelScore;
    stats.maeSum += sample.absError;
    stats.weightUpdates++;
    sample.pattern.transforms.forEach(t => stats.transformSet.add(t));
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
      variant: bestSample.pattern.variant,
      transforms: bestSample.pattern.transforms,
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
