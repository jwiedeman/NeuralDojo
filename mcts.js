// UCT Monte-Carlo Tree Search for Go (9x9 default), random playouts.
// Not NN-backed; intended for browser demo and live stats.
importScripts('go_engine.js');

const { Board, BLACK, WHITE, other } = self.GoEngine;

class Node {
  constructor(board, parent=null, moveFromParent=null, rootPlayer=BLACK) {
    this.board = board;          // board state at this node
    this.parent = parent;        // parent node (null for root)
    this.move = moveFromParent;  // move taken from parent to reach here (index or -1 for pass)
    this.children = [];          // array of child nodes
    this.untriedMoves = board.isTerminal() ? [] : board.legalMoves(true);
    this.visits = 0;
    this.wins = 0;               // accumulate reward from rootPlayer perspective
    this.rootPlayer = rootPlayer;
  }

  isFullyExpanded() { return this.untriedMoves.length === 0; }
  isLeaf() { return this.children.length === 0; }

  selectChild(c=1.41) {
    // UCB1: Q + c * sqrt(ln(N)/n)
    let best = null;
    let bestScore = -Infinity;
    for (const ch of this.children) {
      const q = ch.visits ? (ch.wins / ch.visits) : 0;
      const u = c * Math.sqrt(Math.log(this.visits + 1) / (ch.visits + 1e-6));
      const score = q + u;
      if (score > bestScore) { bestScore = score; best = ch; }
    }
    return best;
  }

  addChild(move, childBoard) {
    const ch = new Node(childBoard, this, move, this.rootPlayer);
    this.children.push(ch);
    // Remove move from untried list
    const k = this.untriedMoves.indexOf(move);
    if (k >= 0) this.untriedMoves.splice(k, 1);
    return ch;
  }
}

function randomChoice(arr) { return arr[(Math.random() * arr.length) | 0]; }

function rollout(board, maxPlayoutMoves=400) {
  // Random legal moves until terminal or move cap
  let steps = 0;
  while (!board.isTerminal() && steps < maxPlayoutMoves) {
    const moves = board.legalMoves(true);
    const m = randomChoice(moves);
    board.play(m);
    steps++;
  }
  const score = board.areaScore(); // black - white(with komi)
  // reward from rootPlayer perspective
  // If score > 0 => Black wins; <0 => White wins.
  const blackWins = score > 0;
  return blackWins ? 1 : 0; // we'll interpret later vs rootPlayer
}

function mctsSearch(rootBoard, timeMs=1200, c=1.41, progressCb=null) {
  const rootPlayer = rootBoard.toPlay;
  const root = new Node(rootBoard.clone(), null, null, rootPlayer);
  const t0 = performance.now();
  let it = 0;
  let nodes = 1;
  let lastPing = t0;
  const trend = [];

  while (performance.now() - t0 < timeMs) {
    // Selection
    let node = root;
    while (!node.board.isTerminal() && node.isFullyExpanded() && node.children.length > 0) {
      node = node.selectChild(c);
    }
    // Expansion
    if (!node.board.isTerminal() && node.untriedMoves.length > 0) {
      const m = randomChoice(node.untriedMoves);
      const bNext = node.board.clone();
      const r = bNext.play(m);
      const child = node.addChild(m, bNext);
      node = child;
      nodes++;
    }
    // Simulation
    const simBoard = node.board.clone();
    const rwdBlack = rollout(simBoard);
    // If rootPlayer is BLACK: reward = rwdBlack; else reward = 1 - rwdBlack
    const reward = (rootPlayer === BLACK) ? rwdBlack : (1 - rwdBlack);

    // Backprop
    while (node) {
      node.visits += 1;
      node.wins += reward;
      node = node.parent;
    }

    it++;
    const now = performance.now();
    if (progressCb && now - lastPing > 120) {
      // root estimate winrate for rootPlayer
      const qRoot = root.visits ? root.wins / root.visits : 0.5;
      trend.push(qRoot);
      // Construct simple heatmap from root children visits
      const heat = new Array(rootBoard.size * rootBoard.size).fill(0);
      let bestMove = null, bestVisits = -1, bestQ = 0;
      for (const ch of root.children) {
        const mv = ch.move;
        if (mv >= 0) heat[mv] = ch.visits;
        if (ch.visits > bestVisits) {
          bestVisits = ch.visits;
          bestMove = ch.move;
          bestQ = ch.visits ? (ch.wins / ch.visits) : 0;
        }
      }
      const elapsed = now - t0;
      const pps = (it * 1000 / elapsed).toFixed(0);
      const avgDepth = (root.visits > 0) ? ((nodes - 1) / root.visits).toFixed(2) : "0";

      progressCb({
        it, nodes, pps, avgDepth,
        winrate: qRoot,
        bestMove, bestVisits, bestQ,
        heat, trend
      });
      lastPing = now;
    }
  }

  // Final pick: child with max visits
  let pick = null, best = -1, bestQ = 0;
  const heat = new Array(rootBoard.size * rootBoard.size).fill(0);
  for (const ch of root.children) {
    if (ch.visits > best) {
      best = ch.visits;
      pick = ch.move;
      bestQ = ch.visits ? (ch.wins / ch.visits) : 0;
    }
    if (ch.move >= 0) heat[ch.move] = ch.visits;
  }
  const qRoot = root.visits ? root.wins / root.visits : 0.5;
  return {
    move: pick !== null ? pick : -1,
    it, nodes, pps: (it * 1000 / (performance.now() - t0)).toFixed(0),
    winrate: qRoot, bestQ, heat, trend
  };
}

// Worker glue
self.onmessage = (ev) => {
  const data = ev.data || {};
  if (data.type === 'search') {
    const { state, config } = data;
    const b = new Board(state.size || 9, state.komi || 6.5);
    b.cells = new Uint8Array(state.cells);
    b.toPlay = state.toPlay;
    b.ko = state.ko;
    b.passes = state.passes;
    b.moveCount = state.moveCount;
    // captures are HUD-only; not needed for search

    const timeMs = config.timeMs || 1200;
    const c = config.c || 1.41;

    const sendProgress = (snap) => {
      self.postMessage({ type: 'progress', snap });
    };

    const res = mctsSearch(b, timeMs, c, sendProgress);
    self.postMessage({ type: 'done', res });
  }
};
