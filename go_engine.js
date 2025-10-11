// Basic 9x9 Go engine with captures, suicide rule, and simple ko.
// Chinese scoring (stones + territory). Designed for clarity over absolute speed.
(() => {
  const BLACK = 1, WHITE = 2, EMPTY = 0;
  const SIZE_DEFAULT = 9;

  function other(p) { return p === BLACK ? WHITE : BLACK; }
  function idxOf(x, y, size) { return y * size + x; }
  function xyOf(i, size) { return [i % size, Math.floor(i / size)]; }

  class Board {
    constructor(size = SIZE_DEFAULT, komi = 6.5) {
      this.size = size;
      this.komi = komi;
      this.cells = new Uint8Array(size * size); // 0 empty, 1 black, 2 white
      this.toPlay = BLACK;
      this.ko = -1;        // point index forbidden by simple ko; -1 if none
      this.passes = 0;     // consecutive passes
      this.moveCount = 0;
      this.capturesB = 0;
      this.capturesW = 0;
      this.history = [];   // minimal record (for undo)
      this._scratchVisited = new Uint8Array(size * size);
    }

    clone() {
      const b = new Board(this.size, this.komi);
      b.cells = new Uint8Array(this.cells);
      b.toPlay = this.toPlay;
      b.ko = this.ko;
      b.passes = this.passes;
      b.moveCount = this.moveCount;
      b.capturesB = this.capturesB;
      b.capturesW = this.capturesW;
      // history not cloned for speed in rollout; not needed
      return b;
    }

    neighbors(i) {
      const s = this.size;
      const x = i % s, y = (i / s) | 0;
      const res = [];
      if (x > 0) res.push(i - 1);
      if (x < s - 1) res.push(i + 1);
      if (y > 0) res.push(i - s);
      if (y < s - 1) res.push(i + s);
      return res;
    }

    // Returns {stones: Int32Array, liberties: Set<int>, countStones, countLibs}
    groupAndLibs(i) {
      const color = this.cells[i];
      if (color === EMPTY) return null;
      const s = this.size;
      const stack = [i];
      const visited = this._scratchVisited;
      visited.fill(0);
      visited[i] = 1;

      const stones = [];
      const libsSet = new Set();

      while (stack.length) {
        const p = stack.pop();
        stones.push(p);
        for (const n of this.neighbors(p)) {
          const c = this.cells[n];
          if (c === EMPTY) {
            libsSet.add(n);
          } else if (c === color && !visited[n]) {
            visited[n] = 1;
            stack.push(n);
          }
        }
      }
      return { stones, liberties: libsSet, countStones: stones.length, countLibs: libsSet.size };
    }

    // Attempt to play at i (0..size*size-1) or pass with i === -1.
    // Returns {ok, pass, captured, ko, msg}
    play(i) {
      if (i === -1) {
        // pass
        this.passes += 1;
        this.toPlay = other(this.toPlay);
        this.ko = -1;
        this.moveCount++;
        this.history.push({ pass: true });
        return { ok: true, pass: true, captured: 0, ko: this.ko };
      }

      if (this.cells[i] !== EMPTY) return { ok: false, msg: "occupied" };
      if (this.ko === i) return { ok: false, msg: "ko" };

      const color = this.toPlay;
      const opp = other(color);

      // Place stone tentatively
      this.cells[i] = color;

      // Capture any adjacent opponent groups with no liberties after move
      let totalCaptured = 0;
      let capturedPositions = [];
      for (const n of this.neighbors(i)) {
        if (this.cells[n] === opp) {
          const g = this.groupAndLibs(n);
          if (g.countLibs === 0) {
            // capture
            totalCaptured += g.countStones;
            capturedPositions.push(...g.stones);
            for (const s of g.stones) this.cells[s] = EMPTY;
          }
        }
      }

      // Check self-capture (suicide) — allowed only if we captured > 0
      const gSelf = this.groupAndLibs(i);
      if (gSelf.countLibs === 0 && totalCaptured === 0) {
        // illegal
        this.cells[i] = EMPTY;
        return { ok: false, msg: "suicide" };
      }

      // Update captures stat
      if (color === BLACK) this.capturesB += totalCaptured;
      else this.capturesW += totalCaptured;

      // Handle simple ko
      if (totalCaptured === 1 && gSelf.countLibs === 1) {
        this.ko = capturedPositions[0];
      } else {
        this.ko = -1;
      }

      // Success
      this.passes = 0;
      this.toPlay = opp;
      this.moveCount++;
      this.history.push({ move: i, color, captured: capturedPositions, prevKo: this.ko });
      return { ok: true, pass: false, captured: totalCaptured, ko: this.ko };
    }

    undo() {
      const rec = this.history.pop();
      if (!rec) return false;
      // reverse last move/pass
      if (rec.pass) {
        this.toPlay = other(this.toPlay);
        this.passes = Math.max(0, this.passes - 1);
        this.moveCount = Math.max(0, this.moveCount - 1);
        this.ko = -1;
        return true;
      } else {
        // Move was at rec.move by rec.color; put back captured stones
        const opp = rec.color === BLACK ? WHITE : BLACK;
        this.toPlay = rec.color;
        this.passes = 0;
        this.moveCount = Math.max(0, this.moveCount - 1);
        this.cells[rec.move] = rec.color;
        // Restore captured
        for (const s of rec.captured) this.cells[s] = opp;
        // Remove the played stone to revert (!)
        // Actually we need to revert to before move: so set played stone to EMPTY
        this.cells[rec.move] = EMPTY;
        // Captures counters are not tracked exactly for undo in this simple revert
        // (only used as HUD, not for rules). For simplicity, reset counters.
        this.recomputeCaptures();
        this.ko = -1;
        return true;
      }
    }

    recomputeCaptures() {
      // Approximate by 0; or recompute by diffing history (costly). We'll zero them.
      this.capturesB = 0;
      this.capturesW = 0;
      for (const rec of this.history) {
        if (rec.move !== undefined) {
          const color = rec.color;
          const nCap = (rec.captured || []).length;
          if (color === BLACK) this.capturesB += nCap;
          else this.capturesW += nCap;
        }
      }
    }

    legalMoves(includePass = true) {
      const s = this.size;
      const moves = [];
      for (let i = 0; i < s * s; i++) {
        if (this.cells[i] !== EMPTY) continue;
        if (i === this.ko) continue;
        // Test for suicide by playing and checking liberties quickly
        // (copy only what's necessary)
        const color = this.toPlay;
        const opp = other(color);
        // Place stone
        this.cells[i] = color;
        // Capture check on neighbors
        let captured = 0;
        for (const n of this.neighbors(i)) {
          if (this.cells[n] === opp) {
            const g = this.groupAndLibs(n);
            if (g.countLibs === 0) captured += g.countStones;
          }
        }
        const gSelf = this.groupAndLibs(i);
        // undo placement
        this.cells[i] = EMPTY;
        if (!(gSelf.countLibs === 0 && captured === 0)) {
          moves.push(i);
        }
      }
      if (includePass) moves.push(-1);
      return moves;
    }

    isTerminal() {
      return this.passes >= 2 || this.moveCount >= this.size * this.size;
    }

    // Chinese area scoring: stones + territory; white gets komi.
    // Returns score = black - white (with komi subtracted from black).
    areaScore() {
      const s = this.size;
      const cells = this.cells;
      const visited = this._scratchVisited;
      visited.fill(0);

      let blackStones = 0, whiteStones = 0;
      for (let i = 0; i < s*s; i++) {
        if (cells[i] === BLACK) blackStones++;
        else if (cells[i] === WHITE) whiteStones++;
      }

      let terrB = 0, terrW = 0;
      for (let i = 0; i < s*s; i++) {
        if (cells[i] !== EMPTY || visited[i]) continue;
        // flood fill empty region
        const q = [i];
        visited[i] = 1;
        const emptyRegion = [i];
        let adjB = false, adjW = false;
        while (q.length) {
          const p = q.pop();
          for (const n of this.neighbors(p)) {
            const c = cells[n];
            if (c === EMPTY && !visited[n]) {
              visited[n] = 1; q.push(n); emptyRegion.push(n);
            } else if (c === BLACK) adjB = true;
            else if (c === WHITE) adjW = true;
          }
        }
        if (adjB && !adjW) terrB += emptyRegion.length;
        else if (adjW && !adjB) terrW += emptyRegion.length;
      }

      const black = blackStones + terrB;
      const white = whiteStones + terrW + this.komi;
      return black - white;
    }
  }

// Expose to both main thread and worker
const _root = (typeof self !== 'undefined')
  ? self
  : (typeof window !== 'undefined' ? window : globalThis);
_root.GoEngine = { Board, BLACK, WHITE, EMPTY, other, idxOf, xyOf };
})();