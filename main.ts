// main.ts — push-optimal Sokoban solver.
// Usage: node main.ts [inputRoot] [outputRoot]   (grader calls it with no args: /input, /output)
//
// Strategy per level, first layer that produces a verified solution wins:
//   1. push-A*, weight 1 — push-optimal. A* over push states (state = canonical player square +
//      sorted box set, cost = pushes); player walks between pushes are rebuilt with BFS so the
//      replay stays legal. Pruning: static dead squares (no goal reachable by pulls) and freeze
//      deadlocks (corner / 2x2 / wall-hugged clusters holding a box off a goal). Heuristic:
//      "ideal distance" — exact single-box push distances over (box square, player region)
//      states, precomputed per goal by backward BFS (walks free, other boxes ignored), combined
//      as a min-cost box->goal assignment (Hungarian). Any greedy goal assignment is one
//      particular bijection, so the matching cost is a valid — and at least as strong — lower
//      bound; one push changes one row by at most 1, so it is consistent and the first goal
//      popped is push-optimal.
//   2. push-A*, weight 3 — same search, f = g + 3h: no optimality guarantee, but far cheaper and
//      it finds a valid solution on the bigger multi-room levels.
//   3. move-level A* (kit baseline port, plus the same deadlock pruning) — valid but wasteful.
//   4. room decomposition: cut the free space at articulation squares, merge components until
//      each body has boxes == goals, push-solve every body on its own tiny grid, stitch the
//      pieces with player walks. Runs only after everything above failed; this is what cracks
//      the big multi-room levels.
//   5. empty solution.txt as the last resort.
// Every candidate solution is replay-checked internally before it is written.
//
// Levels with >= BIG_LEVEL_BOXES boxes try the cheap weighted pass first: the optimal pass is
// the expensive one, so on big boards we only pay for it when the cheap passes fail.
//
// Bounded memory: A* nodes live in flat typed-array pools sized PUSH_POOL_NODES /
// MOVE_POOL_NODES. When a pool fills, the search abandons immediately (status 'budget') instead
// of growing — RSS is a deterministic function of the pool caps, never of the level. Push
// directions are stored as one byte per node and player walk segments are rebuilt by BFS at
// reconstruction time, so no per-node step strings are retained.
//
// Determinism: budgets are node-expansion/commit counts (never wall-clock), ids/boxes/goals are
// sorted, heap tie-breaks are fixed (f, state key, node index), no Math.random / Date / network.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const IN = process.argv[2] ?? '/input';
const OUT = process.argv[3] ?? '/output';
const PUSH_EXPANSIONS = 250_000; // optimal pass, small levels (i003 solves at 179,204)
const BIG_PUSH_EXPANSIONS = 32_000; // optimal pass, big levels: reduced so a give-up is cheap
const PUSH_W_EXPANSIONS = 100_000;
const BIG_W_EXPANSIONS = 50_000; // weighted pass on big levels: cheaper give-up
const PUSH_WEIGHT = 3;
const GREEDY_EXPANSIONS = 100_000;
const MOVE_EXPANSIONS = 150_000;
const BIG_LEVEL_BOXES = 10;
// Decomposition last-resort: per-body and total expansion budgets (deterministic counts).
// Bodies seen so far solve in <= ~600 expansions; the caps bound pathological cases only.
const DECOMP_BODY_CAP = 50_000;
const DECOMP_TOTAL = 250_000;
// Node-pool ceilings. Push search: worst observed need is i003's 968,140 commits at 250k
// expansions (worst commits/expansion rate seen is ~16, so a pool of min(ceiling, cap*16+1024)
// never truncates a search that its expansion budget would have finished). Move search: at most
// 4 children per expansion. Pools scale with the budget so memory stays proportional to the
// work actually allowed; when a pool fills the search abandons gracefully (status 'budget').
const PUSH_POOL_NODES = 1_200_000;
const MOVE_POOL_NODES = 620_000;
const BIG = 10_000_000;

// deltas in row-major grid order: u, d, l, r
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const CH = ['u', 'd', 'l', 'r'];
const CHU = ['U', 'D', 'L', 'R'];

export type Level = {
  W: number;
  n: number;
  wall: Uint8Array;
  goal: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
  dead: Uint8Array;
  minDist: number[];
  regionOf: Int32Array; // regionOf[box * n + cell]: min square of the player region around cell
  exact: Int32Array[]; // exact[goalIndex][box * n + region]: exact single-box push distance
};

export function parse(text: string): Level {
  const rows = text.replace(/\r/g, '').split('\n').filter((r) => r.length > 0);
  const W = Math.max(...rows.map((r) => r.length));
  const H = rows.length;
  const n = W * H;
  const wall = new Uint8Array(n).fill(1);
  const goal = new Uint8Array(n);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  for (let y = 0; y < H; y++) {
    const row = rows[y]!;
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const ch = row[x] ?? '#';
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') {
        goal[i] = 1;
        goals.push(i);
      }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  }
  boxes.sort((a, b) => a - b);

  // Static dead squares: pull-BFS backwards from every goal. A box on a square from which no goal
  // can be reached by pulls can never sit on a goal.
  const minDist = new Array<number>(n).fill(Infinity);
  const dead = new Uint8Array(n);
  {
    const dist = new Array<number>(n).fill(Infinity);
    const q: number[] = [];
    for (const g of goals) {
      dist[g] = 0;
      q.push(g);
    }
    for (let head = 0; head < q.length; head++) {
      const t = q[head]!;
      for (let di = 0; di < 4; di++) {
        const dd = DY[di]! * W + DX[di]!;
        const s = t - dd;
        const pusher = s - dd;
        if (s < 0 || s >= n || wall[s] !== 0 || dist[s] !== Infinity) continue;
        if (pusher < 0 || pusher >= n || wall[pusher] !== 0) continue;
        dist[s] = dist[t]! + 1;
        q.push(s);
      }
    }
    for (let i = 0; i < n; i++) {
      minDist[i] = dist[i]!;
      dead[i] = wall[i] || dist[i] === Infinity ? 1 : 0;
    }
  }

  // Player region ids per box position: regionOf[box * n + cell] = min free square reachable from
  // cell with `box` occupied (walks only). Merges all zero-push player walks into one state.
  const regionOf = new Int32Array(n * n).fill(-1);
  {
    const cells = new Int32Array(n);
    for (let b = 0; b < n; b++) {
      if (wall[b]) continue;
      const base = b * n;
      for (let p = 0; p < n; p++) {
        if (wall[p] || p === b || regionOf[base + p] !== -1) continue;
        let head = 0;
        let tail = 0;
        let min = p;
        cells[tail++] = p;
        regionOf[base + p] = p;
        while (head < tail) {
          const c = cells[head++]!;
          for (let di = 0; di < 4; di++) {
            const t = c + DY[di]! * W + DX[di]!;
            if (t < 0 || t >= n || wall[t] || t === b || regionOf[base + t] !== -1) continue;
            regionOf[base + t] = p;
            cells[tail++] = t;
            if (t < min) min = t;
          }
        }
        for (let k = 0; k < tail; k++) regionOf[base + cells[k]!] = min;
      }
    }
  }

  // Exact single-box push distance per goal over (box square, player region) states: walks are
  // free (regions), other boxes ignored. Backward BFS from the goal over push predecessors.
  const exact: Int32Array[] = goals.map((g) => {
    const dist = new Int32Array(n * n).fill(-1);
    const q: number[] = [];
    for (let p = 0; p < n; p++) {
      if (wall[p] || p === g) continue;
      const r = regionOf[g * n + p]!;
      if (dist[g * n + r] !== -1) continue;
      dist[g * n + r] = 0;
      q.push(g * n + r);
    }
    for (let head = 0; head < q.length; head++) {
      const cur = q[head]!;
      const b = (cur / n) | 0;
      const r = cur % n;
      const k = dist[cur]!;
      for (let di = 0; di < 4; di++) {
        const dd = DY[di]! * W + DX[di]!;
        const b0 = b - dd; // the box came from b0, pushed along dd
        const p0 = b - 2 * dd; // the pusher stood on p0
        if (b0 < 0 || b0 >= n || wall[b0]) continue;
        if (p0 < 0 || p0 >= n || wall[p0]) continue;
        const r0 = regionOf[b0 * n + p0]!;
        const idx = b0 * n + r0;
        if (dist[idx] !== -1) continue;
        dist[idx] = k + 1;
        q.push(idx);
      }
    }
    return dist;
  });

  return { W, n, wall, goal, goals, boxes, player, dead, minDist, regionOf, exact };
}

// --- freeze deadlock, allocation-free (scratch below is single-threaded) --------------------
let FD_wall: Uint8Array;
let FD_goal: Uint8Array;
let FD_dead: Uint8Array;
let FD_W = 0;
let FD_boxes: number[];
const FD_treated: number[] = [];
const FD_cluster: number[] = [];
const fdFrozenAt = (q: number): boolean => {
  FD_treated.push(q);
  const blocked = (d: number): boolean => {
    const a = q + d;
    const b = q - d;
    if (FD_wall[a] !== 0 || FD_wall[b] !== 0) return true;
    if (FD_treated.indexOf(a) >= 0 || FD_treated.indexOf(b) >= 0) return true;
    if (FD_dead[a] && FD_dead[b]) return true;
    if (FD_boxes.indexOf(a) >= 0 && fdFrozenAt(a)) return true;
    if (FD_boxes.indexOf(b) >= 0 && fdFrozenAt(b)) return true;
    return false;
  };
  if (!blocked(1) || !blocked(FD_W)) return false;
  FD_cluster.push(q);
  return true;
};
// True when the box pushed to p freezes a cluster that holds at least one box off a goal.
function frozenDeadlock(
  wall: Uint8Array,
  goal: Uint8Array,
  dead: Uint8Array,
  W: number,
  boxes: number[],
  p: number,
): boolean {
  FD_wall = wall;
  FD_goal = goal;
  FD_dead = dead;
  FD_W = W;
  FD_boxes = boxes;
  FD_treated.length = 0;
  FD_cluster.length = 0;
  if (!fdFrozenAt(p)) return false;
  for (const q of FD_cluster) if (!goal[q]) return true;
  return false;
}

// --- Hungarian assignment, reused scratch, flat row-major matrix ----------------------------
const M_u = new Float64Array(512);
const M_v = new Float64Array(512);
const M_p = new Float64Array(512);
const M_way = new Float64Array(512);
const M_minv = new Float64Array(512);
const M_used = new Uint8Array(512);
function matchingCost(mat: Float64Array, rows: number, cols: number): number {
  const INF = BIG * 4;
  for (let i = 0; i <= rows; i++) M_u[i] = 0;
  for (let j = 0; j <= cols; j++) {
    M_v[j] = 0;
    M_p[j] = 0;
    M_way[j] = 0;
  }
  for (let i = 1; i <= rows; i++) {
    M_p[0] = i;
    let j0 = 0;
    for (let j = 0; j <= cols; j++) {
      M_minv[j] = INF;
      M_used[j] = 0;
    }
    do {
      M_used[j0] = 1;
      const i0 = M_p[j0]!;
      let delta = INF;
      let j1 = -1;
      const rowOff = (i0 - 1) * cols;
      for (let j = 1; j <= cols; j++) {
        if (M_used[j]!) continue;
        const cur = mat[rowOff + j - 1]! - M_u[i0]! - M_v[j]!;
        if (cur < M_minv[j]!) {
          M_minv[j] = cur;
          M_way[j] = j0;
        }
        if (M_minv[j]! < delta) {
          delta = M_minv[j]!;
          j1 = j;
        }
      }
      for (let j = 0; j <= cols; j++) {
        if (M_used[j]) {
          M_u[M_p[j]!] += delta;
          M_v[j] -= delta;
        } else {
          M_minv[j]! -= delta;
        }
      }
      j0 = j1!;
    } while (M_p[j0] !== 0);
    do {
      const j1 = M_way[j0]!;
      M_p[j0] = M_p[j1]!;
      j0 = j1;
    } while (j0 !== 0);
  }
  let total = 0;
  for (let j = 1; j <= cols; j++) if (M_p[j] !== 0) total += mat[(M_p[j]! - 1) * cols + j - 1]!;
  return total;
}

type Flood = {
  stamp: Int32Array;
  mark: Int32Array;
  par: Int32Array;
  q: Int32Array;
  cur: number;
};

type Outcome = { status: 'solved' | 'unsolvable' | 'budget'; moves: string; expansions: number };

// A* over push states. weight 1 = push-optimal; higher weights trade optimality for speed.
export function pushSolve(lv: Level, cap: number, weight: number): Outcome {
  const { W, n, wall, goal, goals, dead, minDist, regionOf, exact } = lv;
  const rootBoxes = lv.boxes.slice();
  const rootPlayer = lv.player;
  const G = goals.length;
  const B = rootBoxes.length;
  const DD = [DY[0]! * W + DX[0]!, DY[1]! * W + DX[1]!, DY[2]! * W + DX[2]!, DY[3]! * W + DX[3]!];

  const mk = (): Flood => ({
    stamp: new Int32Array(n),
    mark: new Int32Array(n),
    par: new Int32Array(n),
    q: new Int32Array(n),
    cur: 0,
  });
  const pa = mk();
  const ca = mk();
  const flood = (c: Flood, start: number, bs: number[]): number => {
    c.cur++;
    const stamp = c.stamp;
    const mark = c.mark;
    const par = c.par;
    const q = c.q;
    for (let i = 0; i < bs.length; i++) mark[bs[i]!] = c.cur;
    let head = 0;
    let tail = 0;
    let min = start;
    q[tail++] = start;
    stamp[start] = c.cur;
    par[start] = -1;
    while (head < tail) {
      const p = q[head++]!;
      if (p < min) min = p;
      for (let di = 0; di < 4; di++) {
        const t = p + DD[di]!;
        if (t < 0 || t >= n || wall[t] !== 0) continue;
        if (stamp[t] === c.cur || mark[t] === c.cur) continue;
        stamp[t] = c.cur;
        par[t] = p;
        q[tail++] = t;
      }
    }
    return min;
  };
  const walkTo = (c: Flood, target: number): string => {
    let s = '';
    for (let p = target; c.par[p] !== -1; p = c.par[p]!) {
      const dd = p - c.par[p]!;
      s = (dd === 1 ? 'r' : dd === -1 ? 'l' : dd === W ? 'd' : 'u') + s;
    }
    return s;
  };

  // Ideal-distance matching lower bound for a box set, seen from the player's square.
  const mat = new Float64Array(Math.max(1, rootBoxes.length) * G);
  const hOf = (bs: number[], pl: number): number => {
    for (let i = 0; i < bs.length; i++) {
      const b = bs[i]!;
      const base = b * n + regionOf[b * n + pl]!;
      const off = i * G;
      for (let gi = 0; gi < G; gi++) {
        const d = exact[gi]![base]!;
        mat[off + gi] = d === -1 ? BIG : d;
      }
    }
    return matchingCost(mat, bs.length, G);
  };

  // Bounded node pools: fixed typed-array slots per node. Dedup uses an open-addressing table
  // with EXACT key comparison (canonical player square + sorted boxes — a hash hit is always
  // verified), so no per-node key strings or Map entries are ever allocated. Heap tie-breaks
  // compare the same (canon, boxes) key element-wise. The pool size is derived from the
  // expansion budget (never wall-clock); when it fills the search abandons (budget) — RSS never
  // depends on how hard the level fights back.
  const MAXN = Math.min(PUSH_POOL_NODES, cap * 16 + 1024);
  const nBoxes = new Int32Array(MAXN * B);
  const nCanon = new Int32Array(MAXN);
  const nPlayerA = new Int32Array(MAXN);
  const nG = new Int32Array(MAXN);
  const nH = new Int32Array(MAXN);
  const nParent = new Int32Array(MAXN);
  const nDir = new Uint8Array(MAXN); // push direction 0..3; 4 = root
  const heapF = new Int32Array(MAXN);
  const heapI = new Int32Array(MAXN);
  let heapLen = 0;
  let cnt = 0;
  const boxMark = new Uint8Array(n);
  let TCAP = 1024;
  while (TCAP < 2 * MAXN) TCAP <<= 1;
  const table = new Int32Array(TCAP); // transposition table: node index + 1, 0 = empty
  const TMASK = TCAP - 1;
  const nodeHash = (t: number): number => {
    let h = 0x811c9dc5 | 0;
    h = Math.imul(h ^ nCanon[t]!, 16777619);
    const off = t * B;
    for (let i = 0; i < B; i++) h = Math.imul(h ^ nBoxes[off + i]!, 16777619);
    return h >>> 0;
  };
  const nodeEq = (t1: number, t2: number): boolean => {
    if (nCanon[t1] !== nCanon[t2]) return false;
    const o1 = t1 * B;
    const o2 = t2 * B;
    for (let i = 0; i < B; i++) if (nBoxes[o1 + i] !== nBoxes[o2 + i]) return false;
    return true;
  };
  const candEq = (t: number, canon: number, bs: number[]): boolean => {
    if (nCanon[t] !== canon) return false;
    const off = t * B;
    for (let i = 0; i < B; i++) if (nBoxes[off + i] !== bs[i]) return false;
    return true;
  };
  const tableFindNode = (t: number): number => {
    let slot = nodeHash(t) & TMASK;
    for (;;) {
      const v = table[slot]!;
      if (v === 0) return -1;
      if (nodeEq(v - 1, t)) return v - 1;
      slot = (slot + 1) & TMASK;
    }
  };
  const tableFindCand = (canon: number, bs: number[]): number => {
    let h = 0x811c9dc5 | 0;
    h = Math.imul(h ^ canon, 16777619);
    for (let i = 0; i < B; i++) h = Math.imul(h ^ bs[i]!, 16777619);
    let slot = (h >>> 0) & TMASK;
    for (;;) {
      const v = table[slot]!;
      if (v === 0) return -1;
      if (candEq(v - 1, canon, bs)) return v - 1;
      slot = (slot + 1) & TMASK;
    }
  };
  const tableSet = (t: number): void => {
    let slot = nodeHash(t) & TMASK;
    for (;;) {
      const v = table[slot]!;
      if (v === 0 || nodeEq(v - 1, t)) {
        table[slot] = t + 1;
        return;
      }
      slot = (slot + 1) & TMASK;
    }
  };
  const keyLess = (t1: number, t2: number): boolean => {
    // same total order as the previous fromCharCode string keys: canon first, then boxes
    if (nCanon[t1] !== nCanon[t2]) return nCanon[t1]! < nCanon[t2]!;
    const o1 = t1 * B;
    const o2 = t2 * B;
    for (let i = 0; i < B; i++) {
      const x = nBoxes[o1 + i]!;
      const y = nBoxes[o2 + i]!;
      if (x !== y) return x < y;
    }
    return false;
  };
  const less = (a: number, b: number): boolean => {
    if (heapF[a]! !== heapF[b]!) return heapF[a]! < heapF[b]!;
    const ia = heapI[a]!;
    const ib = heapI[b]!;
    if (keyLess(ia, ib)) return true;
    if (keyLess(ib, ia)) return false;
    return ia < ib;
  };
  const hswap = (a: number, b: number): void => {
    const f = heapF[a]!;
    const i = heapI[a]!;
    heapF[a] = heapF[b]!;
    heapI[a] = heapI[b]!;
    heapF[b] = f;
    heapI[b] = i;
  };
  const hpush = (i: number): void => {
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(i, p)) break;
      hswap(i, p);
      i = p;
    }
  };
  const hpop = (): number => {
    const top = heapI[0]!;
    heapLen--;
    if (heapLen > 0) {
      heapF[0] = heapF[heapLen]!;
      heapI[0] = heapI[heapLen]!;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heapLen) break;
        if (c + 1 < heapLen && less(c + 1, c)) c++;
        if (!less(c, i)) break;
        hswap(i, c);
        i = c;
      }
    }
    return top;
  };
  const commit = (
    canon: number,
    playerA: number,
    bs: number[],
    g: number,
    h: number,
    parent: number,
    dir: number,
  ): boolean => {
    if (cnt >= MAXN) return false; // pool full: abandon
    const t = cnt++;
    nCanon[t] = canon;
    const off = t * B;
    for (let i = 0; i < B; i++) nBoxes[off + i] = bs[i]!;
    nPlayerA[t] = playerA;
    nG[t] = g;
    nH[t] = h;
    nParent[t] = parent;
    nDir[t] = dir;
    tableSet(t);
    heapF[heapLen] = g + weight * h;
    heapI[heapLen] = t;
    heapLen++;
    hpush(heapLen - 1);
    return true;
  };

  const scratch: number[] = new Array(Math.max(1, B)).fill(0);
  const walkBoxes: number[] = new Array(Math.max(1, B)).fill(0);
  if (rootBoxes.length > G || rootBoxes.some((b) => minDist[b] === Infinity)) {
    return { status: 'unsolvable', moves: '', expansions: 0 };
  }
  if (rootBoxes.every((b) => goal[b])) return { status: 'solved', moves: '', expansions: 0 };
  const rootCanon = flood(ca, rootPlayer, rootBoxes);
  {
    const h0 = hOf(rootBoxes, rootPlayer);
    if (h0 >= BIG) return { status: 'unsolvable', moves: '', expansions: 0 };
    if (!commit(rootCanon, rootPlayer, rootBoxes, 0, h0, -1, 4)) {
      return { status: 'budget', moves: '', expansions: 0 };
    }
  }
  let expanded = 0;
  while (heapLen > 0) {
    const s = hpop();
    if (tableFindNode(s) !== s) continue;
    if (nH[s] === 0) {
      // Rebuild each node's player walk with a fresh BFS over the parent state — deterministic
      // and identical to the flood the expansion originally ran.
      let m = '';
      for (let t = s; t > 0; t = nParent[t]!) {
        const di = nDir[t]!;
        const p = nParent[t]!;
        const off = p * B;
        for (let i = 0; i < B; i++) walkBoxes[i] = nBoxes[off + i]!;
        flood(pa, nPlayerA[p]!, walkBoxes);
        const from = nPlayerA[t]! - DD[di]!;
        m = walkTo(pa, from) + CHU[di]! + m;
      }
      return { status: 'solved', moves: m, expansions: expanded };
    }
    if (expanded >= cap) return { status: 'budget', moves: '', expansions: expanded };
    expanded++;
    const sOff = s * B;
    const g1 = nG[s]! + 1;
    for (let i = 0; i < B; i++) {
      walkBoxes[i] = nBoxes[sOff + i]!;
      boxMark[walkBoxes[i]!] = 1;
    }
    flood(pa, nPlayerA[s]!, walkBoxes);
    const pStamp = pa.stamp;
    const pCur = pa.cur;
    for (let bi = 0; bi < B; bi++) {
      const b = nBoxes[sOff + bi]!;
      for (let di = 0; di < 4; di++) {
        const d = DD[di]!;
        const from = b - d;
        const to = b + d;
        if (wall[from] !== 0 || pStamp[from] !== pCur) continue;
        if (wall[to] !== 0 || boxMark[to] !== 0 || dead[to]) continue;
        for (let i = 0; i < B; i++) scratch[i] = nBoxes[sOff + i]!;
        scratch[bi] = to;
        scratch.sort((x, y) => x - y);
        if (frozenDeadlock(wall, goal, dead, W, scratch, to)) continue;
        const childCanon = flood(ca, b, scratch);
        const old = tableFindCand(childCanon, scratch);
        if (old !== -1 && nG[old]! <= g1) continue;
        const h = hOf(scratch, b);
        if (h >= BIG) continue; // some box can reach no goal from here: unsolvable configuration
        if (!commit(childCanon, b, scratch, g1, h, s, di)) {
          return { status: 'budget', moves: '', expansions: expanded };
        }
      }
    }
    for (let i = 0; i < B; i++) boxMark[nBoxes[sOff + i]!] = 0;
  }
  return { status: 'unsolvable', moves: '', expansions: expanded };
}

// Greedy sequential planner: repeatedly pick the (box, goal) pair with the smallest ideal
// distance among free goals and route that box there with a BFS over (box square, player region)
// states — all other boxes are obstacles, settled boxes are never touched, and every intermediate
// push passes the freeze check so nothing wedges mid-route. No optimality; a cheap way to get a
// valid solution out of room-decomposable levels.
function greedySolve(lv: Level, cap: number): Outcome {
  const { W, n, wall, goal, goals, dead, regionOf, exact } = lv;
  const G = goals.length;
  const DD = [DY[0]! * W + DX[0]!, DY[1]! * W + DX[1]!, DY[2]! * W + DX[2]!, DY[3]! * W + DX[3]!];
  const boxes = lv.boxes.slice();
  let player = lv.player;
  let used = 0;
  const out: string[] = [];
  const settled = new Set<number>();
  const reserved = new Set<number>();
  const boxMark = new Uint8Array(n);
  for (const b of boxes) boxMark[b] = 1;

  const stamp = new Int32Array(n);
  const par = new Int32Array(n);
  const qarr = new Int32Array(n);
  let cur = 0;
  // flood with obstacles wall + boxMark + one extra (the moving box's current square)
  const flood = (start: number, mv: number): number => {
    cur++;
    let head = 0;
    let tail = 0;
    let min = start;
    qarr[tail++] = start;
    stamp[start] = cur;
    par[start] = -1;
    while (head < tail) {
      const p = qarr[head++]!;
      if (p < min) min = p;
      for (let di = 0; di < 4; di++) {
        const t = p + DD[di]!;
        if (t < 0 || t >= n || wall[t] || boxMark[t] || t === mv) continue;
        if (stamp[t] === cur) continue;
        stamp[t] = cur;
        par[t] = p;
        qarr[tail++] = t;
      }
    }
    return min;
  };
  const walkTo = (target: number): string => {
    let s = '';
    for (let p = target; par[p] !== -1; p = par[p]!) {
      const dd = p - par[p]!;
      s = (dd === 1 ? 'r' : dd === -1 ? 'l' : dd === W ? 'd' : 'u') + s;
    }
    return s;
  };
  const scratch: number[] = new Array(40).fill(0);

  // Route one box from fb to gcell; null if no route (or budget spent; exhausted tells which).
  // On success the box lands on gcell, boxMark is updated and the player's final square returned.
  let exhausted = false;
  const planRoute = (fb: number, gcell: number): { steps: string; playerEnd: number } | null => {
    boxMark[fb] = 0;
    const sBox: number[] = [fb];
    const sPl: number[] = [player];
    const sPar: number[] = [-1];
    const sStep: string[] = [''];
    const seen = new Map<number, number>();
    seen.set(fb * n + flood(player, fb)!, 0);
    let head = 0;
    while (head < sBox.length) {
      if (used >= cap) {
        exhausted = true;
        boxMark[fb] = 1;
        return null;
      }
      used++;
      const idx = head++;
      const b = sBox[idx]!;
      const plc = sPl[idx]!;
      flood(plc, b);
      for (let di = 0; di < 4; di++) {
        const d = DD[di]!;
        const from = b - d;
        const to = b + d;
        if (from < 0 || from >= n || stamp[from] !== cur) continue;
        if (to < 0 || to >= n || wall[to] || boxMark[to] || dead[to]) continue;
        scratch.length = boxes.length;
        for (let i = 0; i < boxes.length; i++) scratch[i] = boxes[i]!;
        scratch[boxes.indexOf(fb)] = to;
        scratch.sort((x, y) => x - y);
        if (frozenDeadlock(wall, goal, dead, W, scratch, to)) continue;
        const r = flood(b, to); // child's player sits on b, moving box now on to
        const key = to * n + r;
        if (seen.has(key)) continue;
        const step = walkTo(from) + CHU[di];
        if (to === gcell) {
          boxMark[gcell] = 1;
          return { steps: sStep[idx]! + step, playerEnd: b };
        }
        seen.set(key, sBox.length);
        sBox.push(to);
        sPl.push(b);
        sPar.push(idx);
        sStep.push(sStep[idx]! + step);
      }
    }
    boxMark[fb] = 1;
    return null;
  };

  while (true) {
    let all = true;
    for (const b of boxes) if (!goal[b]) all = false;
    if (all) return { status: 'solved', moves: out.join(''), expansions: used };
    for (const b of boxes) {
      if (!settled.has(b) && goal[b] && !reserved.has(b)) {
        settled.add(b);
        reserved.add(b);
      }
    }
    const cands: { d: number; b: number; g: number }[] = [];
    for (const b of boxes) {
      if (settled.has(b)) continue;
      const r = regionOf[b * n + player]!;
      for (let gi = 0; gi < G; gi++) {
        const gcell = goals[gi]!;
        if (reserved.has(gcell)) continue;
        const d = exact[gi]![b * n + r]!;
        if (d === -1) continue;
        cands.push({ d, b, g: gcell });
      }
    }
    cands.sort((x, y) => x.d - y.d || x.b - y.b || x.g - y.g);
    let applied = false;
    for (const c of cands) {
      if (used >= cap) break;
      const idx = boxes.indexOf(c.b);
      const route = planRoute(c.b, c.g);
      if (route !== null) {
        boxes[idx] = c.g;
        boxes.sort((x, y) => x - y);
        settled.add(c.g);
        reserved.add(c.g);
        player = route.playerEnd;
        out.push(route.steps);
        applied = true;
        break;
      }
    }
    if (!applied) {
      return { status: exhausted ? 'budget' : 'unsolvable', moves: '', expansions: used };
    }
  }
}

// Plain move-level A* (kit baseline port, plus dead-square/freeze pruning): every step costs 1.

// Valid solutions, not always push-optimal. Last solver in the chain. Node storage is a flat
// typed-array pool with a hard cap: abandon (budget) when full, so memory never grows unbounded.
function moveSolve(lv: Level, cap: number): Outcome {
  const { W, n, wall, goal, goals, dead, player } = lv;
  const DD = [DY[0]! * W + DX[0]!, DY[1]! * W + DX[1]!, DY[2]! * W + DX[2]!, DY[3]! * W + DX[3]!];
  const B = lv.boxes.length;
  const goalsSorted = goals.slice().sort((a, b) => a - b);
  const manh = (a: number, b: number): number =>
    Math.abs((a % W) - (b % W)) + Math.abs(Math.floor(a / W) - Math.floor(b / W));
  const goalDist = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    let m = Infinity;
    for (const g of goalsSorted) {
      const d = manh(g, i);
      if (d < m) m = d;
    }
    goalDist[i] = m;
  }
  const est = (bs: number[]): number => {
    let a = 0;
    for (let i = 0; i < bs.length; i++) a += goalDist[bs[i]!]!;
    return a;
  };
  const MAXN = Math.min(MOVE_POOL_NODES, cap * 4 + 1024);
  const sp = new Int32Array(MAXN);
  const sb = new Int32Array(MAXN * B);
  const sg = new Int32Array(MAXN);
  const sh = new Int32Array(MAXN);
  const parent = new Int32Array(MAXN);
  const stepDi = new Uint8Array(MAXN); // 0..3 walk, 4..7 push (di + 4)
  const heapF = new Int32Array(MAXN);
  const heapG = new Int32Array(MAXN);
  const heapI = new Int32Array(MAXN);
  let heapLen = 0;
  let cnt = 0;
  // Transposition table: open addressing, exact key compare (player square + sorted boxes),
  // no per-node strings. Entries are node index + 1; 0 = empty.
  let TCAP = 1024;
  while (TCAP < 2 * MAXN) TCAP <<= 1;
  const table = new Int32Array(TCAP);
  const TMASK = TCAP - 1;
  const nodeHash = (t: number): number => {
    let h = 0x811c9dc5 | 0;
    h = Math.imul(h ^ sp[t]!, 16777619);
    const off = t * B;
    for (let i = 0; i < B; i++) h = Math.imul(h ^ sb[off + i]!, 16777619);
    return h >>> 0;
  };
  const nodeEq = (t1: number, t2: number): boolean => {
    if (sp[t1] !== sp[t2]) return false;
    const o1 = t1 * B;
    const o2 = t2 * B;
    for (let i = 0; i < B; i++) if (sb[o1 + i] !== sb[o2 + i]) return false;
    return true;
  };
  const candEq = (t: number, p: number, bs: number[]): boolean => {
    if (sp[t] !== p) return false;
    const off = t * B;
    for (let i = 0; i < B; i++) if (sb[off + i] !== bs[i]) return false;
    return true;
  };
  const tableFindNode = (t: number): number => {
    let slot = nodeHash(t) & TMASK;
    for (;;) {
      const v = table[slot]!;
      if (v === 0) return -1;
      if (nodeEq(v - 1, t)) return v - 1;
      slot = (slot + 1) & TMASK;
    }
  };
  const tableFindCand = (p: number, bs: number[]): number => {
    let h = 0x811c9dc5 | 0;
    h = Math.imul(h ^ p, 16777619);
    for (let i = 0; i < B; i++) h = Math.imul(h ^ bs[i]!, 16777619);
    let slot = (h >>> 0) & TMASK;
    for (;;) {
      const v = table[slot]!;
      if (v === 0) return -1;
      if (candEq(v - 1, p, bs)) return v - 1;
      slot = (slot + 1) & TMASK;
    }
  };
  const tableSet = (t: number): void => {
    let slot = nodeHash(t) & TMASK;
    for (;;) {
      const v = table[slot]!;
      if (v === 0 || nodeEq(v - 1, t)) {
        table[slot] = t + 1;
        return;
      }
      slot = (slot + 1) & TMASK;
    }
  };
  const less = (a: number, b: number): boolean => {
    if (heapF[a]! !== heapF[b]!) return heapF[a]! < heapF[b]!;
    if (heapG[a]! !== heapG[b]!) return heapG[a]! > heapG[b]!;
    return heapI[a]! < heapI[b]!;
  };
  const hswap = (a: number, b: number): void => {
    let v = heapF[a]!;
    heapF[a] = heapF[b]!;
    heapF[b] = v;
    v = heapG[a]!;
    heapG[a] = heapG[b]!;
    heapG[b] = v;
    v = heapI[a]!;
    heapI[a] = heapI[b]!;
    heapI[b] = v;
  };
  const hpush = (i: number): void => {
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(i, p)) break;
      hswap(i, p);
      i = p;
    }
  };
  const hpop = (): number => {
    const top = heapI[0]!;
    heapLen--;
    if (heapLen > 0) {
      heapF[0] = heapF[heapLen]!;
      heapG[0] = heapG[heapLen]!;
      heapI[0] = heapI[heapLen]!;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heapLen) break;
        if (c + 1 < heapLen && less(c + 1, c)) c++;
        if (!less(c, i)) break;
        hswap(i, c);
        i = c;
      }
    }
    return top;
  };
  // root
  sp[0] = player;
  const rootBoxes = lv.boxes;
  for (let i = 0; i < B; i++) sb[i] = rootBoxes[i]!;
  sg[0] = 0;
  sh[0] = est(rootBoxes);
  parent[0] = -1;
  stepDi[0] = 8; // unused sentinel
  cnt = 1;
  tableSet(0);
  heapF[0] = sh[0]!;
  heapG[0] = 0;
  heapI[0] = 0;
  heapLen = 1;
  const scratch: number[] = new Array(Math.max(1, B)).fill(0);
  const curBoxes: number[] = new Array(Math.max(1, B)).fill(0);
  const boxMark = new Uint8Array(n);
  let expanded = 0;
  while (heapLen > 0) {
    const s = hpop();
    if (tableFindNode(s) !== s) continue;
    const sOff = s * B;
    let all = true;
    for (let i = 0; i < B; i++) {
      if (!goal[sb[sOff + i]!]) {
        all = false;
        break;
      }
    }
    if (all) {
      let moves = '';
      for (let t = s; t > 0; t = parent[t]!) {
        const v = stepDi[t]!;
        moves = (v < 4 ? CH[v]! : CHU[v - 4]!) + moves;
      }
      return { status: 'solved', moves, expansions: expanded };
    }
    if (expanded >= cap) return { status: 'budget', moves: '', expansions: expanded };
    expanded++;
    const p = sp[s]!;
    const g = sg[s]!;
    for (let i = 0; i < B; i++) {
      curBoxes[i] = sb[sOff + i]!;
      boxMark[curBoxes[i]!] = 1;
    }
    for (let di = 0; di < 4; di++) {
      const d = DD[di]!;
      const q = p + d;
      if (q < 0 || q >= n || wall[q]) continue;
      const at = (() => {
        for (let i = 0; i < B; i++) if (sb[sOff + i] === q) return i;
        return -1;
      })();
      let pushed = false;
      let childEst = sh[s]!;
      let childBoxes: number[] | null = null;
      if (at >= 0) {
        const t2 = q + d;
        if (t2 < 0 || t2 >= n || wall[t2] || boxMark[t2] !== 0 || dead[t2]) continue;
        for (let i = 0; i < B; i++) scratch[i] = sb[sOff + i]!;
        scratch[at] = t2;
        scratch.sort((x, y) => x - y);
        if (frozenDeadlock(wall, goal, dead, W, scratch, t2)) continue;
        pushed = true;
        childEst = est(scratch);
        childBoxes = scratch;
      }
      const seen = tableFindCand(q, childBoxes ?? curBoxes);
      if (seen !== -1 && sg[seen]! <= g + 1) continue;
      if (cnt >= MAXN) return { status: 'budget', moves: '', expansions: expanded };
      const t = cnt++;
      sp[t] = q;
      const tOff = t * B;
      if (childBoxes !== null) {
        for (let i = 0; i < B; i++) sb[tOff + i] = childBoxes[i]!;
      } else {
        for (let i = 0; i < B; i++) sb[tOff + i] = sb[sOff + i]!;
      }
      sg[t] = g + 1;
      sh[t] = childEst;
      parent[t] = s;
      stepDi[t] = di + (pushed ? 4 : 0);
      tableSet(t);
      heapF[heapLen] = g + 1 + childEst;
      heapG[heapLen] = g + 1;
      heapI[heapLen] = t;
      heapLen++;
      hpush(heapLen - 1);
    }
    for (let i = 0; i < B; i++) boxMark[sb[sOff + i]!] = 0;
  }
  return { status: 'unsolvable', moves: '', expansions: expanded };
}

// --- room decomposition: split the level into balanced sub-levels and solve each -------------
// Last resort after every whole-level search failed. Single-cell articulation squares (free,
// boxless, goalless) are cut; the remaining components carry supply = boxes - goals; unbalanced
// components greedily absorb neighbors across cut squares until balanced. Each balanced body
// (a small independent sub-level with boxes == goals) is push-solved on its own grid, and the
// pieces are stitched with plain player walks. The stitched string is replay-checked against
// the real level before it is accepted. All budgets are expansion counts, so this stays
// deterministic; totalCap bounds the work across all bodies.
function decomposeSolve(lv: Level, totalCap: number): Outcome {
  const { W, n, wall, goal, goals } = lv;
  const H = Math.floor(n / W);
  const DD = [-W, W, -1, 1];
  const boxSet = new Set<number>(lv.boxes);
  const goalSet = new Set<number>(goals);
  const cellName = (c: number): string => `${c % W},${Math.floor(c / W)}`;

  // articulation squares
  const isCut = new Uint8Array(n);
  for (let c = 0; c < n; c++) {
    if (wall[c] || boxSet.has(c) || goalSet.has(c)) continue;
    const seen = new Uint8Array(n);
    let comps = 0;
    for (let s = 0; s < n && comps < 2; s++) {
      if (wall[s] || s === c || seen[s]) continue;
      comps++;
      const stack = [s];
      seen[s] = 1;
      while (stack.length > 0) {
        const p = stack.pop()!;
        for (const d of DD) {
          const t = p + d;
          if (t < 0 || t >= n || wall[t] || t === c || seen[t]) continue;
          seen[t] = 1;
          stack.push(t);
        }
      }
    }
    if (comps >= 2) isCut[c] = 1;
  }

  // components over free non-cut squares
  const comp = new Int32Array(n).fill(-1);
  const compCells: number[][] = [];
  const compMin: number[] = [];
  for (let s = 0; s < n; s++) {
    if (wall[s] || isCut[s] || comp[s] !== -1) continue;
    const id = compCells.length;
    const cells: number[] = [];
    const stack = [s];
    comp[s] = id;
    while (stack.length > 0) {
      const p = stack.pop()!;
      cells.push(p);
      for (const d of DD) {
        const t = p + d;
        if (t < 0 || t >= n || wall[t] || isCut[t] || comp[t] !== -1) continue;
        comp[t] = id;
        stack.push(t);
      }
    }
    cells.sort((a, b) => a - b);
    compCells.push(cells);
    compMin.push(cells[0]!);
  }
  const nc = compCells.length;
  const supply = compCells.map((cells) => {
    let v = 0;
    for (const c of cells) v += (boxSet.has(c) ? 1 : 0) - (goalSet.has(c) ? 1 : 0);
    return v;
  });

  // node graph: nodes are comps (0..nc-1) and cut squares (nc + square)
  const nNode = nc + n;
  const nodeMin = (nd: number): number => (nd < nc ? compMin[nd]! : nd - nc);
  const adj: number[][] = Array.from({ length: nNode }, () => []);
  const link = (a: number, b: number): void => {
    adj[a]!.push(b);
    adj[b]!.push(a);
  };
  for (let c = 0; c < n; c++) {
    if (wall[c] || !isCut[c]) continue;
    for (const d of DD) {
      const t = c + d;
      if (t < 0 || t >= n || wall[t]) continue;
      if (isCut[t]) link(nc + c, nc + t);
      else link(nc + c, comp[t]!);
    }
  }
  for (const a of adj) a.sort((x, y) => nodeMin(x)! - nodeMin(y)!);

  // merge closure: unbalanced comps absorb nearest comps (through absorbed cut squares)
  const parent = new Int32Array(nNode).map((_, i) => i);
  const find = (x: number): number => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]]!;
      x = parent[x];
    }
    return x;
  };
  const members = new Map<number, Set<number>>();
  for (let i = 0; i < nNode; i++) members.set(i, new Set([i]));
  const rootSupply = (root: number): number => {
    let v = 0;
    for (const nd of members.get(root)!) if (nd < nc) v += supply[nd]!;
    return v;
  };
  const startList = compCells
    .map((_, k) => k)
    .filter((k) => supply[k] !== 0)
    .sort((a, b) => compMin[a]! - compMin[b]!);
  for (const start of startList) {
    const root = find(start);
    if (rootSupply(root) === 0) continue;
    const seen = new Set<number>([root]);
    let frontier = [root];
    let done = false;
    while (!done && frontier.length > 0) {
      const next: number[] = [];
      const hits: number[] = [];
      for (const nd of frontier) {
        for (const t of adj[nd]!) {
          const rt = find(t);
          if (seen.has(rt)) continue;
          seen.add(rt);
          if (rt >= nc) {
            const a = find(start);
            members.get(a)!.add(rt);
            parent[rt] = a;
            next.push(rt);
          } else {
            next.push(rt);
            hits.push(rt);
          }
        }
      }
      next.sort((a, b) => nodeMin(a)! - nodeMin(b)!);
      for (const hit of hits.sort((a, b) => nodeMin(a)! - nodeMin(b)!)) {
        const a = find(start);
        const b = find(hit);
        if (a === b) continue;
        for (const nd of members.get(b)!) {
          members.get(a)!.add(nd);
          parent[nd] = a;
        }
        members.delete(b);
        if (rootSupply(find(start)) === 0) {
          done = true;
          break;
        }
      }
      frontier = next;
    }
    if (!done) return { status: 'budget', moves: '', expansions: 0 };
  }

  // bodies: groups holding boxes; entry = min-index boundary cut square (the player's door)
  type Body = { cells: number[]; boxes: number[]; goalCells: number[]; entry: number };
  const bodies: Body[] = [];
  const bodyCellSets: Set<number>[] = [];
  for (const [, nodes] of members) {
    const cells = new Set<number>();
    const boxes: number[] = [];
    const goalCells: number[] = [];
    let hasComp = false;
    for (const nd of nodes) {
      if (nd >= nc && !isCut[nd - nc]) continue;
      if (nd < nc) hasComp = true;
      const cs = nd < nc ? compCells[nd]! : [nd - nc];
      for (const c of cs) {
        cells.add(c);
        if (boxSet.has(c)) boxes.push(c);
        if (goalSet.has(c)) goalCells.push(c);
      }
    }
    if (!hasComp || boxes.length === 0) continue;
    bodies.push({ cells: [...cells].sort((a, b) => a - b), boxes, goalCells, entry: -1 });
    bodyCellSets.push(cells);
  }
  const cutList: number[] = [];
  for (let c = 0; c < n; c++) if (isCut[c]) cutList.push(c);
  for (let i = 0; i < bodies.length; i++) {
    const b = bodies[i]!;
    const cells = bodyCellSets[i]!;
    for (const c of cutList) {
      if (cells.has(c)) continue;
      let touches = false;
      for (const d of DD) {
        const t = c + d;
        if (t >= 0 && t < n && cells.has(t)) {
          touches = true;
          break;
        }
      }
      if (!touches) continue;
      // cut squares never hold boxes at the end of a solution (they are not goals), so a door
      // may be shared by several bodies
      b.entry = c;
      cells.add(c);
      b.cells = [...cells].sort((x, y) => x - y);
      break;
    }
    if (b.entry === -1 && cells.has(lv.player)) b.entry = lv.player;
    if (b.entry === -1) return { status: 'budget', moves: '', expansions: 0 };
  }
  bodies.sort((a, b) => a.cells[0]! - b.cells[0]!);

  // solve each body on its own grid, walk between entries on the real grid
  const subLevel = (cells: number[], boxes: number[], goalCells: number[], player: number): Level => {
    const keep = new Set(cells);
    const rows: string[] = [];
    for (let y = 0; y < H; y++) {
      let row = '';
      for (let x = 0; x < W; x++) {
        const c = y * W + x;
        if (!keep.has(c)) row += '#';
        else if (boxes.includes(c)) row += '$';
        else if (goalCells.includes(c)) row += '.';
        else if (c === player) row += '@';
        else row += ' ';
      }
      rows.push(row);
    }
    return parse(rows.join('\n') + '\n');
  };
  const walk = (from: number, to: number, boxes: Set<number>): string | null => {
    if (from === to) return '';
    const par = new Int32Array(n).fill(-2);
    par[from] = -1;
    const q = [from];
    for (let h = 0; h < q.length; h++) {
      const p = q[h]!;
      for (const d of DD) {
        const t = p + d;
        if (t < 0 || t >= n || wall[t] || boxes.has(t) || par[t] !== -2) continue;
        par[t] = p;
        if (t === to) {
          let s = '';
          for (let v = to; par[v] !== -1; v = par[v]!) {
            const dd = v - par[v]!;
            s = (dd === 1 ? 'r' : dd === -1 ? 'l' : dd === W ? 'd' : 'u') + s;
          }
          return s;
        }
        q.push(t);
      }
    }
    return null;
  };
  let player = lv.player;
  const boxNow = new Set<number>(lv.boxes);
  const pieces: string[] = [];
  let cap = totalCap;
  let used = 0;
  for (const b of bodies) {
    const w = walk(player, b.entry, boxNow);
    if (w === null) return { status: 'budget', moves: '', expansions: used };
    const bcap = Math.min(cap, DECOMP_BODY_CAP);
    if (bcap <= 0) return { status: 'budget', moves: '', expansions: used };
    const sl = subLevel(b.cells, b.boxes, b.goalCells, b.entry);
    const r = pushSolve(sl, bcap, 1);
    cap -= r.expansions;
    used += r.expansions;
    if (r.status !== 'solved' || !replayOk(sl, r.moves)) {
      return { status: r.status === 'unsolvable' ? 'unsolvable' : 'budget', moves: '', expansions: used };
    }
    pieces.push(w, r.moves);
    let p = b.entry;
    const sim = new Set<number>(b.boxes);
    for (const ch of r.moves) {
      const di = 'udlr'.indexOf(ch.toLowerCase());
      const d = DD[di]!;
      const t = p + d;
      if (sim.has(t)) {
        const t2 = t + d;
        sim.delete(t);
        sim.add(t2);
      }
      p = t;
    }
    player = p;
    for (const bb of b.boxes) boxNow.delete(bb);
    for (const bb of sim) boxNow.add(bb);
  }
  const moves = pieces.join('');
  if (!replayOk(lv, moves)) return { status: 'budget', moves: '', expansions: used };
  return { status: 'solved', moves, expansions: used };
}

// Independent replay check: every step must be legal, every box must end on a goal.
export function replayOk(lv: Level, moves: string): boolean {
  const { W, n, wall, goal } = lv;
  const dd = [DY[0]! * W + DX[0]!, DY[1]! * W + DX[1]!, DY[2]! * W + DX[2]!, DY[3]! * W + DX[3]!];
  const boxes = new Set<number>(lv.boxes);
  let p = lv.player;
  for (const ch of moves) {
    const di = 'udlr'.indexOf(ch.toLowerCase());
    const d = dd[di]!;
    const t = p + d;
    if (t < 0 || t >= n || wall[t]) return false;
    if (boxes.has(t)) {
      const t2 = t + d;
      if (t2 < 0 || t2 >= n || wall[t2] || boxes.has(t2)) return false;
      boxes.delete(t);
      boxes.add(t2);
    }
    p = t;
  }
  for (const b of boxes) if (!goal[b]) return false;
  return true;
}

function solveLevel(text: string): { moves: string; note: string } {
  const lv = parse(text);
  const big = lv.boxes.length >= BIG_LEVEL_BOXES;
  if (!big) {
    const opt = pushSolve(lv, PUSH_EXPANSIONS, 1);
    if (opt.status === 'solved' && replayOk(lv, opt.moves)) {
      return { moves: opt.moves, note: `push-A* x${opt.expansions}` };
    }
    if (opt.status !== 'unsolvable') {
      const fast = pushSolve(lv, PUSH_W_EXPANSIONS, PUSH_WEIGHT);
      if (fast.status === 'solved' && replayOk(lv, fast.moves)) {
        return { moves: fast.moves, note: `push-A* w${PUSH_WEIGHT} x${fast.expansions}` };
      }
      const gr = greedySolve(lv, GREEDY_EXPANSIONS);
      if (gr.status === 'solved' && replayOk(lv, gr.moves)) {
        return { moves: gr.moves, note: `greedy x${gr.expansions}` };
      }
      const mv = moveSolve(lv, MOVE_EXPANSIONS);
      if (mv.status === 'solved' && replayOk(lv, mv.moves)) {
        return { moves: mv.moves, note: `fallback move-A* x${mv.expansions}` };
      }
      const dc = decomposeSolve(lv, DECOMP_TOTAL);
      if (dc.status === 'solved' && replayOk(lv, dc.moves)) {
        return { moves: dc.moves, note: `decomposed x${dc.expansions}` };
      }
    }
    return { moves: '', note: `gave up (${opt.status})` };
  }
  // Big board: the cheap weighted pass usually decides (i004 solves in 53 expansions), so run it
  // first; the optimal pass gets only a reduced budget, and the move-level fallback stays last.
  const fast = pushSolve(lv, BIG_W_EXPANSIONS, PUSH_WEIGHT);
  if (fast.status === 'solved' && replayOk(lv, fast.moves)) {
    return { moves: fast.moves, note: `push-A* w${PUSH_WEIGHT} x${fast.expansions}` };
  }
  if (fast.status !== 'unsolvable') {
    const opt = pushSolve(lv, BIG_PUSH_EXPANSIONS, 1);
    if (opt.status === 'solved' && replayOk(lv, opt.moves)) {
      return { moves: opt.moves, note: `push-A* x${opt.expansions}` };
    }
    const gr = greedySolve(lv, GREEDY_EXPANSIONS);
    if (gr.status === 'solved' && replayOk(lv, gr.moves)) {
      return { moves: gr.moves, note: `greedy x${gr.expansions}` };
    }
    const mv = moveSolve(lv, MOVE_EXPANSIONS);
    if (mv.status === 'solved' && replayOk(lv, mv.moves)) {
      return { moves: mv.moves, note: `fallback move-A* x${mv.expansions}` };
    }
  }
  const dc = decomposeSolve(lv, DECOMP_TOTAL);
  if (dc.status === 'solved' && replayOk(lv, dc.moves)) {
    return { moves: dc.moves, note: `decomposed x${dc.expansions}` };
  }
  return { moves: '', note: `gave up (${fast.status})` };
}

const countPushes = (m: string): number => {
  let c = 0;
  for (const ch of m) if (ch >= 'A' && ch <= 'Z') c++;
  return c;
};

if (pathToFileURL(process.argv[1]!).href === import.meta.url) {
  mkdirSync(OUT, { recursive: true });
  for (const id of readdirSync(IN).sort()) {
    let text: string;
    try {
      text = readFileSync(join(IN, id, 'level.txt'), 'utf8');
    } catch {
      continue;
    }
    const { moves, note } = solveLevel(text);
    mkdirSync(join(OUT, id), { recursive: true });
    writeFileSync(join(OUT, id, 'solution.txt'), `${moves}\n`);
    console.error(`${id}: ${note} pushes=${countPushes(moves)} moves=${moves.length}`);
  }
}
