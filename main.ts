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
//   4. empty solution.txt as the last resort.
// Every candidate solution is replay-checked internally before it is written.
//
// Determinism: budgets are node-expansion counts (never wall-clock), ids/boxes/goals are sorted,
// heap tie-breaks are fixed (f, state key, node index), no Math.random / Date / network.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const IN = process.argv[2] ?? '/input';
const OUT = process.argv[3] ?? '/output';
const PUSH_EXPANSIONS = 250_000;
const PUSH_W_EXPANSIONS = 100_000;
const PUSH_WEIGHT = 3;
const GREEDY_EXPANSIONS = 100_000;
const MOVE_EXPANSIONS = 150_000;
const BIG = 10_000_000;

// deltas in row-major grid order: u, d, l, r
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const CH = ['u', 'd', 'l', 'r'];
const CHU = ['U', 'D', 'L', 'R'];

type Level = {
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

function parse(text: string): Level {
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
function pushSolve(lv: Level, cap: number, weight: number): Outcome {
  const { W, n, wall, goal, goals, dead, minDist, regionOf, exact } = lv;
  const rootBoxes = lv.boxes.slice();
  const rootPlayer = lv.player;
  const G = goals.length;
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

  const nKey: string[] = [];
  const nPlayerA: number[] = [];
  const nBoxes: number[][] = [];
  const nG: number[] = [];
  const nH: number[] = [];
  const nParent: number[] = [];
  const nStep: string[] = [];
  const best = new Map<string, number>();
  const heap: [number, number][] = [];
  const less = (a: [number, number], b: [number, number]): boolean =>
    a[0] < b[0] ||
    (a[0] === b[0] && (nKey[a[1]]! < nKey[b[1]]! || (nKey[a[1]] === nKey[b[1]] && a[1] < b[1])));
  const hpush = (e: [number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const hpop = (): [number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };
  const commit = (key: string, playerA: number, bs: number[], g: number, h: number, parent: number, step: string): void => {
    const t = nKey.length;
    nKey.push(key);
    nPlayerA.push(playerA);
    nBoxes.push(bs);
    nG.push(g);
    nH.push(h);
    nParent.push(parent);
    nStep.push(step);
    best.set(key, t);
    hpush([g + weight * h, t]);
  };

  const scratch: number[] = new Array(40).fill(0);
  if (rootBoxes.length > G || rootBoxes.some((b) => minDist[b] === Infinity)) {
    return { status: 'unsolvable', moves: '', expansions: 0 };
  }
  if (rootBoxes.every((b) => goal[b])) return { status: 'solved', moves: '', expansions: 0 };
  const rootCanon = flood(ca, rootPlayer, rootBoxes);
  {
    const h0 = hOf(rootBoxes, rootPlayer);
    if (h0 >= BIG) return { status: 'unsolvable', moves: '', expansions: 0 };
    commit(String.fromCharCode(rootCanon, ...rootBoxes), rootPlayer, rootBoxes, 0, h0, -1, '');
  }
  let expanded = 0;
  while (heap.length > 0) {
    const [, s] = hpop();
    if (best.get(nKey[s]!) !== s) continue;
    if (nH[s] === 0) {
      let m = '';
      for (let t = s; t > 0; t = nParent[t]!) m = nStep[t]! + m;
      return { status: 'solved', moves: m, expansions: expanded };
    }
    if (expanded >= cap) return { status: 'budget', moves: '', expansions: expanded };
    expanded++;
    const boxes = nBoxes[s]!;
    const B = boxes.length;
    const g1 = nG[s]! + 1;
    flood(pa, nPlayerA[s]!, boxes);
    const pStamp = pa.stamp;
    const pCur = pa.cur;
    for (let bi = 0; bi < B; bi++) {
      const b = boxes[bi]!;
      for (let di = 0; di < 4; di++) {
        const d = DD[di]!;
        const from = b - d;
        const to = b + d;
        if (wall[from] !== 0 || pStamp[from] !== pCur) continue;
        if (wall[to] !== 0 || boxes.indexOf(to) >= 0 || dead[to]) continue;
        scratch.length = B;
        for (let i = 0; i < B; i++) scratch[i] = boxes[i]!;
        scratch[bi] = to;
        scratch.sort((x, y) => x - y);
        if (frozenDeadlock(wall, goal, dead, W, scratch, to)) continue;
        const childCanon = flood(ca, b, scratch);
        const key = String.fromCharCode(childCanon, ...scratch);
        const old = best.get(key);
        if (old !== undefined && nG[old]! <= g1) continue;
        const h = hOf(scratch, b);
        if (h >= BIG) continue; // some box can reach no goal from here: unsolvable configuration
        commit(key, b, scratch.slice(), g1, h, s, walkTo(pa, from) + CHU[di]);
      }
    }
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

// Valid solutions, not always push-optimal. Last solver in the chain.
function moveSolve(lv: Level, cap: number): Outcome {
  const { W, n, wall, goal, goals, dead, player } = lv;
  const DD = [DY[0]! * W + DX[0]!, DY[1]! * W + DX[1]!, DY[2]! * W + DX[2]!, DY[3]! * W + DX[3]!];
  const goalsSorted = goals.slice().sort((a, b) => a - b);
  const manh = (a: number, b: number): number =>
    Math.abs((a % W) - (b % W)) + Math.abs(Math.floor(a / W) - Math.floor(b / W));
  const dist = (b: number): number => Math.min(...goalsSorted.map((g) => manh(g, b)));
  const est = (bs: number[]): number => bs.reduce((a, b) => a + dist(b), 0);
  const key = (p: number, bs: number[]): string => String.fromCharCode(p, ...bs);
  const sp: number[] = [player];
  const sb: number[][] = [lv.boxes.slice()];
  const sg: number[] = [0];
  const parent: number[] = [-1];
  const step: string[] = [''];
  const best = new Map<string, number>([[key(player, sb[0]!), 0]]);
  const heap: [number, number, number][] = [[est(sb[0]!), 0, 0]];
  const less = (a: [number, number, number], b: [number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const hpush = (e: [number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const hpop = (): [number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };
  let expanded = 0;
  while (heap.length > 0) {
    const [, g, s] = hpop();
    if (best.get(key(sp[s]!, sb[s]!)) !== s) continue;
    const bs = sb[s]!;
    if (bs.every((b) => goal[b])) {
      let moves = '';
      for (let t = s; t > 0; t = parent[t]!) moves = step[t]! + moves;
      return { status: 'solved', moves, expansions: expanded };
    }
    if (expanded >= cap) return { status: 'budget', moves: '', expansions: expanded };
    expanded++;
    const p = sp[s]!;
    for (let di = 0; di < 4; di++) {
      const d = DD[di]!;
      const q = p + d;
      if (q < 0 || q >= n || wall[q]) continue;
      let nb = bs;
      let move = CH[di]!;
      const at = bs.indexOf(q);
      if (at >= 0) {
        const t2 = q + d;
        if (t2 < 0 || t2 >= n || wall[t2] || bs.includes(t2) || dead[t2]) continue;
        nb = bs.slice();
        nb[at] = t2;
        nb.sort((a, b) => a - b);
        if (frozenDeadlock(wall, goal, dead, W, nb, t2)) continue;
        move = CHU[di]!;
      }
      const k = key(q, nb);
      const seen = best.get(k);
      if (seen !== undefined && sg[seen]! <= g + 1) continue;
      const t = sp.length;
      sp.push(q);
      sb.push(nb);
      sg.push(g + 1);
      parent.push(s);
      step.push(move);
      best.set(k, t);
      hpush([g + 1 + est(nb), g + 1, t]);
    }
  }
  return { status: 'unsolvable', moves: '', expansions: expanded };
}

// Independent replay check: every step must be legal, every box must end on a goal.
function replayOk(lv: Level, moves: string): boolean {
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
  }
  return { moves: '', note: `gave up (${opt.status})` };
}

const countPushes = (m: string): number => {
  let c = 0;
  for (const ch of m) if (ch >= 'A' && ch <= 'Z') c++;
  return c;
};

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
