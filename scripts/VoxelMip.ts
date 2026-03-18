import * as THREE from "three";

export const VOXEL_RADIUS = 0.5;

// ── Bounding sphere radius for a mip cell at level L ─────────────────────────
// A cell at level L covers a cube of side 2^L voxels.
// Centre = cell_index * 2^L + 2^(L-1)  per axis.
// Bounding sphere radius = half-diagonal = sqrt(3) * 2^(L-1)
// We add VOXEL_RADIUS so leaf spheres (L=0, side=1) have radius VOXEL_RADIUS.
function cellRadius(level: number): number {
  if (level === 0) return VOXEL_RADIUS;
  const half = 1 << (level - 1);          // 2^(L-1)
  return Math.sqrt(3) * half + VOXEL_RADIUS;
}

// ── VoxelMip ──────────────────────────────────────────────────────────────────
export class VoxelMip {
  readonly levels:     Uint8Array[];
  readonly dims:       { w: number; h: number; d: number }[];
  readonly levelCount: number;

  static fromLevels(
    levelBuffers: ArrayBuffer[],
    dims: { w: number; h: number; d: number }[],
  ): VoxelMip {
    const inst = Object.create(VoxelMip.prototype) as VoxelMip;
    (inst as any).levels     = levelBuffers.map(b => new Uint8Array(b));
    (inst as any).dims       = dims;
    (inst as any).levelCount = levelBuffers.length;
    return inst;
  }

  static buildAsync(
    pixels: Uint8Array,
    w: number, h: number, d: number,
  ): Promise<VoxelMip> {
    return new Promise((resolve, reject) => {
      const worker = new Worker(
        new URL('./mip-worker.ts', import.meta.url),
        { type: 'module' },
      );
      worker.onmessage = (e: MessageEvent) => {
        worker.terminate();
        resolve(VoxelMip.fromLevels(e.data.levels, e.data.dims));
      };
      worker.onerror = (err) => { worker.terminate(); reject(err); };
      const copy = new Uint8Array(pixels);
      worker.postMessage({ pixels: copy, w, h, d }, [copy.buffer]);
    });
  }

  constructor(pixels: Uint8Array, w: number, h: number, d: number) {
    this.levels = [];
    this.dims   = [];

    const l0 = new Uint8Array(w * h * d);
    for (let i = 0; i < w * h * d; i++) l0[i] = pixels[i * 4 + 3] > 0 ? 1 : 0;
    this.levels.push(l0);
    this.dims.push({ w, h, d });

    let cw = w, ch = h, cd = d;
    while (cw > 1 || ch > 1 || cd > 1) {
      const pw = cw, ph = ch, pd = cd;
      const prev = this.levels[this.levels.length - 1];
      cw = Math.ceil(cw / 2); ch = Math.ceil(ch / 2); cd = Math.ceil(cd / 2);
      const cur = new Uint8Array(cw * ch * cd);
      for (let tz = 0; tz < cd; tz++) for (let ty = 0; ty < ch; ty++) for (let tx = 0; tx < cw; tx++) {
        let occ = 0;
        for (let dz = 0; dz < 2 && !occ; dz++) { const pz = tz*2+dz; if (pz >= pd) continue;
        for (let dy = 0; dy < 2 && !occ; dy++) { const py = ty*2+dy; if (py >= ph) continue;
        for (let dx = 0; dx < 2 && !occ; dx++) { const px = tx*2+dx; if (px >= pw) continue;
          if (prev[px + pw*(py + ph*pz)]) occ = 1;
        }}}
        cur[tx + cw*(ty + ch*tz)] = occ;
      }
      this.levels.push(cur);
      this.dims.push({ w: cw, h: ch, d: cd });
    }
    this.levelCount = this.levels.length;
  }

  isSolid(tx: number, ty: number, tz: number): boolean {
    const { w, h, d } = this.dims[0];
    if (tx < 0 || tx >= w || ty < 0 || ty >= h || tz < 0 || tz >= d) return false;
    return this.levels[0][tx + w*(ty + h*tz)] === 1;
  }
}

// ── Contact ───────────────────────────────────────────────────────────────────
export interface Contact {
  point:  THREE.Vector3;
  normal: THREE.Vector3;
  depth:  number;
  localA: THREE.Vector3;
  localB: THREE.Vector3;
}

// ── Sphere-only recursive collision ──────────────────────────────────────────
// For a mip cell (level, cx, cy, cz), the bounding sphere in LOCAL space is:
//   centre = (cx + 0.5) * side,  (cy + 0.5) * side,  (cz + 0.5) * side
//   where side = 2^level
//   radius = cellRadius(level)
//
// We transform centres to WORLD space (one vec3 multiply each), then just
// compare distance vs sum-of-radii. No matrix inversions, no corner loops.

// Pre-allocated scratch — avoids GC in hot path
const _wA = new THREE.Vector3();
const _wB = new THREE.Vector3();
const _n  = new THREE.Vector3();

export function queryContacts(
  mipA: VoxelMip, mipB: VoxelMip,
  mwA:  THREE.Matrix4, mwB: THREE.Matrix4,
): Contact[] {
  const out: Contact[] = [];
  const topA = mipA.levelCount - 1;
  const topB = mipB.levelCount - 1;
  _recurse(mipA, topA, 0, 0, 0, mipB, topB, 0, 0, 0, mwA, mwB, out);
  return out;
}

function _recurse(
  mipA: VoxelMip, lA: number, cxA: number, cyA: number, czA: number,
  mipB: VoxelMip, lB: number, cxB: number, cyB: number, czB: number,
  mwA: THREE.Matrix4, mwB: THREE.Matrix4,
  out: Contact[],
): void {
  // ── Occupancy check — prune empty cells immediately ───────────────────────
  const { w: wA, h: hA, d: dA } = mipA.dims[lA];
  const { w: wB, h: hB, d: dB } = mipB.dims[lB];
  if (cxA >= wA || cyA >= hA || czA >= dA) return;
  if (cxB >= wB || cyB >= hB || czB >= dB) return;
  if (!mipA.levels[lA][cxA + wA*(cyA + hA*czA)]) return;
  if (!mipB.levels[lB][cxB + wB*(cyB + hB*czB)]) return;

  // ── Sphere overlap test ───────────────────────────────────────────────────
  // Cell centre in local space = (idx + 0.5) * 2^level
  const sideA = 1 << lA, sideB = 1 << lB;
  _wA.set(
    (cxA + 0.5) * sideA,
    (cyA + 0.5) * sideA,
    (czA + 0.5) * sideA,
  ).applyMatrix4(mwA);
  _wB.set(
    (cxB + 0.5) * sideB,
    (cyB + 0.5) * sideB,
    (czB + 0.5) * sideB,
  ).applyMatrix4(mwB);

  const rA   = cellRadius(lA);
  const rB   = cellRadius(lB);
  const dist = _wA.distanceTo(_wB);
  if (dist > rA + rB) return;   // spheres don't overlap — prune whole subtree

  // ── Both leaves → voxel sphere-sphere contact ─────────────────────────────
  if (lA === 0 && lB === 0) {
    const minDist = VOXEL_RADIUS * 2;
    if (dist < minDist && dist > 1e-6) {
      _n.subVectors(_wA, _wB).divideScalar(dist);
      out.push({
        point:  _wB.clone().addScaledVector(_n, VOXEL_RADIUS),
        normal: _n.clone(),
        depth:  minDist - dist,
        localA: new THREE.Vector3((cxA+0.5)*sideA, (cyA+0.5)*sideA, (czA+0.5)*sideA),
        localB: new THREE.Vector3((cxB+0.5)*sideB, (cyB+0.5)*sideB, (czB+0.5)*sideB),
      });
    }
    return;
  }

  // ── Recurse — split the coarser level ────────────────────────────────────
  if (lA >= lB && lA > 0) {
    for (let dz = 0; dz < 2; dz++)
    for (let dy = 0; dy < 2; dy++)
    for (let dx = 0; dx < 2; dx++)
      _recurse(
        mipA, lA-1, cxA*2+dx, cyA*2+dy, czA*2+dz,
        mipB, lB,   cxB,      cyB,      czB,
        mwA, mwB, out,
      );
  } else {
    for (let dz = 0; dz < 2; dz++)
    for (let dy = 0; dy < 2; dy++)
    for (let dx = 0; dx < 2; dx++)
      _recurse(
        mipA, lA,   cxA,      cyA,      czA,
        mipB, lB-1, cxB*2+dx, cyB*2+dy, czB*2+dz,
        mwA, mwB, out,
      );
  }
}