// ─────────────────────────────────────────────────────────────────────────────
// physics-worker.ts
//
// Runs entirely off the main thread.  No Three.js — all math is inline.
//
// Protocol (main → worker):
//   { type:'register',       id, pixels:Uint8Array, w,h,d, scale,
//                            px,py,pz, qx,qy,qz,qw }
//   { type:'registerStatic', id, pixels:Uint8Array, w,h,d,
//                            mat:[16 floats column-major] }
//   { type:'updateStatic',   id, mat:[16 floats] }
//   { type:'remove',         id }
//   { type:'step',           dt }
//   { type:'wake',           id }
//
// Protocol (worker → main):
//   { type:'transforms', bodies:[{id, mat:[16 floats column-major]}] }
// ─────────────────────────────────────────────────────────────────────────────

// ── Minimal math (no Three.js) ────────────────────────────────────────────────
type V3  = [number, number, number];
type Q4  = [number, number, number, number];  // x y z w
type M16 = Float32Array;                      // column-major 4×4

function v3(x = 0, y = 0, z = 0): V3 { return [x, y, z]; }
function vadd(a: V3, b: V3): V3 { return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
function vsub(a: V3, b: V3): V3 { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function vscale(a: V3, s: number): V3 { return [a[0]*s, a[1]*s, a[2]*s]; }
function vdot(a: V3, b: V3): number { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
function vcross(a: V3, b: V3): V3 {
  return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
}
function vlen(a: V3): number { return Math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]); }
function vnorm(a: V3): V3 { const l = vlen(a); return l > 1e-10 ? vscale(a, 1/l) : v3(); }
function vdist(a: V3, b: V3): number { return vlen(vsub(a, b)); }
function vlensq(a: V3): number { return a[0]*a[0] + a[1]*a[1] + a[2]*a[2]; }

function qnorm(q: Q4): Q4 {
  const l = Math.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
  return l > 1e-10 ? [q[0]/l, q[1]/l, q[2]/l, q[3]/l] : [0, 0, 0, 1];
}

// Transform point by column-major 4×4
function m16pt(m: M16, v: V3): V3 {
  const x=v[0], y=v[1], z=v[2];
  return [
    m[0]*x + m[4]*y + m[8]*z  + m[12],
    m[1]*x + m[5]*y + m[9]*z  + m[13],
    m[2]*x + m[6]*y + m[10]*z + m[14],
  ];
}

// Build column-major 4×4 from corner position + quaternion + uniform scale
function makeMatrix(pos: V3, q: Q4, scale: number): M16 {
  const [qx, qy, qz, qw] = q;
  const s = scale * 2;
  const m = new Float32Array(16);
  m[0]  = (1 - (qy*qy + qz*qz)*s) * scale;
  m[1]  = (qx*qy + qz*qw) * s * scale;
  m[2]  = (qx*qz - qy*qw) * s * scale;
  m[3]  = 0;
  m[4]  = (qx*qy - qz*qw) * s * scale;
  m[5]  = (1 - (qx*qx + qz*qz)*s) * scale;
  m[6]  = (qy*qz + qx*qw) * s * scale;
  m[7]  = 0;
  m[8]  = (qx*qz + qy*qw) * s * scale;
  m[9]  = (qy*qz - qx*qw) * s * scale;
  m[10] = (1 - (qx*qx + qy*qy)*s) * scale;
  m[11] = 0;
  m[12] = pos[0]; m[13] = pos[1]; m[14] = pos[2]; m[15] = 1;
  return m;
}

// ── Sym3: symmetric 3×3 matrix ────────────────────────────────────────────────
// Stored as [I00, I01, I02, I11, I12, I22]  (upper triangle)
type Sym3 = [number, number, number, number, number, number];

// Invert a symmetric 3×3 via cofactor expansion
function sym3Invert(s: Sym3): Sym3 {
  const [a, b, c, e, f, i] = [s[0], s[1], s[2], s[3], s[4], s[5]];
  const A =  e*i - f*f;
  const B = -(b*i - f*c);
  const C =  b*f - e*c;
  const E =  a*i - c*c;
  const F = -(a*f - c*b);
  const I =  a*e - b*b;
  const det = a*A + b*B + c*C;
  if (Math.abs(det) < 1e-12) return [1, 0, 0, 1, 0, 1];
  const inv = 1 / det;
  return [A*inv, B*inv, C*inv, E*inv, F*inv, I*inv];
}

// Rotate symmetric tensor:  R * S * Rᵀ   (R built from quaternion)
// Two-step: first compute T = S * Rᵀ  (i.e. T = S * R because Rᵀ has swapped row/col)
// then result = R * T.
// S is symmetric so S[i][j] == S[j][i].  We index by the upper-triangle layout:
//   s = [s00, s01, s02, s11, s12, s22]
function sym3Rotate(s: Sym3, q: Q4): Sym3 {
  const [qx, qy, qz, qw] = q;
  const x2=qx+qx, y2=qy+qy, z2=qz+qz;
  const xx=qx*x2, xy=qx*y2, xz=qx*z2, yy=qy*y2, yz=qy*z2, zz=qz*z2;
  const wx=qw*x2, wy=qw*y2, wz=qw*z2;
  // Rotation matrix R (row-major for clarity):
  const r00=1-(yy+zz), r01=xy+wz,     r02=xz-wy;
  const r10=xy-wz,     r11=1-(xx+zz), r12=yz+wx;
  const r20=xz+wy,     r21=yz-wx,     r22=1-(xx+yy);
  const [s00, s01, s02, s11, s12, s22] = s;
  // Step 1: T = S * Rᵀ
  // Rᵀ columns are R's rows, so (S*Rᵀ)[i][j] = S[i][k] * R[j][k]
  const t00=s00*r00+s01*r01+s02*r02,  t01=s00*r10+s01*r11+s02*r12,  t02=s00*r20+s01*r21+s02*r22;
  const t10=s01*r00+s11*r01+s12*r02,  t11=s01*r10+s11*r11+s12*r12,  t12=s01*r20+s11*r21+s12*r22;
  const t20=s02*r00+s12*r01+s22*r02,  t21=s02*r10+s12*r11+s22*r12,  t22=s02*r20+s12*r21+s22*r22;
  // Step 2: Result = R * T  (upper triangle only — result is symmetric)
  return [
    r00*t00 + r01*t10 + r02*t20,   // [0,0]
    r00*t01 + r01*t11 + r02*t21,   // [0,1]
    r00*t02 + r01*t12 + r02*t22,   // [0,2]
    r10*t01 + r11*t11 + r12*t21,   // [1,1]
    r10*t02 + r11*t12 + r12*t22,   // [1,2]
    r20*t02 + r21*t12 + r22*t22,   // [2,2]
  ];
}

// Multiply symmetric 3×3 by vector:  S * v
function sym3MulV(s: Sym3, v: V3): V3 {
  return [
    s[0]*v[0] + s[1]*v[1] + s[2]*v[2],
    s[1]*v[0] + s[3]*v[1] + s[4]*v[2],
    s[2]*v[0] + s[4]*v[1] + s[5]*v[2],
  ];
}

// ── Physics constants — identical to VoxelPhysics.ts ─────────────────────────
const VOXEL_MASS      = 0.1;
const GRAVITY         = 9.81;
const LINEAR_DAMPING  = 0.985;
const ANGULAR_DAMPING = 0.980;
const RESTITUTION     = 0.35;
const FRICTION        = 0.6;
const SLEEP_VEL       = 0.02;   // threshold on linVel.lengthSq() and angVel.lengthSq()
const SLEEP_FRAMES    = 90;
const VOXEL_RADIUS    = 0.5;
const FLOOR_Y         = 0;

// Baumgarte position correction — lower scale prevents overshoot oscillation.
// 0.4 (original) was too aggressive: it corrects so much each frame that the
// body re-penetrates on the next step and fires another impulse indefinitely.
const BAUMGARTE_SLOP  = 0.01;
const BAUMGARTE_SCALE = 0.1;

// ── Mip hierarchy ─────────────────────────────────────────────────────────────
interface MipLevel { data: Uint8Array; w: number; h: number; d: number; }

function buildMip(pixels: Uint8Array, w: number, h: number, d: number): MipLevel[] {
  const levels: MipLevel[] = [];
  const l0 = new Uint8Array(w * h * d);
  for (let i = 0; i < w * h * d; i++) l0[i] = pixels[i * 4 + 3] > 0 ? 1 : 0;
  levels.push({ data: l0, w, h, d });
  let cw = w, ch = h, cd = d;
  while (cw > 1 || ch > 1 || cd > 1) {
    const pw = cw, ph = ch, pd = cd;
    const prev = levels[levels.length - 1].data;
    cw = Math.ceil(cw / 2); ch = Math.ceil(ch / 2); cd = Math.ceil(cd / 2);
    const cur = new Uint8Array(cw * ch * cd);
    for (let tz = 0; tz < cd; tz++)
    for (let ty = 0; ty < ch; ty++)
    for (let tx = 0; tx < cw; tx++) {
      let occ = 0;
      for (let dz = 0; dz < 2 && !occ; dz++) { const pz = tz*2+dz; if (pz >= pd) continue;
      for (let dy = 0; dy < 2 && !occ; dy++) { const py = ty*2+dy; if (py >= ph) continue;
      for (let dx = 0; dx < 2 && !occ; dx++) { const px = tx*2+dx; if (px >= pw) continue;
        if (prev[px + pw * (py + ph * pz)]) occ = 1;
      }}}
      cur[tx + cw * (ty + ch * tz)] = occ;
    }
    levels.push({ data: cur, w: cw, h: ch, d: cd });
  }
  return levels;
}

// Bounding sphere radius for a mip cell at level L — same formula as VoxelMip.ts
function cellRadius(level: number): number {
  if (level === 0) return VOXEL_RADIUS;
  const half = 1 << (level - 1);
  return Math.sqrt(3) * half + VOXEL_RADIUS;
}

// ── Contact type ──────────────────────────────────────────────────────────────
interface Contact { point: V3; normal: V3; depth: number; }

// ── Mip contact query — same recursion as VoxelMip.queryContacts ──────────────
function queryContacts(
  mipA: MipLevel[], mipB: MipLevel[],
  mwA: M16, mwB: M16,
  out: Contact[],
): void {
  _mipRecurse(
    mipA, mipA.length - 1, 0, 0, 0,
    mipB, mipB.length - 1, 0, 0, 0,
    mwA, mwB, out,
  );
}

function _mipRecurse(
  mipA: MipLevel[], lA: number, cxA: number, cyA: number, czA: number,
  mipB: MipLevel[], lB: number, cxB: number, cyB: number, czB: number,
  mwA: M16, mwB: M16, out: Contact[],
): void {
  const { w: wA, h: hA, d: dA, data: dataA } = mipA[lA];
  const { w: wB, h: hB, d: dB, data: dataB } = mipB[lB];
  if (cxA >= wA || cyA >= hA || czA >= dA) return;
  if (cxB >= wB || cyB >= hB || czB >= dB) return;
  if (!dataA[cxA + wA * (cyA + hA * czA)]) return;
  if (!dataB[cxB + wB * (cyB + hB * czB)]) return;

  const sA = 1 << lA, sB = 1 << lB;
  const wCentA = m16pt(mwA, [(cxA+0.5)*sA, (cyA+0.5)*sA, (czA+0.5)*sA]);
  const wCentB = m16pt(mwB, [(cxB+0.5)*sB, (cyB+0.5)*sB, (czB+0.5)*sB]);

  // Scale factors from matrix column lengths (handles non-unit scale)
  const scaleA = Math.sqrt(mwA[0]*mwA[0] + mwA[1]*mwA[1] + mwA[2]*mwA[2]);
  const scaleB = Math.sqrt(mwB[0]*mwB[0] + mwB[1]*mwB[1] + mwB[2]*mwB[2]);
  const rA = cellRadius(lA) * scaleA;
  const rB = cellRadius(lB) * scaleB;

  const dist = vdist(wCentA, wCentB);
  if (dist > rA + rB) return;

  // Stop at level 1 — one level above individual voxels.
  // Each level-1 cell covers a 2×2×2 voxel block and gives one smooth contact
  // normal.  Descending to level 0 against large static terrain produces
  // hundreds of noisy, conflicting normals that cause jitter even after
  // reduction.  Level 1 gives stable surface contacts with far fewer points.
  if (lA <= 1 && lB <= 1) {
    const minD = VOXEL_RADIUS * (scaleA + scaleB) * 2;
    if (dist < minD && dist > 1e-6) {
      const n = vnorm(vsub(wCentA, wCentB));
      out.push({
        point:  vadd(wCentB, vscale(n, VOXEL_RADIUS * scaleB)),
        normal: n,
        depth:  minD - dist,
      });
    }
    return;
  }

  // Recurse — split the coarser level
  if (lA >= lB && lA > 0) {
    for (let dz = 0; dz < 2; dz++)
    for (let dy = 0; dy < 2; dy++)
    for (let dx = 0; dx < 2; dx++)
      _mipRecurse(mipA, lA-1, cxA*2+dx, cyA*2+dy, czA*2+dz, mipB, lB, cxB, cyB, czB, mwA, mwB, out);
  } else {
    for (let dz = 0; dz < 2; dz++)
    for (let dy = 0; dy < 2; dy++)
    for (let dx = 0; dx < 2; dx++)
      _mipRecurse(mipA, lA, cxA, cyA, czA, mipB, lB-1, cxB*2+dx, cyB*2+dy, czB*2+dz, mwA, mwB, out);
  }
}

// ── Body types ────────────────────────────────────────────────────────────────
interface DynamicBody {
  id:         number;
  mip:        MipLevel[];
  scale:      number;
  com:        V3;       // local-space centre of mass
  mass:       number;
  invMass:    number;
  invI_loc:   Sym3;     // inverse inertia tensor in local CoM frame
  pos:        V3;       // world-space CoM position
  quat:       Q4;
  linVel:     V3;
  angVel:     V3;
  sleeping:   boolean;
  sleepTimer: number;
  // Needed for resolveFloor — bodyMinLocalY is always 0.5 for tight maps
  // (same assumption as MapManager.stepPhysics in the original code)
}

interface StaticBody {
  id:  number;
  mip: MipLevel[];
  mat: M16;
}

const dynamics = new Map<number, DynamicBody>();
const statics  = new Map<number, StaticBody>();

// ── Body construction — matches VoxelPhysics constructor exactly ───────────────
function makeDynamic(
  id: number, pixels: Uint8Array,
  w: number, h: number, d: number,
  scale: number,
  px: number, py: number, pz: number,
  qx: number, qy: number, qz: number, qw: number,
): DynamicBody {
  // ── Mass + centre of mass ──────────────────────────────────────────────────
  let totalMass = 0;
  let cx = 0, cy = 0, cz = 0;
  for (let tz = 0; tz < d; tz++)
  for (let ty = 0; ty < h; ty++)
  for (let tx = 0; tx < w; tx++) {
    const i = (tx + w * (ty + h * tz)) * 4;
    if (pixels[i + 3] === 0) continue;
    totalMass += VOXEL_MASS;
    cx += (tx + 0.5) * VOXEL_MASS;
    cy += (ty + 0.5) * VOXEL_MASS;
    cz += (tz + 0.5) * VOXEL_MASS;
  }
  if (totalMass < 1e-9) { totalMass = VOXEL_MASS; cx = w/2; cy = h/2; cz = d/2; }
  else { cx /= totalMass; cy /= totalMass; cz /= totalMass; }
  const com: V3 = [cx, cy, cz];

  // ── Inertia tensor about CoM — identical to VoxelPhysics.ts ───────────────
  let Ixx=0, Iyy=0, Izz=0, Ixy=0, Ixz=0, Iyz=0;
  for (let tz = 0; tz < d; tz++)
  for (let ty = 0; ty < h; ty++)
  for (let tx = 0; tx < w; tx++) {
    const i = (tx + w * (ty + h * tz)) * 4;
    if (pixels[i + 3] === 0) continue;
    const rx = (tx + 0.5) - cx;
    const ry = (ty + 0.5) - cy;
    const rz = (tz + 0.5) - cz;
    Ixx += VOXEL_MASS * (ry*ry + rz*rz);
    Iyy += VOXEL_MASS * (rx*rx + rz*rz);
    Izz += VOXEL_MASS * (rx*rx + ry*ry);
    Ixy -= VOXEL_MASS * rx * ry;
    Ixz -= VOXEL_MASS * rx * rz;
    Iyz -= VOXEL_MASS * ry * rz;
  }
  const invI_loc = sym3Invert([Ixx, Ixy, Ixz, Iyy, Iyz, Izz]);
  const mip = buildMip(pixels, w, h, d);

  return {
    id, mip, scale, com,
    mass: totalMass, invMass: 1 / totalMass, invI_loc,
    pos: [px, py, pz], quat: [qx, qy, qz, qw],
    linVel: v3(), angVel: v3(),
    sleeping: false, sleepTimer: 0,
  };
}

// ── World-space inverse inertia — R * I_local⁻¹ * Rᵀ ─────────────────────────
// Direct translation of VoxelPhysics.invInertiaTensorWorld()
function invInertiaTensorWorld(b: DynamicBody): Sym3 {
  return sym3Rotate(b.invI_loc, b.quat);
}

// ── Apply impulse at contact point ────────────────────────────────────────────
// Direct translation of VoxelPhysics.applyImpulse()
function applyImpulse(b: DynamicBody, impulse: V3, r: V3): void {
  if (b.sleeping) { b.sleeping = false; b.sleepTimer = 0; }
  b.linVel = vadd(b.linVel, vscale(impulse, b.invMass));
  // Δω = I_w⁻¹ * (r × J)
  const torqueImpulse = vcross(r, impulse);
  const iInv = invInertiaTensorWorld(b);
  b.angVel = vadd(b.angVel, sym3MulV(iInv, torqueImpulse));
}

// ── Integration — direct translation of VoxelPhysics.integrate() ──────────────
function integrate(b: DynamicBody, dt: number): void {
  if (b.sleeping) return;

  // Gravity
  b.linVel[1] -= GRAVITY * dt;

  // Position
  b.pos = vadd(b.pos, vscale(b.linVel, dt));

  // Orientation:  q += 0.5 * dt * ω⊗q
  // (same formula — ω as pure quaternion multiplied onto q)
  const [qx, qy, qz, qw] = b.quat;
  const [wx, wy, wz] = b.angVel;
  const h = dt * 0.5;
  // wq = (ω as pure quat) * q
  const wqx = h * ( wx*qw + wy*qz - wz*qy);
  const wqy = h * ( wy*qw + wz*qx - wx*qz);
  const wqz = h * ( wz*qw + wx*qy - wy*qx);
  const wqw = h * (-(wx*qx + wy*qy + wz*qz));
  b.quat = qnorm([qx + wqx, qy + wqy, qz + wqz, qw + wqw]);

  // Damping
  b.linVel = vscale(b.linVel, LINEAR_DAMPING);
  b.angVel = vscale(b.angVel, ANGULAR_DAMPING);

  // Sleep check — same threshold and frame count as VoxelPhysics.ts
  if (vlensq(b.linVel) < SLEEP_VEL && vlensq(b.angVel) < SLEEP_VEL) {
    if (++b.sleepTimer > SLEEP_FRAMES) {
      b.sleeping = true;
      b.linVel = v3(); b.angVel = v3();
    }
  } else {
    b.sleepTimer = 0;
  }
}

// ── Floor collision — direct translation of VoxelPhysics.resolveFloor() ───────
// bodyMinLocalY is 0.5 (tight maps, voxel centres start at 0.5 in local space)
function resolveFloor(b: DynamicBody): void {
  const bodyMinLocalY = 0.5;
  const bottomWorld = b.pos[1] - b.com[1] * b.scale
                    + bodyMinLocalY * b.scale
                    - VOXEL_RADIUS * b.scale;

  if (bottomWorld >= FLOOR_Y) return;

  const pen = FLOOR_Y - bottomWorld;
  b.pos[1] += pen;

  if (b.linVel[1] < 0) {
    // Reflect vertical velocity with restitution
    b.linVel[1] = -b.linVel[1] * RESTITUTION;
    // Friction on horizontal components
    b.linVel[0] *= (1 - FRICTION * 0.1);
    b.linVel[2] *= (1 - FRICTION * 0.1);
  }
  if (b.sleeping) { b.sleeping = false; b.sleepTimer = 0; }
}

// ── Contact reduction ─────────────────────────────────────────────────────────
// The mip traversal can return O(100s) of leaf contacts for large overlapping
// surfaces.  Applying a full impulse per contact multiplies the energy injected
// by the contact count, which causes the violent fling-away.
//
// Fix: reduce the contact set to a single representative contact by averaging
// position, normal, and taking the maximum depth.  This matches what the
// original single-map octree naturally produced (it stopped at the first
// overlapping leaf cluster, not every leaf pair).
function reduceContacts(contacts: Contact[]): Contact {
  let px=0, py=0, pz=0;
  let nx=0, ny=0, nz=0;
  let maxDepth = 0;
  const n = contacts.length;
  for (const c of contacts) {
    px += c.point[0];  py += c.point[1];  pz += c.point[2];
    nx += c.normal[0]; ny += c.normal[1]; nz += c.normal[2];
    if (c.depth > maxDepth) maxDepth = c.depth;
  }
  const inv = 1 / n;
  const nl = Math.sqrt(nx*nx + ny*ny + nz*nz);
  const ns: V3 = nl > 1e-10 ? [nx/nl, ny/nl, nz/nl] : [0, 1, 0];
  return {
    point:  [px*inv, py*inv, pz*inv],
    normal: ns,
    depth:  maxDepth,
  };
}

// ── Contact resolution — direct translation of resolveContacts() ───────────────
// Single pass, Baumgarte correction.
// Multiple contacts are reduced to one representative to avoid energy explosion.
function resolveContacts(a: DynamicBody, b: DynamicBody | null, contacts: Contact[]): void {
  if (contacts.length === 0) return;
  // Reduce to one contact so impulse is applied exactly once per collision pair
  const c = contacts.length === 1 ? contacts[0] : reduceContacts(contacts);

  const rA = vsub(c.point, a.pos);
  const rB = b ? vsub(c.point, b.pos) : v3();

  // Relative velocity at contact point (linear + angular contributions)
  let relV = vadd(a.linVel, vcross(a.angVel, rA));
  if (b) relV = vsub(relV, vadd(b.linVel, vcross(b.angVel, rB)));

  const vn = vdot(relV, c.normal);
  if (vn > 0) return;  // separating — skip

  // Impulse denominator: 1/mA + (rA×n)·I_A⁻¹(rA×n) + same for B
  const iA    = invInertiaTensorWorld(a);
  const rAxN  = vcross(rA, c.normal);
  let denom   = a.invMass + vdot(sym3MulV(iA, rAxN), rAxN);

  if (b) {
    const iB   = invInertiaTensorWorld(b);
    const rBxN = vcross(rB, c.normal);
    denom += b.invMass + vdot(sym3MulV(iB, rBxN), rBxN);
  }
  if (denom < 1e-10) return;

  const j       = -(1 + RESTITUTION) * vn / denom;
  const impulse = vscale(c.normal, j);
  applyImpulse(a, impulse, rA);
  if (b) applyImpulse(b, vscale(impulse, -1), rB);

  // Position correction (Baumgarte) — slop 0.005, scale 0.4
  const corr      = Math.max(c.depth - BAUMGARTE_SLOP, 0) * BAUMGARTE_SCALE;
  const totalInvM = a.invMass + (b ? b.invMass : 0);
  if (totalInvM > 1e-10) {
    a.pos = vadd(a.pos, vscale(c.normal,  corr * a.invMass / totalInvM));
    if (b) b.pos = vadd(b.pos, vscale(c.normal, -corr * b.invMass / totalInvM));
  }
}

// ── Build world matrix from dynamic body ──────────────────────────────────────
// The Three.js mesh corner is at (CoM world pos) - R * com_local * scale
function bodyMatrix(b: DynamicBody): M16 {
  // Rotate the local CoM offset into world space then subtract
  const [cx, cy, cz] = b.com;
  const rotCom: V3 = [
    (1 - 2*(b.quat[1]*b.quat[1] + b.quat[2]*b.quat[2])) * cx * b.scale
      + 2*(b.quat[0]*b.quat[1] - b.quat[2]*b.quat[3]) * cy * b.scale
      + 2*(b.quat[0]*b.quat[2] + b.quat[1]*b.quat[3]) * cz * b.scale,
    2*(b.quat[0]*b.quat[1] + b.quat[2]*b.quat[3]) * cx * b.scale
      + (1 - 2*(b.quat[0]*b.quat[0] + b.quat[2]*b.quat[2])) * cy * b.scale
      + 2*(b.quat[1]*b.quat[2] - b.quat[0]*b.quat[3]) * cz * b.scale,
    2*(b.quat[0]*b.quat[2] - b.quat[1]*b.quat[3]) * cx * b.scale
      + 2*(b.quat[1]*b.quat[2] + b.quat[0]*b.quat[3]) * cy * b.scale
      + (1 - 2*(b.quat[0]*b.quat[0] + b.quat[1]*b.quat[1])) * cz * b.scale,
  ];
  const cornerPos = vsub(b.pos, rotCom);
  return makeMatrix(cornerPos, b.quat, b.scale);
}

// ── Physics step ──────────────────────────────────────────────────────────────
function step(dt: number): void {
  const dArr = [...dynamics.values()];

  // 1. Integrate all dynamic bodies
  for (const b of dArr) integrate(b, dt);

  // 2. Floor constraint
  for (const b of dArr) resolveFloor(b);

  // 3+4. Dynamic vs dynamic
  for (let i = 0; i < dArr.length; i++) {
    for (let j = i + 1; j < dArr.length; j++) {
      const a = dArr[i], b = dArr[j];
      if (a.sleeping && b.sleeping) continue;
      // Broad-phase: top-level mip sphere vs sphere
      const rA = cellRadius(a.mip.length - 1) * a.scale;
      const rB = cellRadius(b.mip.length - 1) * b.scale;
      if (vdist(a.pos, b.pos) > rA + rB) continue;
      const mwA = bodyMatrix(a), mwB = bodyMatrix(b);
      const contacts: Contact[] = [];
      queryContacts(a.mip, b.mip, mwA, mwB, contacts);
      if (contacts.length) resolveContacts(a, b, contacts);
    }
  }

  // 3+4. Dynamic vs static
  for (const a of dArr) {
    if (a.sleeping) continue;
    const mwA = bodyMatrix(a);
    const rA  = cellRadius(a.mip.length - 1) * a.scale;
    for (const s of statics.values()) {
      // Centre of the static top-level mip cell in world space.
      // The top level is a single cell covering the whole volume, so its
      // local centre is (w/2, h/2, d/2) at level (mip.length-1), i.e.
      // cell (0,0,0) with side = 2^topLevel.  Use m16pt to transform it.
      const topL  = s.mip.length - 1;
      const side  = 1 << topL;
      const sPos  = m16pt(s.mat, [side * 0.5, side * 0.5, side * 0.5] as V3);
      const scaleS = Math.sqrt(s.mat[0]*s.mat[0] + s.mat[1]*s.mat[1] + s.mat[2]*s.mat[2]);
      const rS    = cellRadius(topL) * scaleS;
      if (vdist(a.pos, sPos) > rA + rS) continue;
      const contacts: Contact[] = [];
      queryContacts(a.mip, s.mip, mwA, s.mat, contacts);
      if (contacts.length) resolveContacts(a, null, contacts);
    }
  }

  // 5. Post transforms back to main thread
  const out: { id: number; mat: number[] }[] = [];
  for (const b of dynamics.values()) {
    out.push({ id: b.id, mat: Array.from(bodyMatrix(b)) });
  }
  self.postMessage({ type: 'transforms', bodies: out });
}

// ── Message handler ───────────────────────────────────────────────────────────
self.onmessage = (e: MessageEvent) => {
  const msg = e.data;
  switch (msg.type) {
    case 'register': {
      const b = makeDynamic(
        msg.id, new Uint8Array(msg.pixels),
        msg.w, msg.h, msg.d, msg.scale,
        msg.px, msg.py, msg.pz,
        msg.qx, msg.qy, msg.qz, msg.qw,
      );
      dynamics.set(msg.id, b);
      break;
    }
    case 'registerStatic': {
      const mip = buildMip(new Uint8Array(msg.pixels), msg.w, msg.h, msg.d);
      statics.set(msg.id, { id: msg.id, mip, mat: new Float32Array(msg.mat) });
      break;
    }
    case 'updateStatic': {
      const s = statics.get(msg.id);
      if (s) s.mat = new Float32Array(msg.mat);
      break;
    }
    case 'remove': {
      dynamics.delete(msg.id);
      statics.delete(msg.id);
      break;
    }
    case 'wake': {
      const b = dynamics.get(msg.id);
      if (b) { b.sleeping = false; b.sleepTimer = 0; }
      break;
    }
    case 'step': {
      step(msg.dt);
      break;
    }
  }
};