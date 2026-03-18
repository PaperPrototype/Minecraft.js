import * as THREE from "three";

// ── Voxel sphere radius ───────────────────────────────────────────────────────
// Each voxel is treated as a sphere of this radius for collision.
// 0.5 means the sphere exactly touches adjacent voxel centres.
export const VOXEL_RADIUS = 0.5;

// ── Node ──────────────────────────────────────────────────────────────────────
export interface OctreeNode {
  // AABB in local voxel space
  minX: number; minY: number; minZ: number;
  maxX: number; maxY: number; maxZ: number;

  // Leaf: voxel centre (integer + 0.5 offset)
  isLeaf:   boolean;
  cx: number; cy: number; cz: number;   // only valid when isLeaf

  left:  OctreeNode | null;
  right: OctreeNode | null;
}

// ── Build ─────────────────────────────────────────────────────────────────────
/**
 * Build a BVH (binary tree partitioned by longest axis median) from all
 * solid voxel centres in the given pixel buffer.
 *
 * Returns null if the buffer contains no solid voxels.
 */
export function buildOctree(
  pixels: Uint8Array,
  w: number, h: number, d: number,
): OctreeNode | null {
  // Collect all solid voxel centres
  const centres: [number, number, number][] = [];
  for (let tz = 0; tz < d; tz++) {
    for (let ty = 0; ty < h; ty++) {
      for (let tx = 0; tx < w; tx++) {
        const i = (tx + w * (ty + h * tz)) * 4;
        if (pixels[i + 3] > 0) centres.push([tx + 0.5, ty + 0.5, tz + 0.5]);
      }
    }
  }
  if (centres.length === 0) return null;
  return _build(centres, 0, centres.length);
}

function _build(
  pts: [number,number,number][],
  start: number,
  end: number,
): OctreeNode {
  // Compute AABB
  let mnx = Infinity, mny = Infinity, mnz = Infinity;
  let mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
  for (let i = start; i < end; i++) {
    const [x,y,z] = pts[i];
    if (x < mnx) mnx = x; if (x > mxx) mxx = x;
    if (y < mny) mny = y; if (y > mxy) mxy = y;
    if (z < mnz) mnz = z; if (z > mxz) mxz = z;
  }

  // Expand AABB by voxel radius so sphere sweeps are covered
  const r = VOXEL_RADIUS;
  const node: OctreeNode = {
    minX: mnx - r, minY: mny - r, minZ: mnz - r,
    maxX: mxx + r, maxY: mxy + r, maxZ: mxz + r,
    isLeaf: false, cx: 0, cy: 0, cz: 0,
    left: null, right: null,
  };

  if (end - start === 1) {
    const [x,y,z] = pts[start];
    node.isLeaf = true;
    node.cx = x; node.cy = y; node.cz = z;
    return node;
  }

  // Split on longest axis at median
  const dx = mxx - mnx, dy = mxy - mny, dz = mxz - mnz;
  const axis = dx >= dy && dx >= dz ? 0 : dy >= dz ? 1 : 2;
  const sub = pts.slice(start, end);
  sub.sort((a, b) => a[axis] - b[axis]);
  for (let i = 0; i < sub.length; i++) pts[start + i] = sub[i];

  const mid = start + Math.floor((end - start) / 2);
  node.left  = _build(pts, start, mid);
  node.right = _build(pts, mid,   end);
  return node;
}

// ── AABB overlap test ─────────────────────────────────────────────────────────
function aabbOverlap(
  a: OctreeNode, b: OctreeNode,
  // b's AABB is already in a's local space (pre-transformed)
): boolean {
  return a.minX <= b.maxX && a.maxX >= b.minX &&
         a.minY <= b.maxY && a.maxY >= b.minY &&
         a.minZ <= b.maxZ && a.maxZ >= b.minZ;
}

// ── Contact ───────────────────────────────────────────────────────────────────
export interface Contact {
  /** World-space contact point (midpoint between sphere surfaces) */
  point:  THREE.Vector3;
  /** World-space normal pointing from B into A */
  normal: THREE.Vector3;
  /** Penetration depth (> 0 means overlapping) */
  depth:  number;
  /** Local-space voxel centre on body A */
  localA: THREE.Vector3;
  /** Local-space voxel centre on body B */
  localB: THREE.Vector3;
}

// ── Broadphase + narrowphase ──────────────────────────────────────────────────
/**
 * Collect all contacts between two voxel bodies.
 *
 * @param rootA  BVH root of body A (already in A's local space)
 * @param rootB  BVH root of body B (already in B's local space)
 * @param mwA    modelWorld matrix of body A  (local→world)
 * @param mwB    modelWorld matrix of body B
 */
export function queryContacts(
  rootA: OctreeNode, rootB: OctreeNode,
  mwA: THREE.Matrix4, mwB: THREE.Matrix4,
): Contact[] {
  const contacts: Contact[] = [];

  // Transform B's tree into A's local space for AABB overlap tests
  const invA  = new THREE.Matrix4().copy(mwA).invert();
  const BtoA  = new THREE.Matrix4().multiplyMatrices(invA, mwB);

  _recurse(rootA, rootB, BtoA, mwA, mwB, contacts);
  return contacts;
}

const _tmpA  = new THREE.Vector3();
const _tmpB  = new THREE.Vector3();
const _tmpAB = new THREE.Vector3();

function _transformNode(n: OctreeNode, m: THREE.Matrix4): OctreeNode {
  // Transform the 8 corners of n's AABB through m, return new AABB
  const corners = [
    [n.minX,n.minY,n.minZ],[n.maxX,n.minY,n.minZ],
    [n.minX,n.maxY,n.minZ],[n.maxX,n.maxY,n.minZ],
    [n.minX,n.minY,n.maxZ],[n.maxX,n.minY,n.maxZ],
    [n.minX,n.maxY,n.maxZ],[n.maxX,n.maxY,n.maxZ],
  ];
  let mnx=Infinity,mny=Infinity,mnz=Infinity;
  let mxx=-Infinity,mxy=-Infinity,mxz=-Infinity;
  const v = new THREE.Vector3();
  for (const [x,y,z] of corners) {
    v.set(x,y,z).applyMatrix4(m);
    if(v.x<mnx)mnx=v.x; if(v.x>mxx)mxx=v.x;
    if(v.y<mny)mny=v.y; if(v.y>mxy)mxy=v.y;
    if(v.z<mnz)mnz=v.z; if(v.z>mxz)mxz=v.z;
  }
  return { ...n, minX:mnx,minY:mny,minZ:mnz, maxX:mxx,maxY:mxy,maxZ:mxz };
}

function _recurse(
  nA: OctreeNode, nB: OctreeNode,
  BtoA: THREE.Matrix4,
  mwA:  THREE.Matrix4,
  mwB:  THREE.Matrix4,
  out:  Contact[],
): void {
  // Transform nB's AABB into A's local space for overlap test
  const nBinA = _transformNode(nB, BtoA);
  if (!aabbOverlap(nA, nBinA)) return;

  // Both leaves → sphere-sphere contact
  if (nA.isLeaf && nB.isLeaf) {
    // World-space positions
    _tmpA.set(nA.cx, nA.cy, nA.cz).applyMatrix4(mwA);
    _tmpB.set(nB.cx, nB.cy, nB.cz).applyMatrix4(mwB);

    const dist = _tmpA.distanceTo(_tmpB);
    const minD = VOXEL_RADIUS * 2;           // sum of radii
    if (dist < minD && dist > 1e-6) {
      const depth = minD - dist;
      const n = _tmpAB.subVectors(_tmpA, _tmpB).divideScalar(dist);
      out.push({
        point:  _tmpB.clone().addScaledVector(n, VOXEL_RADIUS),
        normal: n.clone(),
        depth,
        localA: new THREE.Vector3(nA.cx, nA.cy, nA.cz),
        localB: new THREE.Vector3(nB.cx, nB.cy, nB.cz),
      });
    }
    return;
  }

  // Recurse — split whichever node is larger
  const splitA = !nA.isLeaf && (nB.isLeaf || _volume(nA) >= _volume(nB));

  if (splitA) {
    _recurse(nA.left!,  nB, BtoA, mwA, mwB, out);
    _recurse(nA.right!, nB, BtoA, mwA, mwB, out);
  } else {
    // We need BtoA for nB's children — same matrix, nB.children share nB's space
    _recurse(nA, nB.left!,  BtoA, mwA, mwB, out);
    _recurse(nA, nB.right!, BtoA, mwA, mwB, out);
  }
}

function _volume(n: OctreeNode): number {
  return (n.maxX-n.minX) * (n.maxY-n.minY) * (n.maxZ-n.minZ);
}