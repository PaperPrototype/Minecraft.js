import * as THREE from "three";
import { VoxelMip } from "./VoxelMip";
import { VoxelPhysics } from "./VoxelPhysics";

// ── Fixed AoS map dimensions ──────────────────────────────────────────────────
export const VXL_X = 512;
export const VXL_Y = 512;
export const VXL_Z = 64;

// ── Shaders ───────────────────────────────────────────────────────────────────
const vertexShader = /* glsl */`
  varying vec3 vWorldPos;
  void main() {
    vec4 wp   = modelMatrix * vec4(position, 1.0);
    vWorldPos = wp.xyz;
    gl_Position = projectionMatrix * viewMatrix * wp;
  }
`;

const fragmentShader = /* glsl */`
  precision highp float;
  precision highp sampler3D;

  uniform sampler3D uVoxels;
  uniform vec3      uCamPos;
  uniform mat4      uInvModel;
  uniform vec3      uVolSize;
  uniform vec3      uSunDir;
  uniform vec3      uHighlight;
  uniform int       uSelected;

  varying vec3 vWorldPos;

  vec2 aabbIntersect(vec3 ro, vec3 rd, vec3 bMin, vec3 bMax) {
    vec3 inv  = 1.0 / rd;
    vec3 t0   = (bMin - ro) * inv;
    vec3 t1   = (bMax - ro) * inv;
    vec3 tNr  = min(t0, t1);
    vec3 tFr  = max(t0, t1);
    return vec2(max(max(tNr.x, tNr.y), tNr.z),
                min(min(tFr.x, tFr.y), tFr.z));
  }

  vec4 dda(vec3 ro, vec3 rd, float tStart, float tEnd,
           out vec3 normal, out ivec3 hitVoxel) {
    vec3  entry  = ro + rd * (tStart + 1e-4);
    ivec3 voxel  = ivec3(floor(entry));
    ivec3 step   = ivec3(sign(rd));
    vec3  tDelta = abs(1.0 / rd);
    vec3  nextB  = vec3(voxel) + vec3(max(step, ivec3(0)));
    vec3  tMax   = (nextB - entry) / rd;
    if (abs(rd.x) < 1e-8) tMax.x = 1e30;
    if (abs(rd.y) < 1e-8) tMax.y = 1e30;
    if (abs(rd.z) < 1e-8) tMax.z = 1e30;

    ivec3 volSz  = ivec3(uVolSize);
    normal       = vec3(0.0);
    hitVoxel     = ivec3(-1);

    const int MAX_STEPS = 1100;
    for (int i = 0; i < MAX_STEPS; i++) {
      if (any(lessThan(voxel, ivec3(0))) ||
          any(greaterThanEqual(voxel, volSz))) break;

      vec3 uvw   = (vec3(voxel) + 0.5) / uVolSize;
      vec4 texel = texture(uVoxels, uvw);
      if (texel.a > 0.5) { hitVoxel = voxel; return vec4(texel.rgb, 1.0); }

      if (tMax.x < tMax.y && tMax.x < tMax.z) {
        if (tMax.x > tEnd) break;
        tMax.x += tDelta.x; voxel.x += step.x;
        normal  = vec3(-float(step.x), 0.0, 0.0);
      } else if (tMax.y < tMax.z) {
        if (tMax.y > tEnd) break;
        tMax.y += tDelta.y; voxel.y += step.y;
        normal  = vec3(0.0, -float(step.y), 0.0);
      } else {
        if (tMax.z > tEnd) break;
        tMax.z += tDelta.z; voxel.z += step.z;
        normal  = vec3(0.0, 0.0, -float(step.z));
      }
    }
    return vec4(0.0);
  }

  void main() {
    vec3 worldRd = normalize(vWorldPos - uCamPos);
    vec3 localRo = (uInvModel * vec4(uCamPos,   1.0)).xyz;
    vec3 localRd = (uInvModel * vec4(worldRd,   0.0)).xyz;

    vec2 tHit    = aabbIntersect(localRo, localRd, vec3(0.0), uVolSize);
    if (tHit.x > tHit.y) discard;

    float tStart  = max(tHit.x, 0.0);
    float tEnd    = tHit.y;

    vec3  hitNormal;
    ivec3 hitVoxel;
    vec4  hit = dda(localRo, localRd, tStart, tEnd, hitNormal, hitVoxel);
    if (hit.a < 0.5) discard;

    vec3 worldNormal = normalize((transpose(uInvModel) * vec4(hitNormal, 0.0)).xyz);
    vec3 albedo   = hit.rgb;
    float nDotL   = max(dot(worldNormal, uSunDir), 0.0);
    float faceAO  = (hitNormal.y >  0.5) ? 1.00
                  : (hitNormal.y < -0.5) ? 0.50 : 0.75;
    vec3 colour   = albedo * faceAO * (0.20 + nDotL * 0.80);

    if (uHighlight.x >= 0.0 && hitVoxel == ivec3(uHighlight))
      colour = mix(colour, vec3(1.0, 1.0, 0.3), 0.55);
    if (uSelected == 1)
      colour = mix(colour, vec3(0.3, 0.7, 1.0), 0.08);

    float fogFac = 1.0 - exp(-tStart * 0.0018);
    colour       = mix(colour, vec3(0.502, 0.627, 0.878), clamp(fogFac, 0.0, 1.0));
    gl_FragColor = vec4(colour, 1.0);
  }
`;

// ── Helpers ───────────────────────────────────────────────────────────────────
function bgra2rgb(b: number, g: number, r: number): [number, number, number] {
  return [r, g, b];
}

function makeTex(pixels: Uint8Array, w: number, h: number, d: number): THREE.Data3DTexture {
  const tex           = new THREE.Data3DTexture(pixels, w, h, d);
  tex.format          = THREE.RGBAFormat;
  tex.type            = THREE.UnsignedByteType;
  tex.minFilter       = THREE.NearestFilter;
  tex.magFilter       = THREE.NearestFilter;
  tex.wrapS           = THREE.ClampToEdgeWrapping;
  tex.wrapT           = THREE.ClampToEdgeWrapping;
  tex.wrapR           = THREE.ClampToEdgeWrapping;
  tex.unpackAlignment = 1;
  tex.needsUpdate     = true;
  return tex;
}

function makeMat(tex: THREE.Data3DTexture, w: number, h: number, d: number): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms: {
      uVoxels:    { value: tex },
      uCamPos:    { value: new THREE.Vector3() },
      uInvModel:  { value: new THREE.Matrix4() },
      uVolSize:   { value: new THREE.Vector3(w, h, d) },
      uSunDir:    { value: new THREE.Vector3(0.55, 1.0, 0.35).normalize() },
      uHighlight: { value: new THREE.Vector3(-1, -1, -1) },
      uSelected:  { value: 0 },
    },
    side:       THREE.BackSide,
    depthWrite: true,
    depthTest:  true,
  });
}

function makeGeo(w: number, h: number, d: number): THREE.BufferGeometry {
  const geo = new THREE.BoxGeometry(w, h, d);
  geo.translate(w / 2, h / 2, d / 2);
  return geo;
}

// ── Raycast result ────────────────────────────────────────────────────────────
export interface RaycastHit {
  voxel:  THREE.Vector3;
  normal: THREE.Vector3;
  place:  THREE.Vector3;
  map:    VoxelMap;
}

// ── VoxelMap ──────────────────────────────────────────────────────────────────
/**
 * A raymarched voxel volume of arbitrary dimensions W×H×D.
 *
 * Full AoS maps use W=512, H=64, D=512.
 * Island fragments use tight dimensions matching their actual voxel extents.
 *
 * Local-space convention:
 *   texture x = voxel x  (0 … W-1)
 *   texture y = voxel y  (0 … H-1)   y=0 is sky-top for AoS maps
 *   texture z = voxel z  (0 … D-1)
 *   mesh position = world-space corner of the volume (0,0,0 of local space)
 */
export class VoxelMap extends THREE.Mesh {
  readonly mapName: string;

  // Instance dimensions (may differ from VXL_X/Y/Z for islands)
  readonly w: number;
  readonly h: number;
  readonly d: number;

  pixels: Uint8Array;
  private tex3d: THREE.Data3DTexture;
  declare material: THREE.ShaderMaterial;

  private _invModel = new THREE.Matrix4();

  /** 3D occupancy mipmap for hierarchical collision — built for island maps */
  mip:     VoxelMip | null = null;
  /** Physics state — non-null for island fragments, null for static maps */
  physics: VoxelPhysics | null = null;

  constructor(name: string, w = VXL_X, h = VXL_Z, d = VXL_Y) {
    const pixels = new Uint8Array(w * h * d * 4);
    const tex    = makeTex(pixels, w, h, d);
    const mat    = makeMat(tex, w, h, d);
    const geo    = makeGeo(w, h, d);

    super(geo, mat);

    this.mapName = name;
    this.w       = w;
    this.h       = h;
    this.d       = d;
    this.pixels  = pixels;
    this.tex3d   = tex;
  }

  // ── Loaders ──────────────────────────────────────────────────────────────

  loadFromBuffer(buffer: ArrayBuffer): void {
    const { w, h, d } = this;
    const bytes  = new Uint8Array(buffer);
    this.pixels.fill(0);

    const write = (vx: number, vy: number, vz: number, r: number, g: number, b: number) => {
      // AoS z=0 is sky → texture y = (h-1) - vz
      const idx = (vx + w * ((h - 1 - vz) + h * vy)) * 4;
      this.pixels[idx]     = r;
      this.pixels[idx + 1] = g;
      this.pixels[idx + 2] = b;
      this.pixels[idx + 3] = 255;
    };

    let offset = 0;
    for (let vy = 0; vy < VXL_Y; vy++) {
      for (let vx = 0; vx < VXL_X; vx++) {
        while (true) {
          const n         = bytes[offset];
          const s         = bytes[offset + 1];
          const e         = bytes[offset + 2];
          const numColors = n === 0 ? (e - s + 1) : (n - 1);

          for (let i = 0; i < numColors; i++) {
            const base      = offset + 4 + i * 4;
            const [r, g, b] = bgra2rgb(bytes[base], bytes[base + 1], bytes[base + 2]);
            write(vx, vy, s + i, r, g, b);
          }

          const lastBase     = numColors > 0 ? offset + 4 + (numColors - 1) * 4 : -1;
          const [lr, lg, lb] = lastBase >= 0
            ? bgra2rgb(bytes[lastBase], bytes[lastBase + 1], bytes[lastBase + 2])
            : [136, 136, 136];

          for (let bz = s + numColors; bz <= e; bz++) write(vx, vy, bz, lr, lg, lb);

          offset += 4 + numColors * 4;
          if (n === 0) break;
        }
      }
    }
    this.tex3d.needsUpdate = true;
    // Build mip off the main thread — non-blocking, slots in when ready
    VoxelMip.buildAsync(this.pixels, this.w, this.h, this.d)
      .then(mip => { this.mip = mip; })
      .catch(err => console.warn('VoxelMip build failed:', err));
  }

  private idx(tx: number, ty: number, tz: number): number {
    return (tx + this.w * (ty + this.h * tz)) * 4;
  }

  isSolid(tx: number, ty: number, tz: number): boolean {
    if (tx < 0 || tx >= this.w || ty < 0 || ty >= this.h || tz < 0 || tz >= this.d) return false;
    return this.pixels[this.idx(tx, ty, tz) + 3] > 0;
  }

  setVoxel(tx: number, ty: number, tz: number, r: number, g: number, b: number, solid: boolean): void {
    if (tx < 0 || tx >= this.w || ty < 0 || ty >= this.h || tz < 0 || tz >= this.d) return;
    const i = this.idx(tx, ty, tz);
    this.pixels[i]     = r;
    this.pixels[i + 1] = g;
    this.pixels[i + 2] = b;
    this.pixels[i + 3] = solid ? 255 : 0;
    this.tex3d.needsUpdate = true;
    if (this.mip) {
      VoxelMip.buildAsync(this.pixels, this.w, this.h, this.d)
        .then(mip => { this.mip = mip; })
        .catch(() => {});
    }
  }

  /** Rebuild mip hierarchy after bulk pixel writes */
  rebuildMip(): void {
    VoxelMip.buildAsync(this.pixels, this.w, this.h, this.d)
      .then(mip => { this.mip = mip; })
      .catch(() => {});
  }

  removeVoxel(tx: number, ty: number, tz: number): void { this.setVoxel(tx, ty, tz, 0, 0, 0, false); }
  placeVoxel (tx: number, ty: number, tz: number, r: number, g: number, b: number): void { this.setVoxel(tx, ty, tz, r, g, b, true); }

  // ── CPU DDA raycast ───────────────────────────────────────────────────────

  raycast(worldRo: THREE.Vector3, worldRd: THREE.Vector3, maxDist = 512): RaycastHit | null {
    this.updateMatrixWorld();
    const invM    = new THREE.Matrix4().copy(this.matrixWorld).invert();
    const localRo = worldRo.clone().applyMatrix4(invM);
    const localRd = worldRd.clone().transformDirection(invM);  // w=0 transform, auto-normalised

    const scale      = this.matrixWorld.getMaxScaleOnAxis();
    const localMax   = maxDist / scale;
    const { w, h, d } = this;

    // AABB slab test
    const slabT = (axis: 'x'|'y'|'z', size: number) => {
      const inv = 1 / (localRd as any)[axis];
      const t0  = (0    - (localRo as any)[axis]) * inv;
      const t1  = (size - (localRo as any)[axis]) * inv;
      return [Math.min(t0,t1), Math.max(t0,t1)] as const;
    };
    const [txn,txf] = slabT('x', w);
    const [tyn,tyf] = slabT('y', h);
    const [tzn,tzf] = slabT('z', d);
    const tEntry    = Math.max(txn, tyn, tzn);
    const tExit     = Math.min(txf, tyf, tzf);
    if (tEntry > tExit || tExit < 0) return null;

    const tStart = Math.max(tEntry, 0) + 1e-4;
    const tEnd   = Math.min(tExit, localMax);

    const p  = localRo.clone().addScaledVector(localRd, tStart);
    let vx = Math.floor(p.x), vy = Math.floor(p.y), vz = Math.floor(p.z);
    const sx = localRd.x > 0 ? 1 : -1, sy = localRd.y > 0 ? 1 : -1, sz = localRd.z > 0 ? 1 : -1;
    const tdx = Math.abs(1/localRd.x), tdy = Math.abs(1/localRd.y), tdz = Math.abs(1/localRd.z);
    let tmx = localRd.x !== 0 ? ((sx>0?vx+1:vx) - p.x)/localRd.x : 1e30;
    let tmy = localRd.y !== 0 ? ((sy>0?vy+1:vy) - p.y)/localRd.y : 1e30;
    let tmz = localRd.z !== 0 ? ((sz>0?vz+1:vz) - p.z)/localRd.z : 1e30;
    let nx = 0, ny = 0, nz = 0;

    for (let i = 0; i < 1100; i++) {
      if (vx < 0||vx >= w||vy < 0||vy >= h||vz < 0||vz >= d) break;
      if (this.isSolid(vx, vy, vz)) {
        return {
          voxel:  new THREE.Vector3(vx, vy, vz),
          normal: new THREE.Vector3(nx, ny, nz),
          place:  new THREE.Vector3(vx+nx, vy+ny, vz+nz),
          map:    this,
        };
      }
      if (tmx < tmy && tmx < tmz) {
        if (tmx > tEnd) break; tmx += tdx; vx += sx; nx=-sx; ny=0; nz=0;
      } else if (tmy < tmz) {
        if (tmy > tEnd) break; tmy += tdy; vy += sy; nx=0; ny=-sy; nz=0;
      } else {
        if (tmz > tEnd) break; tmz += tdz; vz += sz; nx=0; ny=0; nz=-sz;
      }
    }
    return null;
  }

  // ── Island extraction ─────────────────────────────────────────────────────
  /**
   * Flood-fill within radius voxels of editedVoxel (in this map's local coords).
   * Any solid voxels not connected to the bottom row or the sub-cube boundary
   * are extracted as new tight VoxelMaps and removed from this one.
   */
  extractIslands(editedVoxel: THREE.Vector3, radius = 50): VoxelMap[] {
    const { w, h, d } = this;

    // ── Sub-volume bounds ───────────────────────────────────────────────────
    const x0 = Math.max(0, Math.floor(editedVoxel.x) - radius);
    const y0 = Math.max(0, Math.floor(editedVoxel.y) - radius);
    const z0 = Math.max(0, Math.floor(editedVoxel.z) - radius);
    const x1 = Math.min(w, Math.floor(editedVoxel.x) + radius + 1);
    const y1 = Math.min(h, Math.floor(editedVoxel.y) + radius + 1);
    const z1 = Math.min(d, Math.floor(editedVoxel.z) + radius + 1);

    const SW = x1 - x0, SH = y1 - y0, SD = z1 - z0;
    if (SW <= 0 || SH <= 0 || SD <= 0) return [];

    const sidx = (lx: number, ly: number, lz: number) => lx + SW * (ly + SH * lz);
    const STOTAL = SW * SH * SD;

    // ── Build solid + visited scratch arrays ────────────────────────────────
    const solid   = new Uint8Array(STOTAL);
    const visited = new Uint8Array(STOTAL);

    for (let lz = 0; lz < SD; lz++)
      for (let ly = 0; ly < SH; ly++)
        for (let lx = 0; lx < SW; lx++)
          if (this.isSolid(x0+lx, y0+ly, z0+lz))
            solid[sidx(lx,ly,lz)] = 1;

    // ── Seed flood fill ─────────────────────────────────────────────────────
    const stack: number[] = [];
    const seed = (lx: number, ly: number, lz: number) => {
      const si = sidx(lx, ly, lz);
      if (solid[si] && !visited[si]) { visited[si] = 1; stack.push(si); }
    };

    // Absolute bedrock (y = h-1 in texture space) within sub-cube
    if (y0 + SH - 1 === h - 1)
      for (let lz = 0; lz < SD; lz++)
        for (let lx = 0; lx < SW; lx++)
          seed(lx, SH - 1, lz);

    // Sub-cube boundary faces → treated as ground-connected (exits radius)
    for (let lz = 0; lz < SD; lz++) for (let ly = 0; ly < SH; ly++) {
      if (x0 > 0)   seed(0,      ly, lz);
      if (x1 < w)   seed(SW - 1, ly, lz);
    }
    for (let lz = 0; lz < SD; lz++) for (let lx = 0; lx < SW; lx++) {
      if (y0 > 0)   seed(lx, 0,      lz);
      if (y1 < h)   seed(lx, SH - 1, lz);
    }
    for (let ly = 0; ly < SH; ly++) for (let lx = 0; lx < SW; lx++) {
      if (z0 > 0)   seed(lx, ly, 0);
      if (z1 < d)   seed(lx, ly, SD - 1);
    }

    const NX = [1,-1,0,0,0,0], NY = [0,0,1,-1,0,0], NZ = [0,0,0,0,1,-1];

    while (stack.length) {
      const si  = stack.pop()!;
      const lz  = Math.floor(si / (SW * SH));
      const rem = si % (SW * SH);
      const ly  = Math.floor(rem / SW);
      const lx  = rem % SW;
      for (let f = 0; f < 6; f++) {
        const nx = lx+NX[f], ny = ly+NY[f], nz = lz+NZ[f];
        if (nx<0||nx>=SW||ny<0||ny>=SH||nz<0||nz>=SD) continue;
        const ni = sidx(nx,ny,nz);
        if (solid[ni] && !visited[ni]) { visited[ni]=1; stack.push(ni); }
      }
    }

    // ── Label floating components ───────────────────────────────────────────
    const label = new Int32Array(STOTAL).fill(-1);
    const components: number[][] = [];

    for (let si = 0; si < STOTAL; si++) {
      if (!solid[si] || visited[si] || label[si] !== -1) continue;
      const comp: number[] = [];
      const cid = components.length;
      components.push(comp);
      const q = [si]; label[si] = cid;
      while (q.length) {
        const cur = q.pop()!; comp.push(cur);
        const lz  = Math.floor(cur / (SW * SH));
        const rem = cur % (SW * SH);
        const ly  = Math.floor(rem / SW);
        const lx  = rem % SW;
        for (let f = 0; f < 6; f++) {
          const nx = lx+NX[f], ny = ly+NY[f], nz = lz+NZ[f];
          if (nx<0||nx>=SW||ny<0||ny>=SH||nz<0||nz>=SD) continue;
          const ni = sidx(nx,ny,nz);
          if (solid[ni] && !visited[ni] && label[ni]===-1) { label[ni]=cid; q.push(ni); }
        }
      }
    }

    if (components.length === 0) return [];

    // ── Build tight VoxelMap per island ─────────────────────────────────────
    const newMaps: VoxelMap[] = [];

    for (const comp of components) {
      // ── Compute tight AABB of this component in parent coords ─────────────
      let mnx = x1, mny = y1, mnz = z1;
      let mxx = x0, mxy = y0, mxz = z0;

      for (const si of comp) {
        const lz  = Math.floor(si / (SW * SH));
        const rem = si % (SW * SH);
        const ly  = Math.floor(rem / SW);
        const lx  = rem % SW;
        const px = x0+lx, py = y0+ly, pz = z0+lz;
        if (px < mnx) mnx = px; if (px > mxx) mxx = px;
        if (py < mny) mny = py; if (py > mxy) mxy = py;
        if (pz < mnz) mnz = pz; if (pz > mxz) mxz = pz;
      }

      const CW = mxx - mnx + 1;
      const CH = mxy - mny + 1;
      const CD = mxz - mnz + 1;

      // ── Allocate tight child map ───────────────────────────────────────────
      const child = new VoxelMap(`${this.mapName}_island`, CW, CH, CD);

      // Child's world position = parent corner + sub-offset, inheriting parent transform
      // We need to map local voxel offset (mnx, mny, mnz) into world space.
      // parent.matrixWorld transforms local→world, so:
      const localOffset = new THREE.Vector3(mnx, mny, mnz);
      const worldOffset = localOffset.applyMatrix4(this.matrixWorld);
      child.position.copy(worldOffset);
      child.rotation.copy(this.rotation);
      child.scale.copy(this.scale);

      // ── Copy voxels into tight child buffer ────────────────────────────────
      for (const si of comp) {
        const lz  = Math.floor(si / (SW * SH));
        const rem = si % (SW * SH);
        const ly  = Math.floor(rem / SW);
        const lx  = rem % SW;

        // Parent texture coords
        const px = x0+lx, py = y0+ly, pz = z0+lz;
        const src = this.idx(px, py, pz);

        // Child texture coords (relative to tight min)
        const cx = px - mnx, cy = py - mny, cz = pz - mnz;
        const dst = child.idx(cx, cy, cz);

        child.pixels[dst]     = this.pixels[src];
        child.pixels[dst + 1] = this.pixels[src + 1];
        child.pixels[dst + 2] = this.pixels[src + 2];
        child.pixels[dst + 3] = this.pixels[src + 3];

        // Erase from parent
        this.pixels[src]     = 0;
        this.pixels[src + 1] = 0;
        this.pixels[src + 2] = 0;
        this.pixels[src + 3] = 0;
      }

      child.tex3d.needsUpdate = true;

      // Build mip off the main thread — small island so this is near-instant
      VoxelMip.buildAsync(child.pixels, CW, CH, CD)
        .then(mip => { child.mip = mip; })
        .catch(() => {});

      // Build physics — inertia tensor from voxel mass distribution
      child.physics = new VoxelPhysics(child.pixels, CW, CH, CD);

      // Initialise physics position = world-space CoM
      // child.position is the world corner; CoM is offset by child.physics.com (scaled)
      const sc = child.scale.x;
      child.physics.position.copy(worldOffset)
        .addScaledVector(child.physics.com, sc);
      child.physics.quaternion.copy(child.quaternion);

      // Small random tumble kick so islands don't fall perfectly straight
      child.physics.linVel.set(
        (Math.random()-0.5)*0.5,
        0.2,
        (Math.random()-0.5)*0.5,
      );
      child.physics.angVel.set(
        (Math.random()-0.5)*0.3,
        (Math.random()-0.5)*0.3,
        (Math.random()-0.5)*0.3,
      );

      newMaps.push(child);
    }

    this.tex3d.needsUpdate = true;
    return newMaps;
  }

  // ── Per-frame ────────────────────────────────────────────────────────────

  updateCamera(camera: THREE.Camera): void {
    camera.getWorldPosition(this.material.uniforms.uCamPos.value);
    this._invModel.copy(this.matrixWorld).invert();
    this.material.uniforms.uInvModel.value.copy(this._invModel);
  }

  /**
   * After physics integration, write the body's position + quaternion back
   * to the mesh transform.
   *
   * VoxelMap.position = world-space volume corner.
   * Physics.position  = world-space CoM.
   * corner = CoM - R * com_local * scale
   */
  syncFromPhysics(): void {
    const ph = this.physics; if (!ph) return;
    const sc = this.scale.x;
    // Rotate local CoM by current physics quaternion
    const rotatedCom = ph.com.clone().applyQuaternion(ph.quaternion);
    this.position.copy(ph.position).addScaledVector(rotatedCom, -sc);
    this.quaternion.copy(ph.quaternion);
  }

  setHighlight(v: THREE.Vector3 | null): void {
    const u = this.material.uniforms.uHighlight.value as THREE.Vector3;
    if (v) u.copy(v); else u.set(-1, -1, -1);
  }

  setSelected(s: boolean): void {
    this.material.uniforms.uSelected.value = s ? 1 : 0;
  }

  // Expose internal idx for extractIslands on child maps
  idx(tx: number, ty: number, tz: number): number {
    return (tx + this.w * (ty + this.h * tz)) * 4;
  }

  dispose(): void {
    this.tex3d.dispose();
    this.material.dispose();
    this.geometry.dispose();
  }
}