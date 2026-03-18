import * as THREE from "three";

// ── Volume dimensions ─────────────────────────────────────────────────────────
const VXL_X = 512;   // map width
const VXL_Y = 512;   // map depth  (vxl y → world Z)
const VXL_Z = 64;    // voxel depth (vxl z=0 sky, z=63 bedrock → world Y)

// World-space AABB
// vxl(x, z, y) → world(x,  VXL_Z - 1 - z,  y)
// So world Y ∈ [0, VXL_Z), world X ∈ [0, VXL_X), world Z ∈ [0, VXL_Y)
export const VOLUME_MIN = new THREE.Vector3(0,     0,     0);
export const VOLUME_MAX = new THREE.Vector3(VXL_X, VXL_Z, VXL_Y);
export const VOLUME_SIZE = VOLUME_MAX.clone().sub(VOLUME_MIN);

// ── Vertex shader ─────────────────────────────────────────────────────────────
// Passes world-space position to the fragment shader for the ray direction.
const vertexShader = /* glsl */`
  varying vec3 vWorldPos;

  void main() {
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPos     = worldPos.xyz;
    gl_Position   = projectionMatrix * viewMatrix * worldPos;
  }
`;

// ── Fragment shader ───────────────────────────────────────────────────────────
const fragmentShader = /* glsl */`
  precision highp float;
  precision highp sampler3D;

  uniform sampler3D uVoxels;   // RGBA volume — a > 0.5 means solid
  uniform vec3      uCamPos;   // world-space camera position
  uniform vec3      uVolMin;
  uniform vec3      uVolMax;
  uniform vec3      uVolSize;  // uVolMax - uVolMin  (== vec3(512, 64, 512))
  uniform vec3      uSunDir;   // normalised sun direction (world space)

  varying vec3 vWorldPos;      // world-space position on the cube surface

  // ── AABB slab intersection ─────────────────────────────────────────────────
  // Returns vec2(tEntry, tExit). Ray misses when tEntry > tExit.
  vec2 aabbIntersect(vec3 ro, vec3 rd) {
    vec3 invRd = 1.0 / rd;
    vec3 t0    = (uVolMin - ro) * invRd;
    vec3 t1    = (uVolMax - ro) * invRd;
    vec3 tNear = min(t0, t1);
    vec3 tFar  = max(t0, t1);
    return vec2(
      max(max(tNear.x, tNear.y), tNear.z),
      min(min(tFar.x,  tFar.y),  tFar.z)
    );
  }

  // ── 3-D DDA voxel traversal ────────────────────────────────────────────────
  // Advances one voxel boundary at a time, cheapest possible sparse traversal.
  //
  //  ro      – ray origin  (world space)
  //  rd      – ray direction (normalised, world space)
  //  tStart  – enter the volume at this t
  //  tEnd    – leave the volume at this t
  //  normal  – (out) surface normal of the hit face
  //
  // Returns vec4(r,g,b,1) on hit, vec4(0) on miss.
  vec4 dda(vec3 ro, vec3 rd, float tStart, float tEnd, out vec3 normal) {

    // Step the entry point just inside so floor() lands in the right voxel
    vec3 entry  = ro + rd * (tStart + 1e-4);
    ivec3 voxel = ivec3(floor(entry));

    // Axis-wise step direction (+1 or -1)
    ivec3 stepDir = ivec3(sign(rd));

    // tDelta: how far along the ray we travel to cross one voxel on each axis
    vec3 tDelta = abs(1.0 / rd);

    // tMax: distance to the next voxel boundary on each axis from entry
    // The next boundary in +step direction from the current voxel floor
    vec3 nextBoundary = vec3(voxel) + vec3(max(stepDir, ivec3(0)));
    vec3 tMax         = (nextBoundary - entry) / rd;

    // Guard zero-component directions (would produce ±inf tMax)
    if (abs(rd.x) < 1e-8) tMax.x = 1e30;
    if (abs(rd.y) < 1e-8) tMax.y = 1e30;
    if (abs(rd.z) < 1e-8) tMax.z = 1e30;

    ivec3 volSize = ivec3(uVolSize);
    normal = vec3(0.0);

    // Worst case: ray traverses full diagonal ≈ sqrt(512²+64²+512²) ≈ 727 steps
    const int MAX_STEPS = 1100;

    for (int i = 0; i < MAX_STEPS; i++) {

      // Out-of-bounds → done
      if (any(lessThan(voxel, ivec3(0))) ||
          any(greaterThanEqual(voxel, volSize))) break;

      // Texture lookup: voxel (ix, iy, iz) → uvw = (ix+0.5)/W etc.
      vec3 uvw   = (vec3(voxel) + 0.5) / uVolSize;
      vec4 texel = texture(uVoxels, uvw);

      if (texel.a > 0.5) {
        // Solid voxel hit — return albedo + face normal
        return vec4(texel.rgb, 1.0);
      }

      // Advance to the nearest slab crossing (the classic DDA "smallest tMax" choice)
      if (tMax.x < tMax.y && tMax.x < tMax.z) {
        if (tMax.x > tEnd) break;
        tMax.x   += tDelta.x;
        voxel.x  += stepDir.x;
        normal    = vec3(-float(stepDir.x), 0.0, 0.0);
      } else if (tMax.y < tMax.z) {
        if (tMax.y > tEnd) break;
        tMax.y   += tDelta.y;
        voxel.y  += stepDir.y;
        normal    = vec3(0.0, -float(stepDir.y), 0.0);
      } else {
        if (tMax.z > tEnd) break;
        tMax.z   += tDelta.z;
        voxel.z  += stepDir.z;
        normal    = vec3(0.0, 0.0, -float(stepDir.z));
      }
    }

    return vec4(0.0); // miss
  }

  void main() {
    // ── Ray setup ────────────────────────────────────────────────────────────
    // When the camera is INSIDE the cube the vertex-interpolated vWorldPos is
    // on the back wall, so the direction is still correct.
    vec3 ro = uCamPos;
    vec3 rd = normalize(vWorldPos - uCamPos);

    // ── AABB entry / exit ─────────────────────────────────────────────────────
    vec2 tHit   = aabbIntersect(ro, rd);
    if (tHit.x > tHit.y) discard;          // ray misses the volume box

    float tStart = max(tHit.x, 0.0);       // clamp: camera may be inside
    float tEnd   = tHit.y;

    // ── DDA march ─────────────────────────────────────────────────────────────
    vec3 hitNormal;
    vec4 hit = dda(ro, rd, tStart, tEnd, hitNormal);

    if (hit.a < 0.5) discard;              // ray exited volume without a hit

    // ── Lighting ──────────────────────────────────────────────────────────────
    vec3 albedo = hit.rgb;

    // Lambertian sun + ambient
    float nDotL   = max(dot(hitNormal, uSunDir), 0.0);
    float ambient = 0.20;

    // Cheap face-AO: side faces slightly darker than top, bottom darkest
    float faceAO  = (hitNormal.y >  0.5) ? 1.00   // top face
                  : (hitNormal.y < -0.5) ? 0.50   // bottom face
                  :                        0.75;   // side faces

    vec3 colour = albedo * faceAO * (ambient + nDotL * 0.80);

    // ── Distance fog ─────────────────────────────────────────────────────────
    float hitDist = tStart; // approximate; good enough for fog
    float fogFac  = 1.0 - exp(-hitDist * 0.0018);
    vec3  skyCol  = vec3(0.502, 0.627, 0.878);
    colour        = mix(colour, skyCol, clamp(fogFac, 0.0, 1.0));

    gl_FragColor  = vec4(colour, 1.0);
  }
`;

// ── Helpers ───────────────────────────────────────────────────────────────────
function bgra2rgb(b: number, g: number, r: number): [number, number, number] {
  return [r, g, b];
}

// ── VoxelWorld ────────────────────────────────────────────────────────────────
/**
 * Single-draw-call voxel renderer.
 *
 * Architecture:
 *   • One inverted BoxGeometry covering the entire world AABB.
 *   • THREE.BackSide so the shader runs on every pixel inside (or aimed at) the box.
 *   • Per-pixel DDA raymarcher in the fragment shader reads a 512×64×512 Data3DTexture.
 *   • AABB slab test gives exact ray entry/exit — no over-marching outside the volume.
 *
 * Usage:
 *   const world = new VoxelWorld();
 *   await world.loadFromURL("/LostValley.vxl");
 *   scene.add(world);
 *
 *   // In your render loop:
 *   world.updateCamera(camera);
 */
export class VoxelWorld extends THREE.Mesh {
  private tex3d: THREE.Data3DTexture | null = null;
  declare material: THREE.ShaderMaterial;

  constructor() {
    const geo = new THREE.BoxGeometry(
      VOLUME_SIZE.x,
      VOLUME_SIZE.y,
      VOLUME_SIZE.z,
    );

    const mat = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        uVoxels:  { value: null },
        uCamPos:  { value: new THREE.Vector3() },
        uVolMin:  { value: VOLUME_MIN.clone() },
        uVolMax:  { value: VOLUME_MAX.clone() },
        uVolSize: { value: VOLUME_SIZE.clone() },
        uSunDir:  { value: new THREE.Vector3(0.55, 1.0, 0.35).normalize() },
      },
      // BackSide: fragment shader runs on the interior faces of the cube.
      // When the camera is outside, it runs on the front-facing back walls;
      // when inside, it still runs because we're looking at back faces.
      side: THREE.BackSide,
      depthWrite: true,
      depthTest:  true,
    });

    super(geo, mat);

    // Translate mesh so its centre aligns with the AABB centre
    // (BoxGeometry is centred at origin; our AABB starts at 0,0,0)
    this.position.copy(VOLUME_MIN.clone().add(VOLUME_MAX).multiplyScalar(0.5));
  }

  // ── Loaders ──────────────────────────────────────────────────────────────

  async loadFromURL(url: string): Promise<void> {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`VoxelWorld: fetch failed (${res.status}) for ${url}`);
    return this.loadFromBuffer(await res.arrayBuffer());
  }

  loadFromBuffer(buffer: ArrayBuffer): void {
    const bytes  = new Uint8Array(buffer);

    // 3D texture dimensions:
    //   width  = VXL_X  (512)   → texture x = vxl x
    //   height = VXL_Z  (64)    → texture y = VXL_Z-1-vxl_z  (flip so y=0 is sky-top)
    //   depth  = VXL_Y  (512)   → texture z = vxl y
    const W = VXL_X;
    const H = VXL_Z;
    const D = VXL_Y;
    const pixels = new Uint8Array(W * H * D * 4); // zeroed = transparent air

    const writeVoxel = (vx: number, vy: number, vz: number, r: number, g: number, b: number) => {
      // texture coords: tx=vx, ty=VXL_Z-1-vz (sky up), tz=vy
      const tx  = vx;
      const ty  = (VXL_Z - 1 - vz);
      const tz  = vy;
      const idx = (tx + W * (ty + H * tz)) * 4;
      pixels[idx]     = r;
      pixels[idx + 1] = g;
      pixels[idx + 2] = b;
      pixels[idx + 3] = 255; // solid marker
    };

    let offset = 0;
    for (let vy = 0; vy < VXL_Y; vy++) {
      for (let vx = 0; vx < VXL_X; vx++) {
        while (true) {
          const n         = bytes[offset];
          const s         = bytes[offset + 1]; // top of solid run (vxl z)
          const e         = bytes[offset + 2]; // bottom of solid run
          const numColors = n === 0 ? (e - s + 1) : (n - 1);

          // Top-surface colours (explicit)
          for (let i = 0; i < numColors; i++) {
            const base      = offset + 4 + i * 4;
            const [r, g, b] = bgra2rgb(bytes[base], bytes[base + 1], bytes[base + 2]);
            writeVoxel(vx, vy, s + i, r, g, b);
          }

          // Buried voxels — inherit the last explicit colour
          const lastBase  = numColors > 0 ? offset + 4 + (numColors - 1) * 4 : -1;
          const [lr, lg, lb] = lastBase >= 0
            ? bgra2rgb(bytes[lastBase], bytes[lastBase + 1], bytes[lastBase + 2])
            : [136, 136, 136];

          for (let bz = s + numColors; bz <= e; bz++) {
            writeVoxel(vx, vy, bz, lr, lg, lb);
          }

          offset += 4 + numColors * 4;
          if (n === 0) break;
        }
      }
    }

    // Build / replace the GPU texture
    if (this.tex3d) this.tex3d.dispose();

    const tex            = new THREE.Data3DTexture(pixels, W, H, D);
    tex.format           = THREE.RGBAFormat;
    tex.type             = THREE.UnsignedByteType;
    tex.minFilter        = THREE.NearestFilter;
    tex.magFilter        = THREE.NearestFilter;
    tex.wrapS            = THREE.ClampToEdgeWrapping;
    tex.wrapT            = THREE.ClampToEdgeWrapping;
    tex.wrapR            = THREE.ClampToEdgeWrapping;
    tex.unpackAlignment  = 1;
    tex.needsUpdate      = true;

    this.tex3d = tex;
    this.material.uniforms.uVoxels.value = tex;
  }

  // ── Per-frame ────────────────────────────────────────────────────────────

  /** Must be called once per frame so the shader knows the camera world position. */
  updateCamera(camera: THREE.Camera): void {
    camera.getWorldPosition(this.material.uniforms.uCamPos.value);
  }
}