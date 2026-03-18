import * as THREE from "three";
import { VOXEL_RADIUS } from "./VoxelMip";
import type { Contact } from "./VoxelMip";

// ── Constants ─────────────────────────────────────────────────────────────────
const VOXEL_MASS      = 0.1;    // kg per voxel
const GRAVITY         = 9.81;   // m/s²
const LINEAR_DAMPING  = 0.985;  // per frame damping (applied after integration)
const ANGULAR_DAMPING = 0.980;
const RESTITUTION     = 0.35;   // bounciness
const FRICTION        = 0.6;
const SLEEP_VEL       = 0.02;   // sleep threshold (linear speed²)
const SLEEP_FRAMES    = 90;     // frames under threshold before sleeping

// ── VoxelPhysics ──────────────────────────────────────────────────────────────
/**
 * Rigid-body physics state for a VoxelMap island.
 *
 * Mass, centre of mass, and the inertia tensor are computed from the voxel
 * distribution at construction time (each voxel = point mass VOXEL_MASS at
 * its local-space centre).
 *
 * Integration: semi-implicit Euler.
 * Collision response: impulse-based, using the prebuilt octree for hierarchical
 * sphere-sphere contacts.
 */
export class VoxelPhysics {
  // ── Inertia / mass ────────────────────────────────────────────────────────
  readonly mass:       number;
  readonly invMass:    number;
  /** Centre of mass in local voxel space */
  readonly com:        THREE.Vector3;
  /** Inertia tensor (3×3) in local CoM space, stored as Matrix3 */
  readonly inertiaCOM: THREE.Matrix3;
  readonly invInertiaCOM: THREE.Matrix3;

  // ── State ─────────────────────────────────────────────────────────────────
  /** World-space position of the body's centre of mass */
  position:   THREE.Vector3;
  /** Orientation quaternion */
  quaternion: THREE.Quaternion;
  linVel:     THREE.Vector3;   // linear velocity  (m/s)
  angVel:     THREE.Vector3;   // angular velocity (rad/s, world space)

  // ── Sleep ─────────────────────────────────────────────────────────────────
  sleeping      = false;
  private _sleepTimer = 0;

  // ── Scratch ───────────────────────────────────────────────────────────────
  private _rotMat    = new THREE.Matrix3();
  private _rotMatT   = new THREE.Matrix3();
  private _invIWorld = new THREE.Matrix3();

  constructor(pixels: Uint8Array, w: number, h: number, d: number) {
    // ── Accumulate mass and first moment ────────────────────────────────────
    let totalMass = 0;
    const com = new THREE.Vector3();

    for (let tz = 0; tz < d; tz++) {
      for (let ty = 0; ty < h; ty++) {
        for (let tx = 0; tx < w; tx++) {
          const i = (tx + w * (ty + h * tz)) * 4;
          if (pixels[i + 3] === 0) continue;
          const cx = tx + 0.5, cy = ty + 0.5, cz = tz + 0.5;
          totalMass += VOXEL_MASS;
          com.x += cx * VOXEL_MASS;
          com.y += cy * VOXEL_MASS;
          com.z += cz * VOXEL_MASS;
        }
      }
    }

    if (totalMass < 1e-9) {
      // Degenerate — give it something so we don't divide by zero
      totalMass = VOXEL_MASS;
      com.set(w / 2, h / 2, d / 2);
    }

    com.divideScalar(totalMass);
    this.mass    = totalMass;
    this.invMass = 1 / totalMass;
    this.com     = com.clone();

    // ── Inertia tensor about CoM (parallel-axis theorem per voxel) ──────────
    // I_xx = Σ m*(y²+z²),  I_yy = Σ m*(x²+z²),  I_zz = Σ m*(x²+y²)
    // I_xy = -Σ m*x*y,  etc.
    let Ixx=0, Iyy=0, Izz=0, Ixy=0, Ixz=0, Iyz=0;

    for (let tz = 0; tz < d; tz++) {
      for (let ty = 0; ty < h; ty++) {
        for (let tx = 0; tx < w; tx++) {
          const i = (tx + w * (ty + h * tz)) * 4;
          if (pixels[i + 3] === 0) continue;
          const rx = (tx + 0.5) - com.x;
          const ry = (ty + 0.5) - com.y;
          const rz = (tz + 0.5) - com.z;
          const m  = VOXEL_MASS;
          Ixx += m * (ry*ry + rz*rz);
          Iyy += m * (rx*rx + rz*rz);
          Izz += m * (rx*rx + ry*ry);
          Ixy -= m * rx * ry;
          Ixz -= m * rx * rz;
          Iyz -= m * ry * rz;
        }
      }
    }

    // THREE.Matrix3 is column-major:  elements[col*3 + row]
    // But setFromMatrix4 etc. use row-major — use .set(row0col0, r0c1, r0c2, r1c0 ...)
    this.inertiaCOM = new THREE.Matrix3();
    this.inertiaCOM.set(
      Ixx, Ixy, Ixz,
      Ixy, Iyy, Iyz,
      Ixz, Iyz, Izz,
    );
    this.invInertiaCOM = this.inertiaCOM.clone().invert();

    // ── Initial state ────────────────────────────────────────────────────────
    this.position   = new THREE.Vector3();   // set by MapManager after creation
    this.quaternion = new THREE.Quaternion();
    this.linVel     = new THREE.Vector3();
    this.angVel     = new THREE.Vector3();
  }

  // ── World-space inverse inertia tensor ────────────────────────────────────
  /**
   * I_world⁻¹ = R * I_local⁻¹ * Rᵀ
   * Must be called after updating the orientation.
   */
  invInertiaTensorWorld(): THREE.Matrix3 {
    // Build rotation matrix from quaternion
    const e  = new THREE.Matrix4().makeRotationFromQuaternion(this.quaternion);
    this._rotMat.setFromMatrix4(e);
    this._rotMatT.copy(this._rotMat).transpose();

    // I_w⁻¹ = R * I_local⁻¹ * Rᵀ
    const result = new THREE.Matrix3()
      .multiplyMatrices(this._rotMat, this.invInertiaCOM)
      .multiply(this._rotMatT);    // wait — THREE.Matrix3.multiply is in-place
    return result;
  }

  // ── Apply impulse ─────────────────────────────────────────────────────────
  /**
   * Apply an impulse at world-space point `contactPt`.
   * `r` = contactPt - comWorldPos
   */
  applyImpulse(impulse: THREE.Vector3, r: THREE.Vector3): void {
    if (this.sleeping) this.wake();
    // Linear
    this.linVel.addScaledVector(impulse, this.invMass);
    // Angular: Δω = I_w⁻¹ * (r × J)
    const torqueImpulse = r.clone().cross(impulse);
    const iInv = this.invInertiaTensorWorld();
    this.angVel.add(torqueImpulse.applyMatrix3(iInv));
  }

  wake(): void {
    this.sleeping    = false;
    this._sleepTimer = 0;
  }

  // ── Integration ───────────────────────────────────────────────────────────
  integrate(dt: number): void {
    if (this.sleeping) return;

    // Gravity
    this.linVel.y -= GRAVITY * dt;

    // Integrate position
    this.position.addScaledVector(this.linVel, dt);

    // Integrate orientation:  q += 0.5 * dt * ω⊗q
    const wq = new THREE.Quaternion(
      this.angVel.x * dt * 0.5,
      this.angVel.y * dt * 0.5,
      this.angVel.z * dt * 0.5,
      0,
    ).multiply(this.quaternion);
    this.quaternion.set(
      this.quaternion.x + wq.x,
      this.quaternion.y + wq.y,
      this.quaternion.z + wq.z,
      this.quaternion.w + wq.w,
    ).normalize();

    // Damping
    this.linVel.multiplyScalar(LINEAR_DAMPING);
    this.angVel.multiplyScalar(ANGULAR_DAMPING);

    // Sleep check
    if (this.linVel.lengthSq() < SLEEP_VEL && this.angVel.lengthSq() < SLEEP_VEL) {
      if (++this._sleepTimer > SLEEP_FRAMES) {
        this.sleeping = true;
        this.linVel.set(0,0,0);
        this.angVel.set(0,0,0);
      }
    } else {
      this._sleepTimer = 0;
    }
  }

  // ── Floor collision ───────────────────────────────────────────────────────
  /**
   * Simple ground-plane constraint at y=0.
   * The mesh's bottom face is at position.y - com.y (since position = world CoM).
   * Actually we track the world-space AABB minimum Y as the lowest voxel sphere.
   */
  resolveFloor(floorY: number, bodyMinLocalY: number, scale: number): void {
    // World-space Y of the lowest voxel sphere bottom
    // approximate: use linVel + position, ignoring rotation for floor
    const bottomWorld = this.position.y - this.com.y * scale + bodyMinLocalY * scale - VOXEL_RADIUS * scale;

    if (bottomWorld < floorY) {
      const pen = floorY - bottomWorld;
      this.position.y += pen;

      if (this.linVel.y < 0) {
        // Reflect vertical velocity with restitution
        this.linVel.y = -this.linVel.y * RESTITUTION;
        // Friction on horizontal
        this.linVel.x *= (1 - FRICTION * 0.1);
        this.linVel.z *= (1 - FRICTION * 0.1);
      }
      this.wake();
    }
  }
}

// ── Resolve contacts between two physics bodies ───────────────────────────────
const _rA = new THREE.Vector3();
const _rB = new THREE.Vector3();
const _relV = new THREE.Vector3();

export function resolveContacts(
  physA: VoxelPhysics, physB: VoxelPhysics | null,   // null = static
  contacts: Contact[],
): void {
  for (const c of contacts) {
    // r vectors: contact point relative to each CoM
    _rA.subVectors(c.point, physA.position);
    if (physB) _rB.subVectors(c.point, physB.position);

    // Relative velocity at contact
    _relV.copy(physA.linVel);
    if (physB) _relV.sub(physB.linVel);

    // Add angular contribution: v_angular = ω × r
    const vaA = physA.angVel.clone().cross(_rA);
    _relV.add(vaA);
    if (physB) {
      const vaB = physB.angVel.clone().cross(_rB);
      _relV.sub(vaB);
    }

    const vn = _relV.dot(c.normal);
    if (vn > 0) continue;   // separating — skip

    // Impulse scalar:  j = -(1+e)*vn / (1/mA + 1/mB + (rA×n)·I_A⁻¹(rA×n) + ...)
    const e  = RESTITUTION;
    const iA = physA.invInertiaTensorWorld();

    const rAxN  = _rA.clone().cross(c.normal);
    const iArAxN = rAxN.clone().applyMatrix3(iA);
    let denom   = physA.invMass + iArAxN.dot(rAxN);

    if (physB) {
      const iB     = physB.invInertiaTensorWorld();
      const rBxN   = _rB.clone().cross(c.normal);
      const iBrBxN = rBxN.clone().applyMatrix3(iB);
      denom += physB.invMass + iBrBxN.dot(rBxN);
    }

    if (denom < 1e-10) continue;
    const j = -(1 + e) * vn / denom;

    // Apply impulse
    const impulse = c.normal.clone().multiplyScalar(j);
    physA.applyImpulse(impulse, _rA);
    if (physB) physB.applyImpulse(impulse.clone().negate(), _rB);

    // Position correction (Baumgarte)
    const slop = 0.005;
    const corr = Math.max(c.depth - slop, 0) * 0.4;
    const totalInvM = physA.invMass + (physB ? physB.invMass : 0);
    if (totalInvM > 1e-10) {
      physA.position.addScaledVector(c.normal,  corr * physA.invMass / totalInvM);
      if (physB) physB.position.addScaledVector(c.normal, -corr * physB.invMass / totalInvM);
    }
  }
}