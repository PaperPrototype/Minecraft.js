import * as THREE from 'three';
import Stats from 'stats.js';

import { VoxelWorld, VOLUME_MAX } from './world';

// ── Stats ─────────────────────────────────────────────────────────────────────
const stats = new Stats();
document.body.appendChild(stats.dom);

// ── Renderer ──────────────────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({ antialias: false });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x80a0e0);
document.body.appendChild(renderer.domElement);

// ── Scene ─────────────────────────────────────────────────────────────────────
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x80a0e0);

// ── Camera ────────────────────────────────────────────────────────────────────
const camera = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.5,
  2000,
);
camera.position.set(VOLUME_MAX.x * 0.5, VOLUME_MAX.y * 0.75, VOLUME_MAX.z * 0.5);

// ── Manual fly camera — Unity scene-view style ────────────────────────────────
//
// Rotation is stored as two independent Euler angles:
//   yaw   — rotation around WORLD Y (left/right), never accumulates tilt
//   pitch — rotation around LOCAL X (up/down), clamped to ±89°
//
// The camera quaternion is rebuilt each frame from scratch:
//   Q = Qyaw * Qpitch
// This guarantees the horizon stays perfectly level no matter what.

let yaw   = 0;   // radians
let pitch = 0;   // radians

const PITCH_LIMIT = Math.PI / 2 - 0.01; // ±89°

// Recompute camera quaternion from current yaw + pitch
function applyRotation() {
  const qYaw   = new THREE.Quaternion().setFromAxisAngle(
    new THREE.Vector3(0, 1, 0), yaw,
  );
  const qPitch = new THREE.Quaternion().setFromAxisAngle(
    new THREE.Vector3(1, 0, 0), pitch,
  );
  camera.quaternion.copy(qYaw).multiply(qPitch);
}
applyRotation(); // initialise upright

// Pointer-lock mouse look
renderer.domElement.addEventListener('click', () => {
  renderer.domElement.requestPointerLock();
});

document.addEventListener('mousemove', (e) => {
  if (document.pointerLockElement !== renderer.domElement) return;
  const SENS = 0.0018;
  yaw   -= e.movementX * SENS;
  pitch -= e.movementY * SENS;
  pitch  = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, pitch));
  applyRotation();
});

// WASD + Space/Q vertical
const keys: Record<string, boolean> = {};
document.addEventListener('keydown', e => { keys[e.code] = true; });
document.addEventListener('keyup',   e => { keys[e.code] = false; });

// Forward/right vectors reused each frame
const _forward = new THREE.Vector3();
const _right   = new THREE.Vector3();

function applyMovement(delta: number) {
  if (document.pointerLockElement !== renderer.domElement) return;

  const sprint = keys['ShiftLeft'] || keys['ShiftRight'];
  const speed  = 60 * (sprint ? 4 : 1) * delta;

  // Extract forward and right from the camera's current orientation,
  // then zero out Y so vertical movement is always world-axis.
  camera.getWorldDirection(_forward);
  _forward.y = 0;
  _forward.normalize();

  _right.crossVectors(_forward, new THREE.Vector3(0, 1, 0)).negate();

  if (keys['KeyW']) camera.position.addScaledVector(_forward,  speed);
  if (keys['KeyS']) camera.position.addScaledVector(_forward, -speed);
  if (keys['KeyD']) camera.position.addScaledVector(_right,    speed);
  if (keys['KeyA']) camera.position.addScaledVector(_right,   -speed);

  // Vertical — pure world Y, no camera tilt involved
  if (keys['Space'] || keys['KeyE']) camera.position.y += speed;
  if (keys['KeyQ'])                  camera.position.y -= speed;
}

// ── World ─────────────────────────────────────────────────────────────────────
const world = new VoxelWorld();
await world.loadFromURL('/LostValley.vxl');
scene.add(world);

// ── Render loop ───────────────────────────────────────────────────────────────
const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);
  const delta = clock.getDelta();
  applyMovement(delta);
  world.updateCamera(camera);
  renderer.render(scene, camera);
  stats.update();
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

animate();