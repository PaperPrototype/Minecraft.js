import * as THREE from 'three';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import Stats from 'stats.js';

import { MapManager } from './MapManager';
import { createMapPanel, setPanelGizmoMode } from './MapPanel';
import type { GizmoMode } from './MapPanel';

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
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.5, 4000);
camera.position.set(256, 48, 256);

// ── Fly camera — Unity scene-view style ──────────────────────────────────────
// Activated by right-mouse-button hold (not pointer lock) so it coexists
// naturally with the TransformControls gizmo which uses left-mouse-button.
let yaw   = 0;
let pitch = 0;
const PITCH_LIMIT = Math.PI / 2 - 0.01;
const _qYaw   = new THREE.Quaternion();
const _qPitch = new THREE.Quaternion();

function applyRotation() {
  _qYaw.setFromAxisAngle(new THREE.Vector3(0,1,0), yaw);
  _qPitch.setFromAxisAngle(new THREE.Vector3(1,0,0), pitch);
  camera.quaternion.copy(_qYaw).multiply(_qPitch);
}
applyRotation();

// RMB held = fly-look mode
let rmbDown = false;

renderer.domElement.addEventListener('mousedown', e => {
  if (e.button === 2) {
    rmbDown = true;
    renderer.domElement.requestPointerLock();
  }
});
renderer.domElement.addEventListener('mouseup', e => {
  if (e.button === 2) {
    rmbDown = false;
    document.exitPointerLock();
  }
});

document.addEventListener('mousemove', e => {
  if (!rmbDown || document.pointerLockElement !== renderer.domElement) return;
  yaw   -= e.movementX * 0.0018;
  pitch -= e.movementY * 0.0018;
  pitch  = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, pitch));
  applyRotation();
});

const keys: Record<string, boolean> = {};
document.addEventListener('keydown', e => { keys[e.code] = true; });
document.addEventListener('keyup',   e => { keys[e.code] = false; });

const _fwd   = new THREE.Vector3();
const _right = new THREE.Vector3();

function applyMovement(delta: number) {
  if (!rmbDown) return;   // only move while RMB held
  const speed = 60 * (keys['ShiftLeft'] || keys['ShiftRight'] ? 4 : 1) * delta;
  camera.getWorldDirection(_fwd);
  _fwd.y = 0; _fwd.normalize();
  _right.crossVectors(_fwd, new THREE.Vector3(0,1,0)).negate();

  if (keys['KeyW']) camera.position.addScaledVector(_fwd,    speed);
  if (keys['KeyS']) camera.position.addScaledVector(_fwd,   -speed);
  if (keys['KeyD']) camera.position.addScaledVector(_right, -speed);
  if (keys['KeyA']) camera.position.addScaledVector(_right,  speed);
  if (keys['KeyE']) camera.position.y += speed;
  if (keys['KeyQ']) camera.position.y -= speed;
}

// ── TransformControls gizmo ───────────────────────────────────────────────────
const gizmo = new TransformControls(camera, renderer.domElement);
gizmo.setMode('translate');
gizmo.setSize(0.6);
scene.add(gizmo.getHelper());   // adds the visual gizmo helper to the scene

// While dragging the gizmo, disable fly-camera mouse interference
gizmo.addEventListener('dragging-changed', (e: any) => {
  // e.value = true while drag is in progress
  // We simply ensure RMB state is cleared so fly won't activate mid-drag
  if (e.value) {
    rmbDown = false;
    document.exitPointerLock();
  }
});

// ── Map manager ───────────────────────────────────────────────────────────────
const manager = new MapManager(scene);

// Attach/detach gizmo when selection changes
manager.on(e => {
  if (e.type === 'selected') {
    const entry = e.id !== null ? manager.get(e.id) : null;
    if (entry) {
      gizmo.attach(entry.map as any);
    } else {
      gizmo.detach();
    }
  }
});

// ── Panel ─────────────────────────────────────────────────────────────────────
function onModeChange(mode: GizmoMode) {
  gizmo.setMode(mode);
}

createMapPanel(manager, onModeChange);

// W / R / S keys switch gizmo mode (only when NOT flying)
document.addEventListener('keydown', e => {
  if (rmbDown) return; // don't interfere with fly controls
  if (e.code === 'KeyW' && !e.repeat) { gizmo.setMode('translate'); setPanelGizmoMode('translate'); }
  if (e.code === 'KeyR' && !e.repeat) { gizmo.setMode('rotate');    setPanelGizmoMode('rotate');    }
  if (e.code === 'KeyS' && !e.repeat) { gizmo.setMode('scale');     setPanelGizmoMode('scale');     }
});

// Load default map
const res = await fetch('/assets/arica.vxl');
if (res.ok) {
  manager.addFromBuffer('arica', await res.arrayBuffer());
}

// ── Crosshair ─────────────────────────────────────────────────────────────────
function makeCrosshairBar(vertical: boolean) {
  const el = document.createElement('div');
  Object.assign(el.style, {
    position: 'fixed', top: '50%', left: '50%',
    transform: 'translate(-50%,-50%)',
    width:  vertical ? '2px' : '18px',
    height: vertical ? '18px' : '2px',
    background: 'rgba(255,255,255,0.85)',
    pointerEvents: 'none', zIndex: '50',
    boxShadow: '0 0 3px rgba(0,0,0,0.7)',
  });
  document.body.appendChild(el);
  return el;
}
makeCrosshairBar(true);
makeCrosshairBar(false);

// ── Dig / Place ───────────────────────────────────────────────────────────────
const _ro = new THREE.Vector3();
const _rd = new THREE.Vector3();

function getCameraRay() {
  camera.getWorldPosition(_ro);
  camera.getWorldDirection(_rd);
  return { ro: _ro.clone(), rd: _rd.clone() };
}

renderer.domElement.addEventListener('mousedown', e => {
  // Only dig/place when NOT interacting with the gizmo and NOT flying
  if (gizmo.dragging) return;
  if (rmbDown) return;

  // LMB = dig, MMB = place
  if (e.button === 0 || e.button === 1) {
    const { ro, rd } = getCameraRay();
    const hit = manager.raycastAll(ro, rd, 256);
    if (!hit) return;
    if (e.button === 0) manager.dig(hit);
    if (e.button === 1) manager.place(hit);
    e.preventDefault();
  }
});

renderer.domElement.addEventListener('contextmenu', e => e.preventDefault());

// ── Render loop ───────────────────────────────────────────────────────────────
const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);
  const delta = clock.getDelta();

  applyMovement(delta);
  manager.stepPhysics(Math.min(delta, 0.05)); // cap dt to avoid spiral
  manager.updateCamera(camera);

  // Crosshair voxel highlight (only when not flying and not dragging gizmo)
  manager.clearHighlights();
  if (!rmbDown && !gizmo.dragging) {
    const { ro, rd } = getCameraRay();
    const hit = manager.raycastAll(ro, rd, 64);
    if (hit) hit.map.setHighlight(hit.voxel);
  }

  renderer.render(scene, camera);
  stats.update();
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

animate();