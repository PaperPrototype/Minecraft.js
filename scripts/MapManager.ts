import * as THREE from "three";
import { VoxelMap, VXL_X, VXL_Y, VXL_Z } from "./VoxelMap";
import type { RaycastHit } from "./VoxelMap";

let _nextId = 1;

export interface MapEntry {
  id:   number;
  name: string;
  map:  VoxelMap;
}

export type MapManagerEvent =
  | { type: 'added';    entry: MapEntry      }
  | { type: 'removed';  id:    number        }
  | { type: 'selected'; id:    number | null }
  | { type: 'changed';  id:    number        };

type Listener = (e: MapManagerEvent) => void;

/**
 * Owns all VoxelMaps in the scene.
 * Handles multi-map raycast, dig/place, and selection.
 *
 * Physics runs entirely in physics-worker.ts off the main thread.
 * Each frame:
 *   1. stepPhysics(dt)  → posts { type:'step', dt } to worker
 *   2. Worker posts back { type:'transforms', bodies:[{id, mat}] }
 *   3. updateCamera()   → applies those matrices to the Three.js meshes
 */
export class MapManager {
  private entries    = new Map<number, MapEntry>();
  private _scene:    THREE.Scene;
  private _sel:      number | null = null;
  private listeners: Listener[]    = [];

  /** Physics worker — runs integration + collision off the main thread */
  private _physWorker: Worker;
  /** Pending transforms from last worker step, applied next updateCamera */
  private _pendingTransforms: { id: number; mat: number[] }[] = [];

  /** RGB colour used when placing new voxels */
  placeColor: [number, number, number] = [100, 180, 80];

  constructor(scene: THREE.Scene) {
    this._scene = scene;
    this._physWorker = new Worker(
      new URL('./physics-worker.ts', import.meta.url),
      { type: 'module' },
    );
    this._physWorker.onmessage = (e: MessageEvent) => {
      if (e.data.type === 'transforms') {
        this._pendingTransforms = e.data.bodies;
      }
    };
  }

  // ── Pub/sub ───────────────────────────────────────────────────────────────
  on(fn: Listener)  { this.listeners.push(fn); }
  off(fn: Listener) { this.listeners = this.listeners.filter(l => l !== fn); }
  private emit(e: MapManagerEvent) { this.listeners.forEach(l => l(e)); }

  // ── Map lifecycle ─────────────────────────────────────────────────────────

  addFromBuffer(name: string, buffer: ArrayBuffer): MapEntry {
    const id  = _nextId++;
    const map = new VoxelMap(name);
    map.loadFromBuffer(buffer);

    // Stagger new maps so they don't all pile on top of each other
    const offset = this.entries.size * 600;
    map.position.set(offset, 0, 0);

    this._scene.add(map);
    const entry: MapEntry = { id, name, map };
    this.entries.set(id, entry);
    this.emit({ type: 'added', entry });
    this.select(id);

    // All maps added via the UI start as static collision bodies.
    // Islands split off by dig/place are registered as dynamic bodies
    // via _registerIsland() in _splitIslands().
    this._registerStatic(entry);
    return entry;
  }

  remove(id: number): void {
    const entry = this.entries.get(id);
    if (!entry) return;
    this._scene.remove(entry.map);
    entry.map.dispose();
    this.entries.delete(id);
    // Tell the worker to drop this body (works for both static and dynamic)
    this._physWorker.postMessage({ type: 'remove', id });
    this.emit({ type: 'removed', id });
    if (this._sel === id) {
      const remaining = [...this.entries.keys()];
      this.select(remaining.length ? remaining[remaining.length - 1] : null);
    }
  }

  get(id: number): MapEntry | undefined { return this.entries.get(id); }
  all(): MapEntry[] { return [...this.entries.values()]; }

  // ── Selection ─────────────────────────────────────────────────────────────

  get selectedId(): number | null { return this._sel; }

  get selected(): MapEntry | null {
    return this._sel !== null ? (this.entries.get(this._sel) ?? null) : null;
  }

  select(id: number | null): void {
    if (this._sel !== null) this.entries.get(this._sel)?.map.setSelected(false);
    this._sel = id;
    if (id !== null)        this.entries.get(id)?.map.setSelected(true);
    this.emit({ type: 'selected', id });
  }

  // ── Transform (gizmo / programmatic) ─────────────────────────────────────
  // These write directly to the Three.js mesh; the static-sync path in
  // stepPhysics() will push the updated matrixWorld to the worker each frame.

  setPosition(id: number, x: number, y: number, z: number): void {
    const e = this.entries.get(id); if (!e) return;
    e.map.position.set(x, y, z);
    this.emit({ type: 'changed', id });
  }

  getPosition(id: number): THREE.Vector3 | null {
    const e = this.entries.get(id); if (!e) return null;
    return e.map.position.clone();
  }

  setRotation(id: number, degX: number, degY: number, degZ: number): void {
    const e = this.entries.get(id); if (!e) return;
    e.map.rotation.set(
      THREE.MathUtils.degToRad(degX),
      THREE.MathUtils.degToRad(degY),
      THREE.MathUtils.degToRad(degZ),
    );
    this.emit({ type: 'changed', id });
  }

  getRotationDeg(id: number): THREE.Vector3 | null {
    const e = this.entries.get(id); if (!e) return null;
    const r = e.map.rotation;
    return new THREE.Vector3(
      THREE.MathUtils.radToDeg(r.x),
      THREE.MathUtils.radToDeg(r.y),
      THREE.MathUtils.radToDeg(r.z),
    );
  }

  setScale(id: number, s: number): void {
    const e = this.entries.get(id); if (!e) return;
    e.map.scale.setScalar(s);
    this.emit({ type: 'changed', id });
  }

  getScale(id: number): number | null {
    const e = this.entries.get(id); if (!e) return null;
    return e.map.scale.x;
  }

  // ── Per-frame ─────────────────────────────────────────────────────────────

  /**
   * Apply physics transforms received from the worker last frame, then
   * update all map matrices and pass the camera position to each shader.
   */
  updateCamera(camera: THREE.Camera): void {
    // Apply transform matrices the physics worker sent back
    for (const { id, mat } of this._pendingTransforms) {
      const entry = this.entries.get(id);
      if (!entry) continue;
      const m = new THREE.Matrix4().fromArray(mat);
      m.decompose(entry.map.position, entry.map.quaternion, entry.map.scale);
    }
    this._pendingTransforms = [];

    for (const { map } of this.entries.values()) {
      map.updateMatrixWorld();
      map.updateCamera(camera);
    }
  }

  /**
   * Send dt to the physics worker.  Non-blocking — the worker posts back
   * transforms which are applied next frame in updateCamera.
   *
   * Also syncs static-map matrices every frame so gizmo drags are reflected
   * in the worker's collision world without a separate event.
   */
  stepPhysics(dt: number): void {
    // Push updated world matrices for every static map so gizmo moves are
    // immediately reflected in the worker's collision world.
    for (const { id, map } of this.entries.values()) {
      if (map.physics === null && map.mip !== null) {
        map.updateMatrixWorld();
        this._physWorker.postMessage({
          type: 'updateStatic',
          id,
          mat: Array.from(map.matrixWorld.elements),
        });
      }
    }
    this._physWorker.postMessage({ type: 'step', dt: Math.min(dt, 0.05) });
  }

  // ── Worker registration helpers ───────────────────────────────────────────

  /**
   * Register a dynamic island body with the worker.
   * Called after VoxelMap.extractIslands() creates a new map with physics state.
   */
  private _registerIsland(entry: MapEntry): void {
    const map = entry.map;
    if (!map.physics) return;
    const ph = map.physics;
    this._physWorker.postMessage({
      type:   'register',
      id:      entry.id,
      pixels:  map.pixels,   // structured-clone copies the buffer
      w: map.w, h: map.h, d: map.d,
      scale:   map.scale.x,
      px: ph.position.x, py: ph.position.y, pz: ph.position.z,
      qx: ph.quaternion.x,   qy: ph.quaternion.y,
      qz: ph.quaternion.z,   qw: ph.quaternion.w,
    });
  }

  /**
   * Register a static map with the worker.
   * Polls until VoxelMap has finished building its mip (async worker),
   * then sends the pixel buffer + current world matrix.
   */
  private _registerStatic(entry: MapEntry): void {
    const map = entry.map;
    const tryRegister = () => {
      if (!map.mip || !map.pixels) { setTimeout(tryRegister, 50); return; }
      map.updateMatrixWorld();
      this._physWorker.postMessage({
        type:   'registerStatic',
        id:      entry.id,
        pixels:  map.pixels,
        w: map.w, h: map.h, d: map.d,
        mat:     Array.from(map.matrixWorld.elements),
      });
    };
    tryRegister();
  }

  // ── Highlights ────────────────────────────────────────────────────────────

  clearHighlights(): void {
    for (const { map } of this.entries.values()) map.setHighlight(null);
  }

  // ── Raycast (nearest hit across all maps) ─────────────────────────────────

  raycastAll(
    worldRo: THREE.Vector3,
    worldRd: THREE.Vector3,
    maxDist = 512,
  ): RaycastHit | null {
    let best: RaycastHit | null = null;
    let bestDist = Infinity;

    for (const { map } of this.entries.values()) {
      const hit = map.raycast(worldRo, worldRd, maxDist);
      if (!hit) continue;

      // Approximate world-space distance to the hit voxel centre
      const lc   = hit.voxel.clone().addScalar(0.5);
      const wc   = lc.applyMatrix4(map.matrixWorld);
      const dist = wc.distanceTo(worldRo);

      if (dist < bestDist) { best = hit; bestDist = dist; }
    }
    return best;
  }

  // ── Dig / Place ───────────────────────────────────────────────────────────

  dig(hit: RaycastHit): void {
    hit.map.removeVoxel(hit.voxel.x, hit.voxel.y, hit.voxel.z);
    hit.map.setHighlight(null);
    const entry = [...this.entries.values()].find(e => e.map === hit.map);
    if (entry) {
      this.emit({ type: 'changed', id: entry.id });
      this._splitIslands(entry, hit.voxel);
    }
  }

  place(hit: RaycastHit): void {
    const [r, g, b] = this.placeColor;
    hit.map.placeVoxel(hit.place.x, hit.place.y, hit.place.z, r, g, b);
    const entry = [...this.entries.values()].find(e => e.map === hit.map);
    if (entry) {
      this.emit({ type: 'changed', id: entry.id });
      this._splitIslands(entry, hit.place);
    }
  }

  private _splitIslands(entry: MapEntry, editedVoxel: THREE.Vector3): void {
    const islands = entry.map.extractIslands(editedVoxel);
    if (islands.length > 0) {
      // Parent lost voxels — rebuild its collision mip so the worker gets
      // a fresh static body next time _registerStatic fires.
      entry.map.rebuildMip();
    }
    for (const islandMap of islands) {
      const id       = _nextId++;
      const newEntry: MapEntry = { id, name: islandMap.mapName, map: islandMap };
      this._scene.add(islandMap);
      this.entries.set(id, newEntry);
      // Islands have physics state set by extractIslands — register dynamic
      this._registerIsland(newEntry);
      this.emit({ type: 'added', entry: newEntry });
    }
  }
}