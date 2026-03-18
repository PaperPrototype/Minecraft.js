import * as THREE from "three";
import { VoxelMap, VXL_X, VXL_Y, VXL_Z } from "./VoxelMap";
import { queryContacts } from "./VoxelMip";
import { resolveContacts } from "./VoxelPhysics";
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
 */
export class MapManager {
  private entries   = new Map<number, MapEntry>();
  private _scene:   THREE.Scene;
  private _sel:     number | null = null;
  private listeners: Listener[]   = [];

  /** RGB colour used when placing new voxels */
  placeColor: [number, number, number] = [100, 180, 80];

  constructor(scene: THREE.Scene) {
    this._scene = scene;
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

    // @ts-expect-error
    this._scene.add(map);
    const entry: MapEntry = { id, name, map };
    this.entries.set(id, entry);
    this.emit({ type: 'added', entry });
    this.select(id);
    return entry;
  }

  remove(id: number): void {
    const entry = this.entries.get(id);
    if (!entry) return;
    //@ts-expect-error
    this._scene.remove(entry.map);
    entry.map.dispose();
    this.entries.delete(id);
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

  // ── Transform ─────────────────────────────────────────────────────────────
  // Position sets the world-space corner of the volume (0,0,0 = corner).
  // VoxelMap centres its BoxGeometry, so we add half-extents here.

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

  updateCamera(camera: THREE.Camera): void {
    for (const { map } of this.entries.values()) {
      map.updateMatrixWorld();
      map.updateCamera(camera);
    }
  }

  /**
   * Advance physics simulation by dt seconds.
   * Call once per frame before updateCamera.
   *
   * Steps:
   *  1. Integrate all dynamic bodies (gravity + velocity)
   *  2. Floor constraint
   *  3. Broadphase: AABB pairs among dynamic bodies
   *  4. Narrowphase: octree contact query + impulse resolution
   *  5. Sync mesh transforms from physics state
   */
  stepPhysics(dt: number, floorY = 0): void {
    const dynamic = [...this.entries.values()].filter(e => e.map.physics !== null);

    // 1. Integrate
    for (const { map } of dynamic) {
      map.physics!.integrate(dt);
    }

    // 2. Floor
    for (const { map } of dynamic) {
      const ph = map.physics!;
      // Lowest local Y among voxel centres = 0.5 (tight maps start at local 0)
      ph.resolveFloor(floorY, 0.5, map.scale.x);
    }

    // 3 + 4. Body-body and body-static contacts
    const allEntries = [...this.entries.values()];
    for (let i = 0; i < dynamic.length; i++) {
      // Against other dynamic bodies
      for (let j = i + 1; j < dynamic.length; j++) {
        const ma = dynamic[i].map, mb = dynamic[j].map;
        if (!ma.mip || !mb.mip) continue;
        if (ma.physics!.sleeping && mb.physics!.sleeping) continue;
        ma.updateMatrixWorld(); mb.updateMatrixWorld();
        // @ts-expect-error
        if (!new THREE.Box3().setFromObject(ma)
            .intersectsBox(new THREE.Box3().setFromObject(mb as any))) continue;
        const contacts = queryContacts(ma.mip, mb.mip, ma.matrixWorld, mb.matrixWorld);
        if (contacts.length > 0) resolveContacts(ma.physics!, mb.physics!, contacts);
      }

      // Against static maps (no physics, but have a mip)
      const dmap = dynamic[i].map;
      if (dmap.physics!.sleeping) continue;
      for (const { map: smap } of allEntries) {
        if (smap.physics !== null) continue; // skip dynamic — handled above
        if (!smap.mip || !dmap.mip) continue;
        dmap.updateMatrixWorld(); smap.updateMatrixWorld();
        // @ts-ignore
        if (!new THREE.Box3().setFromObject(dmap)
            .intersectsBox(new THREE.Box3().setFromObject(smap as any))) continue;
        const contacts = queryContacts(dmap.mip, smap.mip, dmap.matrixWorld, smap.matrixWorld);
        if (contacts.length > 0) resolveContacts(dmap.physics!, null, contacts);
      }
    }

    // 5. Sync mesh transforms
    for (const { map } of dynamic) {
      map.syncFromPhysics();
    }
  }

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

      // Compute approximate world-space distance to hit voxel centre
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
      // Parent lost voxels — rebuild its mip
      entry.map.rebuildMip();
    }
    for (const islandMap of islands) {
      const id  = _nextId++;
      const newEntry: MapEntry = { id, name: islandMap.mapName, map: islandMap };
      this._scene.add(islandMap as any);
      this.entries.set(id, newEntry);
      this.emit({ type: 'added', entry: newEntry });
    }
  }
}