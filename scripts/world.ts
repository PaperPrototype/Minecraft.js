import * as THREE from "three";

const geometry = new THREE.BoxGeometry();
const material = new THREE.MeshLambertMaterial({ color: 0x00d000 });

export class World extends THREE.Group {
  size: number;
  height: number;
  data: { id: number; instanceId: number | null }[][][];

  constructor(size = 32, height = 16) {
    super();
    this.height = height;
    this.size = size;
    this.data = []
  }

  generate() {
    this.generateTerrain();
    this.generateMeshes();
  }

  generateTerrain() {
    this.data = [];
    for (let x = 0; x < this.size; x++) {
      const slice = [];
      for (let y = 0; y < this.height; y++) {
        const row = [];
        for (let z = 0; z < this.size; z++) {
          row.push({
            id: 1,
            instanceId: null
          })
        }
        slice.push(row);
      }
      this.data.push(slice);
    }
  }

  generateMeshes() {
    this.clear();

    const max = this.size * this.height * this.size;
    const mesh = new THREE.InstancedMesh(geometry, material, max);
    mesh.count = 0;

    const matrix = new THREE.Matrix4();
    for (let x = 0; x < this.size; x++) {
      for (let y = 0; y < this.height; y++) {
        for (let z = 0; z < this.size; z++) {
          matrix.setPosition(x + 0.5, y + 0.5, z + 0.5);
          mesh.setMatrixAt(mesh.count++, matrix);
        }
      }
    }

    this.add(mesh);
  }

  getBlock(x: number, y: number, z: number) {
    if (this.inBounds(x, y, z)) {
      return this.data[x][y][z];
    }
    return null;
  }

  inBounds(x: number, y: number, z: number) {
    if (
      x >= 0 && x < this.size &&
      y >= 0 && y < this.height &&
      z >= 0 && z < this.size
    ) {
      return true;
    } else {
      return false;
    }
  }
}