import GUI from 'lil-gui';
import { World } from './world'

export function createUI(world: World) {
  const gui = new GUI();

  gui.add(world, 'size', 8, 128, 1).name('Width');
  gui.add(world, 'height', 8, 64, 1).name('Height');
  // gui.add(world, 'generate')

  gui.onChange(() => {
    world.generate();
  })
}