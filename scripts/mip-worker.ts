// Web Worker: receives a pixel buffer, builds mip levels, transfers them back.
// Messages in:  { pixels: Uint8Array, w, h, d }
// Messages out: { levels: ArrayBuffer[], dims: {w,h,d}[] }

self.onmessage = (e: MessageEvent) => {
  const { pixels, w, h, d } = e.data as {
    pixels: Uint8Array;
    w: number; h: number; d: number;
  };

  const levels: Uint8Array[] = [];
  const dims: { w: number; h: number; d: number }[] = [];

  // Level 0 — solid bit from RGBA alpha channel
  const l0 = new Uint8Array(w * h * d);
  for (let i = 0; i < w * h * d; i++) {
    l0[i] = pixels[i * 4 + 3] > 0 ? 1 : 0;
  }
  levels.push(l0);
  dims.push({ w, h, d });

  let cw = w, ch = h, cd = d;
  while (cw > 1 || ch > 1 || cd > 1) {
    const pw = cw, ph = ch, pd = cd;
    const prev = levels[levels.length - 1];

    cw = Math.ceil(cw / 2);
    ch = Math.ceil(ch / 2);
    cd = Math.ceil(cd / 2);

    const cur = new Uint8Array(cw * ch * cd);
    for (let tz = 0; tz < cd; tz++) {
      for (let ty = 0; ty < ch; ty++) {
        for (let tx = 0; tx < cw; tx++) {
          let occ = 0;
          for (let dz = 0; dz < 2 && !occ; dz++) {
            const pz = tz * 2 + dz; if (pz >= pd) continue;
            for (let dy = 0; dy < 2 && !occ; dy++) {
              const py = ty * 2 + dy; if (py >= ph) continue;
              for (let dx = 0; dx < 2 && !occ; dx++) {
                const px = tx * 2 + dx; if (px >= pw) continue;
                if (prev[px + pw * (py + ph * pz)]) occ = 1;
              }
            }
          }
          cur[tx + cw * (ty + ch * tz)] = occ;
        }
      }
    }
    levels.push(cur);
    dims.push({ w: cw, h: ch, d: cd });
  }

  // Transfer all ArrayBuffers zero-copy back to main thread
  const buffers = levels.map(l => l.buffer);
  self.postMessage({ levels: buffers, dims }, buffers as any);
};