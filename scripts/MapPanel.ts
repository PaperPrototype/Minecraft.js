import { MapManager } from "./MapManager";

const css = `
#map-panel {
  position: fixed;
  top: 12px; right: 12px;
  width: 260px;
  background: rgba(18, 18, 24, 0.92);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 10px;
  color: #e8e8f0;
  font: 13px/1.5 system-ui, sans-serif;
  box-shadow: 0 4px 24px rgba(0,0,0,0.5);
  user-select: none;
  z-index: 100;
  pointer-events: all;
}
#map-panel header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 14px 8px;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  font-weight: 600; font-size: 14px; letter-spacing: .02em;
}
#import-btn {
  background: #3a7bd5; border: none; border-radius: 6px;
  color: #fff; font-size: 12px; font-weight: 600;
  padding: 4px 10px; cursor: pointer;
}
#import-btn:hover { background: #4d8fe0; }

#map-list { max-height: 160px; overflow-y: auto; padding: 6px 0; }
.map-item {
  display: flex; align-items: center; gap: 8px;
  padding: 5px 14px; cursor: pointer;
  border-left: 3px solid transparent; transition: background 0.1s;
}
.map-item:hover  { background: rgba(255,255,255,0.05); }
.map-item.active { border-left-color: #3a7bd5; background: rgba(58,123,213,0.12); }
.map-item .label { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.map-item .del   {
  opacity:0; font-size:16px; line-height:1; cursor:pointer; color:#f66;
  transition: opacity 0.15s;
}
.map-item:hover .del { opacity: 1; }

#gizmo-bar {
  display: flex; gap: 4px; padding: 8px 14px;
  border-top: 1px solid rgba(255,255,255,0.08);
}
.gizmo-btn {
  flex: 1; padding: 5px 0; border: 1px solid rgba(255,255,255,0.12);
  border-radius: 5px; background: rgba(255,255,255,0.04);
  color: #e8e8f0; font-size: 12px; font-weight: 600; cursor: pointer;
  transition: background 0.12s, border-color 0.12s;
}
.gizmo-btn:hover  { background: rgba(255,255,255,0.10); }
.gizmo-btn.active { background: rgba(58,123,213,0.35); border-color: #3a7bd5; }

#bottom-bar {
  display: flex; align-items: center; gap: 8px;
  padding: 8px 14px 10px;
  border-top: 1px solid rgba(255,255,255,0.08);
}
#bottom-bar label { opacity:0.55; font-size:11px; white-space:nowrap; }
#place-color { width:32px; height:22px; border:none; border-radius:4px; cursor:pointer; padding:0; flex-shrink:0; }

#help-hint {
  padding: 6px 14px 10px;
  font-size: 11px; opacity: 0.38;
  border-top: 1px solid rgba(255,255,255,0.06);
  line-height: 1.8;
}
#no-maps { padding: 10px 14px; opacity: 0.4; font-size: 12px; }
`;

function injectCSS(s: string) {
  const el = document.createElement('style');
  el.textContent = s;
  document.head.appendChild(el);
}

export type GizmoMode = 'translate' | 'rotate' | 'scale';
type ModeChangeCallback = (mode: GizmoMode) => void;

export function createMapPanel(
  manager: MapManager,
  onModeChange: ModeChangeCallback,
): void {
  injectCSS(css);

  const panel = document.createElement('div');
  panel.id = 'map-panel';
  panel.innerHTML = `
    <header>
      <div>Maps <span id="map-count"></span></div>
      <button id="import-btn">＋ Import .vxl</button>
    </header>
    <div id="map-list"><div id="no-maps">No maps loaded</div></div>
    <div id="gizmo-bar">
      <button class="gizmo-btn active" data-mode="translate" title="W">Move</button>
      <button class="gizmo-btn"        data-mode="rotate"    title="R">Rotate</button>
      <button class="gizmo-btn"        data-mode="scale"     title="S">Scale</button>
    </div>
    <div id="bottom-bar">
      <label>Place colour</label>
      <input type="color" id="place-color" value="#64b450">
    </div>
    <div id="help-hint">
      RMB drag → fly &nbsp;|&nbsp; LMB dig &nbsp;|&nbsp; MMB place<br>
      WASD move &nbsp;|&nbsp; E/Q up/down &nbsp;|&nbsp; Shift sprint<br>
      W/R/S → gizmo mode &nbsp;|&nbsp; Esc → release cursor
    </div>
  `;
  document.body.appendChild(panel);

  // ── File import ───────────────────────────────────────────────────────────
  const fileInput = document.createElement('input');
  fileInput.type     = 'file';
  fileInput.accept   = '.vxl';
  fileInput.multiple = true;
  fileInput.style.display = 'none';
  document.body.appendChild(fileInput);

  panel.querySelector('#import-btn')!.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', async () => {
    for (const file of Array.from(fileInput.files ?? [])) {
      manager.addFromBuffer(file.name.replace(/\.vxl$/i, ''), await file.arrayBuffer());
    }
    fileInput.value = '';
  });

  // ── Map list ──────────────────────────────────────────────────────────────
  const listEl  = panel.querySelector('#map-list')!;
  const noMaps  = panel.querySelector('#no-maps') as HTMLElement;
  const countEl = panel.querySelector('#map-count')!;

  function rebuildList() {
    const entries = manager.all();
    countEl.textContent = entries.length ? `(${entries.length})` : '';
    noMaps.style.display = entries.length ? 'none' : 'block';

    const existing = new Set<string>();
    for (const el of Array.from(listEl.querySelectorAll('.map-item'))) {
      const id = parseInt((el as HTMLElement).dataset.id!);
      if (!manager.get(id)) el.remove();
      else existing.add(String(id));
    }
    for (const entry of entries) {
      if (existing.has(String(entry.id))) continue;
      const item = document.createElement('div');
      item.className    = 'map-item';
      item.dataset.id   = String(entry.id);
      item.innerHTML    = `<span class="label">${entry.name}</span><span class="del" title="Remove">✕</span>`;
      item.addEventListener('click', (e) => {
        if ((e.target as HTMLElement).classList.contains('del')) manager.remove(entry.id);
        else manager.select(entry.id);
      });
      listEl.appendChild(item);
    }
    syncSelection();
  }

  function syncSelection() {
    const sel = manager.selectedId;
    listEl.querySelectorAll('.map-item').forEach(el => {
      (el as HTMLElement).classList.toggle('active', parseInt((el as HTMLElement).dataset.id!) === sel);
    });
  }

  // ── Gizmo mode buttons ────────────────────────────────────────────────────
  const gizmoBtns = panel.querySelectorAll<HTMLButtonElement>('.gizmo-btn');

  function setMode(mode: GizmoMode) {
    gizmoBtns.forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
    onModeChange(mode);
  }

  gizmoBtns.forEach(btn => {
    btn.addEventListener('click', () => setMode(btn.dataset.mode as GizmoMode));
  });

  // Expose so main.ts can sync W/R/S key shortcuts
  (panel as any).__setMode = setMode;

  // ── Place colour ──────────────────────────────────────────────────────────
  const colorEl = panel.querySelector('#place-color') as HTMLInputElement;
  colorEl.addEventListener('input', () => {
    const h = colorEl.value;
    manager.placeColor = [
      parseInt(h.slice(1,3),16),
      parseInt(h.slice(3,5),16),
      parseInt(h.slice(5,7),16),
    ];
  });

  // ── Manager events ────────────────────────────────────────────────────────
  manager.on(e => {
    if (e.type === 'added' || e.type === 'removed') rebuildList();
    if (e.type === 'selected') syncSelection();
  });

  rebuildList();
}

/** Call from main.ts to sync panel button highlight when using W/R/S keys */
export function setPanelGizmoMode(mode: GizmoMode) {
  const panel = document.getElementById('map-panel');
  if (!panel) return;
  (panel as any).__setMode?.(mode);
}