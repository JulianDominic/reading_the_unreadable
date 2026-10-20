// Browser-style find for the Mirador viewer: Ctrl+F / Cmd+F opens a find bar, every match in the
// OCR word annotations of the visible page is highlighted, and the view jumps to the current one.
//   Enter / Shift+Enter, F3 / Shift+F3, Ctrl+G / Ctrl+Shift+G: next / previous match.  Esc: close.
// Matching is case- and accent-insensitive by default, runs across word boundaries ("the lost"),
// never across text regions, and joins words hyphenated at a line end ("Aban- donment").
// The current match's word is also selected in Mirador's annotation sidebar.
// "All pages" searches every page in /collection.json the same way: Enter steps through the
// matches of all pages (opening each page in the window), and a list shows the pages that match.
const NewspaperFind = (() => {
  const HIGHLIGHT = { fill: "rgba(255, 214, 0, 0.55)", stroke: "rgba(200, 150, 0, 0.8)", width: 1 };
  const CURRENT = { fill: "rgba(255, 120, 0, 0.65)", stroke: "#d84315", width: 2 };
  const READABLE_PX = 12; // a match smaller than this on screen gets zoomed in to
  const TARGET_PX = 30; // on-screen height of a line of text after zooming in

  const wordPages = new Map(); // canvas id -> URL or inline AnnotationPage with the word boxes
  const indexes = new Map(); // canvas id -> Promise of the search index for that canvas
  const imageServices = new Map(); // canvas id -> IIIF image service of its image

  // ---- manifests --------------------------------------------------------------------------

  // Remember where each canvas's words live. Runs before the viewer drops unwanted layers,
  // so search works whichever layers are shown.
  function recordManifest(action) {
    const canvases = action && action.manifestJson && action.manifestJson.items;
    if (!Array.isArray(canvases)) return;
    for (const canvas of canvases) {
      if (canvas.type !== "Canvas") continue;
      const body = (((canvas.items || [])[0] || {}).items || [])[0];
      const service = body && body.body && [].concat(body.body.service || [])[0];
      if (service) imageServices.set(canvas.id, service.id || service["@id"]);
      const pages = canvas.annotations || [];
      const page = pages.find((p) => /\/words\.json$/.test(p.id)) || pages.find((p) => /\/lines\.json$/.test(p.id));
      if (!page) continue;
      wordPages.set(canvas.id, page.items ? page : page.id);
      indexes.delete(canvas.id);
    }
  }

  // ---- index ------------------------------------------------------------------------------

  const FOLDED = { "ſ": "s", "‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"', "–": "-", "—": "-", "¬": "-" };

  // What both the page text and the query are compared as: no accents, uniform quotes and dashes.
  function fold(text, matchCase) {
    const folded = text.normalize("NFD").replace(/\p{M}/gu, "").replace(/[ſ‘’‚“”„–—¬]/g, (c) => FOLDED[c]);
    return matchCase ? folded : folded.toLowerCase();
  }

  // A word's box(es) on its canvas: "canvas#xywh=..." or a SpecificResource whose SVG selector has
  // one rectangle per part (words hyphenated across lines), with an xywh fragment as fallback.
  function parseTarget(target) {
    if (typeof target === "string") {
      const match = target.match(/^([^#]*)#xywh=(?:pixel:)?(\d+),(\d+),(\d+),(\d+)/);
      if (!match) return null;
      const [x, y, w, h] = match.slice(2).map(Number);
      return { canvasId: match[1], x, y, w, h, extra: [] };
    }
    const source = target && target.source && (target.source.id || target.source);
    if (typeof source !== "string") return null;
    const selectors = [].concat(target.selector || []);
    const svg = selectors.find((s) => s.type === "SvgSelector");
    if (svg) {
      const rects = [...svg.value.matchAll(/M(-?[\d.]+),(-?[\d.]+) h(-?[\d.]+) v(-?[\d.]+)/g)]
        .map((m) => ({ x: Number(m[1]), y: Number(m[2]), w: Number(m[3]), h: Number(m[4]) }));
      if (rects.length) return { canvasId: source, ...rects[0], extra: rects.slice(1) };
    }
    const fragment = selectors.find((s) => s.type === "FragmentSelector");
    return fragment ? parseTarget(`${source}#${fragment.value}`) : null;
  }

  function bodyText(body) {
    const bodies = Array.isArray(body) ? body : [body];
    const textual = bodies.find((b) => b && b.type === "TextualBody" && b.purpose !== "tagging");
    return textual ? String(textual.value || "") : "";
  }

  // A hyphen at the end of a line, with the word continuing on the next line of the same region.
  function hyphenatedBreak(word, next) {
    return next && next.region === word.region && /[-‐‑¬–—]$/.test(word.text) && word.text.length > 1
      && next.x < word.x && next.y > word.y + word.h / 2;
  }

  function buildIndex(canvasId, page) {
    const words = [];
    for (const anno of page.items || []) {
      const text = bodyText(anno.body).trim();
      const box = parseTarget(anno.target);
      if (!text || !box || box.canvasId !== canvasId) continue;
      // Word ids end in "<region>-<n>"; a phrase may not run from one region into the next
      const region = String(anno.id || "").replace(/^.*\//, "").replace(/-\d+$/, "");
      words.push({ ...box, id: anno.id, text, region });
    }
    const variants = {};
    // One text per case setting; starts/ends give each word's span in it
    function variant(matchCase) {
      if (variants[matchCase]) return variants[matchCase];
      let text = "";
      const starts = [];
      const ends = [];
      words.forEach((word, i) => {
        const next = words[i + 1];
        const joined = hyphenatedBreak(word, next);
        const piece = fold(joined ? word.text.slice(0, -1) : word.text, matchCase);
        starts.push(text.length);
        text += piece;
        ends.push(text.length);
        if (next) text += next.region !== word.region ? "\n" : joined ? "" : " ";
      });
      variants[matchCase] = { text, starts, ends };
      return variants[matchCase];
    }
    return { canvasId, words, variant };
  }

  function loadIndex(canvasId) {
    if (!indexes.has(canvasId)) {
      const source = wordPages.get(canvasId);
      const promise = !source
        ? Promise.resolve(null)
        : (typeof source === "string" ? fetch(source).then((r) => (r.ok ? r.json() : null)) : Promise.resolve(source))
            .then((page) => (page ? buildIndex(canvasId, page) : null))
            .catch(() => null);
      indexes.set(canvasId, promise);
      promise.then((index) => { if (!index) indexes.delete(canvasId); }); // retry next time
    }
    return indexes.get(canvasId);
  }

  // ---- matching ---------------------------------------------------------------------------

  const WORD_CHAR = /[\p{L}\p{N}]/u;

  // First index i with values[i] > target (values ascending)
  function firstAbove(values, target) {
    let lo = 0;
    let hi = values.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (values[mid] > target) hi = mid; else lo = mid + 1;
    }
    return lo;
  }

  // Merge the words of one match into one box per line
  function lineBoxes(words) {
    const boxes = [];
    for (const word of words) {
      const last = boxes[boxes.length - 1];
      const overlap = last ? Math.min(last.y + last.h, word.y + word.h) - Math.max(last.y, word.y) : 0;
      if (last && word.x >= last.x && overlap > 0.5 * Math.min(last.h, word.h)) {
        const right = Math.max(last.x + last.w, word.x + word.w);
        const bottom = Math.max(last.y + last.h, word.y + word.h);
        last.y = Math.min(last.y, word.y);
        last.w = right - last.x;
        last.h = bottom - last.y;
      } else {
        boxes.push({ x: word.x, y: word.y, w: word.w, h: word.h });
      }
    }
    return boxes;
  }

  function findInIndex(index, query, { matchCase, wholeWord }) {
    const needle = fold(query, matchCase);
    const { text, starts, ends } = index.variant(matchCase);
    const hits = [];
    if (!needle) return hits;
    for (let at = text.indexOf(needle); at !== -1; at = text.indexOf(needle, at + 1)) {
      const end = at + needle.length;
      if (wholeWord && (WORD_CHAR.test(text[at - 1] || "") || WORD_CHAR.test(text[end] || ""))) continue;
      const first = firstAbove(ends, at);
      const last = firstAbove(starts, end - 1) - 1;
      const words = index.words.slice(first, last + 1);
      if (words.length === 0) continue;
      const boxes = lineBoxes(words.flatMap((word) => [word, ...word.extra]));
      const heights = words.map((w) => w.h).sort((a, b) => a - b);
      hits.push({
        canvasId: index.canvasId,
        word: first,
        ids: words.map((w) => w.id),
        center: { x: words[0].x + words[0].w / 2, y: words[0].y + words[0].h / 2 },
        boxes,
        lineHeight: heights[heights.length >> 1],
      });
      at = end - 1; // matches don't overlap, as in a browser
    }
    return hits;
  }

  // ---- viewer state -----------------------------------------------------------------------

  let store = null;

  function target() {
    const state = store.getState();
    const windowId = Mirador.getFocusedWindowId(state) || Mirador.getWindowIds(state)[0];
    if (!windowId) return null;
    const ref = Mirador.OSDReferences.get(windowId);
    return {
      windowId,
      canvasIds: Mirador.getVisibleCanvasIds(state, { windowId }) || [],
      viewer: ref && ref.current,
      world: Mirador.getCurrentCanvasWorld(state, { windowId }),
    };
  }

  // Canvas pixels -> OpenSeadragon viewport (Mirador "world") coordinates
  function toWorld(world, canvasId, box) {
    const { x, y } = world.offsetByCanvas(canvasId);
    const scale = world.canvasScale(canvasId);
    return { x: x + box.x * scale, y: y + box.y * scale, w: box.w * scale, h: box.h * scale };
  }

  // OpenSeadragon isn't a global here, so make its Rects by cloning one from the viewer
  function osdRect(viewer, { x, y, w, h }) {
    return Object.assign(viewer.viewport.getBounds().clone(), { x, y, width: w, height: h, degrees: 0 });
  }

  // ---- find bar ---------------------------------------------------------------------------

  const bar = document.createElement("div");
  bar.className = "find-bar";
  bar.hidden = true;
  bar.setAttribute("role", "search");
  bar.innerHTML = `
    <input type="search" placeholder="Find on page" aria-label="Find on page" spellcheck="false" autocomplete="off">
    <span class="find-count" aria-live="polite"></span>
    <button type="button" class="find-toggle" data-option="matchCase" title="Match case" aria-pressed="false">Aa</button>
    <button type="button" class="find-toggle" data-option="wholeWord" title="Whole words only" aria-pressed="false"><u>ab</u></button>
    <button type="button" class="find-toggle find-scope" data-option="allPages" title="Search all pages" aria-pressed="false">All pages</button>
    <span class="find-divider"></span>
    <button type="button" data-step="-1" title="Previous match (Shift+Enter)" aria-label="Previous match">&#x2303;</button>
    <button type="button" data-step="1" title="Next match (Enter)" aria-label="Next match">&#x2304;</button>
    <button type="button" class="find-close" title="Close (Esc)" aria-label="Close find bar">&#x2715;</button>`;
  const input = bar.querySelector("input");
  const count = bar.querySelector(".find-count");
  const results = document.createElement("div"); // the pages that match, in "All pages" mode
  results.className = "find-results";
  results.hidden = true;

  const options = { matchCase: false, wholeWord: false, allPages: false };
  let hits = [];
  let current = -1;
  let layer = null; // the canvas the highlights are drawn on, see createLayer()
  let shownKey = ""; // window and canvases the current hits belong to
  let searchSeq = 0;

  // All highlights go on one canvas over the image, redrawn from two cached paths (in page
  // coordinates) whenever OpenSeadragon moves the view. Thousands of matches stay smooth, where
  // one DOM overlay per match had OpenSeadragon reposition every element on every frame.
  function createLayer(viewer) {
    const canvas = document.createElement("canvas");
    canvas.className = "find-layer";
    viewer.canvas.appendChild(canvas);
    const redraw = () => {
      const dpr = window.devicePixelRatio || 1;
      const width = viewer.container.clientWidth;
      const height = viewer.container.clientHeight;
      if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
        canvas.width = Math.round(width * dpr);
        canvas.height = Math.round(height * dpr);
      }
      const context = canvas.getContext("2d");
      context.setTransform(1, 0, 0, 1, 0, 0);
      context.clearRect(0, 0, canvas.width, canvas.height);
      if (!layer || !layer.others) return;
      // Page -> screen transform from three points, so rotation and flipping carry over
      const viewport = viewer.viewport;
      const point = (x, y) => viewport.pixelFromPoint(Object.assign(viewport.getCenter(true), { x, y }), true);
      const o = point(0, 0);
      const ax = point(1000, 0);
      const ay = point(0, 1000);
      const [a, b, c, d] = [(ax.x - o.x) / 1000, (ax.y - o.y) / 1000, (ay.x - o.x) / 1000, (ay.y - o.y) / 1000];
      context.setTransform(a * dpr, b * dpr, c * dpr, d * dpr, o.x * dpr, o.y * dpr);
      const pixel = 1 / Math.hypot(a, b); // one screen pixel in page units
      for (const [path, style] of [[layer.others, HIGHLIGHT], [layer.current, CURRENT]]) {
        context.fillStyle = style.fill;
        context.fill(path);
        context.lineWidth = style.width * pixel;
        context.strokeStyle = style.stroke;
        context.stroke(path);
      }
    };
    viewer.addHandler("update-viewport", redraw);
    return { viewer, canvas, redraw, rects: [], others: null, current: null };
  }

  function clearHighlights() {
    if (!layer) return;
    layer.viewer.removeHandler("update-viewport", layer.redraw);
    layer.canvas.remove();
    layer = null;
  }

  function drawHighlights(t) {
    if (!t.viewer || hits.length === 0) {
      clearHighlights();
      return;
    }
    if (!layer || layer.viewer !== t.viewer) {
      clearHighlights();
      layer = createLayer(t.viewer);
    }
    layer.rects = hits.map((hit) => hit.boxes.map((box) => {
      const pad = Math.round(box.h * 0.12);
      return toWorld(t.world, hit.canvasId, { x: box.x - pad, y: box.y - pad, w: box.w + 2 * pad, h: box.h + 2 * pad });
    }));
    markCurrent();
  }

  function markCurrent() {
    if (!layer) return;
    layer.others = new Path2D();
    layer.current = new Path2D();
    layer.rects.forEach((rects, i) => {
      const path = i === current ? layer.current : layer.others;
      for (const r of rects) path.rect(r.x, r.y, r.w, r.h);
    });
    layer.redraw();
  }

  // A new image is reset to the whole page a few times while it opens (OpenSeadragon's goHome,
  // Mirador's fit once the first tile arrives), which would undo a jump. Mirador keeps the same
  // viewer when a window switches pages, so this is tracked per image: returns true when the
  // canvas's image is past that, otherwise calls back once it has fully loaded for the first time.
  const settled = new WeakSet(); // tiled images
  const waiting = new WeakMap(); // viewer -> canvas id it is waiting for

  function imageFor(viewer, canvasId) {
    const want = imageServices.get(canvasId);
    const norm = (id) => String(id || "").replace(/\/info\.json$/, "").replace(/\/$/, "");
    for (let i = 0; i < viewer.world.getItemCount(); i++) {
      const item = viewer.world.getItemAt(i);
      if (!want || norm(item.source && (item.source.id || item.source["@id"])) === norm(want)) return item;
    }
    return null;
  }

  function whenSettled(viewer, canvasId, callback) {
    const item = imageFor(viewer, canvasId);
    if (item && settled.has(item)) return true;
    if (item && item.getFullyLoaded()) {
      settled.add(item);
      return true;
    }
    if (waiting.get(viewer) === canvasId) return false;
    waiting.set(viewer, canvasId);
    const waitFor = (image) => image.addHandler("fully-loaded-change", function loaded(event) {
      if (!event.fullyLoaded) return;
      image.removeHandler("fully-loaded-change", loaded);
      settled.add(image);
      if (waiting.get(viewer) === canvasId) waiting.delete(viewer);
      setTimeout(callback);
    });
    if (item) {
      waitFor(item);
    } else {
      // The page's image isn't in the viewer yet (or the previous page's still is)
      viewer.world.addHandler("add-item", function added() {
        const image = imageFor(viewer, canvasId);
        if (!image) return;
        viewer.world.removeHandler("add-item", added);
        waitFor(image);
      });
    }
    return false;
  }

  // Pan (and zoom in if the text is too small to read) so the current match is on screen,
  // leaving the view alone when it already is, like a browser scrolling to a match.
  function reveal(t) {
    const hit = hits[current];
    if (!hit || !t.viewer) return;
    const settledNow = whenSettled(t.viewer, hit.canvasId, () => {
      const now = target();
      if (now && now.viewer === t.viewer && hits.length) reveal(now);
    });
    if (!settledNow) return;
    const viewport = t.viewer.viewport;
    const boxes = hit.boxes.map((box) => toWorld(t.world, hit.canvasId, box));
    const left = Math.min(...boxes.map((b) => b.x));
    const top = Math.min(...boxes.map((b) => b.y));
    const right = Math.max(...boxes.map((b) => b.x + b.w));
    const bottom = Math.max(...boxes.map((b) => b.y + b.h));
    const lineHeight = hit.lineHeight * t.world.canvasScale(hit.canvasId);

    const view = viewport.getBounds();
    const container = viewport.getContainerSize();
    const pxPerUnit = container.x / view.width;
    const readable = lineHeight * pxPerUnit >= READABLE_PX;
    const marginX = view.width * 0.05;
    const marginY = view.height * 0.05;
    const inView = left >= view.x + marginX && right <= view.x + view.width - marginX
      && top >= view.y + marginY && bottom <= view.y + view.height - marginY;
    if (readable && inView) return;

    let scale = readable ? pxPerUnit : TARGET_PX / lineHeight;
    scale = Math.min(scale, (0.8 * container.x) / (right - left), (0.8 * container.y) / (bottom - top));
    const width = container.x / scale;
    const height = container.y / scale;
    const cx = (left + right) / 2;
    const cy = (top + bottom) / 2;
    viewport.fitBounds(osdRect(t.viewer, { x: cx - width / 2, y: cy - height / 2, w: width, h: height }));
  }

  // ---- annotation sidebar -----------------------------------------------------------------

  let selectedByFind = null; // the annotation find selected, so it only ever clears its own
  let annotationsSeen = null; // the shown annotations the selection was chosen from
  const idSets = new WeakMap(); // annotation list -> Set of its annotation ids

  // The shown annotation for a match: its first word, or with the word layer hidden the smallest
  // shown annotation (a line or block) containing that word. Null while nothing is loaded.
  function annotationFor(hit, lists) {
    for (const list of lists) {
      if (!idSets.has(list)) idSets.set(list, new Set(list.resources.map((r) => r.id)));
      const word = hit.ids.find((id) => idSets.get(list).has(id));
      if (word) return word;
    }
    let best = null;
    let bestArea = Infinity;
    for (const resource of lists.flatMap((list) => list.resources)) {
      const box = resource.targetId === hit.canvasId && resource.fragmentSelector;
      if (!box) continue;
      const [x, y, w, h] = box;
      const { x: cx, y: cy } = hit.center;
      if (cx >= x && cx <= x + w && cy >= y && cy <= y + h && w * h < bestArea) {
        best = resource.id;
        bestArea = w * h;
      }
    }
    return best;
  }

  // Select the current match in the sidebar; Mirador highlights it there and scrolls it into view
  function selectCurrent(t) {
    const state = store.getState();
    const lists = Mirador.getPresentAnnotationsOnSelectedCanvases(state, { windowId: t.windowId });
    annotationsSeen = lists;
    const id = hits[current] ? annotationFor(hits[current], lists) : null;
    const selected = Mirador.getSelectedAnnotationId(state, { windowId: t.windowId });
    if (id && id !== selected) store.dispatch(Mirador.selectAnnotation(t.windowId, id));
    else if (!id && selected && selected === selectedByFind) store.dispatch(Mirador.deselectAnnotation(t.windowId, selected));
    selectedByFind = id;
  }

  function updateCount(status) {
    const total = options.allPages ? all.length : hits.length;
    const position = options.allPages ? allCurrent : current;
    count.textContent = status !== undefined ? status : total ? `${position + 1} of ${total}` : "No results";
    bar.classList.toggle("find-none", input.value.trim() !== "" && total === 0 && status === undefined);
  }

  // The match to start on: keep the reading position while the query is being typed,
  // otherwise the first match on screen, otherwise the first match.
  function startingHit(previous, t) {
    if (previous) {
      const order = (h) => [t.canvasIds.indexOf(h.canvasId), h.word];
      const [pc, pw] = order(previous);
      const i = hits.findIndex((h) => { const [c, w] = order(h); return c > pc || (c === pc && w >= pw); });
      if (i !== -1) return i;
    }
    if (t.viewer) {
      const view = t.viewer.viewport.getBounds();
      const i = hits.findIndex((h) => h.boxes.some((box) => {
        const r = toWorld(t.world, h.canvasId, box);
        return r.x >= view.x && r.y >= view.y && r.x + r.w <= view.x + view.width && r.y + r.h <= view.y + view.height;
      }));
      if (i !== -1) return i;
    }
    return 0;
  }

  async function search({ keepPosition = true } = {}) {
    if (options.allPages) return searchAll({ keepPosition });
    results.hidden = true;
    const seq = ++searchSeq;
    const query = input.value.trim().replace(/\s+/g, " ");
    const t = store && target();
    const previous = keepPosition ? hits[current] : null;
    if (!query || !t || t.canvasIds.length === 0) {
      hits = [];
      current = -1;
      shownKey = "";
      clearHighlights();
      updateCount(query ? "Page not loaded" : "");
      if (t) selectCurrent(t);
      return;
    }
    const loaded = await Promise.all(t.canvasIds.map(loadIndex));
    if (seq !== searchSeq) return; // a newer search started while the words were loading
    if (loaded.every((index) => !index)) {
      hits = [];
      current = -1;
      clearHighlights();
      updateCount("No text on this page");
      selectCurrent(t);
      return;
    }
    hits = loaded.filter(Boolean).flatMap((index) => findInIndex(index, query, options));
    current = hits.length ? startingHit(previous, t) : -1;
    shownKey = targetKey(t);
    drawHighlights(t);
    updateCount();
    reveal(t);
    selectCurrent(t);
  }

  // ---- all pages ----------------------------------------------------------------------------

  let collectionSource = "/collection.json";
  let collection = null; // Promise of [{ manifestId, label }]
  const manifests = new Map(); // manifest id -> Promise of its canvas ids
  let all = []; // every match in the collection: { manifestId, canvasId, word, page }
  let allCurrent = -1;
  let pending = null; // index into all, waiting for the window to open its page

  function label(value) {
    if (!value) return "";
    if (typeof value === "string") return value;
    const values = Object.values(value)[0];
    return Array.isArray(values) ? values.join(" ") : String(values || "");
  }

  function loadCollection() {
    if (!collection) {
      collection = fetch(collectionSource)
        .then((r) => (r.ok ? r.json() : { items: [] }))
        .then((c) => (c.items || []).filter((i) => i.type === "Manifest").map((i) => ({ manifestId: i.id, label: label(i.label) || i.id })))
        .catch(() => []);
    }
    return collection;
  }

  // A manifest's canvases, recording where their words are (as for the manifests Mirador loads)
  function loadManifest(manifestId) {
    if (!manifests.has(manifestId)) {
      manifests.set(manifestId, fetch(manifestId)
        .then((r) => (r.ok ? r.json() : null))
        .then((json) => {
          if (!json) return [];
          if (!(json.items || []).some((c) => wordPages.has(c.id))) recordManifest({ manifestJson: json });
          return (json.items || []).filter((c) => c.type === "Canvas").map((c) => c.id);
        })
        .catch(() => []));
    }
    return manifests.get(manifestId);
  }

  // Run fn over items, a few at a time
  async function eachLimited(items, limit, fn) {
    let next = 0;
    const worker = async () => {
      while (next < items.length) {
        const i = next++;
        await fn(items[i], i);
      }
    };
    await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  }

  async function searchAll({ keepPosition = true } = {}) {
    const seq = ++searchSeq;
    const query = input.value.trim().replace(/\s+/g, " ");
    const previous = keepPosition ? all[allCurrent] : null;
    pending = null;
    if (!query || !store) {
      all = [];
      allCurrent = -1;
      hits = [];
      current = -1;
      shownKey = "";
      clearHighlights();
      results.hidden = true;
      updateCount("");
      return;
    }
    const pages = await loadCollection();
    let done = 0;
    const found = new Array(pages.length);
    await eachLimited(pages, 4, async (page, i) => {
      const canvasIds = await loadManifest(page.manifestId);
      const perPage = [];
      for (const canvasId of canvasIds) {
        const index = await loadIndex(canvasId);
        if (!index) continue;
        for (const hit of findInIndex(index, query, options)) {
          const around = index.words.slice(Math.max(0, hit.word - 6), hit.word + 8).map((w) => w.text).join(" ");
          perPage.push({ manifestId: page.manifestId, canvasId, word: hit.word, page: i, snippet: around });
        }
      }
      found[i] = perPage;
      done++;
      if (seq === searchSeq && done < pages.length) updateCount(`Searching ${done} of ${pages.length} pages`);
    });
    if (seq !== searchSeq) return; // a newer search started meanwhile
    all = found.flat();
    renderResults(pages);
    if (!all.length) {
      allCurrent = -1;
      hits = [];
      current = -1;
      clearHighlights();
      updateCount();
      return;
    }
    // Keep the reading position while typing; otherwise start on the page that is open
    const t = target();
    let start = -1;
    if (previous) start = all.findIndex((g) => g.page > previous.page || (g.page === previous.page && g.word >= previous.word));
    if (start === -1 && t) start = all.findIndex((g) => t.canvasIds.includes(g.canvasId));
    showGlobal(start === -1 ? 0 : start);
  }

  function renderResults(pages) {
    const byPage = new Map();
    all.forEach((g, i) => {
      if (!byPage.has(g.page)) byPage.set(g.page, { first: i, n: 0, snippet: g.snippet });
      byPage.get(g.page).n++;
    });
    results.replaceChildren();
    const head = document.createElement("div");
    head.className = "find-results-head";
    head.textContent = all.length ? `${all.length} matches on ${byPage.size} of ${pages.length} pages` : `No matches on ${pages.length} pages`;
    results.append(head);
    for (const [page, info] of byPage) {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "find-result";
      row.dataset.first = info.first;
      row.dataset.page = page;
      const title = document.createElement("strong");
      title.textContent = `${pages[page].label}  ·  ${info.n}`;
      const snippet = document.createElement("span");
      snippet.textContent = info.snippet;
      row.append(title, snippet);
      results.append(row);
    }
    results.hidden = false;
  }

  // Show match i of the collection: open its page if needed, then highlight it like a page search
  async function showGlobal(i) {
    const g = all[i];
    const t = store && target();
    if (!g || !t) return;
    allCurrent = i;
    for (const row of results.querySelectorAll(".find-result")) row.classList.toggle("find-result-current", Number(row.dataset.page) === g.page);
    if (!t.canvasIds.includes(g.canvasId)) {
      pending = i;
      shownKey = "";
      clearHighlights();
      updateCount("Opening page");
      store.dispatch(Mirador.updateWindow(t.windowId, { manifestId: g.manifestId, canvasId: g.canvasId }));
      return;
    }
    const query = input.value.trim().replace(/\s+/g, " ");
    const loaded = await Promise.all(t.canvasIds.map(loadIndex));
    if (allCurrent !== i) return;
    hits = loaded.filter(Boolean).flatMap((index) => findInIndex(index, query, options));
    current = hits.findIndex((h) => h.canvasId === g.canvasId && h.word === g.word);
    shownKey = targetKey(t);
    drawHighlights(t);
    updateCount();
    reveal(t);
    selectCurrent(t);
  }

  // The user opened another page themselves: carry on from its first match, if it has one
  function followNavigation(t) {
    shownKey = targetKey(t);
    const i = t ? all.findIndex((g) => t.canvasIds.includes(g.canvasId)) : -1;
    if (i !== -1) {
      showGlobal(i);
    } else {
      hits = [];
      current = -1;
      clearHighlights();
      updateCount("No matches on this page");
    }
  }

  function step(direction) {
    if (bar.hidden) open();
    if (options.allPages) {
      if (all.length) showGlobal((allCurrent + direction + all.length) % all.length);
      return;
    }
    if (hits.length === 0) return;
    current = (current + direction + hits.length) % hits.length;
    const t = target();
    if (!layer || layer.viewer !== t.viewer) drawHighlights(t); else markCurrent();
    updateCount();
    reveal(t);
    selectCurrent(t);
  }

  function open() {
    bar.hidden = false;
    input.focus();
    input.select();
    if (input.value.trim()) search({ keepPosition: false });
  }

  function close() {
    bar.hidden = true;
    results.hidden = true;
    pending = null;
    searchSeq++;
    clearHighlights();
    hits = [];
    current = -1;
    shownKey = "";
    updateCount("");
    input.blur();
  }

  function targetKey(t) {
    return t ? `${t.windowId}|${t.canvasIds.join(",")}` : "";
  }

  input.addEventListener("input", () => search());
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      step(event.shiftKey ? -1 : 1);
    } else if (event.key === "Escape") {
      event.preventDefault();
      close();
    }
  });
  bar.addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    if (button.dataset.step) step(Number(button.dataset.step));
    else if (button.classList.contains("find-close")) close();
    else if (button.dataset.option) {
      const option = button.dataset.option;
      options[option] = !options[option];
      button.setAttribute("aria-pressed", String(options[option]));
      if (option === "allPages") input.placeholder = options.allPages ? "Find in all pages" : "Find on page";
      input.focus();
      search({ keepPosition: false });
    }
  });
  results.addEventListener("click", (event) => {
    const row = event.target.closest("[data-first]");
    if (row) showGlobal(Number(row.dataset.first));
  });
  results.addEventListener("mousedown", (event) => event.preventDefault());

  // Clicks on the bar shouldn't reach Mirador (it would take focus from the input)
  bar.addEventListener("mousedown", (event) => { if (event.target !== input) event.preventDefault(); });

  // Capture phase, so the shortcuts win over OpenSeadragon's and the browser's own find
  window.addEventListener("keydown", (event) => {
    const mod = event.ctrlKey || event.metaKey;
    const key = event.key.toLowerCase();
    if (mod && !event.altKey && key === "f") {
      event.preventDefault();
      event.stopPropagation();
      open();
    } else if ((mod && key === "g") || event.key === "F3") {
      if (bar.hidden && !input.value.trim()) return;
      event.preventDefault();
      event.stopPropagation();
      step(event.shiftKey ? -1 : 1);
    }
  }, true);

  // Mirador's fullscreen mode only shows its own element; keep the bar inside it
  document.addEventListener("fullscreenchange", () => {
    (document.fullscreenElement || document.body).append(bar, results);
  });

  // ---- wiring -----------------------------------------------------------------------------

  // Redux calls subscribers inside dispatch. Reacting there (our own dispatches included) in the
  // middle of Mirador's saga chain for a page switch made redux-saga miss the action that requests
  // the new page's image, which then never appeared. So react on the next tick, once per burst.
  let storeChangeQueued = false;
  function onStoreChange() {
    if (storeChangeQueued) return;
    storeChangeQueued = true;
    setTimeout(() => {
      storeChangeQueued = false;
      handleStoreChange();
    }, 0);
  }

  // Re-run the search when the user changes page, window or view type
  function handleStoreChange() {
    if (bar.hidden || !input.value.trim()) return;
    const t = target();
    if (options.allPages && targetKey(t) !== shownKey) {
      if (pending !== null) {
        // Waiting for the window to show the page of the next match
        if (t && all[pending] && t.canvasIds.includes(all[pending].canvasId)) {
          const i = pending;
          pending = null;
          showGlobal(i);
        }
      } else {
        followNavigation(t);
      }
    } else if (targetKey(t) !== shownKey) {
      search({ keepPosition: false });
    } else if (hits.length && t.viewer && (!layer || layer.viewer !== t.viewer)) {
      // The viewer wasn't ready when the search ran, or Mirador replaced it
      drawHighlights(t);
      reveal(t);
    } else if (hits.length && Mirador.getPresentAnnotationsOnSelectedCanvases(store.getState(), { windowId: t.windowId }) !== annotationsSeen) {
      // The sidebar's annotations arrived after the search ran
      selectCurrent(t);
    }
  }

  function attach(miradorStore, { collectionUrl = "/collection.json" } = {}) {
    store = miradorStore;
    collectionSource = collectionUrl;
    document.body.append(bar, results);
    store.subscribe(onStoreChange);
  }

  return { attach, recordManifest };
})();
