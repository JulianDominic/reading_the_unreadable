// Performance fixes for Mirador 4.3.0, applied to its exported classes before the viewer starts.

// 1. CanvasAnnotationDisplay.fragmentContext() calls context.save() and then returns early, without
//    restore(), for every annotation that isn't visible. Word boxes are hidden until "highlight all"
//    is switched on, so a page with 2,800 words grows the canvas state stack by 2,800 entries on
//    every frame. Chrome resolves those saves on the next transform call, so panning and zooming
//    slow down the longer the viewer is open. Skip invisible annotations before save() instead.
(() => {
  const display = Mirador.CanvasAnnotationDisplay.prototype;
  const fragmentContext = display.fragmentContext;
  display.fragmentContext = function () {
    const { palette } = this;
    const current = this.selected ? palette.selected : this.hovered ? palette.hovered : palette.default;
    if (current.globalAlpha === 0) return undefined;
    return fragmentContext.call(this);
  };
})();

// 2. Annotations re-parse their target on every access: each redraw and each mouse move over the
//    image runs a regex per annotation. Annotation objects never change once built, so cache it.
(() => {
  function memoize(cls, name) {
    const getter = Object.getOwnPropertyDescriptor(cls.prototype, name).get;
    const cache = new WeakMap();
    Object.defineProperty(cls.prototype, name, {
      configurable: true,
      get() {
        if (!cache.has(this)) cache.set(this, getter.call(this));
        return cache.get(this);
      },
    });
  }
  for (const cls of [Mirador.AnnotationItem, Mirador.AnnotationResource]) {
    memoize(cls, "fragmentSelector");
    memoize(cls, "targetId");
  }
})();

// 3. Releasing a mouse drag over the image counts as a click: OpenSeadragon raises canvas-click
//    with quick=false, and Mirador's handler only ignores that for touch and pen. So letting go of
//    a pan over a word selected that word, and re-rendering the annotation sidebar (thousands of
//    words) stalled the viewer for about 250 ms. Drop non-quick clicks; OpenSeadragon's own
//    click-to-zoom only acts on quick clicks anyway. OpenSeadragon isn't a global in this build,
//    so the patch goes on the prototype of the first viewer Mirador registers.
(() => {
  let patched = false;
  function patch(viewer) {
    if (patched || !viewer) return;
    patched = true;
    const proto = Object.getPrototypeOf(viewer);
    const raiseEvent = proto.raiseEvent;
    proto.raiseEvent = function (eventName, args) {
      if (eventName === "canvas-click" && args && args.quick === false) return undefined;
      return raiseEvent.call(this, eventName, args);
    };
  }
  const set = Mirador.OSDReferences.set.bind(Mirador.OSDReferences);
  Mirador.OSDReferences.set = (windowId, ref) => {
    set(windowId, ref);
    // apiRef.current is filled in once OpenSeadragon has been created
    const wait = () => (ref && ref.current ? patch(ref.current) : !patched && setTimeout(wait, 50));
    wait();
  };
})();
