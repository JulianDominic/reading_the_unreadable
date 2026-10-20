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
