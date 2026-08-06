/* ================================================================
 * Dotted thought-orb — the `orbits` ("working") painter, ported to
 * plain canvas from thinking-orbs <https://orbs.jakubantalik.com>.
 * MIT License, Copyright (c) 2026 Jakub Antalik.
 *
 * Only this one mode is carried over, with its shipped 64px / 20px
 * tunings pre-resolved into constants (the upstream preset machinery
 * scales counts linearly and radii by a size multiplier; both are
 * already applied below). Grayscale ink is tinted toward the cobalt
 * accent so the mark belongs to this design system.
 *
 * Shared by the control panel (header mark) and the chat page (empty
 * state + streaming indicator), so it is served as its own asset
 * rather than inlined twice.
 * ================================================================ */

const ORB_PRESETS = {
  64: {
    speed: 1.885,
    opts: { orbitN: 12, ghostN: 40, ghostR: 0.9, ghostA: 0.5, particles: 3, partR: 1.2, partRDepth: 1.6, rsPow: 0.6, rMin: 0.3 },
  },
  20: {
    speed: 3.9,
    opts: { orbitN: 3, ghostN: 10, ghostR: 2.16, ghostA: 0.5, particles: 3, partR: 2.88, partRDepth: 3.84, rsPow: 0.6, rMin: 0.3 },
  },
};
/* Multiplied into the grayscale ink value; max channel 1 keeps highlights bright. */
const ORB_TINT = [0.72, 0.84, 1];

/** Deterministic hash in [0, 1). */
function orbHash(a, b) {
  const h = Math.sin(a * 12.9898 + b * 78.233) * 43758.5453;
  return h - Math.floor(h);
}

/** Spin + tilt + orthographic projection. */
function orbProjector(yaw, tilt, cx, cy, scale) {
  const st = Math.sin(tilt), ct = Math.cos(tilt);
  const sy = Math.sin(yaw), cyw = Math.cos(yaw);
  return (x, y, z) => {
    const x1 = x * cyw + z * sy;
    const z1 = -x * sy + z * cyw;
    const y1 = y * ct - z1 * st;
    const z2 = y * st + z1 * ct;
    return [cx + x1 * scale, cy - y1 * scale, z2];
  };
}

/** Dot radii were tuned for a 300pt frame; sub-linear scaling keeps small orbs legible. */
function orbRadiusScale(size, pow) {
  return Math.pow(size / 300, pow);
}

/** z-sort far→near, then fill matte dots. Ink is mirrored for the dark substrate. */
function orbPaint(ctx, dots, rMin) {
  const light =
    typeof document !== 'undefined' &&
    document.documentElement &&
    document.documentElement.dataset.theme === 'light';
  dots.sort((a, b) => a.z - b.z);
  for (const d of dots) {
    const alpha = d.a === undefined ? 1 : d.a;
    if (alpha < 0.02) continue;
    const w = Math.min(1, Math.max(0, d.white));
    const g = (light ? w : 1 - w) * 255;
    const r = Math.round(g * ORB_TINT[0]);
    const gr = Math.round(g * ORB_TINT[1]);
    const b = Math.round(g * ORB_TINT[2]);
    ctx.fillStyle = 'rgba(' + r + ',' + gr + ',' + b + ',' + alpha + ')';
    ctx.beginPath();
    ctx.arc(d.x, d.y, Math.max(rMin, d.r), 0, Math.PI * 2);
    ctx.fill();
  }
}

/** Particles on tilted orbits: a ghost path per orbit plus the ones doing the work. */
function drawOrbits(ctx, size, t, o) {
  const cx = size / 2;
  const cy = size / 2;
  const R = (size / 2) * 0.82;
  const project = orbProjector(t * 0.12, 0.3, cx, cy, 1);
  const rs = orbRadiusScale(size, o.rsPow);
  const dots = [];

  for (let orb = 0; orb < o.orbitN; orb++) {
    const h1 = orbHash(orb, 1.7);
    const h2 = orbHash(orb, 5.2);
    const h3 = orbHash(orb, 8.9);
    const ro = R * (0.45 + 0.52 * h1);
    const th = h1 * 2 * Math.PI;
    const phi = Math.acos(2 * h2 - 1);
    // orbit plane basis (u, v perpendicular to normal n)
    const nx = Math.sin(phi) * Math.cos(th);
    const ny = Math.cos(phi);
    const nz = Math.sin(phi) * Math.sin(th);
    let ux = -ny, uy = nx;
    const uz = 0;
    const ul = Math.max(1e-6, Math.sqrt(ux * ux + uy * uy));
    ux /= ul;
    uy /= ul;
    const vx = ny * uz - nz * uy;
    const vy = nz * ux - nx * uz;
    const vz = nx * uy - ny * ux;
    const speed = (0.25 + 0.55 * h3) * (h3 > 0.5 ? 1 : -1);

    for (let k = 0; k < o.ghostN; k++) {
      const a = (k / o.ghostN) * 2 * Math.PI;
      const ca = Math.cos(a), sa = Math.sin(a);
      const p = project((ux * ca + vx * sa) * ro, (uy * ca + vy * sa) * ro, (uz * ca + vz * sa) * ro);
      const depth = (p[2] / ro + 1) / 2;
      dots.push({ x: p[0], y: p[1], z: p[2], r: o.ghostR * rs, white: 0.72, a: o.ghostA * (0.4 + 0.6 * depth) });
    }
    for (let m = 0; m < o.particles; m++) {
      const a = t * speed + (m / o.particles) * 2 * Math.PI + h2 * 6;
      const ca = Math.cos(a), sa = Math.sin(a);
      const p = project((ux * ca + vx * sa) * ro, (uy * ca + vy * sa) * ro, (uz * ca + vz * sa) * ro);
      const depth = (p[2] / ro + 1) / 2;
      dots.push({ x: p[0], y: p[1], z: p[2], r: (o.partR + o.partRDepth * depth) * rs, white: 0.3 - 0.22 * depth });
    }
  }
  orbPaint(ctx, dots, o.rMin);
}

/**
 * Drive an orb on `canvas`. Returns { play, pause, stop }.
 *
 * With `autoplay: false` the orb paints one static frame and waits — used
 * for the header mark, which only spins while hovered so the control panel
 * chrome stays calm. Pauses on hidden tabs; reduced-motion users always get
 * the single static frame.
 */
function mountOrb(canvas, size, options) {
  const settings = options || {};
  const autoplay = settings.autoplay !== false;
  const preset = ORB_PRESETS[size];
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(size * dpr);
  canvas.height = Math.round(size * dpr);
  canvas.style.width = size + 'px';
  canvas.style.height = size + 'px';
  const ctx = canvas.getContext('2d');
  const noop = { play() {}, pause() {}, stop() {} };
  if (!ctx) return noop;

  const frame = (tSec) => {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);
    drawOrbits(ctx, size, tSec, preset.opts);
  };
  const liveFrame = () => frame((performance.now() / 1000) * preset.speed);

  // Paint one frame up front so the orb is never blank while waiting for
  // the first animation frame (rAF is throttled on hidden/offscreen tabs).
  liveFrame();

  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    frame(0.6);
    return noop;
  }

  let raf = 0;
  let running = false;
  let stopped = false;
  const loop = () => {
    liveFrame();
    if (running) raf = requestAnimationFrame(loop);
  };
  const play = () => {
    if (running || stopped) return;
    running = true;
    raf = requestAnimationFrame(loop);
  };
  const pause = () => {
    running = false;
    cancelAnimationFrame(raf);
  };
  const onVisibility = () => {
    if (document.visibilityState === 'hidden') pause();
    else if (autoplay && !stopped) play();
  };
  document.addEventListener('visibilitychange', onVisibility);
  if (autoplay) play();

  return {
    play,
    pause,
    stop() {
      stopped = true;
      pause();
      document.removeEventListener('visibilitychange', onVisibility);
    },
  };
}
