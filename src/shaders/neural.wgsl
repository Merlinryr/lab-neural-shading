// Every compute pass of the lab lives here, sharing one bind group:
//   showTarget — the ground truth on screen: the Cycles frame for this camera
//   classic — real-time jade the way a game does it (the traditional technique we compare against)
//   train   — forward + backward pass of the MLP on a random batch of pixels
//   adam    — sums the batch gradients and updates the weights
//   infer   — runs the MLP on every pixel (the neural render)
//   score   — per-pixel error and brightness of classic or neural against the Cycles frame
//
// Both real-time shaders share the analytic specular. The network replaces one quantity: the
// light that went into the stone and came back out (classicSubsurface in the game version).
//
// IN, HID, OUT, FREQS, POS_FREQS, BATCH and the scene's lights (SUN*, WORLD) are prepended by
// main.ts so TS, WGSL and the Blender scene agree.

const PI = 3.14159265;
const BG = vec3f(0.02);

// Flat weight layout. Row i of a weight matrix = the weights feeding output neuron i.
const W0 = 0u;                 // HID x IN
const B0 = W0 + HID * IN;      // HID
const W1 = B0 + HID;           // HID x HID
const B1 = W1 + HID * HID;     // HID
const W2 = B1 + HID;           // OUT x HID
const B2 = W2 + OUT * HID;     // OUT
const P  = B2 + OUT;           // MLP parameter count; the neural texture's GRID_P texels follow it in params
const FIX = 16777216.0;        // 2^24: fixed point for the neural texture's atomic gradients

struct Frame {
  camPos: vec3f,
  seed: u32,
  mask: vec4f,      // inference only: albedo / normal / view / baked input groups on (1) or off (0)
  step: u32,        // Adam step, starts at 1
  lr: f32,
  targetFrame: u32, // which Cycles frame is the ground truth
  posMask: f32,     // inference only: position input on (1) or off (0)
  jadeA: vec4f,     // classic shader knobs, fitted by main.ts: wrap, distortion, power, translucency
  jadeB: vec4f,     //   absorption, shadow softness, ambient, specular (shared by both shaders)
  scoreClassic: u32, // score: 1 = the classic shader, 0 = the network
  specMask: f32,    // display only: add the analytic specular (1) or show the subsurface alone (0)
  gridMask: vec2f,  // inference only: the neural texture's coarse / fine levels on (1) or off (0)
}

@group(0) @binding(0) var gAlbedo: texture_2d<f32>;
@group(0) @binding(1) var gNormal: texture_2d<f32>;
@group(0) @binding(2) var gPos: texture_2d<f32>;
@group(0) @binding(3) var<uniform> frame: Frame;
@group(0) @binding(4) var<storage, read_write> params: array<f32>;  // P + GRID_P
@group(0) @binding(5) var<storage, read_write> adamM: array<f32>;   // P + GRID_P
@group(0) @binding(6) var<storage, read_write> adamV: array<f32>;   // P + GRID_P
@group(0) @binding(7) var<storage, read_write> grads: array<f32>;   // BATCH x P, one row per sample
@group(0) @binding(8) var<storage, read_write> losses: array<f32>;  // BATCH per-sample, then the mean
@group(0) @binding(9) var outTarget: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var outNeural: texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var outFeatures: texture_storage_2d<rgba16float, write>;
@group(0) @binding(12) var frames: texture_2d_array<f32>; // Cycles renders, public/jade/frames
@group(0) @binding(13) var gBaked: texture_2d<f32>;
@group(0) @binding(14) var outClassic: texture_storage_2d<rgba16float, write>;
// The neural texture's gradients. WGSL has no float atomics, so samples add fixed-point ints (x FIX).
@group(0) @binding(15) var<storage, read_write> gridGrad: array<atomic<i32>>; // GRID_P
// The other copy of the parameters: the EMA when training (adam writes it), the raw ones when
// displaying. infer and score read `params`, which is the EMA in that bind group: no flicker.
@group(0) @binding(16) var<storage, read_write> ema: array<f32>;  // P + GRID_P
@group(0) @binding(17) var outGrid: texture_storage_2d<rgba16float, write>; // neural texture tiles: coarse .rg, fine .ba

// ---------------------------------------------------------------- G-buffer

struct Surface {
  albedo: vec3f,
  n: vec3f,
  v: vec3f,   // towards the camera
  pos: vec3f,
  baked: vec4f,   // thickness, AO, key sun visibility, back sun visibility (baked per vertex)
  through: vec2f, // distance sunlight travels through the stone to get here, key and back sun
  hit: bool,
}

fn surface(p: vec2i) -> Surface {
  let pos = textureLoad(gPos, p, 0);
  // Background: return harmless values, NaNs would poison the gradients.
  if (pos.w < 0.5) { return Surface(vec3f(0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), vec3f(0.0), vec4f(0.0), vec2f(0.0), false); }
  let albedo = textureLoad(gAlbedo, p, 0);
  let normal = textureLoad(gNormal, p, 0);
  return Surface(
    albedo.rgb,
    normalize(normal.xyz),
    normalize(frame.camPos - pos.xyz),
    pos.xyz,
    textureLoad(gBaked, p, 0),
    vec2f(albedo.a, normal.w),
    true,
  );
}

// The Cycles frame for this camera. Alpha < 1 on the silhouette, where a pixel is part background:
// the G-buffer and Cycles disagree there, so training and scoring skip those pixels.
fn groundTruth(p: vec2i) -> vec4f { return textureLoad(frames, p, frame.targetFrame, 0); }

fn srgb(c: vec3f) -> vec3f { // linear -> display, exactly Blender's Standard view transform
  let x = clamp(c, vec3f(0.0), vec3f(1.0));
  return select(1.055 * pow(x, vec3f(1.0 / 2.4)) - 0.055, 12.92 * x, x <= vec3f(0.0031308));
}
fn linear(c: vec3f) -> vec3f { // display -> linear radiance, the inverse of srgb()
  return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}

// ---------------------------------------------------------------- the easy part: specular

fn ggx(n: vec3f, v: vec3f, l: vec3f, rough: f32) -> f32 {
  let h = normalize(v + l);
  let nh = max(dot(n, h), 0.0);
  let nv = max(dot(n, v), 1e-3);
  let nl = max(dot(n, l), 1e-3);
  let a2 = rough * rough * rough * rough;
  let d = a2 / (PI * pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
  let k = (rough + 1.0) * (rough + 1.0) / 8.0;
  let g = nv / (nv * (1.0 - k) + k) * nl / (nl * (1.0 - k) + k);
  return d * g / (4.0 * nv * nl);
}

fn fresnel(s: Surface) -> f32 { return pow(1.0 - max(dot(s.n, s.v), 0.0), 5.0); }

// Light that bounces off the polished surface: two sun highlights and the sky's reflection.
// It has an exact formula, and a sharp highlight is the kind of spike a small network blurs,
// so both shaders compute it this way (Grassi: don't ask the network for the easy part).
fn specular(s: Surface) -> vec3f {
  let f = mix(F0, 1.0, fresnel(s));
  var c = WORLD * f * s.baked.y;
  c += SUN0_E * s.baked.z * max(dot(s.n, SUN0_DIR), 0.0) * ggx(s.n, s.v, SUN0_DIR, JADE_ROUGH) * f * frame.jadeB.w;
  c += SUN1_E * s.baked.w * max(dot(s.n, SUN1_DIR), 0.0) * ggx(s.n, s.v, SUN1_DIR, JADE_ROUGH) * f * frame.jadeB.w;
  return c;
}

// ---------------------------------------------------------------- the hard part, the game way

// One sun's light that went into the stone: wrapped diffuse for light bleeding past the terminator,
// and the Frostbite "fast translucency" trick (Barre-Brisebois 2011) for light shining through,
// dimmed by how much stone it crossed. A dozen ALU ops, no rays.
fn jadeSun(s: Surface, l: vec3f, e: vec3f, shadow: f32, through: f32) -> vec3f {
  let k = frame.jadeA;
  let q = frame.jadeB;
  let lit = mix(shadow, 1.0, q.y);                                         // light diffuses into its own shadow
  var c = e * s.albedo / PI * max((dot(s.n, l) + k.x) / (1.0 + k.x), 0.0) * lit * (1.0 - fresnel(s));
  let toward = pow(clamp(dot(s.v, -normalize(l + s.n * k.y)), 0.0, 1.0), k.z) * k.w;
  return c + e * s.albedo * toward * exp(-q.x * through);
}

fn classicSubsurface(s: Surface) -> vec3f {
  return WORLD * s.albedo * s.baked.y * frame.jadeB.z                     // sky light, occluded
    + jadeSun(s, SUN0_DIR, SUN0_E, s.baked.z, s.through.x)
    + jadeSun(s, SUN1_DIR, SUN1_E, s.baked.w, s.through.y);
}

fn display(s: Surface, subsurface: vec3f) -> vec3f { return srgb(specular(s) * frame.specMask + subsurface); }

@compute @workgroup_size(8, 8)
fn showTarget(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= textureDimensions(gPos))) { return; }
  let s = surface(vec2i(id.xy));
  textureStore(outTarget, id.xy, vec4f(select(BG, groundTruth(vec2i(id.xy)).rgb, s.hit), 1.0));
}

@compute @workgroup_size(8, 8)
fn classic(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= textureDimensions(gPos))) { return; }
  let s = surface(vec2i(id.xy));
  textureStore(outClassic, id.xy, vec4f(select(BG, display(s, classicSubsurface(s)), s.hit), 1.0));
}
// ---------------------------------------------------------------- the MLP

// Per-invocation scratch: the inputs and every layer's values, kept for backprop.
var<private> x: array<f32, IN>;
var<private> z1: array<f32, HID>; // pre-activation
var<private> a1: array<f32, HID>; // post-activation
var<private> z2: array<f32, HID>;
var<private> a2: array<f32, HID>;
var<private> y: array<f32, OUT>;

// Identity + sin/cos at `freqs` octaves (NeRF-style Fourier features). Lets a tiny
// MLP fit sharper detail than raw xyz would.
fn encode(o: u32, v: vec3f, on: f32, freqs: u32) {
  x[o] = v.x * on; x[o + 1u] = v.y * on; x[o + 2u] = v.z * on;
  for (var k = 0u; k < freqs; k++) {
    let a = v * PI * f32(1u << k);
    let s = sin(a) * on;
    let c = cos(a) * on;
    let i = o + 3u + k * 6u;
    x[i] = s.x; x[i + 1u] = s.y; x[i + 2u] = s.z;
    x[i + 3u] = c.x; x[i + 4u] = c.y; x[i + 5u] = c.z;
  }
}

// ---------------------------------------------------------------- the neural texture

// instant-NGP's multiresolution hash grid over the dragon's box (the mesh has no UVs).
// Coarse levels index their cells directly, fine ones hash them and let the network sort out collisions.
struct GridTap { at: u32, w: f32 } // one cell corner: its first feature in params, its trilinear weight

fn gridTap(pos: vec3f, l: u32, k: u32) -> GridTap {
  let res = floor(GRID_MIN * pow(GRID_GROWTH, f32(l)));
  let t = clamp(pos * 0.5 + 0.5, vec3f(0.0), vec3f(1.0)) * res; // lab space [-1, 1]^3 -> cells
  let corner = vec3u(t) + vec3u(k & 1u, (k >> 1u) & 1u, k >> 2u);
  let w = select(1.0 - fract(t), fract(t), corner > vec3u(t));
  let side = u32(res) + 1u;
  var h = corner.x + side * (corner.y + side * corner.z);
  if (side * side * side > GRID_T) { h = corner.x ^ (corner.y * 2654435761u) ^ (corner.z * 805459861u); }
  return GridTap(P + (l * GRID_T + (h & (GRID_T - 1u))) * GRID_F, w.x * w.y * w.z);
}

fn gridLookup(pos: vec3f, o: u32, on: f32) {
  for (var j = 0u; j < GRID_L * GRID_F; j++) { x[o + j] = 0.0; }
  for (var l = 0u; l < GRID_L; l++) {
    for (var k = 0u; k < 8u; k++) {
      let tap = gridTap(pos, l, k);
      for (var c = 0u; c < GRID_F; c++) { x[o + l * GRID_F + c] += tap.w * params[tap.at + c] * on; }
    }
  }
}

// One step further back than the weights: through the trilinear filter into the texels. Many
// samples hit the same texel, so their gradients add up atomically instead of in per-sample rows.
fn gridBackward(pos: vec3f, d: array<f32, POS_N>) {
  for (var l = 0u; l < GRID_L; l++) {
    for (var k = 0u; k < 8u; k++) {
      let tap = gridTap(pos, l, k);
      for (var c = 0u; c < GRID_F; c++) {
        atomicAdd(&gridGrad[tap.at - P + c], i32(round(clamp(tap.w * d[l * GRID_F + c], -8.0, 8.0) * FIX)));
      }
    }
  }
}

// IN = 3 albedo + 6 baked + POS_N position + encoded normal + encoded reflection (3 + 6 * FREQS each).
// reflect(-v, n) instead of v is feature engineering: view-dependent light depends on it.
fn setInputs(s: Surface, mask: vec4f, posMask: f32) {
  x[0] = s.albedo.r * mask.x; x[1] = s.albedo.g * mask.x; x[2] = s.albedo.b * mask.x;
  for (var i = 0u; i < 4u; i++) { x[3u + i] = s.baked[i] * mask.w; }
  x[7] = s.through.x * mask.w; x[8] = s.through.y * mask.w;
  if (GRID) { gridLookup(s.pos, 9u, posMask); } else { encode(9u, s.pos, posMask, POS_FREQS); }
  let o = 9u + POS_N;
  encode(o, s.n, mask.y, FREQS);
  encode(o + 3u + 6u * FREQS, reflect(-s.v, s.n), mask.z, FREQS);
}

// SiLU: smooth, so no piecewise-linear creases like ReLU leaves on surfaces.
fn act(z: f32) -> f32 { return z / (1.0 + exp(-z)); }
fn actGrad(z: f32) -> f32 { let s = 1.0 / (1.0 + exp(-z)); return s * (1.0 + z * (1.0 - s)); }

fn forward() {
  for (var i = 0u; i < HID; i++) {
    var z = params[B0 + i];
    for (var j = 0u; j < IN; j++) { z += params[W0 + i * IN + j] * x[j]; }
    z1[i] = z; a1[i] = act(z);
  }
  for (var i = 0u; i < HID; i++) {
    var z = params[B1 + i];
    for (var j = 0u; j < HID; j++) { z += params[W1 + i * HID + j] * a1[j]; }
    z2[i] = z; a2[i] = act(z);
  }
  for (var k = 0u; k < OUT; k++) {
    var z = params[B2 + k];
    for (var j = 0u; j < HID; j++) { z += params[W2 + k * HID + j] * a2[j]; }
    y[k] = z; // linear output
  }
}

// Chain rule, layer by layer from the output back. dy = dLoss/dOutput.
// Writes this sample's dLoss/dParam into its own row of `grads` (WGSL has no float atomics).
fn backward(row: u32, dy: array<f32, OUT>, pos: vec3f) {
  let g = row * P;
  var da2: array<f32, HID>; // dLoss/da2, zero-initialised
  for (var k = 0u; k < OUT; k++) {
    grads[g + B2 + k] = dy[k];
    for (var j = 0u; j < HID; j++) {
      grads[g + W2 + k * HID + j] = dy[k] * a2[j];
      da2[j] += params[W2 + k * HID + j] * dy[k];
    }
  }
  var da1: array<f32, HID>;
  for (var i = 0u; i < HID; i++) {
    let dz = da2[i] * actGrad(z2[i]);
    grads[g + B1 + i] = dz;
    for (var j = 0u; j < HID; j++) {
      grads[g + W1 + i * HID + j] = dz * a1[j];
      da1[j] += params[W1 + i * HID + j] * dz;
    }
  }
  var dpos: array<f32, POS_N>; // dLoss/d(position input), only used by the neural texture
  for (var i = 0u; i < HID; i++) {
    let dz = da1[i] * actGrad(z1[i]);
    grads[g + B0 + i] = dz;
    for (var j = 0u; j < IN; j++) { grads[g + W0 + i * IN + j] = dz * x[j]; }
    if (GRID) { for (var j = 0u; j < POS_N; j++) { dpos[j] += params[W0 + i * IN + 9u + j] * dz * POS_ON; } }
  }
  if (GRID) { gridBackward(pos, dpos); }
}

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

// One invocation = one training sample: a random pixel of the G-buffer just rendered.
@compute @workgroup_size(64)
fn train(@builtin(global_invocation_id) id: vec3u) {
  let b = id.x;
  if (b >= BATCH) { return; }
  let size = textureDimensions(gPos);
  var rng = pcg(b ^ pcg(frame.seed));
  var s: Surface;
  var p: vec2i;
  var truth: vec4f;
  for (var t = 0; t < 16; t++) { // retry until we land inside the dragon (not on its silhouette)
    rng = pcg(rng);
    p = vec2i(vec2u(rng % size.x, pcg(rng) % size.y));
    s = surface(p);
    truth = groundTruth(p);
    if (s.hit && truth.a > 0.99) { break; }
  }
  let ok = s.hit && truth.a > 0.99;
  // Target: the linear light Cycles saw minus the analytic specular, untransformed.
  // Computed before forward(), so only 3 floats of the surface stay live across the network.
  let spec = specular(s) * SPEC_ON; // SPEC_ON = 0: the network predicts the highlights too
  let want = linear(truth.rgb) - spec;
  setInputs(s, vec4f(1.0), POS_ON);
  forward();

  // LOSS picks how errors are weighed (main.ts, ?loss=):
  //   0 linear:   (y - want)^2, plain MSE on radiance. Bright pixels dominate.
  //   1 relative: (y - want)^2 / (total^2 + 0.01), total = spec + y held constant. Still raw radiance,
  //               but an error counts relative to how bright the pixel is, roughly as the eye sees it
  //               (the relative L2 of NVIDIA's neural radiance caching, Muller et al. 2021).
  //   2 display:  (srgb(spec + y) - truth)^2, the loss through the display transform.
  var dy: array<f32, OUT>;
  var loss = 0.0;
  for (var k = 0u; k < OUT; k++) {
    let total = spec[k] + y[k];
    var e = y[k] - want[k];
    var de = 1.0; // de/dy
    if (LOSS == 1u) {
      let w = inverseSqrt(total * total + 0.01);
      e *= w; de = w;
    } else if (LOSS == 2u) { // srgb without its clamp: a clamp has no gradient, outputs stuck there would die
      let x = max(total, 0.0);
      e = select(1.055 * pow(x, 1.0 / 2.4) - 0.055, 12.92 * total, total <= 0.0031308) - truth[k];
      de = select(1.055 / 2.4 * pow(max(x, 1e-6), 1.0 / 2.4 - 1.0), 12.92, total <= 0.0031308);
    }
    e = select(0.0, e, ok);
    loss += e * e / f32(OUT);
    dy[k] = 2.0 * e * de / f32(OUT);
  }
  losses[b] = loss;
  backward(b, dy, s.pos);
}

// One invocation per parameter: average its gradient over the batch, then Adam.
@compute @workgroup_size(64)
fn adam(@builtin(global_invocation_id) id: vec3u) {
  let p = id.x;
  if (p == P + GRID_P) { // one spare invocation averages the loss for the UI
    var l = 0.0;
    for (var b = 0u; b < BATCH; b++) { l += losses[b]; }
    losses[BATCH] = l / f32(BATCH);
    return;
  }
  if (p > P + GRID_P) { return; }
  var g = 0.0;
  var lr = frame.lr;
  if (p < P) {
    for (var b = 0u; b < BATCH; b++) { g += grads[b * P + p]; }
  } else { // a texel: skip it if no sample touched it (instant-ngp), else read and clear its gradient
    let raw = atomicLoad(&gridGrad[p - P]);
    if (raw == 0) { emaUpdate(p); return; }
    atomicStore(&gridGrad[p - P], 0);
    g = f32(raw) / FIX;
    lr = LR_GRID;
  }
  g /= f32(BATCH);
  // Adam: momentum (m) + per-parameter step size from gradient variance (v).
  let m = mix(g, adamM[p], 0.9);
  let v = mix(g * g, adamV[p], 0.999);
  adamM[p] = m;
  adamV[p] = v;
  let t = f32(frame.step);
  let mHat = m / (1.0 - pow(0.9, t));
  let vHat = v / (1.0 - pow(0.999, t));
  params[p] -= lr * mHat / (sqrt(vHat) + 1e-8);
  emaUpdate(p);
}

// Exponential moving average of the weights, ~100 steps wide (instant-ngp, NRC). The display reads
// it: each Adam step chases one random view, and the average cancels that noise. The ramp keeps
// it from lagging behind the random init in the first steps.
fn emaUpdate(p: u32) {
  let t = f32(frame.step);
  ema[p] = mix(params[p], ema[p], min(0.99, (1.0 + t) / (10.0 + t)));
}

// The neural render: the same forward pass, on every pixel.
@compute @workgroup_size(8, 8)
fn infer(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(gPos);
  if (any(id.xy >= size)) { return; }
  let s = surface(vec2i(id.xy));
  var c = BG;
  if (s.hit) {
    let spec = specular(s) * frame.specMask * SPEC_ON;
    setInputs(s, frame.mask, frame.posMask);
    if (GRID) {
      for (var j = 0u; j < GRID_L * GRID_F; j++) { x[9u + j] *= select(frame.gridMask.y, frame.gridMask.x, j < GRID_L * GRID_F / 2u); }
    }
    forward();
    c = srgb(spec + vec3f(y[0], y[1], y[2]));
  }
  textureStore(outNeural, id.xy, vec4f(c, 1.0));

  // Debug: the neural texture's features at this pixel, summed over the coarse and the fine levels.
  var g = vec4f(0.0);
  if (GRID && s.hit) {
    for (var l = 0u; l < GRID_L; l++) {
      let f = vec2f(x[9u + l * GRID_F], x[10u + l * GRID_F]);
      if (l < GRID_L / 2u) { g.x += f.x; g.y += f.y; } else { g.z += f.x; g.w += f.y; }
    }
  }
  textureStore(outGrid, id.xy, g);

  // Debug: every 4th pixel writes the first 16 hidden activations into a 4x4 mosaic.
  if (all(id.xy % 4u == vec2u(0u))) {
    let tile = size / 4u;
    for (var k = 0u; k < 16u; k++) {
      let at = vec2u(k % 4u, k / 4u) * tile + id.xy / 4u;
      textureStore(outFeatures, at, vec4f(select(0.0, a1[k], s.hit), 0.0, 0.0, 1.0));
    }
  }
}

// Benchmark, 3 floats per pixel: squared error of the display colour against the Cycles frame
// (-1 where not scored: background, silhouette), then the linear luminance of the render and of
// the truth, for Grassi's brightness ratio. main.ts reads it back. Reuses `grads`.
@compute @workgroup_size(8, 8)
fn score(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(gPos);
  if (any(id.xy >= size)) { return; }
  let p = vec2i(id.xy);
  let s = surface(p);
  let truth = groundTruth(p);
  let i = 3u * (id.y * size.x + id.x);
  grads[i] = -1.0;
  if (s.hit && truth.a > 0.99) {
    var c = specular(s);
    if (frame.scoreClassic == 1u) {
      c += classicSubsurface(s);
    } else {
      c *= SPEC_ON;
      setInputs(s, vec4f(1.0), POS_ON);
      forward();
      c += vec3f(y[0], y[1], y[2]);
    }
    c = srgb(c);
    let d = c - truth.rgb;
    let lum = vec3f(0.2126, 0.7152, 0.0722);
    grads[i] = dot(d, d) / 3.0;
    grads[i + 1u] = dot(linear(c), lum);
    grads[i + 2u] = dot(linear(truth.rgb), lum);
  }
}
