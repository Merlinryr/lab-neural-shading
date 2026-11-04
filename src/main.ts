import gbufferWGSL from './shaders/gbuffer.wgsl?raw'
import neuralWGSL from './shaders/neural.wgsl?raw'
import blitWGSL from './shaders/blit.wgsl?raw'
import { lookAt, multiply, perspective } from './scene'

const query = new URLSearchParams(location.search)
const POS_ON = query.get('pos') === '0' ? 0 : 1
const SPEC_ON = query.get('spec') === '0' ? 0 : 1
const BENCH = +(query.get('bench') ?? 0)
const LOSS = ['linear', 'relative', 'display'].indexOf(query.get('loss') ?? 'relative')

// ---- Network shape: the single source of truth, injected into neural.wgsl ----
const FREQS = 2 // Fourier octaves for the normal and reflection vectors
const GRID = query.get('enc') !== 'fourier'
const GRID_L = 8, GRID_F = 2, GRID_T = 2 ** 17 // levels, features per texel, texels per level
// Cells along the dragon's box, coarsest and finest level. Finer than 128 scores better on the
// training views but worse on held-out ones: it starts memorising them.
const GRID_MIN = 16, GRID_MAX = +(query.get('gridmax') ?? 128)
const GRID_P = GRID ? GRID_L * GRID_T * GRID_F : 0 // learnt texels, after the MLP's weights in `params`
const LR_GRID = 1e-2                            // texels see few samples per step: a bigger step, like instant-ngp
const POS_FREQS = 4                             // Fourier octaves, when not using the grid
const POS_N = GRID ? GRID_L * GRID_F : 3 + 6 * POS_FREQS
const IN = 3 + 6 + POS_N + 2 * (3 + 6 * FREQS) // albedo + baked + position + normal + reflection
const HID = +(query.get('hid') ?? (GRID ? 16 : 32)) // two hidden layers of HID neurons: the neural texture lets a narrow one win
const OUT = 3                            // rgb
const P = HID * IN + HID + HID * HID + HID + OUT * HID + OUT
const MACS = P - HID - HID - OUT         // multiply-adds per pixel
const BATCH = 1024                       // pixels per training step
const STEPS_PER_FRAME = 4
const LR = 2e-3
const SIZE = 512                         // the dataset's resolution: training, scoring, debug panels
// Never trained on: every score in the lab is on these 8 views. Two per orbit height, rotated per
// height, so they cover 8 different angles to the sun.
const HELD_OUT = (i: number) => ((i % 16) - 2 * Math.floor(i / 16) + 13) % 8 === 0
const $ = <T extends HTMLElement>(sel: string) => document.querySelector<T>(sel)!

const adapter = await navigator.gpu?.requestAdapter()
if (!adapter) {
  $('#stats').textContent = 'WebGPU is not available in this browser.'
  throw new Error('no WebGPU')
}
const TIMESTAMPS = adapter.features.has('timestamp-query') // GPU timers for the scoreboard
const device = await adapter.requestDevice({
  requiredFeatures: TIMESTAMPS ? ['timestamp-query'] : [],
  requiredLimits: { maxStorageTexturesPerShaderStage: 5 }, // the default is 4: target, classic, neural, features, grid
})
device.addEventListener('uncapturederror', e => console.error((e as GPUUncapturedErrorEvent).error.message))

// Blender dataset
type Vec3 = [number, number, number]
type View = { file: string, yaw: number, pitch: number, eye: Vec3, seconds: number }
const data: {
  fovY: number, distance: number, samples: number
  suns: { dir: Vec3, irradiance: Vec3 }[], world: Vec3, jade: { roughness: number, ior: number }
  mesh: { vertices: string, indices: string }
  frames: View[]
} = await fetch('/jade/cameras.json').then(r => r.json())
const load = (file: string) => fetch(`/jade/${file}`).then(r => r.arrayBuffer())




// Cycles render frames -> one array layer each
const framesTex = device.createTexture({
  size: [SIZE, SIZE, data.frames.length],
  format: 'rgba8unorm',
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
})
await Promise.all(data.frames.map(async (f, i) => {
  const blob = await fetch(`/jade/${f.file}`).then(r => r.blob())
  const source = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })
  device.queue.copyExternalImageToTexture({ source }, { texture: framesTex, origin: [0, 0, i] }, [SIZE, SIZE])
}))
const views = data.frames.map((_, i) => i)
const trainSet = views.filter(i => !HELD_OUT(i))
const testSet = views.filter(HELD_OUT)
const fitSet = trainSet.filter((_, i) => i % 7 === 0) // the classic shader's knobs are fitted on these
const cyclesSeconds = data.frames.reduce((s, f) => s + f.seconds, 0) / data.frames.length




// Buffers
const buffer = (size: number, usage: number) => device.createBuffer({ size, usage })
const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
const params = buffer((P + GRID_P) * 4, STORAGE)
const adamM = buffer((P + GRID_P) * 4, STORAGE)
const adamV = buffer((P + GRID_P) * 4, STORAGE)
const ema = buffer((P + GRID_P) * 4, STORAGE) // moving average of params: what infer and score read
const gridGrad = buffer(Math.max(GRID_P, 1) * 4, STORAGE)
const grads = buffer(Math.max(BATCH * P, 3 * SIZE * SIZE) * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)
const losses = buffer((BATCH + 1) * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)
const lossRead = buffer(4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)
const cameraBuf = buffer(80, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
const frameBuf = buffer(96, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)

const vertices = new Float32Array(await load(data.mesh.vertices)) // 15 floats a vertex, see gbuffer.wgsl
const indices = new Uint32Array(await load(data.mesh.indices))
const vertexBuf = buffer(vertices.byteLength, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST)
const indexBuf = buffer(indices.byteLength, GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST)
device.queue.writeBuffer(vertexBuf, 0, vertices)
device.queue.writeBuffer(indexBuf, 0, indices)

let step = 0
function resetNetwork() {
  // He-uniform init: keeps activations at a sane scale through the layers. Biases start at 0.
  const w = new Float32Array(P + GRID_P)
  let o = 0
  for (const [fanIn, fanOut] of [[IN, HID], [HID, HID], [HID, OUT]]) {
    const limit = Math.sqrt(6 / fanIn)
    for (let i = 0; i < fanIn * fanOut; i++) w[o++] = (Math.random() * 2 - 1) * limit
    o += fanOut
  }
  for (let i = P; i < P + GRID_P; i++) w[i] = (Math.random() * 2 - 1) * 1e-4 // texels: near zero, as instant-ngp
  device.queue.writeBuffer(params, 0, w)
  device.queue.writeBuffer(ema, 0, w)
  device.queue.writeBuffer(adamM, 0, new Float32Array(P + GRID_P))
  device.queue.writeBuffer(adamV, 0, new Float32Array(P + GRID_P))
  step = 0
}
resetNetwork()




// Pipelines
const cameraLayout = device.createBindGroupLayout({
  entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: {} }],
})
const gbufferModule = device.createShaderModule({ code: gbufferWGSL })
const gbufferPipe = device.createRenderPipeline({
  layout: device.createPipelineLayout({ bindGroupLayouts: [cameraLayout] }),
  vertex: {
    module: gbufferModule,
    buffers: [{
      arrayStride: 60,
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },  // position
        { shaderLocation: 1, offset: 12, format: 'float32x3' }, // normal
        { shaderLocation: 2, offset: 24, format: 'float32x3' }, // albedo
        { shaderLocation: 3, offset: 36, format: 'float32x4' }, // thickness, AO, sun visibility x2
        { shaderLocation: 4, offset: 52, format: 'float32x2' }, // distance through the stone to each sun
      ],
    }],
  },
  fragment: { module: gbufferModule, targets: [0, 1, 2, 3].map(() => ({ format: 'rgba16float' as const })) },
  depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
})
const gbufferBind = device.createBindGroup({
  layout: cameraLayout,
  entries: [{ binding: 0, resource: { buffer: cameraBuf } }],
})

// All compute passes share one explicit layout (see neural.wgsl).
const C = GPUShaderStage.COMPUTE
const computeLayout = device.createBindGroupLayout({
  entries: [
    ...[0, 1, 2].map(binding => ({ binding, visibility: C, texture: {} })),
    { binding: 3, visibility: C, buffer: { type: 'uniform' as const } },
    ...[4, 5, 6, 7, 8].map(binding => ({ binding, visibility: C, buffer: { type: 'storage' as const } })),
    ...[9, 10, 11, 14].map(binding => ({ binding, visibility: C, storageTexture: { format: 'rgba16float' as const } })),
    { binding: 12, visibility: C, texture: { viewDimension: '2d-array' as const } },
    { binding: 13, visibility: C, texture: {} },
    { binding: 15, visibility: C, buffer: { type: 'storage' as const } },
    { binding: 16, visibility: C, buffer: { type: 'storage' as const } },
    { binding: 17, visibility: C, storageTexture: { format: 'rgba16float' as const } },
  ],
})
const vec3 = (v: number[]) => `vec3f(${v.map(x => x.toFixed(6)).join(', ')})`
const F0 = ((data.jade.ior - 1) / (data.jade.ior + 1)) ** 2
const neuralModule = device.createShaderModule({
  code: `const IN = ${IN}u; const HID = ${HID}u; const OUT = ${OUT}u; const FREQS = ${FREQS}u; const POS_FREQS = ${POS_FREQS}u; const BATCH = ${BATCH}u; const POS_ON = ${POS_ON.toFixed(1)}; const LOSS = ${LOSS}u; const SPEC_ON = ${SPEC_ON.toFixed(1)};
const POS_N = ${POS_N}u; const GRID = ${GRID}; const GRID_L = ${GRID_L}u; const GRID_F = ${GRID_F}u; const GRID_T = ${GRID_T}u; const GRID_P = ${GRID_P}u;
const GRID_MIN = ${GRID_MIN.toFixed(1)}; const GRID_GROWTH = ${((GRID_MAX / GRID_MIN) ** (1 / (GRID_L - 1))).toFixed(6)}; const LR_GRID = ${LR_GRID};
const SUN0_DIR = ${vec3(data.suns[0].dir)}; const SUN0_E = ${vec3(data.suns[0].irradiance)};
const SUN1_DIR = ${vec3(data.suns[1].dir)}; const SUN1_E = ${vec3(data.suns[1].irradiance)};
const WORLD = ${vec3(data.world)}; const JADE_ROUGH = ${data.jade.roughness}; const F0 = ${F0.toFixed(5)};\n${neuralWGSL}`,
})
const computePipe = (entryPoint: string) => device.createComputePipeline({
  layout: device.createPipelineLayout({ bindGroupLayouts: [computeLayout] }),
  compute: { module: neuralModule, entryPoint },
})
const targetPipe = computePipe('showTarget')
const classicPipe = computePipe('classic')
const trainPipe = computePipe('train')
const adamPipe = computePipe('adam')
const inferPipe = computePipe('infer')
const scorePipe = computePipe('score')




// Render targets
// `lab` is 512² (training, scoring, tiles); the main view makes its own at the canvas's size.
function renderTargets(width: number, height: number) {
  const texture = (format: GPUTextureFormat, usage: number) => device.createTexture({ size: [width, height], format, usage })
  const RT = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
  const ST = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
  const t = {
    width, height,
    gAlbedo: texture('rgba16float', RT), gNormal: texture('rgba16float', RT),
    gPos: texture('rgba16float', RT), gBaked: texture('rgba16float', RT),
    depth: texture('depth24plus', GPUTextureUsage.RENDER_ATTACHMENT),
    target: texture('rgba16float', ST), classic: texture('rgba16float', ST),
    neural: texture('rgba16float', ST), features: texture('rgba16float', ST), grid: texture('rgba16float', ST),
  }
  // Two bind groups, the parameter copies swapped: `trainBind` trains params and averages them into
  // ema, `bind` (everything else) reads ema as `params`.
  const group = (live: GPUBuffer, other: GPUBuffer) => device.createBindGroup({
    layout: computeLayout,
    entries: [t.gAlbedo.createView(), t.gNormal.createView(), t.gPos.createView(), { buffer: frameBuf },
      { buffer: live }, { buffer: adamM }, { buffer: adamV }, { buffer: grads }, { buffer: losses },
      t.target.createView(), t.neural.createView(), t.features.createView(), framesTex.createView({ dimension: '2d-array' }),
      t.gBaked.createView(), t.classic.createView(), { buffer: gridGrad }, { buffer: other }, t.grid.createView()]
      .map((resource, binding) => ({ binding, resource })),
  })
  return { ...t, bind: group(ema, params), trainBind: group(params, ema), destroy: () => Object.values(t).forEach(x => x instanceof GPUTexture && x.destroy()) }
}
type Targets = ReturnType<typeof renderTargets>
const lab = renderTargets(SIZE, SIZE)




// Panels: one canvas each, blitted with a MODE-specialised pipeline
const format = navigator.gpu.getPreferredCanvasFormat()
const blitModule = device.createShaderModule({ code: blitWGSL })
const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' })
const blitPipes = new Map<number, GPURenderPipeline>()
function blitPipe(mode: number) {
  if (!blitPipes.has(mode)) blitPipes.set(mode, device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: blitModule },
    fragment: { module: blitModule, targets: [{ format }], constants: { MODE: mode } },
  }))
  return blitPipes.get(mode)!
}
const blitBind = (mode: number, a: GPUTexture, b = a) => device.createBindGroup({
  layout: blitPipe(mode).getBindGroupLayout(0),
  entries: [a.createView(), b.createView(), sampler].map((resource, binding) => ({ binding, resource })),
})
const VIEWS: Record<string, [mode: number, a: GPUTexture, b?: GPUTexture]> = {
  albedo: [0, lab.gAlbedo],
  normal: [1, lab.gNormal],
  thickness: [6, lab.gBaked],
  light: [7, lab.gBaked],
  features: [4, lab.features],
  gridCoarse: [8, lab.grid, lab.gPos],
  gridFine: [9, lab.grid, lab.gPos],
  target: [0, lab.target],
  classic: [0, lab.classic],
  neural: [0, lab.neural],
  errClassic: [5, lab.classic, lab.target],
  errNeural: [5, lab.neural, lab.target],
}
const panels = [...document.querySelectorAll<HTMLCanvasElement>('canvas[data-view]')].map(canvas => {
  const [mode, a, b] = VIEWS[canvas.dataset.view!]
  const context = canvas.getContext('webgpu')!
  context.configure({ device, format })
  return { canvas, context, pipeline: blitPipe(mode), bind: blitBind(mode, a, b) }
})

function blit(encoder: GPUCommandEncoder, canvas: HTMLCanvasElement, context: GPUCanvasContext, pipeline: GPURenderPipeline, bind: GPUBindGroup) {
  const w = Math.round(canvas.clientWidth * devicePixelRatio)
  const h = Math.round(canvas.clientHeight * devicePixelRatio)
  if (!w || !h) return // not laid out (yet)
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h }
  const p = encoder.beginRenderPass({
    colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store' }],
  })
  p.setPipeline(pipeline)
  p.setBindGroup(0, bind)
  p.draw(3)
  p.end()
}




// UI
let yaw = 0, pitch = 0.25, zoom = 1, dragging = false, training = true
// Input groups zeroed at inference: header boxes (checked = off) and the G-buffer tiles (click = off).
const mask = [1, 1, 1, 1, POS_ON, 1, 1, 1] // albedo, normal, view, baked, position, specular, neural texture coarse / fine
document.querySelectorAll<HTMLInputElement>('input[data-input]').forEach(box => {
  box.checked = !mask[+box.dataset.input!]
  box.addEventListener('change', () => {
    const i = +box.dataset.input!
    mask[i] = box.checked ? 0 : 1
    document.querySelectorAll(`figure[data-input="${i}"]`).forEach(t => t.classList.toggle('off', !mask[i]))
  })
})
document.querySelectorAll<HTMLElement>('figure[data-input]').forEach(tile => tile.addEventListener('click', () => {
  const i = +tile.dataset.input!
  mask[i] = 1 - mask[i]
  document.querySelectorAll(`figure[data-input="${i}"]`).forEach(t => t.classList.toggle('off', !mask[i])) // thickness and AO share "baked"
  document.querySelectorAll<HTMLInputElement>(`input[data-input="${i}"]`).forEach(b => { b.checked = !mask[i] })
}))
$<HTMLInputElement>('#stop').addEventListener('change', e => { training = !(e.target as HTMLInputElement).checked })
$('#reset').addEventListener('click', resetNetwork)
addEventListener('pointerdown', e => { dragging = (e.target as HTMLElement).tagName === 'CANVAS' })
addEventListener('pointerup', () => { dragging = false })
addEventListener('pointermove', e => {
  if (!dragging) return
  yaw -= e.movementX * 0.01
  pitch = Math.min(1.4, Math.max(-1.4, pitch + e.movementY * 0.01))
})
  + (GRID ? ` · neural texture ${GRID_L} × ${GRID_T / 1024}k × ${GRID_F}` : '')
$('#inputs').textContent = `${IN} numbers a pixel in`
$('#cycles-cost').textContent = `${cyclesSeconds.toFixed(1)} s / frame (${data.samples} samples)`

const fullCanvas = $<HTMLCanvasElement>('#main')
const fullContext = fullCanvas.getContext('webgpu')!
fullContext.configure({ device, format })
let full: (Targets & { blit: GPUBindGroup }) | null = null
fullCanvas.addEventListener('wheel', e => { zoom = Math.min(3, Math.max(0.3, zoom * Math.exp(e.deltaY * 1e-3))) }, { passive: true })

function fullTargets() {
  const w = Math.max(8, Math.round(fullCanvas.clientWidth * devicePixelRatio))
  const h = Math.max(8, Math.round(fullCanvas.clientHeight * devicePixelRatio))
  if (full && full.width === w && full.height === h) return full
  full?.destroy()
  const t = renderTargets(w, h)
  full = { ...t, blit: blitBind(0, t.neural) }
  return full
}




// Frame
const frameData = new ArrayBuffer(96) // matches struct Frame in neural.wgsl
const frameF32 = new Float32Array(frameData)
const frameU32 = new Uint32Array(frameData)
// Classic knobs: wrap, distortion, power, translucency | absorption, shadow softness, ambient, specular.
// The specular scale stays at 1, the physical value: fitted, it inflated to fake missing subsurface.
// The fit is deterministic, so its result is stored here. ?fit=1 refits from the starting guess
// (do it after changing the scene or the classic shader, and paste the logged knobs here).
const FIT = query.get('fit') === '1'
const jade = FIT ? [0.5, 0.2, 4, 0.5, 4, 0.3, 1, 1] : [0.625, 0.063, 1.6, 0.781, 2.56, 0.3, 0.064, 1]
const FITTED = 7
let loss = NaN, lossPending = false, fps = 60, last = performance.now(), frameIndex = 0

function writeCamera(eye: Vec3, aspect = 1) {
  const viewProj = multiply(perspective(data.fovY, aspect, 0.1, 40), lookAt(eye, [0, 0, 0], [0, 1, 0]))
  device.queue.writeBuffer(cameraBuf, 0, new Float32Array([...viewProj, ...eye, 0]))
  frameF32.set(eye, 0)
}

function writeFrame(targetFrame: number, scoreClassic = false) {
  frameF32.set(mask.slice(0, 4), 4)
  frameF32[9] = LR
  frameU32[10] = targetFrame
  frameF32[11] = mask[4]
  frameF32.set(jade, 12)
  frameU32[20] = scoreClassic ? 1 : 0
  frameF32[21] = mask[5]
  frameF32.set(mask.slice(6, 8), 22)
  device.queue.writeBuffer(frameBuf, 0, frameData)
}

function encodeGBuffer(encoder: GPUCommandEncoder, t: Targets) {
  const pass = encoder.beginRenderPass({
    colorAttachments: [t.gAlbedo, t.gNormal, t.gPos, t.gBaked].map(x => ({ view: x.createView(), loadOp: 'clear' as const, storeOp: 'store' as const, clearValue: [0, 0, 0, 0] })),
    depthStencilAttachment: { view: t.depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
  })
  pass.setPipeline(gbufferPipe)
  pass.setBindGroup(0, gbufferBind)
  pass.setVertexBuffer(0, vertexBuf)
  pass.setIndexBuffer(indexBuf, 'uint32')
  pass.drawIndexed(indices.length)
  pass.end()
}

// Each dispatch covers the whole target (8x8 workgroups) or is a 1D count of workgroups.
function encodeCompute(encoder: GPUCommandEncoder, t: Targets, dispatches: (GPUComputePipeline | [GPUComputePipeline, number])[], bind = t.bind) {
  const pass = encoder.beginComputePass()
  pass.setBindGroup(0, bind)
  for (const d of dispatches) {
    if (Array.isArray(d)) { pass.setPipeline(d[0]); pass.dispatchWorkgroups(d[1]) }
    else { pass.setPipeline(d); pass.dispatchWorkgroups(Math.ceil(t.width / 8), Math.ceil(t.height / 8)) }
  }
  pass.end()
}

// One training step: rasterize the G-buffer from a random training camera and fit a batch of its
// pixels to that Cycles frame. One submit per step: writeBuffer lands between submits, so each
// step sees its own camera, seed and step.
function trainStep() {
  const k = trainSet[Math.floor(Math.random() * trainSet.length)]
  writeCamera(data.frames[k].eye)
  frameU32[3] = step * 7919 // seed
  frameU32[8] = ++step
  writeFrame(k)
  const encoder = device.createCommandEncoder()
  encodeGBuffer(encoder, lab)
  encodeCompute(encoder, lab, [[trainPipe, Math.ceil(BATCH / 64)], [adamPipe, Math.ceil((P + GRID_P + 1) / 64)]], lab.trainBind)
  device.queue.submit([encoder.finish()])
}

// Score the classic shader or the network against Cycles, frame by frame (dragon pixels only):
// PSNR of the display colours, and Grassi's brightness ratio (render / truth, linear; 1 = exact).
const scoreRead = buffer(3 * SIZE * SIZE * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)
async function score(frames: number[], classic: boolean) {
  let sum = 0, count = 0, lit = 0, truth = 0
  const perView: { frame: number, psnr: number, ratio: number }[] = []
  for (const i of frames) {
    writeCamera(data.frames[i].eye)
    writeFrame(i, classic)
    const encoder = device.createCommandEncoder()
    encodeGBuffer(encoder, lab)
    encodeCompute(encoder, lab, [scorePipe])
    encoder.copyBufferToBuffer(grads, 0, scoreRead, 0, 3 * SIZE * SIZE * 4)
    device.queue.submit([encoder.finish()])
    await scoreRead.mapAsync(GPUMapMode.READ)
    const px = new Float32Array(scoreRead.getMappedRange())
    let s = 0, n = 0, l = 0, t = 0
    for (let k = 0; k < px.length; k += 3) if (px[k] >= 0) { s += px[k]; n++; l += px[k + 1]; t += px[k + 2] }
    scoreRead.unmap()
    perView.push({ frame: i, psnr: 10 * Math.log10(n / s), ratio: l / t })
    sum += s; count += n; lit += l; truth += t
  }
  return { psnr: 10 * Math.log10(count / sum), ratio: lit / truth, perView }
}

// The classic shader gets its best shot: its knobs are tuned against the training frames by
// coordinate descent, the way an artist would tweak sliders until it looks like the reference.
async function fitClassic() {
  let best = (await score(fitSet, true)).psnr
  for (let sweep = 0; sweep < 3; sweep++)
    for (let i = 0; i < FITTED; i++)
      for (const f of [0.5, 0.8, 1.25, 2]) {
        const old = jade[i]
        jade[i] = i === 5 ? Math.min(old * f, 1) : old * f // shadow softness is a blend weight
        const p = (await score(fitSet, true)).psnr
        if (p > best) best = p
        else jade[i] = old
      }
  console.log('classic knobs', jade.map(v => +v.toFixed(3)))
}

// GPU time of one full-screen pass, 512² pixels: GPU timestamps around 100 back-to-back dispatches.
// Best of 5, because anything else on the GPU only ever makes a run slower.
const querySet = TIMESTAMPS ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null
const queryResolve = buffer(16, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC)
const queryRead = buffer(16, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)
async function gpuMs(pipeline: GPUComputePipeline) {
  if (!querySet) return NaN
  writeCamera(data.frames[testSet[0]].eye)
  writeFrame(testSet[0])
  let best = Infinity
  for (let i = 0; i < 5; i++) {
    const encoder = device.createCommandEncoder()
    encodeGBuffer(encoder, lab)
    const pass = encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } })
    pass.setBindGroup(0, lab.bind)
    pass.setPipeline(pipeline)
    for (let k = 0; k < 100; k++) pass.dispatchWorkgroups(SIZE / 8, SIZE / 8)
    pass.end()
    encoder.resolveQuerySet(querySet, 0, 2, queryResolve, 0)
    encoder.copyBufferToBuffer(queryResolve, 0, queryRead, 0, 16)
    device.queue.submit([encoder.finish()])
    await queryRead.mapAsync(GPUMapMode.READ)
    const [t0, t1] = new BigUint64Array(queryRead.getMappedRange())
    queryRead.unmap()
    best = Math.min(best, Number(t1 - t0) / 1e6 / 100)
  }
  return best
}

function frame(now: number) {
  fps = fps * 0.95 + 50 / Math.max(now - last, 1)
  last = now
  const swing = yaw + now * 1.5e-4 // a slow orbit

  for (let s = 0; training && s < STEPS_PER_FRAME; s++) trainStep()

  // Tiles: snap to the nearest rendered camera, so the target and error tiles are exact.
  // Its own submit: the main view below rewrites the camera.
  let shown = 0
  const dist = (f: View) => Math.atan2(Math.sin(f.yaw - swing), Math.cos(f.yaw - swing)) ** 2 + (f.pitch - pitch) ** 2
  data.frames.forEach((f, i) => { if (dist(f) < dist(data.frames[shown])) shown = i })
  writeCamera(data.frames[shown].eye)
  writeFrame(shown)
  const tiles = device.createCommandEncoder()
  encodeGBuffer(tiles, lab)
  encodeCompute(tiles, lab, [targetPipe, classicPipe, inferPipe])
  for (const { canvas, context, pipeline, bind } of panels) blit(tiles, canvas, context, pipeline, bind)
  device.queue.submit([tiles.finish()])

  // Main view: free camera at the canvas's resolution and aspect, no Cycles frame to compare with.
  const t = fullTargets()
  const d = data.distance * zoom
  writeCamera([d * Math.cos(pitch) * Math.sin(swing), d * Math.sin(pitch), d * Math.cos(pitch) * Math.cos(swing)], t.width / t.height)
  writeFrame(0)
  const encoder = device.createCommandEncoder()
  encodeGBuffer(encoder, t)
  encodeCompute(encoder, t, [inferPipe])
  blit(encoder, fullCanvas, fullContext, blitPipe(0), t.blit)
  const readLoss = !lossPending && training
  if (readLoss) encoder.copyBufferToBuffer(losses, BATCH * 4, lossRead, 0, 4)
  device.queue.submit([encoder.finish()])
  if (readLoss) {
    lossPending = true
    lossRead.mapAsync(GPUMapMode.READ).then(() => {
      loss = new Float32Array(lossRead.getMappedRange())[0]
      lossRead.unmap()
      lossPending = false
    })
  }

  if (frameIndex++ % 10 === 0) {
    $('#res').textContent = `${t.width}×${t.height} px`
    $('#fps').textContent = `${fps.toFixed(0)} fps`
    $('#stats').textContent = `step ${step} · loss ${isNaN(loss) ? '—' : loss.toExponential(2)}`
    $('#target').textContent = `Cycles frame ${shown}${HELD_OUT(shown) ? ' · held out' : ''}`
  }
  requestAnimationFrame(frame)
}




// Start: (re)fit the classic shader if asked, time both shaders, then train (live) or benchmark
if (FIT) {
  $('#stats').textContent = 'fitting the classic shader to the training frames…'
  await fitClassic()
}
const classicTest = await score(testSet, true)
const classicMs = await gpuMs(classicPipe)
const neuralMs = await gpuMs(inferPipe)
$('#classic-cost').textContent = `${classicMs.toFixed(2)} ms / frame`
$('#neural-cost').textContent = `${neuralMs.toFixed(2)} ms / frame · ${MACS} MAC/pixel`
$('#classic-psnr').textContent = `${classicTest.psnr.toFixed(2)} dB · brightness ×${classicTest.ratio.toFixed(2)}`
const showNeural = (r: { psnr: number, ratio: number }) => { $('#neural-psnr').textContent = `${r.psnr.toFixed(2)} dB · brightness ×${r.ratio.toFixed(2)}` }

if (BENCH) {
  const t0 = performance.now()
  for (let s = 0; s < BENCH; s++) {
    trainStep()
    if (s % 64 === 63) {
      await device.queue.onSubmittedWorkDone()
      $('#stats').textContent = `bench · step ${step}/${BENCH}`
    }
  }
  await device.queue.onSubmittedWorkDone()
  const neuralTest = await score(testSet, false)
  // Grassi's sweep: each held-out view by its angle to the back sun (0° = sun behind the camera).
  const back = data.suns[1].dir
  const angle = (i: number) => {
    const e = data.frames[i].eye, l = Math.hypot(...e)
    return Math.round(Math.acos((e[0] * back[0] + e[1] * back[1] + e[2] * back[2]) / l) * 180 / Math.PI)
  }
  const result = {
    hid: HID, encoding: GRID ? 'neural texture' : 'fourier', gridTexels: GRID_P, gridMax: GRID_MAX, position: !!POS_ON, analyticSpecular: !!SPEC_ON, loss: ['linear', 'relative', 'display'][LOSS], params: P, macsPerPixel: MACS, steps: BENCH, trainSeconds: +((performance.now() - t0) / 1000).toFixed(1),
    psnrHeldOut: { classic: +classicTest.psnr.toFixed(2), neural: +neuralTest.psnr.toFixed(2) },
    brightnessHeldOut: { classic: +classicTest.ratio.toFixed(3), neural: +neuralTest.ratio.toFixed(3) },
    psnrTrain: { classic: +(await score(fitSet, true)).psnr.toFixed(2), neural: +(await score(fitSet, false)).psnr.toFixed(2) },
    msPerFrame: { cycles: +(cyclesSeconds * 1000).toFixed(0), classic: +classicMs.toFixed(3), neural: +neuralMs.toFixed(3) },
    sweep: classicTest.perView.map((c, k) => ({
      frame: c.frame, sunAngle: angle(c.frame),
      classic: { psnr: +c.psnr.toFixed(2), ratio: +c.ratio.toFixed(3) },
      neural: { psnr: +neuralTest.perView[k].psnr.toFixed(2), ratio: +neuralTest.perView[k].ratio.toFixed(3) },
    })).sort((a, b) => a.sunAngle - b.sunAngle),
    classicKnobs: jade.map(v => +v.toFixed(3)),
  }
  console.log('BENCH', JSON.stringify(result))
  Object.assign(window, { bench: result })
  showNeural(neuralTest)
  training = false
} else {
  // Live: re-score the network on the held-out views every few seconds while it trains.
  (async () => {
    for (;;) {
      await new Promise(r => setTimeout(r, 4000))
      showNeural(await score(testSet, false))
    }
  })()
}
requestAnimationFrame(frame)
