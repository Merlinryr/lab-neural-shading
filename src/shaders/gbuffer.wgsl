// Pass 1 — G-buffer. Rasterize the mesh once and store, per pixel, everything a
// shader needs to know about the surface. Classic and neural shading both read this.

struct Camera {
  viewProj: mat4x4f,
  pos: vec4f,
}
@group(0) @binding(0) var<uniform> cam: Camera;

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) pos: vec3f,
  @location(1) normal: vec3f,
  @location(2) albedo: vec3f,
  @location(3) baked: vec4f,
  @location(4) through: vec2f,
}

// Vertex layout of public/jade/mesh_vertices.bin (blender/jade_dragon.py).
@vertex
fn vs(@location(0) pos: vec3f, @location(1) normal: vec3f, @location(2) albedo: vec3f, @location(3) baked: vec4f,
      @location(4) through: vec2f) -> VOut {
  return VOut(cam.viewProj * vec4f(pos, 1.0), pos, normal, albedo, baked, through);
}

// Four rgba16float targets is WebGPU's default budget (32 bytes a pixel), so the two
// through-the-stone distances ride in the spare alpha channels.
struct GBuffer {
  @location(0) albedo: vec4f, // rgb base colour, a = distance through the stone towards the key sun
  @location(1) normal: vec4f, // xyz world normal, w = distance through the stone towards the back sun
  @location(2) pos: vec4f,    // xyz world position, w = 1 where there is geometry (cleared to 0)
  @location(3) baked: vec4f,  // thickness, ambient occlusion, key sun visibility, back sun visibility
}

@fragment
fn fs(in: VOut) -> GBuffer {
  return GBuffer(
    vec4f(in.albedo, in.through.x),
    vec4f(normalize(in.normal), in.through.y),
    vec4f(in.pos, 1.0),
    in.baked,
  );
}
