// Draws one texture into one debug panel. MODE is a pipeline-override constant,
// so each panel gets its own specialised pipeline without a uniform buffer.

override MODE: u32 = 0u;

@group(0) @binding(0) var texA: texture_2d<f32>;
@group(0) @binding(1) var texB: texture_2d<f32>; // only read by the error views
@group(0) @binding(2) var samp: sampler;

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) uv: vec2f,
}

// Fullscreen triangle
@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
  return VOut(vec4f(p, 0.0, 1.0), vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5));
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  let a = textureSampleLevel(texA, samp, in.uv, 0.0);
  var c = a.rgb;
  switch MODE {
    case 1u: { c = select(vec3f(0.0), a.xyz * 0.5 + 0.5, dot(a.xyz, a.xyz) > 0.5); } // normal
    case 4u: { c = vec3f(0.675, 1.0, 0.6) * (1.0 - exp(-1.5 * max(a.r, 0.0))); } // feature map, the page's #acff99
    case 5u: {                                                                   // |a - target| x4
      let e = abs(a.rgb - textureSampleLevel(texB, samp, in.uv, 0.0).rgb);
      c = vec3f(1.0, 0.35, 0.1) * (e.r + e.g + e.b) * 4.0 / 3.0;
    }
    case 6u: { c = vec3f(0.3, 1.0, 0.55) * (1.0 - exp(-4.0 * a.r)); }           // baked thickness
    case 7u: { c = a.y * vec3f(0.35) + a.z * vec3f(0.4, 0.38, 0.3) + a.w * vec3f(0.3, 0.27, 0.2); } // AO + sun visibility
    default: {}
  }
  return vec4f(c, 1.0);
}
