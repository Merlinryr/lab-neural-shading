"""Stanford dragon carved in jade, path-traced with Cycles, exported for the lab.

Run headless (from the repo root):
  "C:/Program Files/Blender Foundation/Blender 5.2/blender.exe" -b -P blender/jade_dragon.py -- --frames 64 --samples 512
  ... -- --preview     one frame to blender/jade_preview.png, no export
  ... -- --bake-only   re-export the lab's mesh and bakes, keep the frames

Lab space: Y up, dragon centred on the origin, longest side 2. Blender is Z up: (x, y, z) -> (x, -z, y),
the same convention as the glTF importer, so the conversion is a plain change of axes.

Outputs in public/jade/ (raw little-endian, row-major):
  frames/NNN.png       Cycles renders, sRGB display colours (Standard view, no tonemap), alpha 0 off the dragon
  cameras.json         per-frame camera in lab space, the lights, render time per frame
  mesh_vertices.bin    float32 x 15 per vertex: pos3, normal3, albedo3, thickness, ao, shadow per sun (2),
                       distance through the stone towards each sun (2)
  mesh_indices.bin     uint32 triangles
The lab's mesh is a decimated copy (~300k triangles). Thickness, AO and the sun shadows are baked
per vertex here: what a game would precompute for a static object, and what the classic shader uses.
"""
import json, math, os, sys, time

import bpy
import bmesh
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector, noise
from mathutils.bvhtree import BVHTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLB = os.path.join(ROOT, 'src', 'models', 'dragon.glb')
OUT = os.path.join(ROOT, 'public', 'jade')

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
PREVIEW, BAKE_ONLY = '--preview' in args, '--bake-only' in args
opt = dict(zip(*[iter(a for a in args if a not in ('--preview', '--bake-only'))] * 2))
FRAMES, SAMPLES = int(opt.get('--frames', 64)), int(opt.get('--samples', 512))

# ---- Shared with src/main.ts (through cameras.json) ----
SIZE, FOV_Y, DISTANCE = 512, math.radians(40), 3.5
SUNS = [  # lab-space direction towards the sun, colour x strength = irradiance (W/m^2) on a facing surface
    dict(name='Key', dir=(-0.45, 0.7, 0.55), color=(1.0, 0.95, 0.88), strength=1.3),
    dict(name='Back', dir=(0.5, 0.45, -0.75), color=(1.0, 0.9, 0.72), strength=2.6),
]
SUN_ANGLE = math.radians(3)
WORLD = (0.035, 0.045, 0.055)  # uniform sky radiance
LAYOUT = 'pos3 normal3 albedo3 thickness ao shadow0 shadow1 through0 through1'
JADE = dict(roughness=0.12, ior=1.6, sss_scale=0.12, sss_radius=(0.9, 1.4, 0.6))


def to_blender(v):  # lab (x, y, z) -> Blender (x, -z, y)
    return (v[0], -v[2], v[1])


def jade_color(p):
    """Albedo at lab-space point p: cloudy green stone with darker veins. Linear RGB."""
    cloud = 0.5 + 0.5 * noise.fractal(Vector(p) * 1.8, 0.6, 2.0, 4)
    warp = Vector(p) * 2.2 + 0.6 * noise.noise_vector(Vector(p) * 1.2)
    vein = 1.0 - min(abs(noise.noise(warp)) * 6.0, 1.0)                   # thin bright lines -> 1
    deep, pale, dark = np.array((0.03, 0.2, 0.08)), np.array((0.22, 0.52, 0.28)), np.array((0.015, 0.07, 0.035))
    c = deep + (pale - deep) * min(max(cloud, 0.0), 1.0)
    return tuple(c + (dark - c) * vein ** 3 * 0.8)


# ---------------------------------------------------------------- scene

def import_dragon():
    """Import the glb, join the chunks and move it to lab space (centred, longest side 2)."""
    bpy.ops.import_scene.gltf(filepath=GLB)
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    for o in meshes:
        o.data.transform(o.matrix_world)
        o.parent = None
        o.matrix_world.identity()
    for o in [o for o in bpy.context.scene.objects if o.type != 'MESH']:
        bpy.data.objects.remove(o)
    with bpy.context.temp_override(active_object=meshes[0], selected_editable_objects=meshes):
        bpy.ops.object.join()
    obj = meshes[0]
    obj.name = 'Dragon'
    co = np.zeros(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get('co', co)
    lab = co.reshape(-1, 3)[:, [0, 2, 1]] * (1, 1, -1)                      # Blender -> lab
    lo, hi = lab.min(0), lab.max(0)
    lab = (lab - (lo + hi) / 2) * (2 / (hi - lo).max())
    obj.data.vertices.foreach_set('co', (lab[:, [0, 2, 1]] * (1, -1, 1)).ravel())
    obj.data.update()
    return obj, lab


def set_colors(mesh, lab):
    attr = mesh.color_attributes.new('Col', 'FLOAT_COLOR', 'POINT')
    attr.data.foreach_set('color', np.array([(*jade_color(p), 1.0) for p in lab], np.float32).ravel())


def jade_material():
    m = bpy.data.materials.new('Jade')
    if not m.node_tree:
        m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes['Principled BSDF']
    col = nt.nodes.new('ShaderNodeAttribute')
    col.attribute_name = 'Col'
    nt.links.new(col.outputs['Color'], b.inputs['Base Color'])
    b.subsurface_method = 'RANDOM_WALK'
    b.inputs['Subsurface Weight'].default_value = 1.0
    b.inputs['Subsurface Scale'].default_value = JADE['sss_scale']
    b.inputs['Subsurface Radius'].default_value = JADE['sss_radius']
    b.inputs['Roughness'].default_value = JADE['roughness']
    b.inputs['IOR'].default_value = JADE['ior']
    return m


def add_suns():
    for s in SUNS:
        data = bpy.data.lights.new(s['name'], 'SUN')
        data.color, data.energy, data.angle = s['color'], s['strength'], SUN_ANGLE
        obj = bpy.data.objects.new(s['name'], data)
        obj.rotation_euler = (-Vector(to_blender(s['dir']))).to_track_quat('-Z', 'Y').to_euler()  # shines along -Z
        bpy.context.scene.collection.objects.link(obj)


def setup_render(scene):
    scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    for backend in ('OPTIX', 'CUDA', 'HIP', 'ONEAPI', 'METAL'):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices()
        if any(d.type == backend for d in prefs.devices):
            for d in prefs.devices:
                d.use = d.type == backend
            scene.cycles.device = 'GPU'
            print('Cycles device:', backend)
            break
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = True
    scene.render.resolution_x = scene.render.resolution_y = SIZE
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.view_settings.view_transform = 'Standard'   # plain sRGB, so the lab's shaders can match it exactly
    scene.view_settings.look = 'None'
    world = bpy.data.worlds.new('World')
    if not world.node_tree:
        world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (*WORLD, 1)
    scene.world = world


def lab_eye(yaw, pitch):  # same formula as main.ts
    return (DISTANCE * math.cos(pitch) * math.sin(yaw), DISTANCE * math.sin(pitch), DISTANCE * math.cos(pitch) * math.cos(yaw))


def point_camera(cam, pos):
    cam.location = to_blender(pos)
    cam.rotation_euler = (-Vector(to_blender(pos))).to_track_quat('-Z', 'Y').to_euler()
    bpy.context.view_layer.update()


def self_check(scene, cam, pos):
    """Blender must project lab points to the same pixels as perspective() * lookAt() in src/scene.ts."""
    e = np.array(pos)
    z = e / np.linalg.norm(e)
    x = np.cross((0, 1, 0), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    f = 1 / math.tan(FOV_Y / 2)
    worst = 0.0
    for p in [(0, 0, 0), (0.8, 0.4, 0.2), (-0.9, -0.5, 0.3), (0.3, -0.6, -0.4)]:
        q = np.array(p) - e
        lab = ((f * q @ x / -(q @ z)) * 0.5 + 0.5, 0.5 - (f * q @ y / -(q @ z)) * 0.5)
        u, v, _ = world_to_camera_view(scene, cam, Vector(to_blender(p)))
        worst = max(worst, math.hypot(lab[0] - u, lab[1] - (1 - v)) * SIZE)
    print(f'self-check: lab and Blender projections differ by {worst:.3f} px')
    assert worst < 0.5, 'camera does not match the lab (fov / axes)'


# ---------------------------------------------------------------- the lab's mesh + bakes

def lab_mesh(src):
    """Decimated, welded copy in lab space with smooth vertex normals."""
    bm = bmesh.new()
    bm.from_mesh(src.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)             # weld the glb's chunk seams
    me = bpy.data.meshes.new('LabDragon')
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new('LabDragon', me)
    bpy.context.scene.collection.objects.link(obj)
    dec = obj.modifiers.new('Decimate', 'DECIMATE')
    dec.ratio = 0.35
    with bpy.context.temp_override(object=obj, active_object=obj):
        bpy.ops.object.modifier_apply(modifier='Decimate')
    me.shade_smooth()
    obj.hide_render = True
    return obj


def sphere_dirs(n=48):  # Fibonacci sphere
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    a = math.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(1 - z * z)
    return np.stack([r * np.cos(a), r * np.sin(a), z], 1)


def bake(lab_v, lab_n, faces):
    """Per vertex: thickness (mean distance through the stone along the inward hemisphere),
    ambient occlusion, how much of each sun's disc it sees, and how far sunlight travels through
    the stone to reach it (0 where the surface faces that sun). All in lab space.
    The last one is Grassi's "density toward the sun", the dominant predictor for his clouds."""
    bvh = BVHTree.FromPolygons([tuple(v) for v in lab_v], faces)
    dirs = [Vector(d) for d in sphere_dirs()]
    suns = []
    for s in SUNS:  # the centre of the sun's disc and 4 points on its rim
        l = Vector(s['dir']).normalized()
        t = l.orthogonal().normalized()
        b = l.cross(t)
        suns.append([(l + o * math.tan(SUN_ANGLE / 2)).normalized() for o in (Vector(), t, -t, b, -b)])
    out = np.zeros((len(lab_v), 2 + 2 * len(SUNS)), np.float32)
    t0 = time.time()
    for i, (p, n) in enumerate(zip(lab_v, lab_n)):
        p, n = Vector(p), Vector(n)
        thick = wt = ao = wa = 0.0
        for d in dirs:
            c = d.dot(n)
            if c < 0:   # inward: how far until the ray leaves the stone
                hit = bvh.ray_cast(p - n * 1e-4, d, 1.0)[3]
                thick += -c * (1.0 if hit is None else hit)
                wt += -c
            else:       # outward: is the sky blocked?
                ao += c * (bvh.ray_cast(p + n * 1e-4, d, 0.5)[0] is None)
                wa += c
        out[i, 0], out[i, 1] = thick / wt, ao / wa
        for k, rays in enumerate(suns):
            out[i, 2 + k] = sum(bvh.ray_cast(p + n * 1e-4, d)[0] is None for d in rays) / len(rays)
            through = bvh.ray_cast(p - n * 1e-4, rays[0])[3]                  # from inside, towards the sun
            out[i, 2 + len(SUNS) + k] = 0.0 if through is None else through
        if i % 20000 == 0:
            print(f'bake {i}/{len(lab_v)} ({time.time() - t0:.0f} s)')
    return out


def export_lab_mesh(obj):
    me = obj.data
    nv = len(me.vertices)
    co, nb = np.zeros(nv * 3), np.zeros(nv * 3)
    me.vertices.foreach_get('co', co)
    me.vertex_normals.foreach_get('vector', nb)
    lab_v = co.reshape(-1, 3)[:, [0, 2, 1]] * (1, 1, -1)
    lab_n = nb.reshape(-1, 3)[:, [0, 2, 1]] * (1, 1, -1)
    me.calc_loop_triangles()
    tris = np.zeros(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get('vertices', tris)
    albedo = np.array([jade_color(p) for p in lab_v])
    baked = bake(lab_v, lab_n, tris.reshape(-1, 3).tolist())
    np.concatenate([lab_v, lab_n, albedo, baked], 1).astype(np.float32).tofile(os.path.join(OUT, 'mesh_vertices.bin'))
    tris.astype(np.uint32).tofile(os.path.join(OUT, 'mesh_indices.bin'))
    return nv, tris.size


# ---------------------------------------------------------------- main

def main():
    if bpy.app.background:
        bpy.ops.wm.read_factory_settings(use_empty=True)
    else:  # run through the Blender connector: build in a new scene, leave the user's file alone
        bpy.context.window.scene = bpy.data.scenes.new('JadeDragon')
    scene = bpy.context.scene
    os.makedirs(os.path.join(OUT, 'frames'), exist_ok=True)

    dragon, lab = import_dragon()
    set_colors(dragon.data, lab)
    dragon.data.materials.clear()
    dragon.data.materials.append(jade_material())
    add_suns()
    setup_render(scene)

    cam = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
    cam.data.sensor_fit = 'VERTICAL'
    cam.data.angle_y = FOV_Y
    cam.data.clip_start, cam.data.clip_end = 0.1, 40
    scene.collection.objects.link(cam)
    scene.camera = cam

    if PREVIEW:
        point_camera(cam, lab_eye(0.6, 0.25))
        self_check(scene, cam, lab_eye(0.6, 0.25))
        scene.render.filepath = os.path.join(ROOT, 'blender', 'jade_preview.png')
        t = time.time()
        bpy.ops.render.render(write_still=True)
        print(f'preview rendered in {time.time() - t:.1f} s')
        return

    print('Lab mesh + bakes...')
    nv, ni = export_lab_mesh(lab_mesh(dragon))
    if BAKE_ONLY:
        path = os.path.join(OUT, 'cameras.json')
        meta = json.load(open(path))
        meta['mesh'].update(count=nv, indexCount=ni, layout=LAYOUT)
        json.dump(meta, open(path, 'w'), indent=1)
        print('Done (bake only):', OUT)
        return

    # A full orbit at 4 heights. The lab picks the held-out views (HELD_OUT in main.ts).
    side = 4
    views = [(2 * math.pi * k / (FRAMES // side), p) for p in np.linspace(0.05, 0.65, side) for k in range(FRAMES // side)]
    frames = []
    for i, (yaw, pitch) in enumerate(views):
        pos = lab_eye(yaw, pitch)
        point_camera(cam, pos)
        if i == 0:
            self_check(scene, cam, pos)
        name = f'frames/{i:03d}.png'
        scene.render.filepath = os.path.join(OUT, name)
        t = time.time()
        bpy.ops.render.render(write_still=True)
        frames.append(dict(file=name, yaw=yaw, pitch=float(pitch), eye=pos, seconds=round(time.time() - t, 2)))
        print(f'frame {i + 1}/{len(views)} in {frames[-1]["seconds"]} s')

    json.dump(dict(
        size=SIZE, fovY=FOV_Y, distance=DISTANCE, samples=SAMPLES, colorSpace='sRGB (Standard view transform)',
        suns=[dict(dir=list(np.array(s['dir']) / np.linalg.norm(s['dir'])), irradiance=[c * s['strength'] for c in s['color']]) for s in SUNS],
        world=WORLD, jade=JADE,
        mesh=dict(vertices='mesh_vertices.bin', count=nv, indices='mesh_indices.bin', indexCount=ni,
                  layout=LAYOUT),
        frames=frames,
    ), open(os.path.join(OUT, 'cameras.json'), 'w'), indent=1)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(ROOT, 'blender', 'jade_dragon.blend'))
    print('Done:', OUT)


if __name__ == '__main__':
    main()
