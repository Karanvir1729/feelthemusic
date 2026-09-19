import numpy as np, mujoco, zlib, struct, model as M
def png(path, img):
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
LIDZ = 0.0
def scene(lid_alpha, lid_lift, cam):
    return f"""<mujoco><compiler meshdir="."/><visual><global offwidth="1400" offheight="1000"/><headlight ambient=".45 .45 .45" diffuse=".6 .6 .6"/></visual>
<asset><mesh name="base" file="puck_base.stl" scale=".001 .001 .001"/><mesh name="lid" file="asm_lid.stl" scale=".001 .001 .001"/>
<mesh name="board" file="asm_board.stl" scale=".001 .001 .001"/><mesh name="lf" file="asm_LF.stl" scale=".001 .001 .001"/>
<mesh name="mf" file="asm_MF.stl" scale=".001 .001 .001"/><mesh name="lfi" file="asm_LFi.stl" scale=".001 .001 .001"/></asset>
<worldbody><light pos="0 -0.2 0.4" dir="0 0.4 -1"/><light pos="0.2 0.2 0.3" dir="-0.4 -0.4 -1"/>
<geom type="plane" size=".3 .3 .01" pos="0 0 -0.0017" rgba=".92 .92 .9 1"/>
<geom type="mesh" mesh="base" rgba=".95 .55 .15 1"/>
<geom type="mesh" mesh="lid" pos="0 0 {lid_lift}" rgba=".9 .9 .95 {lid_alpha}"/>
<geom type="mesh" mesh="board" rgba=".1 .35 .15 1"/>
<geom type="mesh" mesh="lf" rgba=".15 .15 .15 1"/><geom type="mesh" mesh="mf" rgba=".35 .35 .35 1"/><geom type="mesh" mesh="lfi" rgba=".75 .1 .1 1"/>
<camera name="c" pos="{cam[0]}" xyaxes="{cam[1]}"/></worldbody></mujoco>"""
views = {
 "puck_render_open.png": (0.35, 0.045, ("0.024 -0.105 0.13", "1 0 0 0 0.78 0.62")),
 "puck_render_closed.png": (1.0, 0.0, ("0.10 -0.09 0.10", "0.68 0.73 0 -0.38 0.35 0.86")),
 "puck_render_top.png": (0.0, 0.2, ("0.024 0.019 0.20", "1 0 0 0 1 0")),
}
for name, (alpha, lift, cam) in views.items():
    m = mujoco.MjModel.from_xml_string(scene(alpha, lift, cam)); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 800, 1100); r.update_scene(d, camera="c"); png(name, r.render()); r.close()
print("rendered")
