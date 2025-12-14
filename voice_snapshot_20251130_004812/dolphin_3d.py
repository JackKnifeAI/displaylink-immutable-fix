#!/usr/bin/env python3
import math
import time
import os
import random

class Dolphin3D:
    def __init__(self):
        self.vertices = []
        self.triangles = []
        self.bubbles = []
        self.generate_mesh()

    def generate_mesh(self):
        # Body profile: (z, radius_x, radius_y, y_offset)
        body_profile = [
            (3.2, 0.02, 0.02, 0),
            (3.0, 0.08, 0.06, 0.01),
            (2.7, 0.18, 0.14, 0.03),
            (2.3, 0.28, 0.22, 0.06),
            (1.8, 0.40, 0.35, 0.10),
            (1.2, 0.52, 0.48, 0.08),
            (0.5, 0.62, 0.58, 0.03),
            (-0.2, 0.58, 0.54, -0.02),
            (-0.9, 0.48, 0.44, -0.06),
            (-1.5, 0.35, 0.30, -0.08),
            (-2.0, 0.22, 0.18, -0.05),
            (-2.4, 0.12, 0.10, -0.02),
            (-2.7, 0.06, 0.05, 0),
        ]

        segments = 16

        # Generate body vertices in rings
        body_rings = []
        for z, rx, ry, y_off in body_profile:
            ring = []
            for i in range(segments):
                angle = 2 * math.pi * i / segments
                x = rx * math.cos(angle)
                y = ry * math.sin(angle) + y_off
                self.vertices.append([x, y, z])
                ring.append(len(self.vertices) - 1)
            body_rings.append(ring)

        # Connect rings with triangles
        for r in range(len(body_rings) - 1):
            ring1 = body_rings[r]
            ring2 = body_rings[r + 1]
            for i in range(segments):
                i_next = (i + 1) % segments
                v0, v1 = ring1[i], ring1[i_next]
                v2, v3 = ring2[i], ring2[i_next]
                self.triangles.append((v0, v2, v1))
                self.triangles.append((v1, v2, v3))

        # Add tail flukes
        self._add_tail_flukes()
        # Add dorsal fin
        self._add_dorsal_fin()
        # Add pectoral fins
        self._add_pectoral_fin(0.15, 1)
        self._add_pectoral_fin(-0.15, -1)

        self.body_rings = body_rings

    def _add_tail_flukes(self):
        # Left fluke
        base = len(self.vertices)
        self.vertices.extend([
            [0, 0, -2.7],
            [0.6, 0.05, -3.3],
            [0.8, 0, -3.6],
            [0.5, -0.05, -3.4],
            [0.2, 0, -3.0],
        ])
        self.triangles.extend([
            (base, base+1, base+4),
            (base+1, base+2, base+4),
            (base+2, base+3, base+4),
        ])

        # Right fluke
        base = len(self.vertices)
        self.vertices.extend([
            [0, 0, -2.7],
            [-0.6, 0.05, -3.3],
            [-0.8, 0, -3.6],
            [-0.5, -0.05, -3.4],
            [-0.2, 0, -3.0],
        ])
        self.triangles.extend([
            (base, base+4, base+1),
            (base+1, base+4, base+2),
            (base+2, base+4, base+3),
        ])

    def _add_dorsal_fin(self):
        base = len(self.vertices)
        self.vertices.extend([
            [0, 0.58, 0.5],
            [0.03, 0.95, -0.2],
            [0, 0.75, -0.8],
            [0.03, 0.55, -0.5],
            [0, 0.48, 0.3],
            [-0.03, 0.95, -0.2],
            [-0.03, 0.55, -0.5],
        ])
        self.triangles.extend([
            (base, base+1, base+4),
            (base+1, base+2, base+3),
            (base+1, base+3, base+4),
            (base, base+4, base+5),
            (base+5, base+2, base+6),
            (base+5, base+6, base+4),
        ])

    def _add_pectoral_fin(self, x_off, side):
        base = len(self.vertices)
        self.vertices.extend([
            [x_off, -0.1, 1.2],
            [x_off + side * 0.7, -0.3, 0.8],
            [x_off + side * 0.8, -0.35, 0.4],
            [x_off + side * 0.5, -0.25, 0.3],
            [x_off + side * 0.2, -0.15, 0.6],
        ])
        if side > 0:
            self.triangles.extend([
                (base, base+1, base+4),
                (base+1, base+2, base+4),
                (base+2, base+3, base+4),
            ])
        else:
            self.triangles.extend([
                (base, base+4, base+1),
                (base+1, base+4, base+2),
                (base+2, base+4, base+3),
            ])


class Bubble:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.vx = random.uniform(-0.02, 0.02)
        self.vy = random.uniform(0.05, 0.15)
        self.vz = random.uniform(-0.02, 0.02)
        self.life = random.uniform(0.5, 1.5)
        self.size = random.choice(['°', '○', '◦', '∘', 'o'])

    def update(self, dt):
        self.x += self.vx * dt * 10
        self.y += self.vy * dt * 10
        self.z += self.vz * dt * 10
        self.life -= dt
        self.vx += random.uniform(-0.01, 0.01)
        self.vz += random.uniform(-0.01, 0.01)
        return self.life > 0


def rotate_x(v, angle):
    c, s = math.cos(angle), math.sin(angle)
    return [v[0], v[1]*c - v[2]*s, v[1]*s + v[2]*c]

def rotate_y(v, angle):
    c, s = math.cos(angle), math.sin(angle)
    return [v[0]*c + v[2]*s, v[1], -v[0]*s + v[2]*c]

def rotate_z(v, angle):
    c, s = math.cos(angle), math.sin(angle)
    return [v[0]*c - v[1]*s, v[0]*s + v[1]*c, v[2]]

def normalize(v):
    mag = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if mag < 0.0001:
        return [0, 0, 1]
    return [v[0]/mag, v[1]/mag, v[2]/mag]

def cross(u, v):
    return [
        u[1]*v[2] - u[2]*v[1],
        u[2]*v[0] - u[0]*v[2],
        u[0]*v[1] - u[1]*v[0]
    ]

def compute_normal(v0, v1, v2):
    ux, uy, uz = v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]
    vx, vy, vz = v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]
    nx = uy*vz - uz*vy
    ny = uz*vx - ux*vz
    nz = ux*vy - uy*vx
    return normalize([nx, ny, nz])


def render_dolphin(dolphin, width, height, angle_x, angle_y, angle_z, swim_phase, bubbles):
    screen = [[' ' for _ in range(width)] for _ in range(height)]
    z_buffer = [[float('-inf') for _ in range(width)] for _ in range(height)]

    light = normalize([0.3, 0.8, 0.5])
    ascii_chars = " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@"

    fov = 35
    distance = 8

    def project(v):
        d = distance - v[2]
        if d <= 0.1:
            d = 0.1
        scale = fov / d
        sx = int(width/2 + v[0] * scale * 2)
        sy = int(height/2 - v[1] * scale)
        return sx, sy, v[2]

    # Animate vertices with swimming motion
    anim_verts = []
    for v in dolphin.vertices:
        nv = v.copy()
        wave = math.sin(swim_phase * 2 - v[2] * 1.2)
        intensity = max(0, (2.5 - v[2]) / 5.5) ** 1.5
        nv[0] += wave * 0.18 * intensity
        nv[1] += math.sin(swim_phase * 2 - v[2] * 0.8) * 0.04 * intensity

        nv = rotate_x(nv, angle_x)
        nv = rotate_y(nv, angle_y)
        nv = rotate_z(nv, angle_z)
        anim_verts.append(nv)

    # Sort triangles by depth
    tri_depths = []
    for i, tri in enumerate(dolphin.triangles):
        avg_z = sum(anim_verts[vi][2] for vi in tri) / 3
        tri_depths.append((avg_z, i))
    tri_depths.sort(key=lambda x: x[0], reverse=True)

    # Render triangles
    for _, tri_idx in tri_depths:
        tri = dolphin.triangles[tri_idx]
        v0 = anim_verts[tri[0]]
        v1 = anim_verts[tri[1]]
        v2 = anim_verts[tri[2]]

        normal = compute_normal(v0, v1, v2)
        dot = normal[0]*light[0] + normal[1]*light[1] + normal[2]*light[2]
        brightness = (dot + 1) / 2 * 0.85 + 0.15
        char_idx = int(brightness * (len(ascii_chars) - 1))
        char = ascii_chars[max(0, min(char_idx, len(ascii_chars)-1))]

        p0 = project(v0)
        p1 = project(v1)
        p2 = project(v2)

        # Get bounding box
        min_x = max(0, min(p0[0], p1[0], p2[0]))
        max_x = min(width-1, max(p0[0], p1[0], p2[0]))
        min_y = max(0, min(p0[1], p1[1], p2[1]))
        max_y = min(height-1, max(p0[1], p1[1], p2[1]))

        # Rasterize triangle
        for py in range(min_y, max_y + 1):
            for px in range(min_x, max_x + 1):
                # Barycentric coordinates
                p0x, p0y = p0[0], p0[1]
                p1x, p1y = p1[0], p1[1]
                p2x, p2y = p2[0], p2[1]

                denom = (p1y - p2y) * (p0x - p2x) + (p2x - p1x) * (p0y - p2y)
                if abs(denom) < 0.001:
                    continue

                w0 = ((p1y - p2y) * (px - p2x) + (p2x - p1x) * (py - p2y)) / denom
                w1 = ((p2y - p0y) * (px - p2x) + (p0x - p2x) * (py - p2y)) / denom
                w2 = 1 - w0 - w1

                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    z = w0 * p0[2] + w1 * p1[2] + w2 * p2[2]
                    if z > z_buffer[py][px]:
                        z_buffer[py][px] = z
                        screen[py][px] = char

    # Render eyes
    eye_positions = [[0.18, 0.15, 2.1], [-0.18, 0.15, 2.1]]
    for ex, ey, ez in eye_positions:
        ev = [ex, ey, ez]
        ev = rotate_x(ev, angle_x)
        ev = rotate_y(ev, angle_y)
        ev = rotate_z(ev, angle_z)
        sx, sy, sz = project(ev)
        if 0 <= sx < width and 0 <= sy < height:
            if sz > z_buffer[sy][sx] - 0.1:
                screen[sy][sx] = '●'

    # Render bubbles
    for b in bubbles:
        bv = rotate_x([b.x, b.y, b.z], angle_x)
        bv = rotate_y(bv, angle_y)
        bv = rotate_z(bv, angle_z)
        sx, sy, sz = project(bv)
        if 0 <= sx < width and 0 <= sy < height:
            if sz > z_buffer[sy][sx]:
                screen[sy][sx] = b.size

    return [''.join(row) for row in screen]


def render_frame_for_panel(width=60, height=20, angle_y=0, swim_phase=0):
    """Render a single frame suitable for embedding in a Rich panel"""
    dolphin = Dolphin3D()
    angle_x = 0.15 * math.sin(swim_phase * 0.5)
    angle_z = 0.08 * math.sin(swim_phase * 0.7)
    frame = render_dolphin(dolphin, width, height, angle_x, angle_y, angle_z, swim_phase, [])
    return '\n'.join(frame)


def main():
    width, height = 120, 50
    dolphin = Dolphin3D()
    bubbles = []

    angle_x = 0.2
    angle_y = 0
    angle_z = 0
    swim_phase = 0
    frame_count = 0

    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')

            frame = render_dolphin(dolphin, width, height, angle_x, angle_y, angle_z, swim_phase, bubbles)

            # Add bubbles from blowhole
            if frame_count % 8 == 0:
                bx = 0.05 * math.cos(angle_y)
                by = 0.4
                bz = 2.5
                bubbles.append(Bubble(bx, by, bz))

            # Update bubbles
            bubbles = [b for b in bubbles if b.update(0.035)]

            # Wave border
            t = time.time()
            wave1 = ''.join(['~' if math.sin(t*3 + i*0.5) > 0 else '≈' for i in range(width-12)])

            print(f"\033[36m  ░▒▓ {wave1} 🐬\033[0m")
            for row in frame:
                print(f"  {row}")
            print(f"\033[36m  ░▒▓ {wave1[::-1]} ▓▒░\033[0m")
            print(f"\n  \033[90mSwimming... [Ctrl+C to release dolphin]\033[0m")

            angle_y += 0.025
            angle_x = 0.15 * math.sin(swim_phase * 0.5)
            angle_z = 0.08 * math.sin(swim_phase * 0.7)
            swim_phase += 0.08
            frame_count += 1

            time.sleep(0.035)

    except KeyboardInterrupt:
        print("\n\033[36m  🐬 *splooosh* The dolphin leaps into the sunset! 🌅\033[0m\n")


if __name__ == "__main__":
    main()
