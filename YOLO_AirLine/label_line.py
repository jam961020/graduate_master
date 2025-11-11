#!/usr/bin/env python3
"""
Semi-automatic MULTI-LINE labeler with CLASS support.
- Draw many welding lines per image
- Per-line class assignment (cycle with [ / ] or number keys)
- Snap (auto-refine) each draft to nearby edges using weighted orthogonal regression
- Save all lines into one JSON per image (segment + normal form + quality + class)

Controls:
  Left Click x2 : draft line (P1, P2)
  h            : snap (auto-refine using edges near the draft)
  Enter        : commit current draft to the list of lines
  Tab/Shift+Tab: select next/prev committed line
  [ / ]        : cycle class for draft or selected line
  0..9         : set class index (if exists)
  d            : delete selected line
  r            : reset current draft
  s            : SAVE labels for current image
  n / p        : next / previous image (does not auto-save)
  q            : quit

Usage:
python label_lines_multiline.py --images ./input_images --out ./labels --classes "longi,transverse,fillet,other" --edge_low 50 --edge_high 150 --band_px 6
"""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

# ------------------------- Utilities -------------------------

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def line_from_points(p1, p2):
    v = p2 - p1
    if np.linalg.norm(v) < 1e-6:
        return None
    n = unit(np.array([-v[1], v[0]], dtype=np.float32))  # unit normal
    b = float(n @ p1)
    return n, b

def refine_with_edges(img_gray, p1, p2, canny_low, canny_high, band_px=6):
    """Refine draft line using edge pixels near it (band distance).
    Returns p1r, p2r, inlier_count, residual (mean abs perp dist)."""
    edges = cv2.Canny(img_gray, canny_low, canny_high, L2gradient=True)
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    lf = line_from_points(p1, p2)
    if lf is None:
        return None, None, 0, None
    n, b = lf

    ys, xs = np.where(edges > 0)
    if xs.size == 0:
        return None, None, 0, None
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    dists = pts @ n - b
    mask = np.abs(dists) <= band_px
    near = pts[mask]
    if near.shape[0] < 20:
        return None, None, 0, None

    wts = mag[near[:,1].astype(int), near[:,0].astype(int)].astype(np.float32)
    wts = np.clip(wts, 1.0, None)
    W = float(wts.sum())
    mu = (near * wts[:,None]).sum(axis=0) / W
    C = ((near - mu) * np.sqrt(wts[:,None])).T @ ((near - mu) * np.sqrt(wts[:,None])) / W
    eigvals, eigvecs = np.linalg.eigh(C)
    direction = unit(eigvecs[:, np.argmax(eigvals)])

    def project(pt, origin, direction):
        t = (pt - origin) @ direction
        return origin + t * direction

    p1r = project(p1, mu, direction)
    p2r = project(p2, mu, direction)

    n_ref = unit(np.array([-direction[1], direction[0]], dtype=np.float32))
    b_ref = float(n_ref @ mu)
    residual = float(np.mean(np.abs(near @ n_ref - b_ref)))
    return p1r, p2r, int(near.shape[0]), residual

# ------------------------- Main App -------------------------

PALETTE = [
    ( 60, 180, 255),  # 0 sky
    ( 80, 220, 100),  # 1 green
    (200, 160,  60),  # 2 amber
    (180,  80, 200),  # 3 purple
    ( 80, 160, 240),  # 4 blue
    ( 50,  50, 220),  # 5 red-ish
    (220,  80,  80),  # 6 coral
    (120, 120, 120),  # 7 gray
    ( 40, 200, 200),  # 8 teal
    (160,  80,  80),  # 9 brown
]

class LineLabeler:
    def __init__(self, img_paths, out_dir, classes, edge_low, edge_high, band_px=6):
        self.img_paths = img_paths
        self.idx = 0
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.classes = classes
        self.edge_low = edge_low
        self.edge_high = edge_high
        self.band_px = band_px
        self.reset_draft()
        self.lines = []  # list of dicts: p1,p2,p1r,p2r,cls,inl,res,snap
        self.sel = -1

    # ---------- Draft state ----------
    def reset_draft(self):
        self.p1 = None; self.p2 = None
        self.snap = None  # (p1r,p2r,inl,res)
        self.cls = 0

    # ---------- Mouse ----------
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.p1 is None:
                self.p1 = np.array([x, y], dtype=np.float32)
            elif self.p2 is None:
                self.p2 = np.array([x, y], dtype=np.float32)
            else:
                # start a new draft on third click
                self.p1 = np.array([x, y], dtype=np.float32)
                self.p2 = None
                self.snap = None

    # ---------- Rendering ----------
    def draw(self, img):
        vis = img.copy()
        H, W = vis.shape[:2]
        # committed lines
        for i, L in enumerate(self.lines):
            color = PALETTE[L['cls'] % len(PALETTE)] if self.classes else (0,255,255)
            p1 = (L.get('p1r') if L.get('p1r') is not None else L['p1']).astype(int)
            p2 = (L.get('p2r') if L.get('p2r') is not None else L['p2']).astype(int)
            thickness = 3 if i == self.sel else 2
            cv2.line(vis, tuple(p1), tuple(p2), color, thickness)
            tag = self.classes[L['cls']] if self.classes else str(L['cls'])
            mid = ((p1+p2)//2).astype(int)
            cv2.putText(vis, f"{tag}", (int(mid[0])+6, int(mid[1])-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # draft
        if self.p1 is not None:
            cv2.circle(vis, tuple(self.p1.astype(int)), 4, (0,255,0), -1)
        if self.p2 is not None:
            cv2.circle(vis, tuple(self.p2.astype(int)), 4, (0,255,0), -1)
            cv2.line(vis, tuple(self.p1.astype(int)), tuple(self.p2.astype(int)), (0,255,0), 2)
        if self.snap is not None and self.p1 is not None and self.p2 is not None:
            p1r, p2r, inl, res = self.snap
            if p1r is not None:
                cv2.line(vis, tuple(p1r.astype(int)), tuple(p2r.astype(int)), (0,128,255), 2)
                cv2.putText(vis, f"snap inl={inl} res={res:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,128,255), 2)
        # HUD
        help1 = "Clicks:P1,P2 | h:snap | Enter:commit | [/] or 0-9:class | Tab/Shift+Tab:select | d:del | r:reset | s:save | n/p:nav | q:quit"
        cv2.putText(vis, help1, (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        return vis

    # ---------- Actions ----------
    def snap_draft(self, gray):
        if self.p1 is None or self.p2 is None:
            return
        self.snap = refine_with_edges(gray, self.p1, self.p2, self.edge_low, self.edge_high, self.band_px)

    def commit_draft(self):
        if self.p1 is None or self.p2 is None:
            return
        p1, p2 = self.p1.copy(), self.p2.copy()
        p1r = p2r = None; inl = 0; res = None; used = False
        if self.snap is not None and self.snap[0] is not None:
            p1r, p2r, inl, res = self.snap
            used = True
        self.lines.append({
            'p1': p1, 'p2': p2, 'p1r': p1r, 'p2r': p2r,
            'cls': int(self.cls), 'inl': int(inl), 'res': (None if res is None else float(res)), 'snap': bool(used)
        })
        self.sel = len(self.lines)-1
        self.reset_draft()

    def cycle_class(self, delta):
        if self.p1 is None and self.sel >= 0:
            self.lines[self.sel]['cls'] = (self.lines[self.sel]['cls'] + delta) % max(1, len(self.classes))
        else:
            self.cls = (self.cls + delta) % max(1, len(self.classes))

    def set_class(self, idx):
        if idx < 0 or idx >= max(1, len(self.classes)):
            return
        if self.p1 is None and self.sel >= 0:
            self.lines[self.sel]['cls'] = idx
        else:
            self.cls = idx

    def select_next(self, delta):
        if not self.lines:
            self.sel = -1
            return
        self.sel = (self.sel + delta) % len(self.lines)

    def delete_selected(self):
        if 0 <= self.sel < len(self.lines):
            self.lines.pop(self.sel)
            self.sel = min(self.sel, len(self.lines)-1)

    def save_json(self, img_path, H, W):
        out = {
            'image': os.path.basename(img_path),
            'size': {'w': int(W), 'h': int(H)},
            'lines': [],
            'meta': {'time': datetime.utcnow().isoformat()+'Z', 'classes': self.classes}
        }
        for L in self.lines:
            # choose refined if available
            p1 = L['p1']; p2 = L['p2']
            if L['p1r'] is not None and L['p2r'] is not None:
                p1 = L['p1r']; p2 = L['p2r']
            lf = line_from_points(p1, p2)
            if lf is None:
                continue
            n, b = lf
            out['lines'].append({
                'class_id': int(L['cls']),
                'class_name': (self.classes[L['cls']] if self.classes else str(L['cls'])),
                'segment': {'x1': float(p1[0]), 'y1': float(p1[1]), 'x2': float(p2[0]), 'y2': float(p2[1])},
                'normal_form': {'nx': float(n[0]), 'ny': float(n[1]), 'b': float(b)},
                'quality': {'snap_used': bool(L['snap']), 'inlier_count': int(L['inl']), 'residual': (None if L['res'] is None else float(L['res']))}
            })
        out_path = self.out_dir / (Path(img_path).stem + '.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Saved {out_path} ({len(out['lines'])} lines)")

    # ---------- Main loop ----------
    def run(self):
        cv2.namedWindow('labeler', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('labeler', self.on_mouse)
        while 0 <= self.idx < len(self.img_paths):
            path = self.img_paths[self.idx]
            img = cv2.imread(path)
            if img is None:
                print(f"Failed to read {path}")
                self.idx += 1
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imshow('labeler', self.draw(img))
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                self.snap_draft(gray)
            elif key in (13, 10):  # Enter
                self.commit_draft()
            elif key == 9:  # Tab
                self.select_next(+1)
            elif key == 353:  # Shift+Tab (platform dependent; may not fire)
                self.select_next(-1)
            elif key == ord('['):
                self.cycle_class(-1)
            elif key == ord(']'):
                self.cycle_class(+1)
            elif key == ord('d'):
                self.delete_selected()
            elif key == ord('r'):
                self.reset_draft()
            elif key == ord('s'):
                H, W = img.shape[:2]
                self.save_json(path, H, W)
            elif key == ord('n'):
                self.idx += 1
                self.reset_draft(); self.lines = []; self.sel = -1
            elif key == ord('p'):
                self.idx = max(0, self.idx-1)
                self.reset_draft(); self.lines = []; self.sel = -1
            else:
                # number keys 0..9
                if ord('0') <= key <= ord('9'):
                    self.set_class(key - ord('0'))
            # redraw
            cv2.imshow('labeler', self.draw(img))
        cv2.destroyAllWindows()

# ------------------------- Entry -------------------------

def list_images(folder):
    exts = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
    return [str(p) for p in sorted(Path(folder).glob('**/*')) if p.suffix.lower() in exts]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--classes', default='')
    ap.add_argument('--edge_low', type=int, default=50)
    ap.add_argument('--edge_high', type=int, default=150)
    ap.add_argument('--band_px', type=int, default=6)
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(',') if c.strip()] if args.classes else []
    imgs = list_images(args.images)
    if not imgs:
        raise SystemExit('No images found. Check --images path.')

    app = LineLabeler(imgs, args.out, classes, args.edge_low, args.edge_high, args.band_px)
    app.run()
