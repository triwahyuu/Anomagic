import os
import cv2
import numpy as np
import json
import random
import math
import argparse
from PIL import Image


# --- Crack-shaped mask primitive ---------------------------------------------------------
# NOT part of Anomagic's own released code. The shipped generate_adaptive_mask() below only
# offers rectangle/ellipse/polygon/irregular-blob primitives (verified by reading this file
# directly) -- none of them elongated/line-like. This is new code added to give the shape
# primitive pool an actual crack-like option, wired in as one more entry in the existing
# shape_type random.choice() pool (see generate_adaptive_mask()'s 'crack' branch below), not a
# replacement for the existing shapes. Algorithm: a random-walk / midpoint-displacement main
# path (the same technique used to generate mountain-range/coastline silhouettes), tapered in
# width from base to tip, with a small number of shorter sub-branches forking off the main
# path at random points along it.

def _midpoint_displace(p1, p2, roughness, depth):
    """Recursively displace the midpoint of segment p1-p2 perpendicular to itself, to turn a
    straight line into a jagged crack-like path. Returns an ordered list of points p1..p2."""
    if depth <= 0:
        return [p1, p2]
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    seg_len = math.hypot(dx, dy)
    if seg_len < 1e-6:
        return [p1, p2]
    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    nx, ny = -dy / seg_len, dx / seg_len
    offset = random.uniform(-1, 1) * roughness * seg_len
    mid = (mx + nx * offset, my + ny * offset)
    left = _midpoint_displace(p1, mid, roughness * 0.65, depth - 1)
    right = _midpoint_displace(mid, p2, roughness * 0.65, depth - 1)
    return left[:-1] + right


def _draw_tapered_polyline(canvas, path, max_width):
    """Draws `path` as a sequence of line segments whose thickness tapers from max_width at
    the start down to ~30% of max_width at the far end, giving the line a crack-like "thick
    near the root, thin at the tip" profile."""
    n = len(path)
    if n < 2:
        return
    for i in range(n - 1):
        t = i / max(1, n - 2)
        w = max(1, int(round(max_width * (1 - 0.7 * t))))
        p1 = (int(round(path[i][0])), int(round(path[i][1])))
        p2 = (int(round(path[i + 1][0])), int(round(path[i + 1][1])))
        cv2.line(canvas, p1, p2, 255, thickness=w, lineType=cv2.LINE_AA)


def generate_crack_mask(size, x, y, width, height):
    """Procedural elongated crack-shaped mask, drawn inside the same (x, y, width, height)
    bounding box convention generate_adaptive_mask() already uses for its other shape
    primitives -- so it plugs into the existing per-anomaly placement/overlap logic unchanged.
    """
    canvas = np.zeros(size, dtype=np.uint8)

    angle = random.uniform(0, 2 * math.pi)
    length = max(width, height) * random.uniform(0.9, 1.4)
    cx, cy = x + width / 2, y + height / 2
    p1 = (cx - math.cos(angle) * length / 2, cy - math.sin(angle) * length / 2)
    p2 = (cx + math.cos(angle) * length / 2, cy + math.sin(angle) * length / 2)

    depth = random.randint(4, 6)
    path = _midpoint_displace(p1, p2, roughness=0.5, depth=depth)

    max_line_width = max(2, int(min(width, height) * random.uniform(0.06, 0.16)))
    _draw_tapered_polyline(canvas, path, max_width=max_line_width)

    num_branches = random.randint(1, 3)
    for _ in range(num_branches):
        t = random.uniform(0.2, 0.8)
        idx = int(t * (len(path) - 1))
        origin = path[idx]
        branch_angle = angle + random.choice([-1, 1]) * random.uniform(math.pi / 6, math.pi / 3)
        branch_len = length * random.uniform(0.2, 0.45)
        branch_end = (
            origin[0] + math.cos(branch_angle) * branch_len,
            origin[1] + math.sin(branch_angle) * branch_len,
        )
        branch_path = _midpoint_displace(origin, branch_end, roughness=0.5, depth=max(2, depth - 2))
        _draw_tapered_polyline(canvas, branch_path, max_width=max(1, max_line_width // 2))

    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    _, canvas = cv2.threshold(canvas, 50, 255, cv2.THRESH_BINARY)
    return canvas


def generate_adaptive_mask(template, info, max_attempts=100, shape_pool=None):
    """shape_pool: optional override of the shape-type choices (default: the full
    ['rectangle', 'ellipse', 'polygon', 'irregular', 'crack'] mix). Pass e.g. ['crack'] to
    force every generated anomaly instance to use the crack primitive -- useful for a
    single-defect-category use case (KDW's pipe cracks) or for testing, without disturbing
    the default random mix any other caller relies on."""

    size = (256, 256)
    template_mask = (template == 255)
    best_mask = None
    best_overlap = 0

    width_range = info[0] if isinstance(info, list) and len(info) >= 3 else [0.2, 0.3]
    height_range = info[1] if isinstance(info, list) and len(info) >= 3 else [0.2, 0.3]
    need_rotation = info[2] == "True" if isinstance(info, list) and len(info) >= 3 else True


    grid_size = 6
    grid_positions = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    random.shuffle(grid_positions)

    for attempt in range(max_attempts):
        mask = np.zeros(size, dtype=np.uint8)
        used_positions = set()

        num_anomalies = random.randint(1, 3)

        for idx in range(num_anomalies):

            if idx < len(grid_positions):
                grid_i, grid_j = grid_positions[idx]
                used_positions.add((grid_i, grid_j))
            else:

                grid_i, grid_j = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)

            cell_width = size[0] // grid_size
            cell_height = size[1] // grid_size
            base_x = grid_i * cell_width
            base_y = grid_j * cell_height

            x_offset = random.randint(0, cell_width // 3)
            y_offset = random.randint(0, cell_height // 3)
            x = min(base_x + x_offset, size[0] - 10)
            y = min(base_y + y_offset, size[1] - 10)

            shape_type = random.choice(shape_pool or ['rectangle', 'ellipse', 'polygon', 'irregular', 'crack'])

            width = random.randint(int(width_range[0] * 256), int(width_range[1] * 256))
            height = random.randint(int(height_range[0] * 256), int(height_range[1] * 256))
            width = max(width, 8)
            height = max(height, 8)

            x = max(0, min(x, size[0] - width))
            y = max(0, min(y, size[1] - height))

            if shape_type == 'rectangle':
                cv2.rectangle(mask, (x, y), (x + width, y + height), 255, -1)
            elif shape_type == 'ellipse':
                center = (x + width // 2, y + height // 2)
                axes = (width // 2, height // 2)
                angle = random.randint(0, 180) if need_rotation else 0
                cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
            elif shape_type == 'polygon':
                num_vertices = random.randint(3, 6)
                vertices = []
                center_x, center_y = x + width // 2, y + height // 2
                radius_x, radius_y = width // 2, height // 2
                for i in range(num_vertices):
                    angle = 2 * math.pi * i / num_vertices
                    offset_x = radius_x * math.cos(angle) * random.uniform(0.7, 1.3)
                    offset_y = radius_y * math.sin(angle) * random.uniform(0.7, 1.3)
                    vertices.append([center_x + offset_x, center_y + offset_y])
                vertices = np.array(vertices, dtype=np.int32)
                cv2.fillPoly(mask, [vertices], 255)
            elif shape_type == 'irregular':
                temp_mask = np.zeros(size, dtype=np.uint8)
                num_points = random.randint(4, 6)
                points = []
                for _ in range(num_points):
                    px = x + random.randint(0, width)
                    py = y + random.randint(0, height)
                    points.append((px, py))
                points = np.array(points, dtype=np.int32)
                cv2.polylines(temp_mask, [points], isClosed=True, color=255, thickness=random.randint(2, 5))
                contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(temp_mask, contours, -1, 255, -1)
                temp_mask = cv2.GaussianBlur(temp_mask, (9, 9), 0)
                _, temp_mask = cv2.threshold(temp_mask, 50, 255, cv2.THRESH_BINARY)
                mask = cv2.bitwise_or(mask, temp_mask)
            elif shape_type == 'crack':
                temp_mask = generate_crack_mask(size, x, y, width, height)
                mask = cv2.bitwise_or(mask, temp_mask)

            if need_rotation and random.random() > 0.5:
                center = (x + width // 2, y + height // 2)
                angle = random.randint(0, 180)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                mask = cv2.warpAffine(mask, M, size, flags=cv2.INTER_NEAREST)

        final_mask = np.where((mask == 255) & template_mask, 255, 0).astype(np.uint8)
        overlap_ratio = np.count_nonzero(final_mask) / max(1, np.count_nonzero(mask))

        if overlap_ratio > best_overlap:
            best_mask = final_mask
            best_overlap = overlap_ratio

        if overlap_ratio >= 0.6:
            break

    if best_mask is None or np.count_nonzero(best_mask) == 0:
        best_mask = np.zeros(size, dtype=np.uint8)
        for _ in range(10):
            x = random.randint(20, 220)
            y = random.randint(20, 220)
            width = random.randint(10, 30)
            height = random.randint(10, 30)
            cv2.rectangle(best_mask, (x, y), (x + width, y + height), 255, -1)
        best_mask = np.where((best_mask == 255) & template_mask, 255, 0).astype(np.uint8)

    if random.random() > 0.5:
        kernel_size = random.randint(1, 5)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if random.random() > 0.5:
            best_mask = cv2.erode(best_mask, kernel, iterations=1)
        else:
            best_mask = cv2.dilate(best_mask, kernel, iterations=1)

    return best_mask


def process_images(data_path, output_root, json_path, dataset_type="mvtec"):

    with open(json_path, "r") as file:
        mask_info = json.load(file)

    if dataset_type == "mvtec":
        classes = [
            "bottle", "cable", "capsule", "carpet", "grid",
            "hazelnut", "leather", "metal_nut", "pill", "screw",
            "tile", "toothbrush", "transistor", "wood", "zipper"
        ]
    elif dataset_type == "visa":
        classes = [
            "candle", "capsules", "cashew", "chewinggum", "fryum",
            "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
        ]
    else:

        classes = list(mask_info.keys())
        print(f"No dataset type specified, using all categories from JSON: {len(classes)} categories")

    for c in classes:
        if c not in mask_info:
            print(f"Warning: Category '{c}' is not in the JSON configuration, skipping")
            continue

        anomaly_types = list(mask_info[c].keys())

        if dataset_type == "mvtec":
            root_path = os.path.join(data_path, c, "train/good")
        elif dataset_type == "visa":
            root_path = os.path.join(data_path, c, "Data/Images/Normal")
        else:

            possible_paths = [
                os.path.join(data_path, c, "train/good"),
                os.path.join(data_path, c, "Data/Images/Normal"),
                os.path.join(data_path, c, "normal"),
                os.path.join(data_path, c, "good")
            ]

            root_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    root_path = path
                    break

            if root_path is None:
                print(f"Normal image directory for category '{c}' not found, skipping")
                continue

        if not os.path.exists(root_path):
            print(f"Directory does not exist: {root_path}")
            continue

        image_files = [f for f in os.listdir(root_path) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]

        for img_name in image_files:
            img_path = os.path.join(root_path, img_name)
            name = os.path.splitext(img_name)[0]

            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue

            img = cv2.resize(img, (256, 256))

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if thresh[0][0] == 255:
                thresh = cv2.bitwise_not(thresh)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            for anomaly_type in anomaly_types:
                object_dir = os.path.join(output_root, c)
                defect_dir = os.path.join(object_dir, anomaly_type, name)
                os.makedirs(defect_dir, exist_ok=True)

                for i in range(5):
                    save_path = os.path.join(defect_dir, f"{i}.png")
                    mask = generate_adaptive_mask(closing, mask_info[c][anomaly_type])
                    cv2.imwrite(save_path, mask)
                    print(f"Generated mask {i} for {anomaly_type} at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate masks from existing anomaly masks")
    parser.add_argument("--json_path", type=str, required=False, default="visa.json",
                        help="Path to JSON file with mask info")
    parser.add_argument("--data_path", type=str, required=False,
                        default="",
                        help="Path to data directory")
    parser.add_argument("--output_root", type=str, required=False,
                        default="",
                        help="Output root directory")
    parser.add_argument("--dataset_type", type=str, required=False,
                        default="visa", choices=["mvtec", "visa"],
                        help="Dataset type: 'mvtec' or 'visa'")


    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    process_images(args.data_path, args.output_root, args.json_path, args.dataset_type)