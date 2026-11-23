import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.path import Path


st.set_page_config(page_title="МСЕ — варіант 6", layout="wide")
st.title("МСЕ: триангуляція та формування матриць (варіант 6)")


# Площа, кути та центр описаного кола трикутника.
def triangle_properties(p):
    A, B, C = p
    a = np.linalg.norm(B - C)
    b = np.linalg.norm(A - C)
    c = np.linalg.norm(A - B)
    sides = (a, b, c)
    cross = (B[0]-A[0])*(C[1]-A[1]) - (B[1]-A[1])*(C[0]-A[0])
    area = abs(cross) / 2

    if min(sides) < 1e-12 or area < 1e-12:
        return 0.0, area, None

    angles = [
        np.degrees(np.arccos(np.clip((y*y + z*z - x*x) / (2*y*z), -1, 1)))
        for x, y, z in ((a, b, c), (b, a, c), (c, a, b))
    ]
    delta = 2 * cross
    ux = ((A @ A)*(B[1]-C[1]) + (B @ B)*(C[1]-A[1]) + (C @ C)*(A[1]-B[1])) / delta
    uy = ((A @ A)*(C[0]-B[0]) + (B @ B)*(A[0]-C[0]) + (C @ C)*(B[0]-A[0])) / delta
    return min(angles), area, np.array([ux, uy])


# Побудова та уточнення трикутної сітки методом Делоне.
def generate_mesh(polygon, min_angle, max_area, max_iter=500):
    step = 1.5 * np.sqrt(max_area)
    points = []
    for i in range(len(polygon)):
        p1, p2 = polygon[i], polygon[(i + 1) % len(polygon)]
        pieces = max(1, int(np.ceil(np.linalg.norm(p2 - p1) / step)))
        points.extend(p1 + (p2 - p1) * j / pieces for j in range(pieces))
    points = np.array(points)
    polygon_path = Path(polygon)

    for _ in range(max_iter):
        triangles = Delaunay(points).simplices
        inside = polygon_path.contains_points(points[triangles].mean(axis=1), radius=-1e-6)
        triangles = triangles[inside]

        bad = []
        for tri in triangles:
            angle, area, center = triangle_properties(points[tri])
            if center is None or (area <= max_area and angle >= min_angle):
                continue
            score = max(0.0, area - max_area) + max(0.0, min_angle - angle)
            bad.append((score, center, points[tri].mean(axis=0)))

        if not bad:
            break
        added = False
        for _, center, centroid in sorted(bad, reverse=True, key=lambda item: item[0]):
            candidate = center if polygon_path.contains_point(center) else centroid
            if min(np.linalg.norm(points - candidate, axis=1)) > 1e-3:
                points = np.vstack((points, candidate))
                added = True
                break
        if not added:
            break

    triangles = Delaunay(points).simplices
    inside = polygon_path.contains_points(points[triangles].mean(axis=1), radius=-1e-6)
    return points, triangles[inside]


# Локальні K, M, F для одного трикутника.
def local_matrices(coords, a11, a22, d, f):
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    # Площа за координатами трьох вершин.
    area = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)) / 2
    if area < 1e-12:
        raise ValueError("Знайдено трикутник із нульовою площею")

    # Коефіцієнти для похідних функцій форми.
    b = np.array([y2-y3, y3-y1, y1-y2])
    c = np.array([x3-x2, x1-x3, x2-x1])

    # Матриця жорсткості K.
    K = (a11 * np.outer(b, b) + a22 * np.outer(c, c)) / (4 * area)
    template = np.array([[2., 1., 1.], [1., 2., 1.], [1., 1., 2.]])
    # Матриця мас M.
    M = d * area / 12 * template
    # Вектор правої частини F.
