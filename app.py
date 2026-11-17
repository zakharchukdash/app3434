import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.path import Path

st.set_page_config(page_title="МСЕ — варіант 6", layout="wide")
st.title("МСЕ: триангуляція та формування матриць (варіант 6)")


# Геометрія трикутника: мінімальний кут, площа, центр описаного кола.
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


def generate_mesh(polygon, min_angle, max_area, max_iter=500):
    # Розбиваємо кожну сторону контуру на відрізки.
    step = 1.5 * np.sqrt(max_area)
    points = []
    for i in range(len(polygon)):
        p1, p2 = polygon[i], polygon[(i + 1) % len(polygon)]
        pieces = max(1, int(np.ceil(np.linalg.norm(p2 - p1) / step)))
        points.extend(p1 + (p2 - p1) * j / pieces for j in range(pieces))
    points = np.array(points)
    polygon_path = Path(polygon)

    # Уточнюємо сітку, доки трикутники не задовольняють обмеження.
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


def point_on_side(p, a, b, tol=1e-5):
    return abs(np.linalg.norm(p-a) + np.linalg.norm(p-b) - np.linalg.norm(b-a)) < tol


def get_boundary_edges(vertices, triangles, polygon):
    # Ребро зовнішнє, якщо воно входить тільки в один трикутник.
    counts = {}
    for tri in triangles:
        for i in range(3):
            edge = tuple(sorted((tri[i], tri[(i+1) % 3])))
            counts[edge] = counts.get(edge, 0) + 1

    result = []
    for (n1, n2), count in counts.items():
        if count != 1:
            continue
        midpoint = (vertices[n1] + vertices[n2]) / 2
        for side in range(len(polygon)):
            if point_on_side(midpoint, polygon[side], polygon[(side+1) % len(polygon)]):
                length = np.linalg.norm(vertices[n1] - vertices[n2])
                result.append((n1, n2, side, length))
                break
    return result


def local_matrices(coords, a11, a22, d, f):
    # 3 вузли одного трикутника; порядок вузлів не впливає на правильність.
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    area = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)) / 2
    if area < 1e-12:
        raise ValueError("Знайдено трикутник із нульовою площею")

    b = np.array([y2-y3, y3-y1, y1-y2])
    c = np.array([x3-x2, x1-x3, x2-x1])

    K = (a11 * np.outer(b, b) + a22 * np.outer(c, c)) / (4 * area)
    template = np.array([[2., 1., 1.], [1., 2., 1.], [1., 1., 2.]])
    M = d * area / 12 * template
    F = f * area / 3 * np.ones(3)  # У варіанті 6 f — константа.
    return K, M, F


def assemble(vertices, triangles, boundary_edges, a11, a22, d, f,
             bc_types, beta, delta, u_c):
    n = len(vertices)
    A = np.zeros((n, n))
    rhs = np.zeros(n)

    # Збираємо локальні внески в глобальну матрицю і вектор.
    for tri in triangles:
        K, M, F = local_matrices(vertices[tri], a11, a22, d, f)
        for i in range(3):
            rhs[tri[i]] += F[i]
            for j in range(3):
                A[tri[i], tri[j]] += K[i, j] + M[i, j]

    # 3-й тип: beta * Nu + delta * (u-u_c) = 0.
    for n1, n2, side, length in boundary_edges:
        if bc_types[side] != "3° Робіна":
            continue
        R = delta / beta * length / 6 * np.array([[2., 1.], [1., 2.]])
        P = delta * u_c / beta * length / 2 * np.ones(2)
        nodes = [n1, n2]
        for i in range(2):
            rhs[nodes[i]] += P[i]
            for j in range(2):
                A[nodes[i], nodes[j]] += R[i, j]

    A_full, rhs_full = A.copy(), rhs.copy()

    # 1-й тип: u=0. Для нульового значення зануляємо рядок і стовпець.
    dirichlet_nodes = set()
    for n1, n2, side, _ in boundary_edges:
        if bc_types[side] == "1° Діріхле":
            dirichlet_nodes.update((n1, n2))
    for node in dirichlet_nodes:
        A[node, :] = 0
        A[:, node] = 0
        A[node, node] = 1
        rhs[node] = 0

    return A_full, rhs_full, A, rhs


left, right = st.columns([1, 2])
with left:
    st.subheader("Параметри (варіант 6)")
    p1, p2 = st.columns(2)
    a11 = p1.number_input("a₁₁", value=4.0)
    a22 = p2.number_input("a₂₂", value=3.0)
    d = p1.number_input("d", value=1.0)
    f = p2.number_input("f", value=2.0)
    st.caption("a₁₂ = a₂₁ = 0")

    st.subheader("Триангуляція")
    min_angle = st.slider("Мінімальний кут", 0, 33, 20)
    max_area = st.number_input("Максимальна площа елемента", min_value=0.01,
                               max_value=1.0, value=0.25, step=0.01)
    show_labels = st.checkbox("Показувати номери вузлів і трикутників", True)

    st.subheader("Контур")
    default_polygon = pd.DataFrame({"X": [0., 1., 1., 0.], "Y": [0., 0., 2., 2.]})
    polygon_table = st.data_editor(default_polygon, num_rows="dynamic", hide_index=True)
    polygon = polygon_table[["X", "Y"]].dropna().to_numpy(dtype=float)

    st.subheader("Граничні умови")
    choices = ["1° Діріхле", "2° Неймана", "3° Робіна"]
    presets = ["3° Робіна", "1° Діріхле", "3° Робіна", "2° Неймана"]
    bc_types = {}
    for i in range(len(polygon)):
        j = (i + 1) % len(polygon)
        label = (f"Сторона {i+1}: "
                 f"({polygon[i, 0]:g}, {polygon[i, 1]:g}) → "
                 f"({polygon[j, 0]:g}, {polygon[j, 1]:g})")
        default = presets[i] if i < 4 else choices[0]
        bc_types[i] = st.selectbox(label, choices, choices.index(default), key=f"bc_{i}")

    st.caption("Для Робіна: βNu + δ(u−u_c) = 0")
    c1, c2, c3 = st.columns(3)
    beta = c1.number_input("β", min_value=0.01, value=1.0)
    delta = c2.number_input("δ", value=1.0)
    u_c = c3.number_input("u_c", value=1.0)

with right:
    if len(polygon) < 3:
        st.warning("Для області потрібно щонайменше три точки.")
        st.stop()

    try:
        vertices, triangles = generate_mesh(polygon, min_angle, max_area)
        if len(triangles) == 0:
            raise ValueError("Не вдалося створити трикутники")
        boundary_edges = get_boundary_edges(vertices, triangles, polygon)
        angles = [triangle_properties(vertices[tri])[0] for tri in triangles]
        areas = [triangle_properties(vertices[tri])[1] for tri in triangles]
        A_full, b_full, A, rhs = assemble(
            vertices, triangles, boundary_edges, a11, a22, d, f,
            bc_types, beta, delta, u_c
        )

        tab1, tab2, tab3, tab4 = st.tabs([
            "Сітка", "Вузли та топологія", "Локальні матриці", "Глобальні матриці"
        ])
        with tab1:
            fig, ax = plt.subplots(figsize=(3.8, 5.2), dpi=100)
            ax.triplot(vertices[:, 0], vertices[:, 1], triangles, color="black",
                       linewidth=0.8, marker="o", markersize=3)
            if show_labels:
                for i, p in enumerate(vertices):
                    ax.text(p[0], p[1]+0.03, str(i), color="darkred", fontsize=8,
                            ha="center")
                for i, tri in enumerate(triangles):
                    center = vertices[tri].mean(axis=0)
                    ax.text(center[0], center[1], str(i), color="blue", fontsize=8,
                            ha="center")
            contour = np.vstack((polygon, polygon[0]))
            ax.plot(contour[:, 0], contour[:, 1], "g--", linewidth=1.5)
            ax.set_aspect("equal")
            # Додаємо невеликі поля навколо контуру, щоб номери вузлів не обрізались.
            xmin, xmax = polygon[:, 0].min(), polygon[:, 0].max()
            ymin, ymax = polygon[:, 1].min(), polygon[:, 1].max()
            dx, dy = max(xmax - xmin, 0.1), max(ymax - ymin, 0.1)
            ax.set_xlim(xmin - 0.12 * dx, xmax + 0.12 * dx)
            ax.set_ylim(ymin - 0.08 * dy, ymax + 0.08 * dy)
            ax.grid(alpha=0.3)
            fig.tight_layout(pad=0.5)
            # Явно обмежуємо ширину в пікселях. Streamlit сам збереже пропорції,
            # тому вся висота прямокутної області поміститься на екрані.
            st.pyplot(fig, width=330)
            plt.close(fig)

        with tab2:
            st.write(f"Вузлів: **{len(vertices)}**, трикутників: **{len(triangles)}**")
            st.write(f"Мінімальний кут: **{min(angles):.2f}°**; "
                     f"найбільша площа: **{max(areas):.4f}**")
            if min(angles) + 1e-6 < min_angle or max(areas) > max_area + 1e-6:
                st.warning("Генератор сітки не досяг заданих обмежень. "
                           "Спробуйте інші параметри.")
            table = pd.DataFrame(triangles, columns=["Вузол 1", "Вузол 2", "Вузол 3"])
            table["Мін. кут"] = np.round(angles, 2)
            table["Площа"] = np.round(areas, 4)
            st.dataframe(table, use_container_width=True)

            node_rows = []
            for p in vertices:
                sides = [i for i in range(len(polygon))
                         if point_on_side(p, polygon[i], polygon[(i+1) % len(polygon)])]
                labels = ", ".join(bc_types[i] for i in sides) or "Внутрішній"
                node_rows.append((p[0], p[1], labels))
            st.dataframe(pd.DataFrame(node_rows, columns=["X", "Y", "Граничні умови"]),
                         use_container_width=True)

        with tab3:
            for i, tri in enumerate(triangles):
                with st.expander(f"Трикутник {i}, вузли {list(tri)}", expanded=(i == 0)):
                    K, M, F = local_matrices(vertices[tri], a11, a22, d, f)
                    c1, c2, c3 = st.columns([2, 2, 1])
                    c1.write("Матриця K")
                    c1.dataframe(np.round(K, 4))
                    c2.write("Матриця M")
                    c2.dataframe(np.round(M, 4))
                    c3.write("Вектор F")
                    c3.dataframe(np.round(F, 4))

            st.subheader("Внески ребер Робіна")
            for n1, n2, side, length in boundary_edges:
                if bc_types[side] == "3° Робіна":
                    R = delta / beta * length / 6 * np.array([[2., 1.], [1., 2.]])
                    P = delta * u_c / beta * length / 2 * np.ones(2)
                    with st.expander(f"Ребро {n1}—{n2}, сторона {side+1}"):
                        c1, c2 = st.columns(2)
                        c1.write("Матриця R")
                        c1.dataframe(np.round(R, 4))
                        c2.write("Вектор P")
                        c2.dataframe(np.round(P, 4))

        with tab4:
            matrix_kind = st.radio("Показати матрицю:", [
                "До накладання умови Діріхле", "Після накладання умови Діріхле"
            ])
            shown_A, shown_b = (A_full, b_full) if matrix_kind.startswith("До") else (A, rhs)
            labels = [f"u{i}" for i in range(len(vertices))]
            c1, c2 = st.columns([3, 1])
            with c1:
                st.write(f"Матриця A ({len(vertices)} × {len(vertices)})")
                st.dataframe(pd.DataFrame(np.round(shown_A, 4), index=labels,
                                          columns=labels), use_container_width=True)
            with c2:
                st.write("Вектор b")
                st.dataframe(pd.DataFrame(np.round(shown_b, 4), index=labels,
                                          columns=["b"]), use_container_width=True)
    except (ValueError, RuntimeError) as exc:
        st.error(f"Помилка під час обчислення: {exc}")
