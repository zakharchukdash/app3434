import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.path import Path

st.set_page_config(page_title="Генератор сітки", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    </style>
    <h3 style='text-align: center; color: #2C3E50; margin-bottom: 0;'>
    Система розбиття області на скінченні елементи</h3>
""", unsafe_allow_html=True)


def calculate_triangle_properties(pts):
    """Площа, мінімальний кут та центр описаного кола."""
    A, B, C = pts
    a = np.linalg.norm(B - C)
    b = np.linalg.norm(A - C)
    c = np.linalg.norm(A - B)

    s = (a + b + c) / 2
    area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))

    angles = []
    for x, y, z in [(a, b, c), (b, a, c), (c, a, b)]:
        val = np.clip((y**2 + z**2 - x**2) / (2 * y * z), -1, 1)
        angles.append(np.degrees(np.arccos(val)))

    D = 2 * (A[0] * (B[1] - C[1]) +
             B[0] * (C[1] - A[1]) +
             C[0] * (A[1] - B[1]))

    if abs(D) < 1e-10:
        return min(angles), area, None

    Ux = ((A[0]**2 + A[1]**2) * (B[1] - C[1]) +
          (B[0]**2 + B[1]**2) * (C[1] - A[1]) +
          (C[0]**2 + C[1]**2) * (A[1] - B[1])) / D

    Uy = ((A[0]**2 + A[1]**2) * (C[0] - B[0]) +
          (B[0]**2 + B[1]**2) * (A[0] - C[0]) +
          (C[0]**2 + C[1]**2) * (B[0] - A[0])) / D

    return min(angles), area, [Ux, Uy]


def generate_custom_mesh(polygon, min_angle, max_area, max_iter=500):
    """Генерація сітки з контролем якості."""
    target_len = np.sqrt(max_area) * 1.5
    points = []

    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        L = np.linalg.norm(p2 - p1)
        n_segs = max(1, int(np.ceil(L / target_len)))

        for j in range(n_segs):
            points.append(p1 + (p2 - p1) * (j / n_segs))

    points = np.array(points)
    path = Path(polygon)

    for _ in range(max_iter):
        tri = Delaunay(points)
        simplices = tri.simplices

        centroids = np.mean(points[simplices], axis=1)
        simplices = simplices[
            path.contains_points(centroids, radius=-1e-5)
        ]

        bad_triangles = []

        for s in simplices:
            angle, area, cc = calculate_triangle_properties(points[s])

            if cc is not None and (area > max_area or angle < min_angle):
                score = max(0, area - max_area) + max(0, min_angle - angle)
                bad_triangles.append((score, cc, np.mean(points[s], axis=0)))

        if not bad_triangles:
            break

        bad_triangles.sort(key=lambda x: x[0], reverse=True)

        for _, cc, centroid in bad_triangles:
            candidate = cc if path.contains_point(cc) else centroid

            if np.min(np.linalg.norm(points - candidate, axis=1)) > 1e-3:
                points = np.vstack([points, candidate])
                break
        else:
            break

    tri = Delaunay(points)
    simplices = tri.simplices
    centroids = np.mean(points[simplices], axis=1)

    return points, simplices[
        path.contains_points(centroids, radius=-1e-6)
    ]


def point_on_segment(p, a, b, tol=1e-5):
    """Перевіряє, чи лежить точка на відрізку."""
    ab = b - a
    ap = p - a
    cross = abs(ab[0] * ap[1] - ab[1] * ap[0])
    dot = np.dot(ap, ab)

    return (cross < tol and
            -tol <= dot <= np.dot(ab, ab) + tol)


def get_edge_markers(p, polygon_pts):
    """Визначає грань, на якій знаходиться вузол."""
    markers = []

    for i in range(len(polygon_pts)):
        p1 = polygon_pts[i]
        p2 = polygon_pts[(i + 1) % len(polygon_pts)]

        if point_on_segment(p, p1, p2):
            markers.append(f"Грань {i + 1}")

    return " + ".join(markers) if markers else "Внутрішній (0)"


def mark_boundary_part(vertices, start, end):
    """Маркує вузли на заданій частині границі."""
    return [int(point_on_segment(p, start, end)) for p in vertices]


# ---------- Інтерфейс ----------

col_left, col_right = st.columns([1, 2.2])

with col_left:
    st.subheader("⚙️ Налаштування")

    min_angle = st.slider(
        "Мінімальний кут (градуси)", 0.0, 33.0, 20.0, step=1.0
    )

    max_area = st.number_input(
        "Обмеження площі (згущення)", 0.01, 1.0, 0.05, step=0.01
    )

    show_labels = st.checkbox("Показувати підписи", value=True)

    st.subheader("📍 Координати контуру")

    default_df = pd.DataFrame({
        'X': [0.0, 0.0, 2.0, 1.0],
        'Y': [1.0, 2.0, 0.0, 0.0]
    })

    edited_df = st.data_editor(
        default_df, num_rows="dynamic",
        use_container_width=True, height=200
    )

    st.subheader("📌 Частина границі")

    start_x = st.number_input("X початкової точки", value=1.0)
    start_y = st.number_input("Y початкової точки", value=0.0)
    end_x = st.number_input("X кінцевої точки", value=2.0)
    end_y = st.number_input("Y кінцевої точки", value=0.0)


# ---------- Розрахунок та вивід ----------

with col_right:
    try:
        pts = edited_df[['X', 'Y']].dropna().to_numpy()

        if len(pts) < 3:
            st.warning("Додайте мінімум 3 точки для утворення контуру.")
            st.stop()

        vertices, triangles = generate_custom_mesh(
            pts, min_angle, max_area
        )

        edge_markers = [
            get_edge_markers(v, pts) for v in vertices
        ]

        start = np.array([start_x, start_y])
        end = np.array([end_x, end_y])
        boundary_markers = mark_boundary_part(vertices, start, end)

        m1, m2, m3 = st.columns(3)
        m1.metric("🔴 Вузлів", len(vertices))
        m2.metric("🔺 Трикутників", len(triangles))
        m3.metric("✅ Делоне", "Виконано")

        tab1, tab2 = st.tabs(["📊 Візуалізація", "🗄 Дані (Матриці)"])

        # ---------- Графік ----------

        with tab1:
            fig, ax = plt.subplots(figsize=(8, 4))

            if len(triangles) > 0:
                ax.triplot(
                    vertices[:, 0], vertices[:, 1], triangles,
                    linewidth=1, marker='o', markersize=4
                )

                if show_labels:
                    for i, p_val in enumerate(vertices):
                        ax.text(
                            p_val[0], p_val[1] + 0.03,
                            f"{i}", fontsize=8,
                            ha='center'
                        )

                    for i, t in enumerate(triangles):
                        pt = np.mean(vertices[t], axis=0)
                        ax.text(
                            pt[0], pt[1], f"{i}",
                            fontsize=8, ha='center', va='center'
                        )

            polygon_pts = np.vstack((pts, pts[0]))
            ax.plot(
                polygon_pts[:, 0], polygon_pts[:, 1],
                'g--', linewidth=1.5, alpha=0.5,
                label="Контур"
            )

            selected = np.array(boundary_markers, dtype=bool)

            if np.any(selected):
                ax.scatter(
                    vertices[selected, 0],
                    vertices[selected, 1],
                    s=50, label="Вибрана границя"
                )

            ax.scatter(
                [start_x, end_x], [start_y, end_y],
                marker='x', s=60, label="Початок / кінець"
            )

            ax.set_aspect('equal')
            ax.margins(0.1)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.legend()

            st.pyplot(fig, use_container_width=False)

        # ---------- Таблиці ----------

        with tab2:
            col_t1, col_t2 = st.columns(2)

            with col_t1:
                st.markdown("#### Координати та маркування границь")

                df_nodes = pd.DataFrame(vertices, columns=['X', 'Y'])
                df_nodes['Розташування'] = edge_markers
                df_nodes['Вибрана границя'] = boundary_markers

                st.dataframe(
                    df_nodes,
                    use_container_width=True
                )

            with col_t2:
                st.markdown("#### Масив зв'язності")

                if len(triangles) > 0:
                    df_tri = pd.DataFrame(
                        triangles,
                        columns=['Вуз A', 'Вуз B', 'Вуз C']
                    )

                    st.dataframe(
                        df_tri,
                        use_container_width=True
                    )

    except Exception as e:
        st.error(
            f"Помилка генерації: {e}. "
            "Спробуйте змінити параметри."
        )
