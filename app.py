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
    F = f * area / 3 * np.ones(3)  
    return K, M, F


# Перевірка математичних властивостей локальних матриць.
def check_local_matrices(coords, K, M, F, d, f):
    area = triangle_properties(coords)[1]
    symmetry_K = np.allclose(K, K.T, rtol=1e-9, atol=1e-10)
    symmetry_M = np.allclose(M, M.T, rtol=1e-9, atol=1e-10)
    eigenvalues_M = np.linalg.eigvalsh(M)
    row_sums_K = K.sum(axis=1)
    rows_correct = np.allclose(row_sums_K, 0.0, rtol=0, atol=1e-9 * max(1.0, np.max(np.abs(K))))
    sum_F = float(np.sum(F))
    expected_F = float(f * area)
    load_correct = np.isclose(sum_F, expected_F, rtol=1e-9, atol=1e-10)

    if d > 0:
        mass_property = "Додатно визначена"
        mass_correct = bool(np.all(eigenvalues_M > 0))
    elif d == 0:
        mass_property = "Нульова (d = 0)"
        mass_correct = bool(np.allclose(eigenvalues_M, 0.0, atol=1e-10))
    else:
        mass_property = "Від’ємно визначена (d < 0)"
        mass_correct = bool(np.all(eigenvalues_M < 0))

    return {
        "Симетричність K": bool(symmetry_K),
        "Симетричність M": bool(symmetry_M),
        "Властивість M": mass_property,
        "Власні значення M": eigenvalues_M,
        "Суми рядків K": row_sums_K,
        "Нульові суми рядків K": bool(rows_correct),
        "Сума F": sum_F,
        "Очікувана сума F": expected_F,
        "Коректна сума F": bool(load_correct),
        "Коректна властивість M": mass_correct,
        "Усі перевірки": bool(symmetry_K and symmetry_M and mass_correct and rows_correct and load_correct),
    }


# Складання локальних внесків у глобальні A та rhs.
def assemble(vertices, triangles, a11, a22, d, f):
    n = len(vertices)
    A = np.zeros((n, n))  
    rhs = np.zeros(n)      

    # Проходимо всі трикутники сітки.
    for tri in triangles:
        K, M, F = local_matrices(vertices[tri], a11, a22, d, f)
        for i in range(3):
            # Додаємо внесок у глобальний вектор.
            rhs[tri[i]] += F[i]  
            for j in range(3):
                # Додаємо K + M за глобальними номерами вузлів.
                A[tri[i], tri[j]] += K[i, j] + M[i, j]

    return A, rhs


# Панель параметрів і панель результатів.
left, right = st.columns([1, 2])
with left:
    # Коефіцієнти варіанта 6.
    st.subheader("Параметри (варіант 6)")
    p1, p2 = st.columns(2)
    a11 = p1.number_input("a₁₁", value=4.0)
    a22 = p2.number_input("a₂₂", value=3.0)
    d = p1.number_input("d", value=1.0)
    f = p2.number_input("f", value=2.0)
    st.caption("a₁₂ = a₂₁ = 0")

    # Обмеження на якість трикутників.
    st.subheader("Триангуляція")
    min_angle = st.slider("Мінімальний кут", 0, 33, 20)
    max_area = st.number_input("Максимальна площа елемента", min_value=0.01,
                               max_value=1.0, value=0.25, step=0.01)
    show_labels = st.checkbox("Показувати номери вузлів і трикутників", True)

    st.subheader("Контур")
    # Початкова область: прямокутник 1 × 2.
    default_polygon = pd.DataFrame({"X": [0., 1., 1., 0.], "Y": [0., 0., 2., 2.]})
    polygon_table = st.data_editor(default_polygon, num_rows="dynamic", hide_index=True)
    polygon = polygon_table[["X", "Y"]].dropna().to_numpy(dtype=float)


with right:
    if len(polygon) < 3:
        st.warning("Для області потрібно щонайменше три точки.")
        st.stop()

    try:
        # Генеруємо сітку та обчислюємо глобальну систему.
        vertices, triangles = generate_mesh(polygon, min_angle, max_area)
        if len(triangles) == 0:
            raise ValueError("Не вдалося створити трикутники")
        angles = [triangle_properties(vertices[tri])[0] for tri in triangles]
        areas = [triangle_properties(vertices[tri])[1] for tri in triangles]
        A, rhs = assemble(vertices, triangles, a11, a22, d, f)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Сітка", "Вузли та топологія", "Локальні матриці", "Глобальні матриці",
            "Перевірка коректності локальних матриць"
        ])
        # Відображення трикутної сітки.
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
            xmin, xmax = polygon[:, 0].min(), polygon[:, 0].max()
            ymin, ymax = polygon[:, 1].min(), polygon[:, 1].max()
            dx, dy = max(xmax - xmin, 0.1), max(ymax - ymin, 0.1)
            ax.set_xlim(xmin - 0.12 * dx, xmax + 0.12 * dx)
            ax.set_ylim(ymin - 0.08 * dy, ymax + 0.08 * dy)
            ax.grid(alpha=0.3)
            fig.tight_layout(pad=0.5)
            st.pyplot(fig, width=330)
            plt.close(fig)

        # Таблиця вузлів і характеристик трикутників.
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

            st.subheader("Координати вузлів")
            st.dataframe(pd.DataFrame(vertices, columns=["X", "Y"]),
                         use_container_width=True)

        # Локальні матриці кожного трикутника.
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

        # Глобальна матриця A та вектор rhs.
        with tab4:
            labels = [f"u{i}" for i in range(len(vertices))]
            c1, c2 = st.columns([3, 1])
            with c1:
                st.write(f"Глобальна матриця A ({len(vertices)} × {len(vertices)})")
                st.dataframe(pd.DataFrame(np.round(A, 4), index=labels,
                                          columns=labels), use_container_width=True)
            with c2:
                st.write("Глобальний вектор b")
                st.dataframe(pd.DataFrame(np.round(rhs, 4), index=labels,
                                          columns=["b"]), use_container_width=True)
            st.caption("Система сформована з локальних K, M, F без накладання граничних умов.")

        # Автоматична перевірка коректності локальних матриць.
        with tab5:
            st.subheader("Перевірка коректності локальних матриць")
            st.write("Для кожного трикутника перевіряються симетричність K і M, "
                     "власні значення M, суми рядків K та відповідність суми F величині f·S.")
            checks = []
            for i, tri in enumerate(triangles):
                K, M, F = local_matrices(vertices[tri], a11, a22, d, f)
                result = check_local_matrices(vertices[tri], K, M, F, d, f)
                checks.append((i, tri, result))

            overview = pd.DataFrame([
                {
                    "Трикутник": i,
                    "Симетрична K": "Так" if result["Симетричність K"] else "Ні",
                    "Симетрична M": "Так" if result["Симетричність M"] else "Ні",
                    "Властивість M": "Так" if result["Коректна властивість M"] else "Ні",
                    "Суми рядків K ≈ 0": "Так" if result["Нульові суми рядків K"] else "Ні",
                    "ΣF = f·S": "Так" if result["Коректна сума F"] else "Ні",
                    "Підсумок": "Пройдено" if result["Усі перевірки"] else "Помилка",
                }
                for i, tri, result in checks
            ])
            st.dataframe(overview, use_container_width=True, hide_index=True)

            if all(result["Усі перевірки"] for _, _, result in checks):
                st.success("Усі локальні матриці пройшли перевірки.")
            else:
                st.warning("Деякі перевірки не пройдено. Перегляньте деталі нижче.")

            for i, tri, result in checks:
                with st.expander(f"Трикутник {i}, вузли {list(tri)}"):
                    st.write(f"**Симетричність K:** {result['Симетричність K']}")
                    st.write(f"**Симетричність M:** {result['Симетричність M']}")
                    st.write(f"**Властивість M:** {result['Властивість M']}")
                    st.write(f"**Власні значення M:** {np.round(result['Власні значення M'], 8)}")
                    st.write(f"**Суми рядків K:** {np.round(result['Суми рядків K'], 10)}")
                    st.write(f"**Сума компонент F:** {result['Сума F']:.10g}")
                    st.write(f"**Очікуване f·S:** {result['Очікувана сума F']:.10g}")
                    st.write("**Перевірка:** " + ("Пройдено" if result["Усі перевірки"] else "Помилка"))
            st.caption("Числові порівняння враховують похибку обчислень з рухомою комою.")
    except (ValueError, RuntimeError) as exc:
        st.error(f"Помилка під час обчислення: {exc}")
