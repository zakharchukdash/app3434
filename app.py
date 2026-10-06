import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.path import Path

st.set_page_config(page_title="Генератор сітки", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    </style>
    <h3 style='text-align: center; color: #2C3E50; margin-bottom: 0;'>Система розбиття області на скінченні елементи</h3>
""", unsafe_allow_html=True)


# Ця функція бере один трикутник і перевіряє чи він якісний
def calculate_triangle_properties(pts):
    A, B, C = pts[0], pts[1], pts[2]
    
    # Рахуємо довжини сторін(евклідова норма вектора).
    a = np.linalg.norm(B - C)
    b = np.linalg.norm(A - C)
    c = np.linalg.norm(A - B)
    
    s = (a + b + c) / 2.0
    area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))
    
    angles = []
    for x, y, z in [(a,b,c), (b,a,c), (c,a,b)]:
        val = np.clip((y**2 + z**2 - x**2) / (2 * y * z), -1.0, 1.0)
        angles.append(np.degrees(np.arccos(val)))
        
    # Рахуємо координати центру описаного кола 
    D = 2 * (A[0]*(B[1]-C[1]) + B[0]*(C[1]-A[1]) + C[0]*(A[1]-B[1]))
    
    if abs(D) < 1e-10:
        return min(angles), area, None
        
    Ux = ((A[0]**2 + A[1]**2)*(B[1]-C[1]) + (B[0]**2 + B[1]**2)*(C[1]-A[1]) + (C[0]**2 + C[1]**2)*(A[1]-B[1])) / D
    Uy = ((A[0]**2 + A[1]**2)*(C[0]-B[0]) + (B[0]**2 + B[1]**2)*(A[0]-C[0]) + (C[0]**2 + C[1]**2)*(B[0]-A[0])) / D
    
    return min(angles), area, [Ux, Uy]


def generate_custom_mesh(polygon, min_angle, max_area, max_iter=500):
    # нарізаємо контур фігури на початкові точки
    target_len = np.sqrt(max_area) * 1.5     # бажана відстань між точками (залежить від обмеження площі).
    points = []
    
    # Йдемо по периметру багатокутника і рівномірно розставляємо точки вздовж кожного ребра
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i+1) % len(polygon)]
        L = np.linalg.norm(p2 - p1)
        n_segs = max(1, int(np.ceil(L / target_len)))
        for j in range(n_segs):
            points.append(p1 + (p2 - p1) * (j / n_segs))
            
    points = np.array(points)
    path = Path(polygon)  # Об'єкт контуру
    
    for _ in range(max_iter):
        # Будуємо сітку Делоне на наявних точках
        tri = Delaunay(points)
        simplices = tri.simplices  # списки трійок вузлів для кожного трикутника
        
        # Відкидаємо трикутники, які вилізли за межі фігури
        centroids = np.mean(points[simplices], axis=1)
        mask = path.contains_points(centroids, radius=-1e-5)
        simplices = simplices[mask]
        
        # Шукаємо трикутники які порушують вимоги
        bad_triangles = []
        for s in simplices:
            angle, area, cc = calculate_triangle_properties(points[s])
            if cc is not None:
                if area > max_area or angle < min_angle:
                    # наскільки сильно трикутник порушує норму.
                    score = max(0, area - max_area) + max(0, min_angle - angle)
                    centroid = np.mean(points[s], axis=0)
                    bad_triangles.append((score, cc, centroid))
                    
        if not bad_triangles:
            break
            
        # Сортуємо дефектні трикутники, спочатку виправляємо найгірші
        bad_triangles.sort(key=lambda x: x[0], reverse=True)
        added = False
        
        for _, cc, centroid in bad_triangles:
            # ставимо нову точку в центр описаного кола.
            candidate = cc if path.contains_point(cc) else centroid
            
            # не додаємо точку якщо вона ближче ніж 0.001 до вже наявних
            if np.min(np.linalg.norm(points - candidate, axis=1)) > 1e-3:
                points = np.vstack([points, candidate])  # Додаємо нову точку в масив
                added = True
                break
                
        if not added:
            break
            
    # Фінальна побудова сітки з усіма новими точками.
    tri = Delaunay(points)
    simplices = tri.simplices
    centroids = np.mean(points[simplices], axis=1)
    mask = path.contains_points(centroids, radius=-1e-6)
    
    # Повертаємо координати всіх точок і масив зв'язності
    return points, simplices[mask]


# функція визначає на якій грані лежить кожна точка
def get_edge_indices(p, polygon_pts):
    tol = 1e-5  
    edge_ids = []
    n = len(polygon_pts)
    
    # Перевіряємо кожне ребро багатокутника.
    for i in range(n):
        p1 = polygon_pts[i]
        p2 = polygon_pts[(i+1) % n]
        
        # Рахуємо відстані: від початку ребра до точки, від точки до кінця і повну довжину ребра
        d_p1_p = np.linalg.norm(p - p1)
        d_p_p2 = np.linalg.norm(p2 - p)
        d_p1_p2 = np.linalg.norm(p2 - p1)
        
        # якщо d(P1, P) + d(P, P2) == d(P1, P2) точка точно лежить на цьому відрізку
        if abs(d_p1_p + d_p_p2 - d_p1_p2) < tol:
            edge_ids.append(i)
            
    # Повертає список номерів граней
    return edge_ids


col_left, col_right = st.columns([1.1, 2.1])

with col_left:
    st.subheader("⚙️ Налаштування")
    # Повзунки для зміни порогів якості сітки.
    min_angle = st.slider("Мінімальний кут (градуси)", 0.0, 33.0, 20.0, step=1.0)
    max_area = st.number_input("Обмеження площі (згущення)", 0.01, 1.0, 0.05, step=0.01)
    show_labels = st.checkbox("Показувати підписи", value=True)
    
    st.subheader("📍 Контур та граничні умови")
    # Задаємо початкові координати вершин контуру Варіанта 6: прямокутник 1 на 2.
    default_df = pd.DataFrame({
        'X': [0.0, 1.0, 1.0, 0.0],
        'Y': [0.0, 0.0, 2.0, 2.0]
    })
    # Таблиця, де користувач може вручну змінити координати точок прямо в браузері.
    edited_df = st.data_editor(default_df, num_rows="dynamic", use_container_width=True, height=160)
    
    # Випадаючі списки для вибору фізичних умов на кожній грані (за Варіантом 6).
    bc_options = ["1° род (u=0)", "2° род (Nu=0)", "3° род (βNu+δ(u-uc)=0)"]
    v6_presets = [bc_options[2], bc_options[0], bc_options[2], bc_options[1]]
    
    pts_current = edited_df[['X', 'Y']].dropna().to_numpy()
    boundary_types = {}

    if len(pts_current) >= 3:
        st.markdown("**Типи умов на гранях:**")
        n_edges = len(pts_current)
        for i in range(n_edges):
            p_s = f"({pts_current[i][0]:.1f}, {pts_current[i][1]:.1f})"
            p_e = f"({pts_current[(i+1) % n_edges][0]:.1f}, {pts_current[(i+1) % n_edges][1]:.1f})"
            default_val = v6_presets[i] if i < len(v6_presets) else bc_options[0]
            boundary_types[i] = st.selectbox(
                f"Грань {i+1}: {p_s} → {p_e}",
                bc_options,
                index=bc_options.index(default_val),
                key=f"edge_bc_{i}"
            )

with col_right:
    try:
        pts = edited_df[['X', 'Y']].dropna().to_numpy()
        if len(pts) < 3:
            st.warning("Додайте мінімум 3 точки для утворення контуру.")
            st.stop()

        vertices, triangles = generate_custom_mesh(pts, min_angle, max_area)

        custom_markers = []
        for v in vertices:
            edge_ids = get_edge_indices(v, pts)
            if not edge_ids:
                custom_markers.append("Внутрішній (0)")
            else:
                labels = [f"Грань {e+1} [{boundary_types.get(e, '1° род').split()[0]}]" for e in edge_ids]
                custom_markers.append(" + ".join(labels))

        m1, m2, m3 = st.columns(3)
        m1.metric("🔴 Вузлів", len(vertices))
        m2.metric("🔺 Трикутників", len(triangles))

        tab1, tab2 = st.tabs(["📊 Візуалізація", "🗄 Дані (Матриці)"])

        with tab1:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.set_facecolor('#ffffff')
            
            if len(triangles) > 0:
                ax.triplot(vertices[:,0], vertices[:,1], triangles, color='#2C3E50', linewidth=1, 
                           marker='o', markersize=4, markerfacecolor='#E74C3C', markeredgecolor='#E74C3C')
                
                if show_labels:
                    for i, p_val in enumerate(vertices):
                        ax.text(p_val[0], p_val[1]+0.03, f"{i}", color='#C0392B', fontsize=8, fontweight='bold', ha='center')
                        
                    for i, t in enumerate(triangles):
                        pt = np.mean(vertices[t], axis=0)
                        ax.text(pt[0], pt[1], f"{i}", color='#2980B9', fontsize=8, ha='center', va='center')
            
            polygon_pts = np.vstack((pts, pts[0]))
            ax.plot(polygon_pts[:,0], polygon_pts[:,1], 'g--', linewidth=1.5, alpha=0.5, label="Контур")

            ax.set_aspect('equal') 
            ax.margins(0.1)
            ax.grid(True, linestyle='--', alpha=0.5, color='#BDC3C7')
            ax.set_xlabel('X', fontweight='bold')
            ax.set_ylabel('Y', fontweight='bold')
            
            st.pyplot(fig, use_container_width=False)

        with tab2:
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("#### Координати та маркування границь")
                df_nodes = pd.DataFrame(vertices, columns=['X', 'Y'])
                df_nodes['Розташування'] = custom_markers
                
                def highlight_edges(val):
                    return 'background-color: #d1ecf1' if 'Грань' in str(val) else ''
                
                st.dataframe(df_nodes.style.map(highlight_edges, subset=['Розташування']), use_container_width=True)
                
            with col_t2:
                st.markdown("#### Масив зв'язності")
                if len(triangles) > 0:
                    df_tri = pd.DataFrame(triangles, columns=['Вуз A', 'Вуз B', 'Вуз C'])
                    st.dataframe(df_tri.style.set_properties(**{'background-color': '#f8f9fa'}), use_container_width=True)

    except Exception as e:
        st.error(f"Помилка генерації: {e}. Спробуйте змінити параметри.")
