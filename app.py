import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.path import Path

# Налаштування базового вигляду сторінки у браузері
st.set_page_config(page_title="Генератор сітки", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    </style>
    <h3 style='text-align: center; color: #2C3E50; margin-bottom: 0;'>Система розбиття області на скінченні елементи</h3>
""", unsafe_allow_html=True)

def calculate_triangle_properties(pts):
    """Обчислює геометричні параметри трикутника: площу, мінімальний кут та центр описаного кола."""
    A, B, C = pts[0], pts[1], pts[2]
    # Довжини сторін
    a = np.linalg.norm(B - C)
    b = np.linalg.norm(A - C)
    c = np.linalg.norm(A - B)
    
    # Обчислення площі за формулою Герона
    s = (a + b + c) / 2.0
    area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))
    
    # Знаходження мінімального кута через теорему косинусів
    angles = []
    for x, y, z in [(a,b,c), (b,a,c), (c,a,b)]:
        val = np.clip((y**2 + z**2 - x**2) / (2 * y * z), -1.0, 1.0)
        angles.append(np.degrees(np.arccos(val)))
        
    # Координати центру описаного кола (Ux, Uy)
    D = 2 * (A[0]*(B[1]-C[1]) + B[0]*(C[1]-A[1]) + C[0]*(A[1]-B[1]))
    if abs(D) < 1e-10:
        return min(angles), area, None
        
    Ux = ((A[0]**2 + A[1]**2)*(B[1]-C[1]) + (B[0]**2 + B[1]**2)*(C[1]-A[1]) + (C[0]**2 + C[1]**2)*(A[1]-B[1])) / D
    Uy = ((A[0]**2 + A[1]**2)*(C[0]-B[0]) + (B[0]**2 + B[1]**2)*(A[0]-C[0]) + (C[0]**2 + C[1]**2)*(B[0]-A[0])) / D
    
    return min(angles), area, [Ux, Uy]

def generate_custom_mesh(polygon, min_angle, max_area, max_iter=500):
    """Основний алгоритм генерації сітки з контролем якості."""
    # 1. Розбиття зовнішнього контуру на дрібні відрізки для рівномірної сітки
    target_len = np.sqrt(max_area) * 1.5
    points = []
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i+1) % len(polygon)]
        L = np.linalg.norm(p2 - p1)
        n_segs = max(1, int(np.ceil(L / target_len)))
        for j in range(n_segs):
            points.append(p1 + (p2 - p1) * (j / n_segs))
            
    points = np.array(points)
    path = Path(polygon)
    
    # 2. Ітеративне покращення сітки (додавання нових точок)
    for _ in range(max_iter):
        tri = Delaunay(points)
        simplices = tri.simplices
        
        # Відсікаємо трикутники, які згенерувалися поза межами фігури (наприклад, у вирізах)
        centroids = np.mean(points[simplices], axis=1)
        mask = path.contains_points(centroids, radius=-1e-5)
        simplices = simplices[mask]
        
        # Шукаємо "погані" трикутники, що занадто великі або занадто гострі
        bad_triangles = []
        for s in simplices:
            angle, area, cc = calculate_triangle_properties(points[s])
            if cc is not None:
                if area > max_area or angle < min_angle:
                    score = max(0, area - max_area) + max(0, min_angle - angle)
                    centroid = np.mean(points[s], axis=0)
                    bad_triangles.append((score, cc, centroid))
                    
        # Якщо всі трикутники якісні – зупиняємо цикл
        if not bad_triangles:
            break
            
        # Сортуємо від найгіршого трикутника і пробуємо його розбити
        bad_triangles.sort(key=lambda x: x[0], reverse=True)
        added = False
        for _, cc, centroid in bad_triangles:
            # Якщо центр кола всередині, беремо його, інакше беремо звичайний центр мас
            candidate = cc if path.contains_point(cc) else centroid
            
            # Перевіряємо, щоб нова точка не зливалася з уже існуючими
            if np.min(np.linalg.norm(points - candidate, axis=1)) > 1e-3:
                points = np.vstack([points, candidate])
                added = True
                break
                
        if not added:
            break
            
    # Фінальна підготовка масивів вузлів та елементів
    tri = Delaunay(points)
    simplices = tri.simplices
    centroids = np.mean(points[simplices], axis=1)
    mask = path.contains_points(centroids, radius=-1e-6)
    return points, simplices[mask]

def get_edge_markers(p, polygon_pts):
    """Перевіряє, на якій грані лежить вузол (для накладання граничних умов 1, 2 або 3 роду)."""
    tol = 1e-5
    markers = []
    for i in range(len(polygon_pts)):
        p1 = polygon_pts[i]
        p2 = polygon_pts[(i+1) % len(polygon_pts)]
        
        # Якщо сума відстаней від точки до кінців відрізка дорівнює його довжині, точка лежить на ньому
        d_p1_p = np.linalg.norm(p - p1)
        d_p_p2 = np.linalg.norm(p2 - p)
        d_p1_p2 = np.linalg.norm(p2 - p1)
        
        if abs(d_p1_p + d_p_p2 - d_p1_p2) < tol:
            markers.append(f"Грань {i+1}")
            
    if not markers:
        return "Внутрішній (0)"
    # Якщо точка в кутку, вона отримає маркування обох граней (напр. "Грань 1 + Грань 2")
    return " + ".join(markers)

# --- Інтерфейс програми ---
col_left, col_right = st.columns([1, 2.2])

# Ліва панель: Введення параметрів користувачем
with col_left:
    st.subheader("⚙️ Налаштування")
    min_angle = st.slider("Мінімальний кут (градуси)", 0.0, 33.0, 20.0, step=1.0)
    max_area = st.number_input("Обмеження площі (згущення)", 0.01, 1.0, 0.05, step=0.01)
    show_labels = st.checkbox("Показувати підписи", value=True)
    
    st.subheader("📍 Координати контуру")
    default_df = pd.DataFrame({
    'X': [0.0, 1.0, 1.0, 0.0],
    'Y': [0.0, 0.0, 2.0, 2.0]
})
    edited_df = st.data_editor(default_df, num_rows="dynamic", use_container_width=True, height=200)

# Права панель: Візуалізація та вивід таблиць
with col_right:
    try:
        # Зчитуємо координати з таблиці
        pts = edited_df[['X', 'Y']].dropna().to_numpy()
        if len(pts) < 3:
            st.warning("Додайте мінімум 3 точки для утворення контуру.")
            st.stop()

        # Запускаємо генератор сітки
        vertices, triangles = generate_custom_mesh(pts, min_angle, max_area)

        # Визначаємо типи границь для кожного згенерованого вузла
        custom_markers = [get_edge_markers(v, pts) for v in vertices]

        # Статистика зверху
        m1, m2, m3 = st.columns(3)
        m1.metric("🔴 Вузлів", len(vertices))
        m2.metric("🔺 Трикутників", len(triangles))
        m3.metric("✅ Делоне", "Виконано")

        tab1, tab2 = st.tabs(["📊 Візуалізація", "🗄 Дані (Матриці)"])

        # Вкладка 1: Малювання графіка
        with tab1:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.set_facecolor('#ffffff')
            
            if len(triangles) > 0:
                # Малюємо сітку трикутників
                ax.triplot(vertices[:,0], vertices[:,1], triangles, color='#2C3E50', linewidth=1, 
                           marker='o', markersize=4, markerfacecolor='#E74C3C', markeredgecolor='#E74C3C')
                
                # Підписуємо номери вузлів та елементів, якщо увімкнено
                if show_labels:
                    for i, p_val in enumerate(vertices):
                        ax.text(p_val[0], p_val[1]+0.03, f"{i}", color='#C0392B', fontsize=8, fontweight='bold', ha='center')
                        
                    for i, t in enumerate(triangles):
                        pt = np.mean(vertices[t], axis=0)
                        ax.text(pt[0], pt[1], f"{i}", color='#2980B9', fontsize=8, ha='center', va='center')
            
            # Обводимо початковий контур пунктиром
            polygon_pts = np.vstack((pts, pts[0]))
            ax.plot(polygon_pts[:,0], polygon_pts[:,1], 'g--', linewidth=1.5, alpha=0.5, label="Контур")

            ax.set_aspect('equal')
            ax.margins(0.1)
            ax.grid(True, linestyle='--', alpha=0.5, color='#BDC3C7')
            ax.set_xlabel('X', fontweight='bold')
            ax.set_ylabel('Y', fontweight='bold')
            
            st.pyplot(fig, use_container_width=False)

        # Вкладка 2: Таблиці (Матриці) для методу скінченних елементів
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
        st.error(f"Помилка генерації: {e}. Спробуйте змінити параметри (наприклад, зменшити вимоги до кута).")
