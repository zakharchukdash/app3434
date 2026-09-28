import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import triangle as tr

st.set_page_config(page_title="Генератор сітки", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    </style>
    <h3 style='text-align: center; color: #2C3E50; margin-bottom: 0;'>Система розбиття області на скінченні елементи</h3>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 2.2])

with col_left:
    st.subheader("⚙️ Налаштування")
    min_angle = st.slider("Мінімальний кут (градуси)", 0.0, 33.0, 20.0, step=1.0)
    max_area = st.number_input("Обмеження площі (згущення)", 0.01, 1.0, 0.05, step=0.01)
    show_labels = st.checkbox("Показувати підписи", value=True)
    
    st.subheader("📍 Координати контуру")
    default_df = pd.DataFrame({'X': [0.0, 0.0, 2.0, 1.0], 'Y': [1.0, 2.0, 0.0, 0.0]})
    edited_df = st.data_editor(default_df, num_rows="dynamic", use_container_width=True, height=200)

with col_right:
    try:
        pts = edited_df[['X', 'Y']].dropna().to_numpy()
        if len(pts) < 3:
            st.warning("Додайте мінімум 3 точки.")
            st.stop()

        seg = np.array([[i, (i+1)%len(pts)] for i in range(len(pts))])
        A = dict(vertices=pts, segments=seg)
        mesh = tr.triangulate(A, f'pq{min_angle}a{max_area}')

        vertices = mesh['vertices']
        triangles = mesh.get('triangles', [])
        
        def get_edge_marker(p, polygon_pts):
            tol = 1e-5
            for i in range(len(polygon_pts)):
                p1 = polygon_pts[i]
                p2 = polygon_pts[(i+1) % len(polygon_pts)]
                d_p1_p = np.linalg.norm(p - p1)
                d_p_p2 = np.linalg.norm(p2 - p)
                d_p1_p2 = np.linalg.norm(p2 - p1)
                if abs(d_p1_p + d_p_p2 - d_p1_p2) < tol:
                    return f"Грань {i+1}"
            return "Внутрішній (0)"

        custom_markers = [get_edge_marker(v, pts) for v in vertices]

        m1, m2, m3 = st.columns(3)
        m1.metric("🔴 Вузлів", len(vertices))
        m2.metric("🔺 Трикутників", len(triangles))
        m3.metric("✅ Делоне", "Виконано")

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
                    
            ax.set_aspect('equal')
            ax.margins(0.1)
            ax.grid(True, linestyle='--', alpha=0.5, color='#BDC3C7')
            ax.set_xlabel('X', fontweight='bold')
            ax.set_ylabel('Y', fontweight='bold')
            ax.set_axisbelow(True)
            
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
        st.error(f"Помилка: {e}")
