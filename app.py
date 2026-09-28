import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import triangle as tr
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# Налаштування сторінки
st.set_page_config(page_title="Генератор сітки", layout="wide", initial_sidebar_state="collapsed")

st.markdown("<h2 style='text-align: center; color: #2C3E50;'>Система розбиття області на скінченні елементи</h2>", unsafe_allow_html=True)

# Верхній блок налаштувань замість бокової панелі
with st.expander("⚙️ Налаштування параметрів та координат контуру", expanded=True):
    col_in1, col_in2 = st.columns([1, 2])
    with col_in1:
        min_angle = st.slider("Мінімальний кут (градуси)", 0.0, 33.0, 20.0, step=1.0)
        max_area = st.number_input("Обмеження площі (згущення)", 0.01, 1.0, 0.05, step=0.01)
    with col_in2:
        default_df = pd.DataFrame({'X': [0.0, 0.0, 2.0, 1.0], 'Y': [1.0, 2.0, 0.0, 0.0]})
        edited_df = st.data_editor(default_df, num_rows="dynamic", use_container_width=True, height=140)

# Основна логіка
try:
    pts = edited_df[['X', 'Y']].dropna().to_numpy()
    if len(pts) < 3:
        st.warning("Додайте мінімум 3 точки для побудови області.")
        st.stop()

    seg = np.array([[i, (i+1)%len(pts)] for i in range(len(pts))])
    A = dict(vertices=pts, segments=seg)
    command = f'pq{min_angle}a{max_area}'
    mesh = tr.triangulate(A, command)

    vertices = mesh['vertices']
    triangles = mesh.get('triangles', [])
    markers = mesh.get('vertex_markers', np.zeros((len(vertices), 1)))

    # Статистика нагорі
    m1, m2, m3 = st.columns(3)
    m1.metric("🔴 Кількість вузлів", len(vertices))
    m2.metric("🔺 Кількість трикутників", len(triangles))
    m3.metric("✅ Умова Делоне", "Виконано")

    # Вкладки для розділення контенту
    tab1, tab2 = st.tabs(["📊 Візуалізація", "🗄 Матриці та масиви"])

    with tab1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('#ffffff')
        
        if len(triangles) > 0:
            # Залиті трикутники замість звичайних ліній
            patches = [Polygon(vertices[t], closed=True) for t in triangles]
            p = PatchCollection(patches, facecolor='#17A2B8', edgecolor='white', alpha=0.4, linewidths=2)
            ax.add_collection(p)
            
            # Квадратні вузли
            ax.scatter(vertices[:,0], vertices[:,1], c='#DC3545', s=60, zorder=5, marker='s')
            
            for i, p_val in enumerate(vertices):
                ax.text(p_val[0], p_val[1]+0.06, f"N{i}", color='darkred', fontsize=10, fontweight='bold', ha='center')
                
            for i, t in enumerate(triangles):
                pt = np.mean(vertices[t], axis=0)
                ax.text(pt[0], pt[1], f"E{i}", color='#343A40', fontsize=9, ha='center', va='center')
                
        ax.set_aspect('equal')
        ax.axis('off') # Приховуємо осі для стильного вигляду
        st.pyplot(fig)

    with tab2:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("#### Координати вузлів та границі")
            df_nodes = pd.DataFrame(vertices, columns=['X', 'Y'])
            df_nodes['Маркер межі'] = markers
            st.dataframe(df_nodes.style.highlight_max(axis=0, color='#e2e3e5'), use_container_width=True, height=350)
            
        with col_t2:
            st.markdown("#### Масив зв'язності (Елементи)")
            if len(triangles) > 0:
                df_tri = pd.DataFrame(triangles, columns=['Вузол A', 'Вузол B', 'Вузол C'])
                st.dataframe(df_tri.style.set_properties(**{'background-color': '#f8f9fa'}), use_container_width=True, height=350)

except Exception as e:
    st.error(f"Помилка розрахунку: {e}")
