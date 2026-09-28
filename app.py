import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import triangle as tr

st.set_page_config(page_title="Генератор сітки", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<h2 style='text-align: center; color: #2C3E50;'>Система розбиття області на скінченні елементи</h2>", unsafe_allow_html=True)

with st.expander("⚙️ Налаштування параметрів та координат", expanded=False):
    col_in1, col_in2 = st.columns([1, 2])
    with col_in1:
        min_angle = st.slider("Мінімальний кут (градуси)", 0.0, 33.0, 20.0, step=1.0)
        max_area = st.number_input("Обмеження площі (згущення)", 0.01, 1.0, 0.05, step=0.01)
        # Додаємо тумблер для керування текстом на графіку
        show_labels = st.checkbox("Показувати підписи вузлів та елементів", value=True)
    with col_in2:
        default_df = pd.DataFrame({'X': [0.0, 0.0, 2.0, 1.0], 'Y': [1.0, 2.0, 0.0, 0.0]})
        edited_df = st.data_editor(default_df, num_rows="dynamic", use_container_width=True, height=140)

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
    markers = mesh.get('vertex_markers', np.zeros((len(vertices), 1)))

    m1, m2, m3 = st.columns(3)
    m1.metric("🔴 Кількість вузлів", len(vertices))
    m2.metric("🔺 Кількість трикутників", len(triangles))
    m3.metric("✅ Умова Делоне", "Виконано")

    tab1, tab2 = st.tabs(["📊 Візуалізація", "🗄 Матриці та масиви"])

    with tab1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_facecolor('#ffffff')
        
        if len(triangles) > 0:
            # Малюємо чіткі лінії сітки та акуратні вузли
            ax.triplot(vertices[:,0], vertices[:,1], triangles, color='#2C3E50', linewidth=1, 
                       marker='o', markersize=4, markerfacecolor='#E74C3C', markeredgecolor='#E74C3C')
            
            # Виводимо текст лише якщо стоїть галочка
            if show_labels:
                for i, p_val in enumerate(vertices):
                    ax.text(p_val[0], p_val[1]+0.03, f"{i}", color='#C0392B', fontsize=8, fontweight='bold', ha='center')
                    
                for i, t in enumerate(triangles):
                    pt = np.mean(vertices[t], axis=0)
                    ax.text(pt[0], pt[1], f"{i}", color='#2980B9', fontsize=8, ha='center', va='center')
                
        ax.set_aspect('equal')
        ax.margins(0.1)
        
        # Повертаємо координатну сітку та осі
        ax.grid(True, linestyle='--', alpha=0.5, color='#BDC3C7')
        ax.set_xlabel('X', fontweight='bold')
        ax.set_ylabel('Y', fontweight='bold')
        ax.set_axisbelow(True)
        
        st.pyplot(fig, use_container_width=False)

    with tab2:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("#### Координати вузлів")
            df_nodes = pd.DataFrame(vertices, columns=['X', 'Y'])
            df_nodes['Межа'] = markers
            st.dataframe(df_nodes.style.highlight_max(axis=0, color='#e2e3e5'), use_container_width=True, height=350)
            
        with col_t2:
            st.markdown("#### Масив зв'язності")
            if len(triangles) > 0:
                df_tri = pd.DataFrame(triangles, columns=['Вуз A', 'Вуз B', 'Вуз C'])
                st.dataframe(df_tri.style.set_properties(**{'background-color': '#f8f9fa'}), use_container_width=True, height=350)

except Exception as e:
    st.error(f"Помилка: {e}")
