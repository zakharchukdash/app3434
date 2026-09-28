import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import triangle as tr

st.set_page_config(layout="wide")
st.title("Триангуляція області (Варіант 6)")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Налаштування")
    min_angle = st.slider("Мінімальний кут", 0.0, 33.0, 20.0)
    max_area = st.number_input("Максимальна площа трикутника (згущення)", 0.001, 1.0, 0.05, step=0.01)
    
    st.subheader("Довільна область (x, y)")
    default_vertices = "0.0, 1.0\n0.0, 2.0\n2.0, 0.0\n1.0, 0.0"
    vertices_input = st.text_area("Координати контуру", default_vertices, height=150)

with col2:
    try:
        # Парсинг координат
        pts = np.array([list(map(float, line.split(','))) for line in vertices_input.strip().split('\n')])
        seg = np.array([[i, (i+1)%len(pts)] for i in range(len(pts))])
        A = dict(vertices=pts, segments=seg)
        
        # Генерація сітки
        command = f'pq{min_angle}a{max_area}'
        mesh = tr.triangulate(A, command)
        
        vertices = mesh['vertices']
        triangles = mesh.get('triangles', [])
        
        # Візуалізація
        fig, ax = plt.subplots(figsize=(6, 6))
        if len(triangles) > 0:
            ax.triplot(vertices[:,0], vertices[:,1], triangles, color='blue', marker='o', markersize=4)
            for i, p in enumerate(vertices):
                ax.text(p[0]+0.02, p[1]+0.02, str(i), color='red', fontsize=10)
            for i, t in enumerate(triangles):
                pt = np.mean(vertices[t], axis=0)
                ax.text(pt[0], pt[1], str(i), color='darkgreen', fontsize=9, ha='center', va='center')
                
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Помилка: {e}")

st.header("Дані розбиття")
d_col1, d_col2 = st.columns(2)

with d_col1:
    st.subheader("Вузли та маркування границь")
    markers = mesh.get('vertex_markers', np.zeros((len(vertices), 1)))
    df_nodes = pd.DataFrame(vertices, columns=['X', 'Y'])
    df_nodes['Маркер границі'] = markers
    st.dataframe(df_nodes, use_container_width=True)

with d_col2:
    st.subheader("Масив зв'язності")
    if len(triangles) > 0:
        df_tri = pd.DataFrame(triangles, columns=['Вузол 1', 'Вузол 2', 'Вузол 3'])
        st.dataframe(df_tri, use_container_width=True)
