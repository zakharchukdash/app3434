import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import triangle as tr

st.set_page_config(layout="wide")
st.title("Триангуляція області (Варіант 6)")

# Переносимо налаштування в бокову панель
with st.sidebar:
    st.header("Налаштування")
    min_angle = st.slider("Мінімальний кут", 0.0, 33.0, 20.0)
    max_area = st.number_input("Максимальна площа", 0.001, 1.0, 0.05, step=0.01)
    
    st.subheader("Координати контуру")
    default_vertices = "0.0, 1.0\n0.0, 2.0\n2.0, 0.0\n1.0, 0.0"
    vertices_input = st.text_area("x, y", default_vertices, height=150)

# Основна область на 3 колонки
col1, col2, col3 = st.columns([1.5, 1, 1])

try:
    pts = np.array([list(map(float, line.split(','))) for line in vertices_input.strip().split('\n')])
    seg = np.array([[i, (i+1)%len(pts)] for i in range(len(pts))])
    A = dict(vertices=pts, segments=seg)
    
    command = f'pq{min_angle}a{max_area}'
    mesh = tr.triangulate(A, command)
    
    vertices = mesh['vertices']
    triangles = mesh.get('triangles', [])
    
    with col1:
        st.subheader("Графік")
        fig, ax = plt.subplots(figsize=(4, 4)) # Зменшений розмір графіка
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
        
    with col2:
        st.subheader("Вузли та маркери")
        markers = mesh.get('vertex_markers', np.zeros((len(vertices), 1)))
        df_nodes = pd.DataFrame(vertices, columns=['X', 'Y'])
        df_nodes['Маркер'] = markers
        st.dataframe(df_nodes, use_container_width=True, height=400)

    with col3:
        st.subheader("Зв'язність")
        if len(triangles) > 0:
            df_tri = pd.DataFrame(triangles, columns=['Вуз 1', 'Вуз 2', 'Вуз 3'])
            st.dataframe(df_tri, use_container_width=True, height=400)
            
except Exception as e:
    st.error(f"Помилка: {e}")
