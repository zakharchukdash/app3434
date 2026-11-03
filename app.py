import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.path import Path

st.set_page_config(page_title="МСЕ: Розбиття та матриці", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    </style>
    <h3 style='text-align: center; color: #2C3E50; margin-bottom: 0;'>
        Система розбиття області та формування матриць МСЕ
    </h3>
""", unsafe_allow_html=True)

def calculate_triangle_properties(pts):
    A, B, C = pts[0], pts[1], pts[2]
    a = np.linalg.norm(B - C)
    b = np.linalg.norm(A - C)
    c = np.linalg.norm(A - B)
    
    s = (a + b + c) / 2.0
    area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))
    
    angles = []
    for x, y, z in [(a,b,c), (b,a,c), (c,a,b)]:
        val = np.clip((y**2 + z**2 - x**2) / (2 * y * z), -1.0, 1.0)
        angles.append(np.degrees(np.arccos(val)))
        
    D = 2 * (A[0]*(B[1]-C[1]) + B[0]*(C[1]-A[1]) + C[0]*(A[1]-B[1]))
    if abs(D) < 1e-10:
        return min(angles), area, None
        
    Ux = ((A[0]**2 + A[1]**2)*(B[1]-C[1]) + (B[0]**2 + B[1]**2)*(C[1]-A[1]) + (C[0]**2 + C[1]**2)*(A[1]-B[1])) / D
    Uy = ((A[0]**2 + A[1]**2)*(C[0]-B[0]) + (B[0]**2 + B[1]**2)*(A[0]-C[0]) + (C[0]**2 + C[1]**2)*(B[0]-A[0])) / D
    
    return min(angles), area, [Ux, Uy]


def generate_custom_mesh(polygon, min_angle, max_area, max_iter=500):
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
    
    for _ in range(max_iter):
        tri = Delaunay(points)
        simplices = tri.simplices
        
        centroids = np.mean(points[simplices], axis=1)
        mask = path.contains_points(centroids, radius=-1e-5)
        simplices = simplices[mask]
        
        bad_triangles = []
        for s in simplices:
            angle, area, cc = calculate_triangle_properties(points[s])
            if cc is not None:
                if area > max_area or angle < min_angle:
                    score = max(0, area - max_area) + max(0, min_angle - angle)
                    centroid = np.mean(points[s], axis=0)
                    bad_triangles.append((score, cc, centroid))
                    
        if not bad_triangles:
            break
            
        bad_triangles.sort(key=lambda x: x[0], reverse=True)
        added = False
        
        for _, cc, centroid in bad_triangles:
            candidate = cc if path.contains_point(cc) else centroid
            if np.min(np.linalg.norm(points - candidate, axis=1)) > 1e-3:
                points = np.vstack([points, candidate])
                added = True
                break
                
        if not added:
            break
            
    tri = Delaunay(points)
    simplices = tri.simplices
    centroids = np.mean(points[simplices], axis=1)
    mask = path.contains_points(centroids, radius=-1e-6)
    
    return points, simplices[mask]


def get_edge_indices(p, polygon_pts):
    tol = 1e-5  
    edge_ids = []
    n = len(polygon_pts)
    for i in range(n):
        p1 = polygon_pts[i]
        p2 = polygon_pts[(i+1) % n]
        if abs(np.linalg.norm(p - p1) + np.linalg.norm(p2 - p) - np.linalg.norm(p2 - p1)) < tol:
            edge_ids.append(i)
    return edge_ids

def compute_element_matrices(nodes_coords, a11, a22, d_coef, f_func):
    (x1, y1), (x2, y2), (x3, y3) = nodes_coords
    
    delta = (x2*y3 - x3*y2) - (x1*y3 - x3*y1) + (x1*y2 - x2*y1)
    
    if delta < 0:
        x2, y2, x3, y3 = x3, y3, x2, y2
        delta = -delta

    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])
    
    Ke = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            Ke[i, j] = (a11 * b[i] * b[j] + a22 * c[i] * c[j]) / (2.0 * delta)
            
    mass_template = np.array([
        [2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0]
    ])
    Me = d_coef * (delta / 24.0) * mass_template
    
    # 3. Вектор правої частини Qe
    f_vals = np.array([f_func(x1, y1), f_func(x2, y2), f_func(x3, y3)])
    if abs(d_coef) > 1e-12:
        Qe = (Me / d_coef) @ f_vals
    else:
        Qe = (delta / 24.0) * (mass_template @ f_vals)
        
    return Ke, Me, Qe


def find_boundary_edges(triangles, vertices, polygon_pts):
    edge_dict = {}
    for elem_id, tri in enumerate(triangles):
        for i in range(3):
            n1, n2 = tri[i], tri[(i + 1) % 3]
            edge = tuple(sorted([n1, n2]))
            edge_dict.setdefault(edge, []).append(elem_id)
            
    boundary_edges = []
    tol = 1e-5
    n_poly = len(polygon_pts)
    
    for (n1, n2), elems in edge_dict.items():
        if len(elems) == 1:
            p1 = vertices[n1]
            p2 = vertices[n2]
            p_mid = 0.5 * (p1 + p2)
            
            for edge_idx in range(n_poly):
                poly1 = polygon_pts[edge_idx]
                poly2 = polygon_pts[(edge_idx + 1) % n_poly]
                d1 = np.linalg.norm(p_mid - poly1)
                d2 = np.linalg.norm(poly2 - p_mid)
                d12 = np.linalg.norm(poly2 - poly1)
                if abs(d1 + d2 - d12) < tol:
                    length = np.linalg.norm(p2 - p1)
                    boundary_edges.append((n1, n2, edge_idx, length))
                    break
                    
    return boundary_edges


def assemble_global_fem(vertices, triangles, boundary_edges, a11, a22, d_coef, bc_types, bc_params, f_func):
    num_nodes = len(vertices)
    A_global = np.zeros((num_nodes, num_nodes))
    b_global = np.zeros(num_nodes)
    
    for elem_idx, tri in enumerate(triangles):
        coords = vertices[tri]
        Ke, Me, Qe = compute_element_matrices(coords, a11, a22, d_coef, f_func)
        
        for local_i, global_i in enumerate(tri):
            b_global[global_i] += Qe[local_i]
            for local_j, global_j in enumerate(tri):
                A_global[global_i, global_j] += (Ke[local_i, local_j] + Me[local_i, local_j])
                
    for n1, n2, edge_idx, gamma_len in boundary_edges:
        if "3°" in bc_types.get(edge_idx, ''):
            params = bc_params[edge_idx]
            beta = params["beta"]
            sigma = params["sigma"]
            psi = params["psi"]
            
            Re = (sigma / beta) * (gamma_len / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
            Pe = (psi / beta) * (gamma_len / 2.0) * np.array([1.0, 1.0])
            
            boundary_nodes = [n1, n2]
            for loc_i, glob_i in enumerate(boundary_nodes):
                b_global[glob_i] += Pe[loc_i]
                for loc_j, glob_j in enumerate(boundary_nodes):
                    A_global[glob_i, glob_j] += Re[loc_i, loc_j]
                    
    # 3. Врахування умов 1-го роду (Діріхле)
    dirichlet_nodes = set()
    for n1, n2, edge_idx, _ in boundary_edges:
        if "1°" in bc_types.get(edge_idx, ''):
            dirichlet_nodes.add(n1)
            dirichlet_nodes.add(n2)
            
    for node in dirichlet_nodes:
        A_global[node, :] = 0.0
        A_global[:, node] = 0.0
        A_global[node, node] = 1.0
        b_global[node] = 0.0
                    
    return A_global, b_global


col_left, col_right = st.columns([1.1, 2.1])

with col_left:
    st.subheader("⚙️ Параметри рівняння")
    col_p1, col_p2 = st.columns(2)
    a11 = col_p1.number_input("a11 (дифузія X)", value=1.0, step=0.1)
    a22 = col_p2.number_input("a22 (дифузія Y)", value=1.0, step=0.1)
    d_coef = col_p1.number_input("d (реакція)", value=1.0, step=0.1)
    f_const = col_p2.number_input("f(x, y) = const", value=1.0, step=0.1)
    f_func = lambda x, y: f_const

    st.subheader("🕸️ Налаштування сітки")
    min_angle = st.slider("Мінімальний кут (градуси)", 0.0, 33.0, 20.0, step=1.0)
    max_area = st.number_input("Обмеження площі (згущення)", 0.01, 1.0, 0.25, step=0.01)
    show_labels = st.checkbox("Показувати підписи вузлів та елементів", value=True)
    
    st.subheader("📍 Контур та граничні умови")
    default_df = pd.DataFrame({
        'X': [0.0, 1.0, 1.0, 0.0],
        'Y': [0.0, 0.0, 2.0, 2.0]
    })
    edited_df = st.data_editor(default_df, num_rows="dynamic", use_container_width=True, height=140)
    
    bc_options = [
        "1° рід (u=0)", 
        "2° рід (Nu=0)", 
        "3° рід (βNu+σu=ψ)"
    ]
    v6_presets = [bc_options[2], bc_options[0], bc_options[2], bc_options[1]]
    
    pts_current = edited_df[['X', 'Y']].dropna().to_numpy()
    boundary_types = {}
    bc_numerical_params = {}

    if len(pts_current) >= 3:
        st.markdown("**Умови на гранях:**")
        n_edges = len(pts_current)
        for i in range(n_edges):
            p_s = f"({pts_current[i][0]:.1f}, {pts_current[i][1]:.1f})"
            p_e = f"({pts_current[(i+1) % n_edges][0]:.1f}, {pts_current[(i+1) % n_edges][1]:.1f})"
            default_val = v6_presets[i] if i < len(v6_presets) else bc_options[0]
            
            chosen_bc = st.selectbox(
                f"Грань {i+1}: {p_s} → {p_e}",
                bc_options,
                index=bc_options.index(default_val),
                key=f"edge_bc_{i}"
            )
            boundary_types[i] = chosen_bc
            bc_numerical_params[i] = {"beta": 1.0, "sigma": 1.0, "psi": 1.0}

with col_right:
    try:
        pts = edited_df[['X', 'Y']].dropna().to_numpy()
        if len(pts) < 3:
            st.warning("Додайте мінімум 3 точки для утворення контуру.")
            st.stop()

        vertices, triangles = generate_custom_mesh(pts, min_angle, max_area)
        boundary_edges = find_boundary_edges(triangles, vertices, pts)

        # Рахуємо кути кожного трикутника для таблиці Topology
        tri_angles = []
        for t in triangles:
            ang, _, _ = calculate_triangle_properties(vertices[t])
            tri_angles.append(ang)
        overall_min_angle = min(tri_angles) if len(tri_angles) > 0 else 0.0

        # Формування міток для таблиці крайових міток
        custom_markers = []
        numeric_markers = []
        condition_labels = []
        for v in vertices:
            edge_ids = get_edge_indices(v, pts)
            if not edge_ids:
                numeric_markers.append(0)
                condition_labels.append("Внутрішній")
            else:
                primary_edge = edge_ids[0]
                numeric_markers.append(primary_edge + 1)
                bc_name = boundary_types.get(primary_edge, '')
                if "1°" in bc_name:
                    condition_labels.append("1 (Діріхле)")
                elif "2°" in bc_name:
                    condition_labels.append("2 (Неймана)")
                else:
                    condition_labels.append("3 (Робіна)")

        A_glob, b_glob = assemble_global_fem(
            vertices, triangles, boundary_edges, a11, a22, d_coef, boundary_types, bc_numerical_params, f_func
        )

        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Візуалізація", 
            "📋 Сітка та топологія", 
            "🗄 Локальні матриці", 
            "🌐 Глобальна СЛАР (A, b)"
        ])

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
            st.metric(
                label="Мінімальний кут",
                value=f"{overall_min_angle:.2f}°",
                delta=f"{overall_min_angle - min_angle:.2f}°"
            )
            st.markdown(f"**Вузлів:** {len(vertices)} | **Елементів:** {len(triangles)}")

            st.markdown("#### Таблиця зв'язності (Topology)")
            if len(triangles) > 0:
                df_topo = pd.DataFrame(triangles, columns=['N1', 'N2', 'N3'])
                df_topo['Мін. кут (°)'] = np.round(tri_angles, 1)
                st.dataframe(df_topo, use_container_width=True)

            st.markdown("#### Таблиця вузлів та крайових міток")
            df_nodes_tags = pd.DataFrame({
                'X': np.round(vertices[:, 0], 4),
                'Y': np.round(vertices[:, 1], 4),
                'Маркер': numeric_markers,
                'Умова': condition_labels
            })
            st.dataframe(df_nodes_tags, use_container_width=True)

        with tab3:
            st.markdown("### Локальні матриці скінченних елементів ($\Omega_e$)")
            for elem_id, tri in enumerate(triangles):
                with st.expander(f"🔺 Елемент №{elem_id} (Вузли: {tri[0]}, {tri[1]}, {tri[2]})", expanded=(elem_id == 0)):
                    Ke, Me, Qe = compute_element_matrices(vertices[tri], a11, a22, d_coef, f_func)
                    
                    c_k, c_m, c_q = st.columns([1.2, 1.2, 0.8])
                    with c_k:
                        st.write(f"**Матриця жорсткості $K_{{{elem_id}}}$:**")
                        st.dataframe(pd.DataFrame(np.round(Ke, 4)), use_container_width=True)
                    with c_m:
                        st.write(f"**Матриця мас $M_{{{elem_id}}}$:**")
                        st.dataframe(pd.DataFrame(np.round(Me, 4)), use_container_width=True)
                    with c_q:
                        st.write(f"**Вектор $Q_{{{elem_id}}}$:**")
                        st.dataframe(pd.DataFrame(np.round(Qe, 4), columns=["Значення"]), use_container_width=True)

            st.markdown("---")
            st.markdown("### Граничні матриці для ребер 3-го типу (Робіна)")
            st.info("Значення $R_e$ та $P_e$ залежать від довжини відрізка на межі.")
            
            robin_edges = [e for e in boundary_edges if "3°" in boundary_types.get(e[2], '')]
            if len(robin_edges) == 0:
                st.write("Немає ребер з умовою 3-го роду.")
            else:
                for idx, (n1, n2, e_idx, gamma_len) in enumerate(robin_edges):
                    with st.expander(f"🔹 Граничне ребро №{idx+1} (Між вузлами [{n1}, {n2}])"):
                        params = bc_numerical_params[e_idx]
                        beta, sigma, psi = params["beta"], params["sigma"], params["psi"]
                        
                        Re = (sigma / beta) * (gamma_len / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
                        Pe = (psi / beta) * (gamma_len / 2.0) * np.array([1.0, 1.0])
                        
                        c_r, c_p = st.columns([1.2, 0.8])
                        with c_r:
                            st.write(f"**$R_{{{idx+1}}}$:**")
                            st.dataframe(pd.DataFrame(np.round(Re, 4)), use_container_width=True)
                        with c_p:
                            st.write(f"**$P_{{{idx+1}}}$:**")
                            st.dataframe(pd.DataFrame(np.round(Pe, 4), columns=["Значення"]), use_container_width=True)

        with tab4:
            st.markdown("#### Глобальна матриця $A$ та вектор $b$")
            node_labels = [f"U_{i}" for i in range(len(vertices))]
            
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.write(f"**Глобальна матриця $A$ ({len(vertices)} × {len(vertices)}):**")
                st.dataframe(pd.DataFrame(np.round(A_glob, 4), index=node_labels, columns=node_labels), use_container_width=True)
            with col_b:
                st.write(f"**Вектор $b$:**")
                st.dataframe(pd.DataFrame(np.round(b_glob, 4), index=node_labels, columns=["b"]), use_container_width=True)

    except Exception as e:
        st.error(f"Помилка розрахунку: {e}. Спробуйте змінити параметри.")
