import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
import tensorflow as tf

# 設定網頁標題與版面
st.set_page_config(page_title="ANN 電子級 PMA 優化系統", layout="wide")

# --- Google Analytics 匿名統計 ---
GA_ID = 'G-7TKCC4EV45'
ga_injection = f"""
    <script>
        var script = window.parent.document.createElement('script');
        script.async = true;
        script.src = "https://www.googletagmanager.com/gtag/js?id={GA_ID}";
        window.parent.document.head.appendChild(script);
        window.parent.window.dataLayer = window.parent.window.dataLayer || [];
        function gtag(){{window.parent.window.dataLayer.push(arguments);}}
        gtag('js', new Date());
        gtag('config', '{GA_ID}');
    </script>
"""
st.components.v1.html(ga_injection, height=0)

st.title("人工類神經網路（ANN）應用於電子級 PMA 製程之產率預測與參數優化")
st.write("本平台為碩士研究相關之電子級 PMA 製程預測與操作條件分析展示頁面，使用 Streamlit 建立。")

if os.path.exists("電子級PMA 製程流程圖.png"):
    st.image("電子級PMA 製程流程圖.png", caption="電子級 PMA 生產製程流程圖", use_container_width=True)

MW_PMA = 132.16

@st.cache_resource
def load_models():
    model_dir = 'models_v5'
    m = {}
    try:
        m['s_pu'] = pickle.load(open(os.path.join(model_dir, 'purity_scalers.pkl'), 'rb'))
        m['mod_pu'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_purity_expert_log.h5'), compile=False)
        m['s_fl'] = pickle.load(open(os.path.join(model_dir, 'flow_master_scalers.pkl'), 'rb'))
        m['mod_fl'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_flow_master_final.h5'), compile=False)
        m['s_aa'] = pickle.load(open(os.path.join(model_dir, 'aa_ppm_scalers.pkl'), 'rb'))
        m['mod_aa'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_aa_ppm_hifi.h5'), compile=False)
        m['s_r1'] = pickle.load(open(os.path.join(model_dir, 'x_scaler.pkl'), 'rb'))
        m['s_r1_y'] = pickle.load(open(os.path.join(model_dir, 'y_scalers.pkl'), 'rb'))
        m['mod_r_ene'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_Heater_Energy_Consumption_kW.h5'), compile=False)
        m['mod_r_pma'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_Reactor_PMA_Flow_kmol_h.h5'), compile=False)
        m['mod_r_aa'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_Reactor_AA_Flow_kmol_h.h5'), compile=False)
        m['mod_r_pgme'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_Reactor_PGME_Flow_kmol_h.h5'), compile=False)
        m['s_ene_hifi'] = pickle.load(open(os.path.join(model_dir, 'energy_hifi_scalers.pkl'), 'rb'))
        m['mod_d_ene'] = {
            'C1_Cond': tf.keras.models.load_model(os.path.join(model_dir, 'model_ann_energy_C1_Cond_relu_log.h5'), compile=False),
            'C1_Reb': tf.keras.models.load_model(os.path.join(model_dir, 'model_ann_energy_C1_Reb_relu_log.h5'), compile=False),
            'C2_Cond': tf.keras.models.load_model(os.path.join(model_dir, 'model_ann_energy_C2_Cond_relu_log.h5'), compile=False),
            'C2_Reb': tf.keras.models.load_model(os.path.join(model_dir, 'model_ann_energy_C2_Reb_relu_log.h5'), compile=False)
        }
    except Exception as e:
        st.error(f"加載模型出錯: {e}"); st.stop()
    return m

with st.sidebar:
    st.header("⚙️ 操作參數輸入")
    with st.expander("🔹 反應器階段", expanded=True):
        T = st.number_input("加熱器出口溫度 (°C)", 70.0, 110.0, 108.62, step=0.01)
        Fin = st.number_input("總進料流量 (kmol/h)", 1.0, 300.0, 73.04, step=0.1)
        Raa = st.number_input("進料比 (AA 分率)", 0.0, 1.0, 0.464, step=0.001, format="%.3f")
    with st.expander("🔹 蒸餾塔 C1", expanded=True):
        C1_R, C1_B, C1_P = st.number_input("C1 回流比", 0.1, 30.0, 5.87), st.number_input("C1 蒸氣比", 0.1, 30.0, 9.77), st.number_input("C1 壓力", 1000.0, 50000.0, 9118.0)
    with st.expander("🔹 蒸餾塔 C2", expanded=True):
        C2_R, C2_B, C2_P = st.number_input("C2 回流比", 0.1, 30.0, 8.69), st.number_input("C2 蒸氣比", 0.1, 30.0, 9.95), st.number_input("C2 壓力", 1000.0, 50000.0, 7941.0)

try:
    M = load_models()
    r_in_s = M['s_r1'].transform([[T, Fin, Raa, 1-Raa]])
    h_ene = M['s_r1_y']['Heater Energy Consumption (kW)'].inverse_transform(M['mod_r_ene'].predict(r_in_s, verbose=0))[0,0]
    r_pma_out = np.expm1(M['s_r1_y']['Reactor PMA Flow (kmol/h)'].inverse_transform(M['mod_r_pma'].predict(r_in_s, verbose=0)))[0,0]
    aa_in, pg_in = Fin*Raa, Fin*(1-Raa)
    r_aa_out, r_pg_out = max(0, aa_in - r_pma_out), max(0, pg_in - r_pma_out)
    aa_conv, pgme_conv = (aa_in-r_aa_out)/(aa_in+1e-9)*100, (pg_in-r_pg_out)/(pg_in+1e-9)*100
    lim_reagent = "AA (醋酸)" if aa_in < pg_in else "PGME (丙二醇甲醚)"

    x_df_e2e = pd.DataFrame([[T, Fin, Raa, C1_R, C1_B, C1_P, C2_R, C2_B, C2_P]], columns=['T', 'Flow_In', 'Ratio_AA', 'C1_R', 'C1_B', 'C1_P', 'C2_R', 'C2_B', 'C2_P'])
    m_flow = np.minimum(np.expm1(M['s_fl']['y_s'].inverse_transform(M['mod_fl'].predict(M['s_fl']['x_s'].transform(x_df_e2e), verbose=0)))[0,0], r_pma_out*0.99)
    aa_ppm = np.expm1(M['s_aa']['y_s'].inverse_transform(M['mod_aa'].predict(M['s_aa']['x_s'].transform(x_df_e2e), verbose=0)))[0,0]
    
    x_pu_in = pd.DataFrame([[r_aa_out, r_pg_out, r_pma_out, C1_R, C1_B, C1_P, C2_R, C2_B, C2_P]], columns=['R_AA', 'R_PGME', 'R_PMA', 'C1_R', 'C1_B', 'C1_P', 'C2_R', 'C2_B', 'C2_P'])
    purity = np.clip(100.0001 - (10**M['s_pu']['y_scaler'].inverse_transform(M['mod_pu'].predict(M['s_pu']['x_scaler'].transform(x_pu_in), verbose=0))[0,0]), 0, 100)
    
    total_sep, ene_vals = 0, {}
    for t in ['C1_Cond', 'C1_Reb', 'C2_Cond', 'C2_Reb']:
        p_log = M['mod_d_ene'][t].predict(M['s_ene_hifi'][t]['sx'].transform(x_pu_in), verbose=0)[0,0]
        v = 10**p_log; ene_vals[t] = v; total_sep += v

    limiting_in_mol = min(aa_in, pg_in)
    total_yield = (m_flow / (limiting_in_mol + 1e-9)) * 100
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("✨ 預測純度", f"{purity:.4f} %"); col1.progress(float(np.clip(purity/100, 0.0, 1.0)))
    with col2:
        st.metric("📈 總產率", f"{total_yield:.2f} %")
        with st.popover("🧮 計算過程"):
            st.latex(r"X_{AA} = " + f"{aa_conv:.2f}\%"); st.latex(r"Y_{Total} = \frac{" + f"{m_flow:.4f}" + r"}{\min(" + f"{aa_in:.4f}, {pg_in:.4f}" + r")} = " + f"{total_yield:.2f}\%")
    col3.metric("📦 質量流率", f"{m_flow*MW_PMA:.2f} kg/h", f"{m_flow:.4f} kmol/h")
    col4.metric("🧪 AA 含量", f"{aa_ppm:.2f} ppm")
    col5.metric("⚡ 系統總能耗", f"{h_ene + total_sep:.2f} kW")

    st.write("---")
    t1, t2, t3 = st.tabs(["📌 反應器詳情", "📌 分離塔詳情", "🏆 最佳優化方案 (v7)"])
    with t1:
        cc1, cc2 = st.columns(2)
        with cc1:
            st.table(pd.DataFrame({"組分": ["AA", "PGME", "PMA", "H2O"], "進料": [f"{aa_in:.4f}", f"{pg_in:.4f}", "0.0000", "0.0000"], "出口": [f"{r_aa_out:.4f}", f"{r_pg_out:.4f}", f"{r_pma_out:.4f}", f"{r_pma_out:.4f}"]}))
        with cc2:
            st.info(f"**限量試劑:** {lim_reagent}"); st.info(f"**AA 轉化率:** {aa_conv:.2f} %"); st.info(f"**Heater 能耗:** {h_ene:.2f} kW")
    with t2:
        st.table(pd.DataFrame({"換熱器": ["C1 Cond", "C1 Reb", "C2 Cond", "C2 Reb"], "負荷 (kW)": [f"{ene_vals['C1_Cond']:.2f}", f"{ene_vals['C1_Reb']:.2f}", f"{ene_vals['C2_Cond']:.2f}", f"{ene_vals['C2_Reb']:.2f}"]}))
    with t3:
        st.markdown("#### **v7 ReLU 物理嚴謹版最佳建議方案**")
        st.table(pd.DataFrame({
            "指標": ["溫度", "流量", "進料比_AA", "R1/B1/P1", "R2/B2/P2", "純度 (%)", "AA (ppm)", "產率 (%)", "能耗 (kW)"],
            "Case 1 (節能)": ["82.72", "13.84", "0.528", "5.85/9.88/8181", "6.99/6.99/7022", "99.9912", "78.43", "32.97", "1518.97"],
            "Case 2 (產能)": ["101.07", "112.50", "0.534", "5.70/9.71/8480", "8.17/9.17/7084", "99.9903", "88.12", "41.46", "14664.96"],
            "Case 3 (平衡)": ["108.62", "73.04", "0.464", "5.87/9.77/9118", "8.69/9.95/7941", "99.9949", "62.45", "49.19", "9064.29"]
        }))
except Exception as e:
    st.error(f"運行錯誤: {e}")
st.caption("PMA AI Optimization System © 2023 | 驅動於 v7 ReLU 高精度模型")
