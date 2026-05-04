import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
import tensorflow as tf

# 設定網頁標題與版面
st.set_page_config(page_title="ANN 電子級 PMA 優化系統", layout="wide")

# --- Google Analytics 匿名統計 (穿透注入版) ---
GA_ID = 'G-7TKCC4EV45'
ga_injection = f"""
    <script>
        // 1. 建立 GA 腳本標籤
        var script = window.parent.document.createElement('script');
        script.async = true;
        script.src = "https://www.googletagmanager.com/gtag/js?id={GA_ID}";
        window.parent.document.head.appendChild(script);

        // 2. 初始化 GA 邏輯
        window.parent.window.dataLayer = window.parent.window.dataLayer || [];
        function gtag(){{window.parent.window.dataLayer.push(arguments);}}
        gtag('js', new Date());
        gtag('config', '{GA_ID}');
        
        console.log("GA4 Injected into parent head.");
    </script>
"""
# 執行隱形注入
st.components.v1.html(ga_injection, height=0)

st.title("人工類神經網路（ANN）應用於電子級 PMA 製程之產率預測與參數優化")
st.write("本平台為碩士研究相關之電子級 PMA 製程預測與操作條件 analysis 展示頁面，使用 Streamlit 建立。")

# 插入製程流程圖
if os.path.exists("電子級PMA 製程流程圖.png"):
    st.image("電子級PMA 製程流程圖.png", caption="電子級 PMA 生產製程流程圖", use_container_width=True)

MW_PMA = 132.16

# =========================================================
# 1. 載入模型 (快取處理)
# =========================================================
@st.cache_resource
def load_models():
    model_dir = 'models_v5'
    if not os.path.exists(model_dir):
        st.error(f"找不到模型資料夾: {model_dir}")
        st.stop()
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
            'C1_Cond': tf.keras.models.load_model(os.path.join(model_dir, 'model_stage2_C1_Cond_optimized.h5'), compile=False),
            'C1_Reb': tf.keras.models.load_model(os.path.join(model_dir, 'model_stage2_C1_Reb_optimized.h5'), compile=False),
            'C2_Cond': tf.keras.models.load_model(os.path.join(model_dir, 'model_stage2_C2_Cond_optimized.h5'), compile=False),
            'C2_Reb': tf.keras.models.load_model(os.path.join(model_dir, 'model_stage2_C2_Reb_optimized.h5'), compile=False)
        }
    except Exception as e:
        st.error(f"加載模型出錯: {e}")
        st.stop()
    return m

# =========================================================
# 2. 側邊欄控制
# =========================================================
with st.sidebar:
    st.header("⚙️ 操作參數輸入")
    with st.expander("🔹 反應器階段", expanded=True):
        T = st.number_input("加熱器出口溫度 (°C)", 70.0, 110.0, 109.03, step=0.01)
        Fin = st.number_input("總進料流量 (kmol/h)", 1.0, 300.0, 71.25, step=0.1)
        Raa = st.number_input("進料比 (AA 分率)", 0.0, 1.0, 0.509, step=0.001, format="%.3f")
    with st.expander("🔹 蒸餾塔 C1", expanded=True):
        C1_R = st.number_input("C1 回流比 (R1)", 0.1, 30.0, 5.07)
        C1_B = st.number_input("C1 蒸氣比 (B1)", 0.1, 30.0, 9.31)
        C1_P = st.number_input("C1 壓力 (N/m2)", 1000.0, 50000.0, 9619.0)
    with st.expander("🔹 蒸餾塔 C2", expanded=True):
        C2_R = st.number_input("C2 回流比 (R2)", 0.1, 30.0, 9.89)
        C2_B = st.number_input("C2 蒸氣比 (B2)", 0.1, 30.0, 7.64)
        C2_P = st.number_input("C2 壓力 (N/m2)", 1000.0, 50000.0, 7102.0)

# =========================================================
# 3. 預測核心
# =========================================================
try:
    M = load_models()
    r_in_s = M['s_r1'].transform([[T, Fin, Raa, 1-Raa]])
    h_ene = M['s_r1_y']['Heater Energy Consumption (kW)'].inverse_transform(M['mod_r_ene'].predict(r_in_s, verbose=0))[0,0]
    r_pma_out = np.expm1(M['s_r1_y']['Reactor PMA Flow (kmol/h)'].inverse_transform(M['mod_r_pma'].predict(r_in_s, verbose=0)))[0,0]
    aa_in_flow = Fin * Raa
    pgme_in_flow = Fin * (1 - Raa)
    r_aa_out = max(0, aa_in_flow - r_pma_out)
    r_pg_out = max(0, pgme_in_flow - r_pma_out)
    aa_conv = (aa_in_flow - r_aa_out) / (aa_in_flow + 1e-9) * 100
    pgme_conv = (pgme_in_flow - r_pg_out) / (pgme_in_flow + 1e-9) * 100

    if aa_in_flow < pgme_in_flow: limiting_reagent = "AA (醋酸)"
    elif pgme_in_flow < aa_in_flow: limiting_reagent = "PGME (丙二醇甲醚)"
    else: limiting_reagent = "等摩爾進料"

    x_distill = [C1_R, C1_B, C1_P, C2_R, C2_B, C2_P]
    x_df_e2e = pd.DataFrame([[T, Fin, Raa] + x_distill], columns=['T', 'Flow_In', 'Ratio_AA', 'C1_R', 'C1_B', 'C1_P', 'C2_R', 'C2_B', 'C2_P'])
    m_flow_raw = np.expm1(M['s_fl']['y_s'].inverse_transform(M['mod_fl'].predict(M['s_fl']['x_s'].transform(x_df_e2e), verbose=0)))[0,0]
    m_flow = np.minimum(m_flow_raw, r_pma_out * 0.99)
    aa_ppm = np.expm1(M['s_aa']['y_s'].inverse_transform(M['mod_aa'].predict(M['s_aa']['x_s'].transform(x_df_e2e), verbose=0)))[0,0]
    x_pu_in = pd.DataFrame([[r_aa_out, r_pg_out, r_pma_out] + x_distill], columns=['R_AA', 'R_PGME', 'R_PMA', 'C1_R', 'C1_B', 'C1_P', 'C2_R', 'C2_B', 'C2_P'])
    p_log = M['s_pu']['y_scaler'].inverse_transform(M['mod_pu'].predict(M['s_pu']['x_scaler'].transform(x_pu_in), verbose=0))[0,0]
    purity = np.clip(100.0001 - (10**p_log), 0, 100)
    
    total_sep = 0
    ene_vals = {}
    for t in ['C1_Cond', 'C1_Reb', 'C2_Cond', 'C2_Reb']:
        p_val = M['s_ene_hifi'][t]['sy'].inverse_transform(M['mod_d_ene'][t].predict(M['s_ene_hifi'][t]['sx'].transform(x_pu_in), verbose=0))[0,0]
        ene_vals[t] = abs(p_val)
        total_sep += abs(p_val)

    # 【修正】基於限量試劑計算產率
    limiting_in_mol = min(aa_in_flow, pgme_in_flow)
    total_yield = (m_flow / (limiting_in_mol + 1e-9)) * 100
    total_sys_ene = h_ene + total_sep

    # =========================================================
    # 4. UI
    # =========================================================
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("✨ 預測純度", f"{purity:.4f} %"); c1.progress(float(np.clip(purity/100, 0.0, 1.0)))
    with c2:
        st.metric("📈 總產率", f"{total_yield:.2f} %")
        with st.popover("🧮 計算過程詳解"):
            st.markdown("### **製程指標動態計算式**")
            pma_rec = (m_flow / (r_pma_out + 1e-9)) * 100
            st.markdown("#### **1. 反應器轉化率**")
            st.latex(r"X = \frac{\dot{n}_{in} - \dot{n}_{out}}{\dot{n}_{in}} \times 100\%")
            st.code(f"AA: ({aa_in_flow:.4f}-{r_aa_out:.4f})/{aa_in_flow:.4f}={aa_conv:.2f}%", language="text")
            st.code(f"PGME: ({pgme_in_flow:.4f}-{r_pg_out:.4f})/{pgme_in_flow:.4f}={pgme_conv:.2f}%", language="text")
            st.markdown("#### **2. 總產率 (基於限量試劑)**")
            st.latex(r"Y_{Total} = \frac{\dot{n}_{PMA, product}}{\min(\dot{n}_{AA, feed}, \dot{n}_{PGME, feed})} \times 100\%")
            st.code(f"計算: {m_flow:.4f} / {limiting_in_mol:.4f} × 100% = {total_yield:.2f}%", language="text")
            st.info(f"💡 目前限量試劑: **{limiting_reagent}**")

    c3.metric("📦 質量流率", f"{m_flow*MW_PMA:.2f} kg/h", f"{m_flow:.4f} kmol/h")
    c4.metric("🧪 AA 含量", f"{aa_ppm:.2f} ppm")
    c5.metric("⚡ 系統總能耗", f"{total_sys_ene:.2f} kW")

    st.write("---")
    t1, t2, t3 = st.tabs(["📌 反應器詳情", "📌 分離塔詳情", "🏆 最佳優化方案 (v6)"])
    with t1:
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("#### **物料衡算 (kmol/h)**")
            st.table(pd.DataFrame({"組分": ["AA", "PGME", "PMA", "H2O"], "進料": [f"{aa_in_flow:.4f}", f"{pgme_in_flow:.4f}", "0.0000", "0.0000"], "出口": [f"{r_aa_out:.4f}", f"{r_pg_out:.4f}", f"{r_pma_out:.4f}", f"{r_pma_out:.4f}"]}))
        with cc2:
            st.markdown("#### **反應指標**")
            st.info(f"**限量試劑:** {limiting_reagent}"); st.info(f"**AA 轉化率:** {aa_conv:.2f} %"); st.info(f"**PGME 轉化率:** {pgme_conv:.2f} %"); st.info(f"**Heater 能耗:** {h_ene:.2f} kW")
    with t2:
        cc3, cc4 = st.columns(2)
        with cc3:
            st.markdown("#### **分離能耗詳情 (kW)**")
            st.table(pd.DataFrame({"換熱器": ["C1 Cond", "C1 Reb", "C2 Cond", "C2 Reb"], "負荷": [f"{ene_vals['C1_Cond']:.2f}", f"{ene_vals['C1_Reb']:.2f}", f"{ene_vals['C2_Cond']:.2f}", f"{ene_vals['C2_Reb']:.2f}"]}))
        with cc4:
            st.markdown("#### **最終產品與能耗**")
            st.success(f"**產率 (限量基準):** {total_yield:.2f} %"); st.success(f"**最終純度:** {purity:.4f} %"); st.success(f"**系統總耗能:** {total_sys_ene:.2f} kW")
    with t3:
        st.markdown("#### **最佳建議方案 (純度 ≥ 99.99% & 物理嚴謹 v6)**")
        st.table(pd.DataFrame({
            "指標": ["溫度", "流量", "進料比_AA", "R1/B1/P1", "R2/B2/P2", "純度 (%)", "AA (ppm)", "PMA (kmol/h)", "產率 (%)", "能耗 (kW)"],
            "Case 1 (節能)": ["82.72", "13.84", "0.528", "5.85/9.88/8181", "6.99/6.99/7022", "99.9912", "78.43", "2.1547", "33.00", "1304.43"],
            "Case 2 (產能)": ["101.07", "112.50", "0.534", "5.70/9.71/8480", "8.17/9.17/7084", "99.9903", "88.12", "21.7156", "41.42", "14964.14"],
            "Case 3 (平衡)": ["109.03", "71.25", "0.509", "5.07/9.31/9619", "9.89/7.64/7102", "99.9917", "62.45", "16.2179", "46.36", "9077.31"]
        }))
except Exception as e:
    st.error(f"錯誤: {e}")
st.markdown("---")
st.caption("PMA AI Optimization System © 2023 | 數據由 DWSIM 模擬與 集成專家模型 驅動")
