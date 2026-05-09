import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
import tensorflow as tf

# 設定網頁標題與版面
st.set_page_config(page_title="ANN 電子級 PMA 優化系統 - 校正版", layout="wide")

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
st.write("本平台為碩士研究相關之電子級 PMA 製程預測與操作條件分析展示頁面。模型已完成 DWSIM 數據校正與物理載荷優化。")

if os.path.exists("電子級PMA 製程流程圖.png"):
    st.image("電子級PMA 製程流程圖.png", caption="電子級 PMA 生產製程流程圖", use_container_width=True)

MW_PMA = 132.16

@st.cache_resource
def load_models():
    model_dir = 'models_v5'
    opt_dir = '分析結果0423_專屬優化'
    m = {}
    try:
        # 1. 核心與品質模型 (保持 v5 高精度版)
        m['s_pu'] = pickle.load(open(os.path.join(model_dir, 'purity_scalers.pkl'), 'rb'))
        m['mod_pu'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_purity_expert_log.h5'), compile=False)
        m['s_aa'] = pickle.load(open(os.path.join(model_dir, 'aa_ppm_scalers.pkl'), 'rb'))
        m['mod_aa'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_aa_ppm_hifi.h5'), compile=False)
        
        # 2. 最新校正後的產品流量與能耗模型 (256, 128, 64)
        m['s_ene_opt_y'] = pickle.load(open(os.path.join(opt_dir, 'y2_scalers_optimized.pkl'), 'rb'))
        m['s_ene_opt_x'] = pickle.load(open(os.path.join(opt_dir, 'x2_scaler_optimized.pkl'), 'rb'))
        m['mod_fl'] = tf.keras.models.load_model(os.path.join(opt_dir, 'model_stage2_Product_Flow_optimized.h5'), compile=False)
        m['mod_d_ene'] = {
            'C1_Cond': tf.keras.models.load_model(os.path.join(opt_dir, 'model_stage2_C1_Cond_optimized.h5'), compile=False),
            'C1_Reb': tf.keras.models.load_model(os.path.join(opt_dir, 'model_stage2_C1_Reb_optimized.h5'), compile=False),
            'C2_Cond': tf.keras.models.load_model(os.path.join(opt_dir, 'model_stage2_C2_Cond_optimized.h5'), compile=False),
            'C2_Reb': tf.keras.models.load_model(os.path.join(opt_dir, 'model_stage2_C2_Reb_optimized.h5'), compile=False)
        }
        
        # 3. 反應器階段模型
        m['s_r1'] = pickle.load(open(os.path.join(model_dir, 'x_scaler.pkl'), 'rb'))
        m['s_r1_y'] = pickle.load(open(os.path.join(model_dir, 'y_scalers.pkl'), 'rb'))
        m['mod_r_ene'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_Heater_Energy_Consumption_kW.h5'), compile=False)
        m['mod_r_pma'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_Reactor_PMA_Flow_kmol_h.h5'), compile=False)
        m['mod_r_aa'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_Reactor_AA_Flow_kmol_h.h5'), compile=False)
        m['mod_r_pgme'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_Reactor_PGME_Flow_kmol_h.h5'), compile=False)
        
    except Exception as e:
        st.error(f"加載模型出錯: {e}"); st.stop()
    return m

with st.sidebar:
    st.header("⚙️ 操作參數輸入")
    with st.expander("🔹 反應器階段", expanded=True):
        T = st.number_input("加熱器出口溫度 (°C)", 70.0, 110.0, 92.64, step=0.01)
        Fin = st.number_input("總進料流量 (kmol/h)", 1.0, 300.0, 80.17, step=0.1)
        Raa = st.number_input("進料比 (AA 分率)", 0.0, 1.0, 0.535, step=0.001, format="%.3f")
    with st.expander("🔹 蒸餾塔 C1", expanded=True):
        C1_R = st.number_input("C1 回流比", 0.1, 30.0, 1.59)
        C1_B = st.number_input("C1 蒸氣比", 0.1, 30.0, 7.76)
        C1_P = st.number_input("C1 壓力 (N/m²)", 1000.0, 50000.0, 9187.0)
    with st.expander("🔹 蒸餾塔 C2", expanded=True):
        C2_R = st.number_input("C2 回流比", 0.1, 30.0, 8.46)
        C2_B = st.number_input("C2 蒸氣比", 0.1, 30.0, 3.41)
        C2_P = st.number_input("C2 壓力 (N/m²)", 1000.0, 50000.0, 7954.0)

try:
    M = load_models()
    # 反應器預測
    r_in_s = M['s_r1'].transform(pd.DataFrame([[T, Fin, Raa, 1-Raa]], columns=['Temperature (°C)', 'Total Feed Molar Flow (kmol/h)', 'Feed Molar Fraction (AA)', 'Feed Molar Fraction (PGME)']))
    h_ene = M['s_r1_y']['Heater Energy Consumption (kW)'].inverse_transform(M['mod_r_ene'].predict(r_in_s, verbose=0))[0,0]
    r_pma_out = np.expm1(M['s_r1_y']['Reactor PMA Flow (kmol/h)'].inverse_transform(M['mod_r_pma'].predict(r_in_s, verbose=0)))[0,0]
    r_aa_out = np.expm1(M['s_r1_y']['Reactor AA Flow (kmol/h)'].inverse_transform(M['mod_r_aa'].predict(r_in_s, verbose=0)))[0,0]
    r_pg_out = np.expm1(M['s_r1_y']['Reactor PGME Flow (kmol/h)'].inverse_transform(M['mod_r_pgme'].predict(r_in_s, verbose=0)))[0,0]
    
    aa_in, pg_in = Fin*Raa, Fin*(1-Raa)
    aa_conv = (aa_in - r_aa_out) / (aa_in + 1e-9) * 100
    lim_reagent = "AA (醋酸)" if aa_in < pg_in else "PGME (丙二醇甲醚)"

    # 分離塔物理特徵
    C1_Load = r_pma_out * (C1_R + 1.0)
    C2_Load = r_pma_out * (C2_R + 1.0)
    C1_Vap = r_pma_out * C1_B
    C2_Vap = r_pma_out * C2_B
    
    x_opt_in = pd.DataFrame([[r_aa_out, r_pg_out, r_pma_out, C1_R, C1_B, C1_P, C2_R, C2_B, C2_P, C1_Load, C2_Load, C1_Vap, C2_Vap]], 
                            columns=['R_AA', 'R_PGME', 'R_PMA', 'C1_R', 'C1_B', 'C1_P', 'C2_R', 'C2_B', 'C2_P', 'C1_Load', 'C2_Load', 'C1_Vap', 'C2_Vap'])
    x_opt_s = M['s_ene_opt_x'].transform(x_opt_in)

    # 產品流量與能耗預測 (使用校正後的 Log+StandardScaler 邏輯)
    s_y_fl = M['s_ene_opt_y']['Product_Flow']
    m_flow_raw = np.expm1(s_y_fl.inverse_transform(M['mod_fl'].predict(x_opt_s, verbose=0).reshape(-1, 1))[0,0])
    m_flow = min(m_flow_raw, r_pma_out * 0.99)
    
    total_sep, ene_vals = 0, {}
    for t in ['C1_Cond', 'C1_Reb', 'C2_Cond', 'C2_Reb']:
        s_y_ene = M['s_ene_opt_y'][t]
        pred_val = M['mod_d_ene'][t].predict(x_opt_s, verbose=0)[0,0]
        v = np.expm1(s_y_ene.inverse_transform([[pred_val]])[0,0])
        v_abs = abs(v)
        ene_vals[t] = v_abs; total_sep += v_abs

    # 品質預測 (AA PPM & Purity)
    x_in_aa = pd.DataFrame([[T, Fin, Raa, C1_R, C1_B, C1_P, C2_R, C2_B, C2_P]], 
                           columns=['T', 'Flow_In', 'Ratio_AA', 'C1_R', 'C1_B', 'C1_P', 'C2_R', 'C2_B', 'C2_P'])
    aa_ppm = np.expm1(M['s_aa']['y_s'].inverse_transform(M['mod_aa'].predict(M['s_aa']['x_s'].transform(x_in_aa), verbose=0)))[0,0]
    
    x_pu_in = x_opt_in[['R_AA', 'R_PGME', 'R_PMA', 'C1_R', 'C1_B', 'C1_P', 'C2_R', 'C2_B', 'C2_P']]
    p_log_res = M['s_pu']['y_scaler'].inverse_transform(M['mod_pu'].predict(M['s_pu']['x_scaler'].transform(x_pu_in), verbose=0))[0,0]
    purity = np.clip(100.0001 - (10**p_log_res), 0, 100)

    total_sys_ene = h_ene + total_sep
    limiting_in_mol = min(aa_in, pg_in)
    total_yield = (m_flow / (limiting_in_mol + 1e-9)) * 100
    
    # UI 主面版
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("✨ 預測純度", f"{purity:.4f} %"); col1.progress(float(np.clip(purity/100, 0.0, 1.0)))
    with col2:
        st.metric("📈 總產率", f"{total_yield:.2f} %")
        with st.popover("🧮 計算方式"):
            st.write(f"限量試劑進料: {limiting_in_mol:.4f} kmol/h")
            st.latex(r"Yield = \frac{\text{Product Flow}}{\min(\text{AA, PGME})} \times 100\%")
    col3.metric("📦 質量流率", f"{m_flow*MW_PMA:.2f} kg/h", f"{m_flow:.4f} kmol/h")
    col4.metric("🧪 AA 含量", f"{aa_ppm:.2f} ppm")
    col5.metric("⚡ 系統總能耗", f"{total_sys_ene:.2f} kW")

    st.write("---")
    t1, t2, t3 = st.tabs(["📌 反應器詳情", "📌 分離塔詳情", "🏆 最佳優化方案 (校正版)"])
    with t1:
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("#### **組分流量 (kmol/h)**")
            st.table(pd.DataFrame({
                "組分": ["AA", "PGME", "PMA", "H2O"], 
                "反應器進料": [f"{aa_in:.4f}", f"{pg_in:.4f}", "0.0000", "0.0000"], 
                "反應器出口": [f"{r_aa_out:.4f}", f"{r_pg_out:.4f}", f"{r_pma_out:.4f}", f"{r_pma_out:.4f}"]
            }))
        with cc2:
            st.info(f"**限量試劑:** {lim_reagent}"); st.info(f"**AA 轉化率:** {aa_conv:.2f} %"); st.info(f"**Heater 能耗:** {h_ene:.2f} kW")
    with t2:
        cc3, cc4 = st.columns(2)
        with cc3:
            st.markdown("#### **分離能耗詳情 (kW)**")
            st.table(pd.DataFrame({
                "項目": ["C1 Condenser", "C1 Reboiler", "C2 Condenser", "C2 Reboiler"], 
                "預測負荷 (kW)": [f"{ene_vals['C1_Cond']:.2f}", f"{ene_vals['C1_Reb']:.2f}", f"{ene_vals['C2_Cond']:.2f}", f"{ene_vals['C2_Reb']:.2f}"]
            }))
        with cc4:
            st.markdown("#### **物理載荷 (Physics Load)**")
            st.write(f"C1 處理負荷: {C1_Load:.2f}"); st.write(f"C2 處理負荷: {C2_Load:.2f}")
            st.success(f"**單位能耗:** {total_sys_ene/(m_flow*MW_PMA+1e-9):.4f} kW/kg")
    with t3:
        st.markdown("#### **校正後 [256, 128, 64] 架構最佳建議方案**")
        st.table(pd.DataFrame({
            "指標": ["反應溫度", "進料流量", "AA 進料比", "C1 R/B/P", "C2 R/B/P", "純度 (%)", "AA (ppm)", "產品流量 (kmol/h)", "總能耗 (kW)", "產率 (%)"],
            "Case 1 (節能優先)": ["85.49", "24.49", "0.473", "1.55/9.72/9720", "8.90/3.29/8113", "99.9919", "34.4", "2.42", "1328.7", "20.90"],
            "Case 2 (產能優先)": ["109.84", "184.22", "0.548", "2.65/9.61/11284", "8.20/7.54/10755", "99.9904", "18.5", "22.86", "16112.5", "27.45"],
            "Case 3 (綜合平衡)": ["92.64", "80.17", "0.535", "1.59/7.76/9187", "8.46/3.41/7954", "99.9912", "40.3", "13.97", "5143.4", "37.47"]
        }))

except Exception as e:
    st.error(f"運行錯誤: {e}")
st.caption("PMA AI Optimization System © 2026 | 模型架構 [256, 128, 64] | 物理載荷校正版")
