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
st.write("本平台為碩士論文研究開發之電子級 PMA 製程預測與操作條件分析系統")

# 檢查圖片是否存在
if os.path.exists("電子級PMA 製程流程圖.png"):
    st.image("電子級PMA 製程流程圖.png", caption="電子級 PMA 生產製程流程圖", use_container_width=True)

# 分子量常數
MW = {
    'AA': 60.05,
    'PGME': 90.12,
    'PMA': 132.16,
    'Water': 18.02
}

@st.cache_resource
def load_models():
    model_dir = 'deploy_models'
    m = {}
    # 檢查資料夾是否存在
    if not os.path.exists(model_dir):
        st.error(f"❌ 錯誤: 找不到 '{model_dir}' 資料夾，請確認已上傳至 GitHub。")
        st.stop()
        
    try:
        # 1. 品質模型
        m['s_pu'] = pickle.load(open(os.path.join(model_dir, 'purity_scalers.pkl'), 'rb'))
        m['mod_pu'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_purity_expert_log.h5'), compile=False)
        m['s_aa'] = pickle.load(open(os.path.join(model_dir, 'aa_ppm_scalers.pkl'), 'rb'))
        m['mod_aa'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_aa_ppm_hifi.h5'), compile=False)
        
        # 2. 最新校正後的產品流量與能耗模型
        m['s_ene_opt_y'] = pickle.load(open(os.path.join(model_dir, 'y2_scalers_optimized.pkl'), 'rb'))
        m['s_ene_opt_x'] = pickle.load(open(os.path.join(model_dir, 'x2_scaler_optimized.pkl'), 'rb'))
        m['mod_fl'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_stage2_Product_Flow_optimized.h5'), compile=False)
        m['mod_d_ene'] = {
            'C1_Cond': tf.keras.models.load_model(os.path.join(model_dir, 'model_stage2_C1_Cond_optimized.h5'), compile=False),
            'C1_Reb': tf.keras.models.load_model(os.path.join(model_dir, 'model_stage2_C1_Reb_optimized.h5'), compile=False),
            'C2_Cond': tf.keras.models.load_model(os.path.join(model_dir, 'model_stage2_C2_Cond_optimized.h5'), compile=False),
            'C2_Reb': tf.keras.models.load_model(os.path.join(model_dir, 'model_stage2_C2_Reb_optimized.h5'), compile=False)
        }
        
        # 3. 反應器模型
        m['s_r1'] = pickle.load(open(os.path.join(model_dir, 'x_scaler.pkl'), 'rb'))
        m['s_r1_y'] = pickle.load(open(os.path.join(model_dir, 'y_scalers.pkl'), 'rb'))
        m['mod_r_ene'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_Heater_Energy_Consumption_kW.h5'), compile=False)
        m['mod_r_pma'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_Reactor_PMA_Flow_kmol_h.h5'), compile=False)
        m['mod_r_aa'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_Reactor_AA_Flow_kmol_h.h5'), compile=False)
        m['mod_r_pgme'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_Reactor_PGME_Flow_kmol_h.h5'), compile=False)
        
    except Exception as e:
        st.error(f"❌ 模型檔案載入失敗: {e}")
        st.write("請確認 deploy_models 資料夾內包含所有 17 個 .h5 與 .pkl 檔案。")
        st.stop()
    return m

# 側邊欄輸入
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
    r_in_data = pd.DataFrame([[T, Fin, Raa, 1-Raa]], 
                             columns=['Temperature (°C)', 'Total Feed Molar Flow (kmol/h)', 'Feed Molar Fraction (AA)', 'Feed Molar Fraction (PGME)'])
    r_in_s = M['s_r1'].transform(r_in_data)
    
    h_ene = M['s_r1_y']['Heater Energy Consumption (kW)'].inverse_transform(M['mod_r_ene'].predict(r_in_s, verbose=0))[0,0]
    r_pma_out = np.expm1(M['s_r1_y']['Reactor PMA Flow (kmol/h)'].inverse_transform(M['mod_r_pma'].predict(r_in_s, verbose=0)))[0,0]
    r_aa_out = np.expm1(M['s_r1_y']['Reactor AA Flow (kmol/h)'].inverse_transform(M['mod_r_aa'].predict(r_in_s, verbose=0)))[0,0]
    r_pg_out = np.expm1(M['s_r1_y']['Reactor PGME Flow (kmol/h)'].inverse_transform(M['mod_r_pgme'].predict(r_in_s, verbose=0)))[0,0]
    r_water_out = r_pma_out # 產率 1:1
    
    aa_in, pg_in = Fin*Raa, Fin*(1-Raa)
    aa_conv = (aa_in - r_aa_out) / (aa_in + 1e-9) * 100
    pgme_conv = (pg_in - r_pg_out) / (pg_in + 1e-9) * 100
    lim_reagent = "AA (醋酸)" if aa_in < pg_in else "PGME (丙二醇甲醚)"
    lim_in_mol = min(aa_in, pg_in)

    # 分離塔物理特徵
    C1_Load = r_pma_out * (C1_R + 1.0)
    C2_Load = r_pma_out * (C2_R + 1.0)
    C1_Vap = r_pma_out * C1_B
    C2_Vap = r_pma_out * C2_B
    
    x_opt_in = pd.DataFrame([[r_aa_out, r_pg_out, r_pma_out, C1_R, C1_B, C1_P, C2_R, C2_B, C2_P, C1_Load, C2_Load, C1_Vap, C2_Vap]], 
                            columns=['R_AA', 'R_PGME', 'R_PMA', 'C1_R', 'C1_B', 'C1_P', 'C2_R', 'C2_B', 'C2_P', 'C1_Load', 'C2_Load', 'C1_Vap', 'C2_Vap'])
    x_opt_s = M['s_ene_opt_x'].transform(x_opt_in)

    # 產品流量與能耗預測
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

    # 品質預測
    x_in_aa = pd.DataFrame([[T, Fin, Raa, C1_R, C1_B, C1_P, C2_R, C2_B, C2_P]], 
                           columns=['T', 'Flow_In', 'Ratio_AA', 'C1_R', 'C1_B', 'C1_P', 'C2_R', 'C2_B', 'C2_P'])
    aa_ppm = np.expm1(M['s_aa']['y_s'].inverse_transform(M['mod_aa'].predict(M['s_aa']['x_s'].transform(x_in_aa), verbose=0)))[0,0]
    
    x_pu_in = x_opt_in[['R_AA', 'R_PGME', 'R_PMA', 'C1_R', 'C1_B', 'C1_P', 'C2_R', 'C2_B', 'C2_P']]
    p_log_res = M['s_pu']['y_scaler'].inverse_transform(M['mod_pu'].predict(M['s_pu']['x_scaler'].transform(x_pu_in), verbose=0))[0,0]
    purity = np.clip(100.0001 - (10**p_log_res), 0, 100)

    total_sys_ene = h_ene + total_sep
    total_yield = (m_flow / (lim_in_mol + 1e-9)) * 100
    
    # UI 呈現 (Top Metrics)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("✨ 預測純度", f"{purity:.4f} %")
    with c2:
        st.metric("📈 總產率", f"{total_yield:.2f} %")
        with st.popover("📖 產率定義"):
            st.markdown("**總產率 (Overall Yield) 定義：**")
            st.latex(r"Yield = \frac{n_{PMA, product}}{n_{limiting, feed}} \times 100\%")
            st.write(f"- 限量試劑: {lim_reagent}")
            st.write(f"- 限量試劑進料: {lim_in_mol:.4f} kmol/h")
            st.write(f"- 最終產品流量: {m_flow:.4f} kmol/h")
    c3.metric("📦 產量 (kg/h)", f"{m_flow*MW['PMA']:.2f}")
    c4.metric("🧪 AA (ppm)", f"{aa_ppm:.2f}")
    c5.metric("⚡ 總能耗 (kW)", f"{total_sys_ene:.2f}")

    st.write("---")
    t1, t2, t3 = st.tabs(["📌 反應器詳情", "📌 分離塔詳情", "🏆 最佳優化方案"])
    
    with t1:
        st.markdown("#### **組分流量詳細分析 (kmol/h & kg/h)**")
        reactor_df = pd.DataFrame({
            "組分": ["AA (醋酸)", "PGME (丙二醇甲醚)", "PMA (產品)", "Water (水)"],
            "莫耳流量 (kmol/h)": [f"{r_aa_out:.4f}", f"{r_pg_out:.4f}", f"{r_pma_out:.4f}", f"{r_water_out:.4f}"],
            "質量流率 (kg/h)": [f"{r_aa_out*MW['AA']:.2f}", f"{r_pg_out*MW['PGME']:.2f}", f"{r_pma_out*MW['PMA']:.2f}", f"{r_water_out*MW['Water']:.2f}"]
        })
        st.table(reactor_df)
        st.markdown(f"**限量試劑:** {lim_reagent} | **加熱器能耗:** {h_ene:.2f} kW")
        cc1, cc2 = st.columns(2)
        cc1.info(f"**AA 轉化率:** {aa_conv:.2f} %")
        cc2.info(f"**PGME 轉化率:** {pgme_conv:.2f} %")

    with t2:
        st.markdown("#### **分離塔 Y 輸出指標與能耗詳情**")
        sep_col1, sep_col2 = st.columns(2)
        with sep_col1:
            st.write("**分離效能指標**")
            st.success(f"產品莫耳流量: {m_flow:.4f} kmol/h")
            st.success(f"產品質量流量: {m_flow*MW['PMA']:.2f} kg/h")
            st.success(f"最終產品純度: {purity:.4f} %")
            st.success(f"AA 殘留量: {aa_ppm:.2f} ppm")
        with sep_col2:
            st.write("**設備熱負荷詳情 (kW)**")
            st.table(pd.DataFrame({
                "設備名稱": ["C1 冷凝器", "C1 再沸器", "C2 冷凝器", "C2 再沸器"],
                "熱負荷 (kW)": [f"{ene_vals['C1_Cond']:.2f}", f"{ene_vals['C1_Reb']:.2f}", f"{ene_vals['C2_Cond']:.2f}", f"{ene_vals['C2_Reb']:.2f}"]
            }))

    with t3:
        st.markdown("#### **搜尋5000組後最佳優化方案**")
        st.table(pd.DataFrame({
            "指標": ["反應溫度", "進料流量", "AA 進料比", "C1 R/B/P", "C2 R/B/P", "純度 (%)", "AA (ppm)", "產品流量 (kmol/h)", "總能耗 (kW)", "產率 (%)"],
            "Case 1 (節能優先)": ["85.49", "24.49", "0.473", "1.55/9.72/9720", "8.90/3.29/8113", "99.9919", "34.4", "2.42", "1328.7", "20.90"],
            "Case 2 (產能優先)": ["109.84", "184.22", "0.548", "2.65/9.61/11284", "8.20/7.54/10755", "99.9904", "18.5", "22.86", "16112.5", "27.45"],
            "Case 3 (綜合平衡)": ["92.64", "80.17", "0.535", "1.59/7.76/9187", "8.46/3.41/7954", "99.9912", "40.3", "13.97", "5143.4", "37.47"]
        }))

except Exception as e:
    st.error(f"執行出錯: {e}")

st.caption("PMA AI Optimization System (c) 2026 | 碩士論文研究成果展示 | 模型架構 [256, 128, 64]")
