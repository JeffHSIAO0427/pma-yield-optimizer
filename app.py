import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
import tensorflow as tf

# 設定網頁標題與版面
st.set_page_config(page_title="ANN 電子級 PMA 優化系統", layout="wide")

st.title("人工類神經網路（ANN）應用於電子級 PMA 製程之產率預測與參數優化")

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
        st.error(f"找不到模型資料夾: {model_dir}。請確認 GitHub 上資料夾名稱已改為 models_v5。")
        st.stop()
    
    m = {}
    try:
        # 品質與流量
        m['s_pu'] = pickle.load(open(os.path.join(model_dir, 'purity_scalers.pkl'), 'rb'))
        m['mod_pu'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_purity_expert_log.h5'), compile=False)
        m['s_fl'] = pickle.load(open(os.path.join(model_dir, 'flow_master_scalers.pkl'), 'rb'))
        m['mod_fl'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_flow_master_final.h5'), compile=False)
        m['s_aa'] = pickle.load(open(os.path.join(model_dir, 'aa_ppm_scalers.pkl'), 'rb'))
        m['mod_aa'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_aa_ppm_hifi.h5'), compile=False)
        
        # 反應器
        m['s_r1'] = pickle.load(open(os.path.join(model_dir, 'x_scaler.pkl'), 'rb'))
        m['s_r1_y'] = pickle.load(open(os.path.join(model_dir, 'y_scalers.pkl'), 'rb'))
        m['mod_r_ene'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_Heater_Energy_Consumption_kW.h5'), compile=False)
        m['mod_r_pma'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_Reactor_PMA_Flow_kmol_h.h5'), compile=False)
        m['mod_r_aa'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_Reactor_AA_Flow_kmol_h.h5'), compile=False)
        m['mod_r_pgme'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_Reactor_PGME_Flow_kmol_h.h5'), compile=False)
        
        # 分離能耗 (v5 高精度)
        m['s_ene_hifi'] = pickle.load(open(os.path.join(model_dir, 'energy_hifi_scalers.pkl'), 'rb'))
        m['mod_d_ene'] = {
            'C1_Cond': tf.keras.models.load_model(os.path.join(model_dir, 'model_stage2_C1_Cond_optimized.h5'), compile=False),
            'C1_Reb': tf.keras.models.load_model(os.path.join(model_dir, 'model_stage2_C1_Reb_optimized.h5'), compile=False),
            'C2_Cond': tf.keras.models.load_model(os.path.join(model_dir, 'model_stage2_C2_Cond_optimized.h5'), compile=False),
            'C2_Reb': tf.keras.models.load_model(os.path.join(model_dir, 'model_stage2_C2_Reb_optimized.h5'), compile=False)
        }
    except Exception as e:
        st.error(f"加載模型檔案時發生錯誤: {e}")
        st.stop()
    return m

# =========================================================
# 2. 側邊欄控制
# =========================================================
with st.sidebar:
    st.header("⚙️ 操作參數輸入")
    with st.expander("🔹 反應器階段", expanded=True):
        T = st.number_input("加熱器出口溫度 (°C)", 70.0, 110.0, 109.41, step=0.01)
        Fin = st.number_input("總進料流量 (kmol/h)", 1.0, 300.0, 72.80, step=0.1)
        Raa = st.number_input("進料比 (AA 分率)", 0.0, 1.0, 0.470, step=0.001, format="%.3f")
    
    with st.expander("🔹 蒸餾塔 C1", expanded=True):
        C1_R = st.number_input("C1 回流比 (R1)", 0.1, 30.0, 5.64)
        C1_B = st.number_input("C1 蒸氣比 (B1)", 0.1, 30.0, 9.13)
        C1_P = st.number_input("C1 壓力 (N/m2)", 1000.0, 50000.0, 9629.0)
    
    with st.expander("🔹 蒸餾塔 C2", expanded=True):
        C2_R = st.number_input("C2 回流比 (R2)", 0.1, 30.0, 8.44)
        C2_B = st.number_input("C2 蒸氣比 (B2)", 0.1, 30.0, 7.97)
        C2_P = st.number_input("C2 壓力 (N/m2)", 1000.0, 50000.0, 7520.0)

# =========================================================
# 3. 預測核心
# =========================================================
try:
    M = load_models()
    
    # --- 1. 反應器預測 ---
    r_in_s = M['s_r1'].transform([[T, Fin, Raa, 1-Raa]])
    h_ene = M['s_r1_y']['Heater Energy Consumption (kW)'].inverse_transform(M['mod_r_ene'].predict(r_in_s, verbose=0))[0,0]
    r_aa_out = np.expm1(M['s_r1_y']['Reactor AA Flow (kmol/h)'].inverse_transform(M['mod_r_aa'].predict(r_in_s, verbose=0)))[0,0]
    r_pg_out = np.expm1(M['s_r1_y']['Reactor PGME Flow (kmol/h)'].inverse_transform(M['mod_r_pgme'].predict(r_in_s, verbose=0)))[0,0]
    r_pma_out = np.expm1(M['s_r1_y']['Reactor PMA Flow (kmol/h)'].inverse_transform(M['mod_r_pma'].predict(r_in_s, verbose=0)))[0,0]
    
    aa_in_flow = Fin * Raa
    aa_conv = (aa_in_flow - r_aa_out) / (aa_in_flow + 1e-9) * 100
    
    # --- 2. 分離塔預測 ---
    x_distill = [C1_R, C1_B, C1_P, C2_R, C2_B, C2_P]
    x_df_e2e = pd.DataFrame([[T, Fin, Raa] + x_distill], columns=['T', 'Flow_In', 'Ratio_AA', 'C1_R', 'C1_B', 'C1_P', 'C2_R', 'C2_B', 'C2_P'])
    
    # 產量與品質
    m_flow = np.expm1(M['s_fl']['y_s'].inverse_transform(M['mod_fl'].predict(M['s_fl']['x_s'].transform(x_df_e2e), verbose=0)))[0,0]
    aa_ppm = np.expm1(M['s_aa']['y_s'].inverse_transform(M['mod_aa'].predict(M['s_aa']['x_s'].transform(x_df_e2e), verbose=0)))[0,0]
    
    # 純度與能耗 (串聯)
    x_pu_in = pd.DataFrame([[r_aa_out, r_pg_out, r_pma_out] + x_distill], columns=['R_AA', 'R_PGME', 'R_PMA', 'C1_R', 'C1_B', 'C1_P', 'C2_R', 'C2_B', 'C2_P'])
    p_log = M['s_pu']['y_scaler'].inverse_transform(M['mod_pu'].predict(M['s_pu']['x_scaler'].transform(x_pu_in), verbose=0))[0,0]
    purity = np.clip(100.0001 - (10**p_log), 0, 100)
    
    # 能耗詳情 (v5)
    total_sep = 0
    ene_vals = {}
    for t in ['C1_Cond', 'C1_Reb', 'C2_Cond', 'C2_Reb']:
        sx = M['s_ene_hifi'][t]['sx']
        sy = M['s_ene_hifi'][t]['sy']
        v = sy.inverse_transform(M['mod_d_ene'][t].predict(sx.transform(x_pu_in), verbose=0))[0,0]
        ene_vals[t] = abs(v)
        total_sep += abs(v)
    
    # 總產率與總能耗
    total_yield = (m_flow / (aa_in_flow + 1e-9)) * 100
    total_sys_ene = h_ene + total_sep

    # =========================================================
    # 4. UI 渲染
    # =========================================================
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("✨ 預測純度", f"{purity:.4f} %"); col1.progress(float(np.clip(purity/100, 0.0, 1.0)))
    col2.metric("📈 總產率", f"{total_yield:.2f} %")
    col3.metric("📦 質量流率", f"{m_flow*MW_PMA:.2f} kg/h", f"{m_flow:.4f} kmol/h")
    col4.metric("🧪 AA 含量", f"{aa_ppm:.2f} ppm")
    col5.metric("⚡ 系統總能耗", f"{total_sys_ene:.2f} kW")

    st.write("---")
    t1, t2, t3 = st.tabs(["📌 反應器詳情", "📌 分離塔詳情", "🏆 最佳優化方案 (v5)"])
    
    with t1:
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("#### **物料衡算 (kmol/h)**")
            st.table(pd.DataFrame({
                "組分": ["AA (醋酸)", "PGME (丙二醇甲醚)", "PMA (產品)", "H2O (水)"],
                "進料": [f"{aa_in_flow:.4f}", f"{Fin-aa_in_flow:.4f}", "0.0000", "0.0000"],
                "反應器出口": [f"{r_aa_out:.4f}", f"{r_pg_out:.4f}", f"{r_pma_out:.4f}", f"{r_pma_out:.4f}"]
            }))
        with cc2:
            st.markdown("#### **關鍵指標**")
            st.info(f"**AA 轉化率:** {aa_conv:.2f} %")
            st.info(f"**Heater 能耗:** {h_ene:.2f} kW")

    with t2:
        cc3, cc4 = st.columns(2)
        with cc3:
            st.markdown("#### **分離能耗詳情 (kW)**")
            st.table(pd.DataFrame({
                "換熱器": ["C1 Condenser", "C1 Reboiler", "C2 Condenser", "C2 Reboiler"],
                "負荷 (kW)": [f"{ene_vals['C1_Cond']:.2f}", f"{ene_vals['C1_Reb']:.2f}", f"{ene_vals['C2_Cond']:.2f}", f"{ene_vals['C2_Reb']:.2f}"]
            }))
        with cc4:
            st.markdown("#### **最終產品與能耗**")
            st.success(f"**PMA 質量流率:** {m_flow*MW_PMA:.2f} kg/h")
            st.success(f"**最終產品純度:** {purity:.4f} %")
            st.success(f"**系統總耗能:** {total_sys_ene:.2f} kW")

    with t3:
        st.markdown("#### **根據 30 萬組全域搜尋獲得的最佳建議方案 (純度 ≥ 99.99%)**")
        st.table(pd.DataFrame({
            "指標": ["溫度 (°C)", "流量 (kmol/h)", "進料比_AA", "R1/B1/P1", "R2/B2/P2", "預測純度 (%)", "AA 含量 (ppm)", "PMA 摩爾流量 (kmol/h)", "質量流率 (kg/h)", "總產率 (%)", "總能耗 (kW)"],
            "Case 1 (節能)": ["97.42", "14.93", "0.489", "5.06/9.73/7237", "7.28/8.56/7247", "99.9902", "72.16", "2.0259", "267.75", "27.75", "1250.52"],
            "Case 2 (產能)": ["110.00", "105.91", "0.484", "6.33/9.66/8484", "5.10/8.70/8154", "99.9913", "67.63", "20.7866", "2747.16", "40.55", "13877.90"],
            "Case 3 (平衡)": ["109.41", "72.80", "0.470", "5.64/9.13/9629", "8.44/7.97/7520", "99.9922", "48.63", "16.9144", "2235.41", "49.43", "9569.12"]
        }))

except Exception as e:
    st.error(f"系統運行錯誤: {e}")

st.markdown("---")
st.caption("PMA AI Production Optimization System © 2023 | 數據由 DWSIM 模擬與 集成專家模型 驅動")
