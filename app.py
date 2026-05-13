import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
import tensorflow as tf

# ==============================================================================
# 0. 系統設定與基礎資訊
# ==============================================================================
st.set_page_config(page_title="ANN 電子級 PMA 優化系統", layout="wide")

# Google Analytics 追蹤
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

# ==============================================================================
# 1. 首頁平台說明與學術聲明
# ==============================================================================
st.title("人工類神經網路（ANN）應用於電子級 PMA 製程之產率預測與參數優化")
st.markdown("""
本平台結合 **DWSIM 模擬資料**與**雙階段人工神經網路模型**，可快速預測電子級 PMA 製程於不同操作條件下之產品流量、產品純度、AA 殘留量與能耗表現。本系統旨在提供化學工程師作為候選操作條件之初步篩選與代表性操作模式比較之決策輔助工具。
""")

# 檢查並顯示製程流程圖
if os.path.exists("電子級PMA 製程流程圖.png"):
    st.image("電子級PMA 製程流程圖.png", caption="電子級 PMA 生產製程流程圖", use_container_width=True)

st.info("""
**📖 模型適用範圍與限制：**
本預測模型係基於本研究 DWSIM 模擬數據集訓練而成。若輸入條件超出以下建議之模型訓練範圍，預測結果可能產生偏差，建議僅供初步工程評估，最終仍需透過模擬器或實驗驗證。
""")

# ==============================================================================
# 2. 模型載入與常數定義
# ==============================================================================
MW = {'AA': 60.05, 'PGME': 90.12, 'PMA': 132.16, 'Water': 18.02}

@st.cache_resource
def load_models():
    model_dir = 'deploy_models'
    m = {}
    if not os.path.exists(model_dir):
        st.error(f"❌ 錯誤: 找不到 '{model_dir}' 資料夾，請確認已上傳至 GitHub。")
        st.stop()
        
    try:
        # 品質與能耗模型
        m['s_pu'] = pickle.load(open(os.path.join(model_dir, 'purity_scalers.pkl'), 'rb'))
        m['mod_pu'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_purity_expert_log.h5'), compile=False)
        m['s_aa'] = pickle.load(open(os.path.join(model_dir, 'aa_ppm_scalers.pkl'), 'rb'))
        m['mod_aa'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_aa_ppm_hifi.h5'), compile=False)
        m['s_ene_opt_y'] = pickle.load(open(os.path.join(model_dir, 'y2_scalers_optimized.pkl'), 'rb'))
        m['s_ene_opt_x'] = pickle.load(open(os.path.join(model_dir, 'x2_scaler_optimized.pkl'), 'rb'))
        m['mod_fl'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_stage2_Product_Flow_optimized.h5'), compile=False)
        m['mod_d_ene'] = {
            'C1_Cond': tf.keras.models.load_model(os.path.join(model_dir, 'model_stage2_C1_Cond_optimized.h5'), compile=False),
            'C1_Reb': tf.keras.models.load_model(os.path.join(model_dir, 'model_stage2_C1_Reb_optimized.h5'), compile=False),
            'C2_Cond': tf.keras.models.load_model(os.path.join(model_dir, 'model_stage2_C2_Cond_optimized.h5'), compile=False),
            'C2_Reb': tf.keras.models.load_model(os.path.join(model_dir, 'model_stage2_C2_Reb_optimized.h5'), compile=False)
        }
        # 反應器模型
        m['s_r1'] = pickle.load(open(os.path.join(model_dir, 'x_scaler.pkl'), 'rb'))
        m['s_r1_y'] = pickle.load(open(os.path.join(model_dir, 'y_scalers.pkl'), 'rb'))
        m['mod_r_ene'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_Heater_Energy_Consumption_kW.h5'), compile=False)
        m['mod_r_pma'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_Reactor_PMA_Flow_kmol_h.h5'), compile=False)
        m['mod_r_aa'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_Reactor_AA_Flow_kmol_h.h5'), compile=False)
        m['mod_r_pgme'] = tf.keras.models.load_model(os.path.join(model_dir, 'model_Reactor_PGME_Flow_kmol_h.h5'), compile=False)
    except Exception as e:
        st.error(f"模型載入失敗: {e}"); st.stop()
    return m

# ==============================================================================
# 3. 輸入欄位管理
# ==============================================================================
with st.sidebar:
    st.header("🛠️ 操作參數設定")
    
    with st.expander("🔹 A. 反應段條件", expanded=True):
        T = st.number_input("Reaction Temperature (°C)", 50.0, 150.0, 92.64, help="訓練範圍: 70–110 °C")
        Fin = st.number_input("Total Feed Molar Flow (kmol/h)", 1.0, 300.0, 80.17, help="訓練範圍: 10–200 kmol/h")
        Raa = st.number_input("Feed Molar Fraction (AA)", 0.0, 1.0, 0.535, step=0.001, format="%.3f", help="建議範圍: 0.45–0.55")
        Rpg = 1.0 - Raa
        st.caption(f"Feed Molar Fraction (PGME): **{Rpg:.3f}**")

    with st.expander("🔹 B. 第一塔條件 (C1)", expanded=True):
        C1_R = st.number_input("C1 Reflux Ratio", 0.1, 20.0, 1.59, help="訓練範圍: 1.0–10.0")
        C1_B = st.number_input("C1 Vapor Rise Ratio", 0.1, 20.0, 7.76, help="訓練範圍: 1.0–10.0")
        C1_P = st.number_input("C1 Pressure (Pa)", 1000.0, 20000.0, 9187.0, help="訓練範圍: 7000–12000 Pa")

    with st.expander("🔹 C. 第二塔條件 (C2)", expanded=True):
        C2_R = st.number_input("C2 Reflux Ratio", 0.1, 20.0, 8.46, help="訓練範圍: 1.0–10.0")
        C2_B = st.number_input("C2 Vapor Rise Ratio", 0.1, 20.0, 3.41, help="訓練範圍: 1.0–10.0")
        C2_P = st.number_input("C2 Pressure (Pa)", 1000.0, 20000.0, 7954.0, help="訓練範圍: 7000–12000 Pa")

# 範圍警告
is_out = (not (70 <= T <= 110) or not (10 <= Fin <= 200) or not (0.45 <= Raa <= 0.55) or
          not (1 <= C1_R <= 10) or not (1 <= C1_B <= 10) or not (7000 <= C1_P <= 12000) or
          not (1 <= C2_R <= 10) or not (1 <= C2_B <= 10) or not (7000 <= C2_P <= 12000))
if is_out:
    st.warning("⚠️ **警告**：輸入條件已超出本研究 ANN 模型之核心訓練範圍。預測結果可能不具備可靠性。")

# ==============================================================================
# 4. 預測計算流程
# ==============================================================================
try:
    M = load_models()
    # [Reaction Stage]
    r_in_s = M['s_r1'].transform(pd.DataFrame([[T, Fin, Raa, Rpg]], columns=['Temperature (°C)', 'Total Feed Molar Flow (kmol/h)', 'Feed Molar Fraction (AA)', 'Feed Molar Fraction (PGME)']))
    h_ene = M['s_r1_y']['Heater Energy Consumption (kW)'].inverse_transform(M['mod_r_ene'].predict(r_in_s, verbose=0))[0,0]
    r_pma = np.expm1(M['s_r1_y']['Reactor PMA Flow (kmol/h)'].inverse_transform(M['mod_r_pma'].predict(r_in_s, verbose=0)))[0,0]
    r_aa = np.expm1(M['s_r1_y']['Reactor AA Flow (kmol/h)'].inverse_transform(M['mod_r_aa'].predict(r_in_s, verbose=0)))[0,0]
    r_pg = np.expm1(M['s_r1_y']['Reactor PGME Flow (kmol/h)'].inverse_transform(M['mod_r_pgme'].predict(r_in_s, verbose=0)))[0,0]
    
    # [Separation Stage]
    C1_L, C2_L = r_pma * (C1_R + 1.0), r_pma * (C2_R + 1.0)
    C1_V, C2_V = r_pma * C1_B, r_pma * C2_B
    x_opt_s = M['s_ene_opt_x'].transform(pd.DataFrame([[r_aa, r_pg, r_pma, C1_R, C1_B, C1_P, C2_R, C2_B, C2_P, C1_L, C2_L, C1_V, C2_V]], 
                                        columns=['R_AA', 'R_PGME', 'R_PMA', 'C1_R', 'C1_B', 'C1_P', 'C2_R', 'C2_B', 'C2_P', 'C1_Load', 'C2_Load', 'C1_Vap', 'C2_Vap']))

    m_flow = min(np.expm1(M['s_ene_opt_y']['Product_Flow'].inverse_transform(M['mod_fl'].predict(x_opt_s, verbose=0).reshape(-1, 1))[0,0]), r_pma * 0.99)
    ene_dict = {t: abs(np.expm1(M['s_ene_opt_y'][t].inverse_transform([[M['mod_d_ene'][t].predict(x_opt_s, verbose=0)[0,0]]])[0,0])) for t in ['C1_Cond', 'C1_Reb', 'C2_Cond', 'C2_Reb']}
    aa_ppm = np.expm1(M['s_aa']['y_s'].inverse_transform(M['mod_aa'].predict(M['s_aa']['x_s'].transform(pd.DataFrame([[T, Fin, Raa, C1_R, C1_B, C1_P, C2_R, C2_B, C2_P]], columns=['T', 'Flow_In', 'Ratio_AA', 'C1_R', 'C1_B', 'C1_P', 'C2_R', 'C2_B', 'C2_P'])), verbose=0)))[0,0]
    p_log_val = M['s_pu']['y_scaler'].inverse_transform(M['mod_pu'].predict(M['s_pu']['x_scaler'].transform(pd.DataFrame([[r_aa, r_pg, r_pma, C1_R, C1_B, C1_P, C2_R, C2_B, C2_P]], columns=['R_AA', 'R_PGME', 'R_PMA', 'C1_R', 'C1_B', 'C1_P', 'C2_R', 'C2_B', 'C2_P'])), verbose=0))[0,0]
    purity = np.clip(100.0001 - (10**p_log_val), 0, 100)

    # [Extra Calculation]
    total_yield = (m_flow / (min(Fin*Raa, Fin*Rpg) + 1e-9)) * 100
    sep_yield = (m_flow / (r_pma + 1e-9)) * 100
    total_energy = h_ene + sum(ene_dict.values())
    is_spec = (purity >= 99.99 and aa_ppm <= 100)

    # ==============================================================================
    # 5. 輸出結果呈現
    # ==============================================================================
    st.subheader("📋 系統預測結果分析")
    res_q, res_p, res_e = st.columns(3)
    with res_q:
        st.markdown("#### **A. 產品品質**")
        st.metric("PMA Product Purity", f"{purity:.4f} %")
        st.metric("AA Residue", f"{aa_ppm:.2f} ppm")
        if is_spec:
            st.success("✅ 符合電子級規格")
        else:
            st.error("❌ 未達電子級規格")

    with res_p:
        st.markdown("#### **B. 產能表現**")
        st.metric("Product Flow (Molar)", f"{m_flow:.4f} kmol/h")
        st.metric("Product Flow (Mass)", f"{m_flow*MW['PMA']:.2f} kg/h")
        st.metric("Total Overall Yield", f"{total_yield:.2f} %")

    with res_e:
        st.markdown("#### **C. 能耗表現**")
        st.metric("Total Energy Duty", f"{total_energy:.2f} kW")
        st.write(f"• **HT-1 Heater:** {h_ene:.2f} kW")
        st.write(f"• **C1 Duty:** {ene_dict['C1_Cond']+ene_dict['C1_Reb']:.2f} kW")
        st.write(f"• **C2 Duty:** {ene_dict['C2_Cond']+ene_dict['C2_Reb']:.2f} kW")

    st.divider()
    tabs = st.tabs(["💡 推薦操作模式", "📊 反應器詳情", "🏗️ 分離塔(蒸餾)詳情"])
    
    with tabs[0]:
        st.markdown("#### **搜尋 5000 組後之代表性操作模式比較**")
        st.table(pd.DataFrame({
            "指標": ["反應溫度 (°C)", "進料流量 (kmol/h)", "AA 進料比", "C1 R/B/P", "C2 R/B/P", "純度 (%)", "AA (ppm)", "產品流量 (kmol/h)", "總能耗 (kW)", "產率 (%)"],
            "Case 1 (節能優先)": ["85.49", "24.49", "0.473", "1.55/9.72/9720", "8.90/3.29/8113", "99.9919", "34.4", "2.42", "1328.7", "20.90"],
            "Case 2 (產能優先)": ["109.84", "184.22", "0.548", "2.65/9.61/11284", "8.20/7.54/10755", "99.9904", "18.5", "22.86", "16112.5", "27.45"],
            "Case 3 (綜合平衡)": ["92.64", "80.17", "0.535", "1.59/7.76/9187", "8.46/3.41/7954", "99.9912", "40.3", "13.97", "5143.4", "37.47"]
        }))

    with tabs[1]:
        st.markdown("#### **反應器反應效能分析**")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("⚖️ 限量試劑", lim_reagent)
        cc2.metric("📉 AA 轉化率", f"{aa_conv:.2f} %")
        cc3.metric("📉 PGME 轉化率", f"{pgme_conv:.2f} %")
        
        st.markdown("---")
        st.markdown("#### **反應器組分流率詳細分析 (Outlet)**")
        st.table(pd.DataFrame({
            "組分 (Component)": ["AA (醋酸)", "PGME (丙二醇甲醚)", "PMA (產品)", "Water (副產物)"],
            "莫耳流量 (kmol/h)": [f"{r_aa:.4f}", f"{r_pg:.4f}", f"{r_pma:.4f}", f"{r_pma:.4f}"],
            "質量流率 (kg/h)": [f"{r_aa*MW['AA']:.2f}", f"{r_pg*MW['PGME']:.2f}", f"{r_pma*MW['PMA']:.2f}", f"{r_pma*MW['Water']:.2f}"]
        }))

    with tabs[2]:
        st.markdown("#### **分離塔(蒸餾)階段質量平衡與回收率**")
        st.table(pd.DataFrame({
            "指標項目": ["最終產品莫耳流量 (kmol/h)", "最終產品質量流率 (kg/h)", "分離階段產率 (Separation Yield %)", "C1 冷凝器負荷 (kW)", "C1 再沸器負荷 (kW)", "C2 冷凝器負荷 (kW)", "C2 再沸器負荷 (kW)"],
            "預測數值": [f"{m_flow:.4f}", f"{m_flow*MW['PMA']:.2f}", f"{sep_yield:.2f}%", f"{ene_dict['C1_Cond']:.2f}", f"{ene_dict['C1_Reb']:.2f}", f"{ene_dict['C2_Cond']:.2f}", f"{ene_dict['C2_Reb']:.2f}"]
        }))
        with st.popover("📖 分離產率定義"):
            st.latex(r"Separation Yield = \frac{n_{PMA, product}}{n_{PMA, reactor\_out}} \times 100\%")
            st.write(f"- 分子 (最終產品): {m_flow:.4f} kmol/h")
            st.write(f"- 分母 (反應器產出): {r_pma:.4f} kmol/h")

except Exception as e:
    st.error(f"系統運行錯誤：{e}")

st.divider()
st.caption("PMA AI Optimization System (c) 2026 | 碩士論文研究成果展示 | 模型架構：[256, 128, 64] | 數據來源：DWSIM")
