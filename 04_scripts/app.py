import streamlit as st
import joblib
import pandas as pd
import os 

# 設定網頁標題與圖示
st.set_page_config(page_title="真偽職缺偵測系統", page_icon="🕵️")

# 1. 載入模型 (使用自動路徑偵測)
@st.cache_resource
def load_model():
    # 取得目前 app.py 所在的絕對路徑 (即 04_scripts 資料夾)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 建立指向 02_models 的正確路徑
    # 邏輯：從 04_scripts 往上一層 (..)，進入 02_models，找到檔案
    model_path = os.path.join(current_dir, "..", "02_models", "xgb_model_precision.pkl")
    
    # 如果您想確認路徑是否正確，取消下面這行的註解：
    # st.write(f"正在嘗試載入模型：{model_path}")
    
    return joblib.load(model_path)

# 執行載入
try:
    model = load_model()
except Exception as e:
    st.error(f"❌ 模型載入失敗！請檢查檔案是否存在。")
    st.info(f"詳細錯誤訊息: {e}")

# 2. 側邊欄：輸入輔助資訊
st.sidebar.header("職缺輔助資訊")
has_logo = st.sidebar.selectbox("是否有公司 Logo?", ["否", "是"])
has_questions = st.sidebar.selectbox("是否有篩選問題?", ["否", "是"])

# 轉換為 0 或 1
logo_val = 1 if has_logo == "是" else 0
ques_val = 1 if has_questions == "是" else 0

# 3. 主界面
st.title("🕵️ 職缺真偽 AI 偵測系統")
st.write("輸入職缺描述，AI 將為您分析潛在風險。")

# 輸入框
job_text = st.text_area("請貼入職缺內容 (Description):", height=300, placeholder="例如：URGENT HIRE! Work from home...")

if st.button("開始偵測"):
    if job_text.strip() == "":
        st.warning("請先輸入內容再進行測試。")
    else:
        # 建立資料框架
        input_df = pd.DataFrame({
            'text': [job_text],
            'has_company_logo': [logo_val],
            'has_questions': [ques_val]
        })
        
        # 預測機率
        prob = model.predict_proba(input_df)[0][1]
        
        # 顯示結果
        st.subheader("分析報告")
        if prob >= 0.5:
            st.error(f"🚨 警示：這可能是一個【詐騙職缺】")
        else:
            st.success(f"✅ 判定：目前看來較為【安全】")
            
        st.write(f"系統計算的詐騙可能性：**{prob:.2%}**")