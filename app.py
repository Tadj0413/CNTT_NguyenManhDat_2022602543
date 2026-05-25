import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

st.set_page_config(page_title="Fraud Detection System", layout="wide")

# Session state khởi tạo ở đầu file
defaults = {
    'selected_model': 'rf',
    'selected_method': 'smote', 
    'probs': None,
    'X_data': None,
    'df_display': None,
    'true_labels': None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

@st.cache_resource
def load_model(model_name: str, method_name: str):
    name_map = {
        "rf": "random forest",
        "xgb": "xgboost",
        "lgbm": "lightgbm"
    }
    actual_model_name = name_map.get(model_name, model_name)
    path = f"models/model_{actual_model_name}_{method_name}.pkl"
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None

def validate_csv(df) -> tuple[bool, str]:
    required_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return False, f"Thiếu {len(missing)} cột: {missing[:5]}{'...' if len(missing)>5 else ''}"
    if len(df) == 0:
        return False, "File không có dòng dữ liệu nào"
    return True, "OK"

st.title("Hệ Thống Phát Hiện Gian Lận Thẻ Tín Dụng")

tab1, tab2 = st.tabs(["⚙️ Chọn Mô Hình", "📊 Phân Tích CSV"])

# --- TAB 1 ---
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        sel_m = st.selectbox(
            "Thuật Toán", ["rf", "xgb", "lgbm"], 
            index=["rf", "xgb", "lgbm"].index(st.session_state['selected_model']),
            format_func=lambda x: {"rf": "Random Forest", "xgb": "XGBoost", "lgbm": "LightGBM"}[x]
        )
    with col2:
        sel_b = st.selectbox(
            "Phương Pháp", ["smote", "adasyn"],
            index=["smote", "adasyn"].index(st.session_state['selected_method']),
            format_func=lambda x: x.upper()
        )
        
    # Reset predictions if model changes
    if sel_m != st.session_state['selected_model'] or sel_b != st.session_state['selected_method']:
        st.session_state['selected_model'] = sel_m
        st.session_state['selected_method'] = sel_b
        st.session_state['probs'] = None
        st.session_state['X_data'] = None
        st.session_state['df_display'] = None
        st.session_state['true_labels'] = None
    
    model = load_model(sel_m, sel_b)
    if model:
        st.success(f"Đã tải thành công mô hình: {sel_m.upper()} + {sel_b.upper()}")
    else:
        st.warning("Mô hình chưa sẵn sàng. Vui lòng kiểm tra thư mục models/.")
        
    st.markdown("### Bảng hiệu suất tổng hợp")
    if os.path.exists("models/performance_metrics.csv"):
        df_perf = pd.read_csv("models/performance_metrics.csv")
    else:
        df_perf = pd.DataFrame([
            {"model": "rf", "method": "smote", "model_label": "Random Forest", "method_label": "SMOTE", "precision": 0.9312, "recall": 0.8211, "f1": 0.8726, "pr_auc": 0.8534, "roc_auc": 0.9821, "train_time": 48},
            {"model": "rf", "method": "adasyn", "model_label": "Random Forest", "method_label": "ADASYN", "precision": 0.9401, "recall": 0.8374, "f1": 0.8858, "pr_auc": 0.8712, "roc_auc": 0.9847, "train_time": 52},
            {"model": "xgb", "method": "smote", "model_label": "XGBoost", "method_label": "SMOTE", "precision": 0.9228, "recall": 0.8455, "f1": 0.8825, "pr_auc": 0.8801, "roc_auc": 0.9863, "train_time": 22},
            {"model": "xgb", "method": "adasyn", "model_label": "XGBoost", "method_label": "ADASYN", "precision": 0.9344, "recall": 0.8577, "f1": 0.8944, "pr_auc": 0.8893, "roc_auc": 0.9871, "train_time": 25},
            {"model": "lgbm", "method": "smote", "model_label": "LightGBM", "method_label": "SMOTE", "precision": 0.9267, "recall": 0.8496, "f1": 0.8865, "pr_auc": 0.8821, "roc_auc": 0.9868, "train_time": 11},
            {"model": "lgbm", "method": "adasyn", "model_label": "LightGBM", "method_label": "ADASYN", "precision": 0.9389, "recall": 0.8618, "f1": 0.8987, "pr_auc": 0.8934, "roc_auc": 0.9879, "train_time": 13}
        ])
        
    # Tạo normalize column để dễ dàng so sánh
    df_perf['model_norm'] = df_perf['model'].str.lower().replace({'random forest': 'rf', 'xgboost': 'xgb', 'lightgbm': 'lgbm'})
    df_perf['method_norm'] = df_perf['method'].str.lower()
    
    # Đổi tên cột
    rename_dict = {
        'model_label': 'Mô hình',
        'method_label': 'Phương pháp',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1-Score',
        'pr_auc': 'PR-AUC',
        'roc_auc': 'ROC-AUC',
        'train_time': 'Train Time'
    }
    df_perf = df_perf.rename(columns=rename_dict)
    
    display_cols = ['Mô hình', 'Phương pháp', 'Precision', 'Recall', 'F1-Score', 'PR-AUC', 'ROC-AUC', 'Train Time', 'model_norm', 'method_norm']
    
    df_display_perf = df_perf[display_cols]
    
    def highlight_selected_row(row):
        if row['model_norm'] == sel_m and row['method_norm'] == sel_b:
            return ['background-color: lightblue; color: black'] * len(row)
        return [''] * len(row)
        
    styled_df = df_display_perf.style.apply(highlight_selected_row, axis=1)
    
    st.dataframe(
        styled_df, 
        use_container_width=True,
        column_config={
            "model_norm": None,
            "method_norm": None
        }
    )
    
    # Ưu/nhược điểm
    model_pros_cons = {
        "rf": (["Ổn định, ít overfit", "Feature importance rõ ràng", "Kháng nhiễu tốt"],
               ["Train chậm nhất", "PR-AUC thấp hơn boosting", "Bộ nhớ lớn"]),
        "xgb": (["F1 & PR-AUC cao nhất", "Built-in regularization", "Phổ biến trong industry"],
                ["Cần tuning nhiều", "Dễ overfit nếu lr cao", "Chậm hơn LightGBM"]),
        "lgbm": (["Nhanh nhất", "PR-AUC cao", "Tiết kiệm bộ nhớ"],
                 ["Dễ overfit data nhỏ", "num_leaves cần điều chỉnh", "Ít tài liệu tiếng Việt"])
    }
    pc_c1, pc_c2 = st.columns(2)
    pros, cons = model_pros_cons[sel_m]
    with pc_c1:
        st.markdown("**Ưu điểm:**\n" + "\n".join([f"- ✅ {p}" for p in pros]))
    with pc_c2:
        st.markdown("**Nhược điểm:**\n" + "\n".join([f"- ⚠️ {c}" for c in cons]))

# --- TAB 2 ---
with tab2:
    st.markdown("### Upload & Validation")
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file:
        df_uploaded = pd.read_csv(uploaded_file)
        # Bỏ qua Unnamed: 0 nếu có
        if 'Unnamed: 0' in df_uploaded.columns:
            df_uploaded.drop('Unnamed: 0', axis=1, inplace=True)
            
        is_valid, msg = validate_csv(df_uploaded)
        if is_valid:
            st.success(f"File hợp lệ! Preview 5 dòng đầu:")
            st.dataframe(df_uploaded.head(), use_container_width=True)
            
            if st.button("⚡ CHẠY PHÂN TÍCH", type="primary", use_container_width=True):
                if model is None:
                    st.error("Mô hình chưa sẵn sàng. Vui lòng chọn mô hình ở Tab 1.")
                else:
                    X = df_uploaded.drop(columns=['Class'], errors='ignore')
                    
                    try:
                        probs = model.predict_proba(X)[:, 1]
                        st.session_state['probs'] = probs
                        st.session_state['X_data'] = X
                        st.session_state['df_display'] = df_uploaded
                        st.session_state['true_labels'] = df_uploaded['Class'] if 'Class' in df_uploaded.columns else None
                        st.success("Phân tích hoàn tất!")
                    except Exception as e:
                        st.error(f"Lỗi khi dự đoán: {str(e)}")
        else:
            st.error(msg)
            
    if st.session_state['probs'] is not None:
        probs = st.session_state['probs']
        df_display = st.session_state['df_display']
        true_labels = st.session_state['true_labels']
        
        st.markdown("---")
        st.markdown("### Tổng Quan Dữ Liệu")
        st.dataframe(df_display.describe(), use_container_width=True)

        st.markdown("---")
        st.markdown("### Kết quả Phân Tích")
        threshold = st.slider("Ngưỡng phân loại (Threshold)", 0.01, 0.99, 0.50, 0.01)
        
        preds = (probs >= threshold).astype(int)
        
        total = len(preds)
        fraud = sum(preds)
        safe = total - fraud
        rate = (fraud / total * 100) if total > 0 else 0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tổng giao dịch", f"{total:,}")
        c2.metric("An toàn ✅", f"{safe:,}")
        c3.metric("Cảnh báo 🚨", f"{fraud:,}")
        c4.metric("Tỉ lệ cảnh báo %", f"{rate:.2f}%")
        
        if true_labels is not None:
            tp = sum((preds == 1) & (true_labels == 1))
            tn = sum((preds == 0) & (true_labels == 0))
            fp = sum((preds == 1) & (true_labels == 0))
            fn = sum((preds == 0) & (true_labels == 1))
            
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("True Positives (Bắt đúng fraud)", tp)
            c6.metric("False Positives (Báo nhầm fraud)", fp)
            c7.metric("False Negatives (Bỏ lọt fraud)", fn)
            c8.metric("True Negatives (Đúng legit)", tn)
            
        df_res = df_display.copy()
        df_res['Fraud_Probability'] = (probs * 100).round(2)
        df_res['Predicted_Label'] = ["🚨 FRAUD" if p == 1 else "✅ LEGIT" for p in preds]
        
        display_cols = ['Time', 'Amount', 'Fraud_Probability', 'Predicted_Label']
        
        if true_labels is not None:
            df_res['True_Label'] = ["🔴 FRAUD" if t == 1 else "🟢 LEGIT" for t in true_labels]
            df_res['Correct'] = ["✓" if p == t else "✗" for p, t in zip(preds, true_labels)]
            display_cols.extend(['True_Label', 'Correct'])
            
        # Thêm các cột V1-V28 vào cuối
        v_cols = [c for c in df_res.columns if c.startswith('V')]
        display_cols.extend(v_cols)
            
        st.dataframe(df_res[display_cols].head(500), use_container_width=True)
        
        csv_data = df_res[display_cols].to_csv(index=False).encode('utf-8')
        st.download_button("Tải kết quả CSV", data=csv_data, file_name="results.csv", mime="text/csv")
        
        st.markdown("---")
        st.markdown("### Xem Chi Tiết Giao Dịch")
        
        col_left, col_right = st.columns([6, 4])
        
        with col_left:
            filter_opt = st.radio("Lọc giao dịch", ["Tất cả", "Chỉ Fraud", "Chỉ Legit"], horizontal=True)
            
            indices = list(range(total))
            if filter_opt == "Chỉ Fraud":
                indices = [i for i, p in enumerate(preds) if p == 1]
            elif filter_opt == "Chỉ Legit":
                indices = [i for i, p in enumerate(preds) if p == 0]
                
            if indices:
                def format_func(i):
                    amt = df_display.iloc[i].get('Amount', 0)
                    prob = probs[i] * 100
                    lbl = '🚨' if preds[i] == 1 else '✅'
                    return f"#{i} — ${amt:.2f} | {prob:.1f}% fraud [{lbl}]"
                    
                selected_idx = st.selectbox("Chọn giao dịch:", indices, format_func=format_func)
                
                if selected_idx is not None:
                    # Hiển thị Card
                    prob_val = probs[selected_idx] * 100
                    is_fraud = preds[selected_idx] == 1
                    lbl_text = "🚨 FRAUD" if is_fraud else "✅ LEGIT"
                    amt = df_display.iloc[selected_idx].get('Amount', 0)
                    time_val = df_display.iloc[selected_idx].get('Time', 0)
                    
                    with st.container(border=True):
                        st.subheader(f"{lbl_text}")
                        st.progress(probs[selected_idx], text=f"Xác suất Fraud: {prob_val:.2f}%")
                        st.markdown(f"**Amount:** ${amt:.2f}  |  **Time:** {time_val}")
                        
                        if true_labels is not None:
                            t_lbl = true_labels.iloc[selected_idx]
                            correct_mark = "✓" if is_fraud == t_lbl else "✗"
                            st.markdown(f"**Nhãn thật:** {'🔴 FRAUD' if t_lbl == 1 else '🟢 LEGIT'} {correct_mark}")
            else:
                st.info("Không có giao dịch nào thỏa mãn bộ lọc.")
                selected_idx = None

        # Hiển thị top 8 đặc trưng ảnh hưởng          
        with col_right:
            if 'selected_idx' in locals() and selected_idx is not None and model is not None:
                try:
                    single_X = df_display.iloc[[selected_idx]].drop(columns=['Class'], errors='ignore')
                    
                    if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps and 'classifier' in model.named_steps:
                        preprocessor = model.named_steps['preprocessor']
                        classifier = model.named_steps['classifier']
                        #Biến  time và amount thành 2 biến đã scale còn lại giữ nguyên v1-v28 
                        # để tính đóng góp của từng đặc trưng vào dự đoán của mô hình
                        X_transformed = preprocessor.transform(single_X)
                        if hasattr(X_transformed, 'toarray'):
                            row_vals = X_transformed.toarray()[0]
                        else:
                            row_vals = np.array(X_transformed)[0]
                        # row_vals làvector feature của giao dịch đã được scale
                        if hasattr(classifier, 'feature_importances_'):
                            importances = classifier.feature_importances_
                            # feature_importances_ là vector độ quan trọng của từng feature được model sinh ra sau khi train
                            # tính đóng góp feature = | giá trị feature * độ quan trọng feature | 
                            #row_vals chính là một mảng 1 chiều (1D numpy array) chứa các giá trị đặc trưng
                            # của một giao dịch duy nhất sau khi đã được tiền xử lý (chuẩn hóa/scale).
                            contributions = np.abs(row_vals * importances)
                            sum_contrib = np.sum(contributions)
                            
                            if sum_contrib > 0:
                                contributions_pct = (contributions / sum_contrib) * 100
                            else:
                                contributions_pct = contributions
                                
                            feature_names = ['scaled_time', 'scaled_amount'] + [f'V{i}' for i in range(1, 29)]
                            df_contrib = pd.DataFrame({
                                'Feature': feature_names[:len(contributions)],
                                'Contribution': contributions_pct
                            }).sort_values('Contribution', ascending=False).head(8)
                            
                            st.markdown("##### Đóng góp Feature cho giao dịch này (Top 8)")
                            
                            fig = px.bar(
                                df_contrib.sort_values('Contribution', ascending=True), 
                                x='Contribution', 
                                y='Feature', 
                                orientation='h',
                                color='Contribution',
                                color_continuous_scale=['#10b981', '#f43f5e']
                            )
                            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300)
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption("Feature contribution = giá trị feature × tầm quan trọng (sau chuẩn hóa)")
                        else:
                            st.info("Thuật toán đang chọn không hỗ trợ trích xuất feature importances.")
                    else:
                        st.info("Mô hình không đúng định dạng pipeline dự kiến (preprocessor -> classifier).")
                except Exception as e:
                    st.error(f"Lỗi hiển thị feature contribution: {str(e)}")
