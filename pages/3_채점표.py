import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# 1. 페이지 설정
st.set_page_config(
    page_title="뇌졸중 모델 채점표",
    page_icon="📝",
    layout="wide"
)

# 2. 데이터 불러오기 함수
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jangyeonu913-ui/data/refs/heads/main/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

st.title("📝 뇌졸중 모델 채점표")
st.markdown("테스트 데이터를 활용하여 각 모델의 혼동행렬과 평가 지표(정확도, 재현율, 정밀도, F1 점수)를 비교 평가합니다.")
st.markdown("---")

# 3. 속성 선택 및 한글 매핑
feature_map = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}
inv_feature_map = {v: k for k, v in feature_map.items()}
all_features = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
default_features = ["age", "avg_glucose_level", "hypertension", "heart_disease"]
default_korean = [feature_map[f] for f in default_features]

st.subheader("⚙️ 모델 학습 속성 설정")
selected_korean = st.multiselect(
    "입력으로 사용할 속성을 선택하세요 (2개 이상 선택)",
    options=[feature_map[f] for f in all_features],
    default=default_korean
)

if len(selected_korean) < 2:
    st.warning("⚠️ 속성을 2개 이상 선택해 주세요.")
    st.stop()

selected_features = [inv_feature_map[k] for k in selected_korean]

# 4. 데이터 분할 (번호 순 정렬 후 10명씩 묶어 앞 3명 테스트용, 나머지 7명 학습용)
df_sorted = df.reset_index(drop=True)
test_indices = []
train_indices = []

for i in range(len(df_sorted)):
    if (i % 10) < 3:
        test_indices.append(i)
    else:
        train_indices.append(i)

train_df = df_sorted.iloc[train_indices].copy()
test_df = df_sorted.iloc[test_indices].copy()

# 5. 체질량지수(bmi) 결측치 처리 (선택된 경우에만 훈련용 중앙값으로 채우기)
if "bmi" in selected_features:
    bmi_median = train_df["bmi"].median()
    train_df["bmi"] = train_df["bmi"].fillna(bmi_median)
    test_df["bmi"] = test_df["bmi"].fillna(bmi_median)
    st.info(f"💡 체질량지수(bmi) 결측치는 훈련용 데이터의 중앙값({bmi_median:.2f})으로 대체되었습니다.")

X_train = train_df[selected_features]
y_train = train_df["stroke"]
X_test = test_df[selected_features]
y_test = test_df["stroke"]

# 스케일링 (로지스틱 회귀용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. 모델 학습
# 1) 로지스틱 회귀
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

# 2) 의사결정트리
dt_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# 3) Baseline (훈련용에서 많은 쪽으로만 답하는 모델)
majority_class = y_train.mode()[0]
y_pred_base = np.full(len(y_test), majority_class)

# 예측 결과를 test_df에 보관
test_df["pred_lr"] = y_pred_lr
test_df["pred_dt"] = y_pred_dt
test_df["pred_base"] = y_pred_base

st.markdown("---")

# 7. 혼동행렬 계산 함수
def get_cm_components(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn

tp_lr, fp_lr, fn_lr, tn_lr = get_cm_components(y_test, y_pred_lr)
tp_dt, fp_dt, fn_dt, tn_dt = get_cm_components(y_test, y_pred_dt)
tp_base, fp_base, fn_base, tn_base = get_cm_components(y_test, y_pred_base)

# 혼동행렬 2x2 표 시각화
st.subheader("📊 모델별 혼동행렬 (Confusion Matrix)")
st.caption("행은 실제값(Actual), 열은 예측값(Predicted)입니다.")

col_cm1, col_cm2 = st.columns(2)

def create_cm_df(tp, fp, fn, tn):
    actual_1_sum = tp + fn
    actual_0_sum = fp + tn
    pred_1_sum = tp + fp
    pred_0_sum = fn + tn
    total_sum = tp + fp + fn + tn
    
    cm_data = {
        "예측: 뇌졸중(1)": [f"{tp}명 (TP)", f"{fp}명 (FP)", f"{pred_1_sum}명"],
        "예측: 아니오(0)": [f"{fn}명 (FN)", f"{tn}명 (TN)", f"{pred_0_sum}명"],
        "행 합계": [f"{actual_1_sum}명", f"{actual_0_sum}명", f"{total_sum}명"]
    }
    return pd.DataFrame(cm_data, index=["실제: 뇌졸중(1)", "실제: 아니오(0)", "열 합계"])

with col_cm1:
    st.markdown("##### 1. 로지스틱 회귀(확률로 답하는 모델)")
    st.dataframe(create_cm_df(tp_lr, fp_lr, fn_lr, tn_lr), use_container_width=True)

with col_cm2:
    st.markdown("##### 2. 의사결정트리(질문으로 답하는 모델)")
    st.dataframe(create_cm_df(tp_dt, fp_dt, fn_dt, tn_dt), use_container_width=True)

st.markdown("---")

# 8. 평가지표 표 산출 함수
def calculate_metrics(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total > 0 else 0.0
    
    # 재현율 (Recall) = TP / (TP + FN) [실제 뇌졸중인 사람 수]
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # 정밀도 (Precision) = TP / (TP + FP) [뇌졸중이라 예측한 사람 수]
    pred_positive = tp + fp
    if pred_positive == 0:
        precision_str = "계산할 수 없음"
        f1_str = "계산할 수 없음"
    else:
        precision = tp / pred_positive
        precision_str = f"{precision * 100:.2f}%"
        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
            f1_str = f"{f1 * 100:.2f}%"
        else:
            f1_str = "0.00%"
            
    acc_str = f"{acc * 100:.2f}%"
    recall_str = f"{recall * 100:.2f}%"
    
    return acc_str, recall_str, precision_str, f1_str

lr_metrics = calculate_metrics(tp_lr, fp_lr, fn_lr, tn_lr)
dt_metrics = calculate_metrics(tp_dt, fp_dt, fn_dt, tn_dt)
base_metrics = calculate_metrics(tp_base, fp_base, fn_base, tn_base)

metrics_df = pd.DataFrame({
    "모델": [
        "로지스틱 회귀(확률로 답하는 모델)",
        "의사결정트리(질문으로 답하는 모델)",
        "훈련용에서 많은 쪽으로만 답하는 모델"
    ],
    "정확도 (Accuracy)": [lr_metrics[0], dt_metrics[0], base_metrics[0]],
    "재현율 (Recall)": [lr_metrics[1], dt_metrics[1], base_metrics[1]],
    "정밀도 (Precision)": [lr_metrics[2], dt_metrics[2], base_metrics[2]],
    "F1 점수 (F1-Score)": [lr_metrics[3], dt_metrics[3], base_metrics[3]]
})

st.subheader("📈 모델 성능 평가 지표 비교")
st.dataframe(metrics_df, use_container_width=True, hide_index=True)

st.markdown("---")

# 9. 오답(놓친 환자 / 헛짚은 사람) 세부 목록
st.subheader("🔍 오답 목록 분석 (놓친 환자 & 헛짚은 사람)")

total_test_cnt = len(test_df)
stroke_test_cnt = int(y_test.sum())
st.markdown(f"**테스트용 사람 수:** {total_test_cnt:,}명 | **그중 실제 뇌졸중인 사람 수:** {stroke_test_cnt:,}명")

model_choice = st.selectbox(
    "오답 목록을 확인할 모델을 선택하세요:",
    options=[
        "로지스틱 회귀(확률로 답하는 모델)",
        "의사결정트리(질문으로 답하는 모델)",
        "훈련용에서 많은 쪽으로만 답하는 모델"
    ]
)

if model_choice == "로지스틱 회귀(확률로 답하는 모델)":
    pred_col = "pred_lr"
elif model_choice == "의사결정트리(질문으로 답하는 모델)":
    pred_col = "pred_dt"
else:
    pred_col = "pred_base"

# 놓친 환자 (FN): 실제 뇌졸중(1)인데 예측(0)
fn_df = test_df[(test_df["stroke"] == 1) & (test_df[pred_col] == 0)].copy()
# 헛짚은 사람 (FP): 실제 정상(0)인데 예측(1)
fp_df = test_df[(test_df["stroke"] == 0) & (test_df[pred_col] == 1)].copy()

# 나이 많은 순서 정렬
fn_df = fn_df.sort_values(by="age", ascending=False)
fp_df = fp_df.sort_values(by="age", ascending=False)

fn_count = len(fn_df)
fp_count = len(fp_df)

# 표 표시용 서식
display_cols = ["id", "age", "avg_glucose_level", "hypertension", "heart_disease"]
col_rename = {
    "id": "번호",
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "hypertension": "고혈압 여부",
    "heart_disease": "심장병 여부"
}

fn_display = fn_df[display_cols].rename(columns=col_rename).head(20)
fp_display = fp_df[display_cols].rename(columns=col_rename).head(20)

col_list1, col_list2 = st.columns(2)

with col_list1:
    st.markdown(f"##### 🚨 놓친 환자 (실제 뇌졸중인데 아니라고 예측) - 총 {fn_count}명")
    if fn_count > 20:
        st.caption("※ 나이가 많은 순서대로 상위 20명만 표시합니다.")
    st.dataframe(fn_display, use_container_width=True, hide_index=True)

with col_list2:
    st.markdown(f"##### ⚠️ 헛짚은 사람 (실제 아닌데 뇌졸중이라고 예측) - 총 {fp_count}명")
    if fp_count > 20:
        st.caption("※ 나이가 많은 순서대로 상위 20명만 표시합니다.")
    st.dataframe(fp_display, use_container_width=True, hide_index=True)

st.info(f"💡 두 목록의 사람 수(놓친 환자: {fn_count}명, 헛짚은 사람: {fp_count}명)는 위 혼동행렬의 FN, FP 칸의 수치와 정확히 일치합니다.")
