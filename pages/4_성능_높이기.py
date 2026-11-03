import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# 1. 페이지 설정
st.set_page_config(
    page_title="뇌졸중 모델 성능 높이기",
    page_icon="⚖️",
    layout="wide"
)

# 2. 데이터 불러오기 함수
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jangyeonu913-ui/data/refs/heads/main/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

st.title("⚖️ 뇌졸중 모델 성능 높이기 (클래스 가중치 조절)")
st.markdown("뇌졸중 데이터의 불균형(정상인에 비해 환자가 매우 적음)을 해결하기 위해 가중치 조절(`class_weight='balanced'`) 전후의 성능을 비교합니다.")
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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. 모델 학습 (가중치 조절 전 / 후)
# 1) 로지스틱 회귀
lr_before = LogisticRegression(random_state=42)
lr_before.fit(X_train_scaled, y_train)
y_pred_lr_b = lr_before.predict(X_test_scaled)

lr_after = LogisticRegression(class_weight='balanced', random_state=42)
lr_after.fit(X_train_scaled, y_train)
y_pred_lr_a = lr_after.predict(X_test_scaled)

# 2) 의사결정트리
dt_before = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_before.fit(X_train, y_train)
y_pred_dt_b = dt_before.predict(X_test)

dt_after = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, class_weight='balanced', random_state=42)
dt_after.fit(X_train, y_train)
y_pred_dt_a = dt_after.predict(X_test)

st.markdown("---")

# 7. 혼동행렬 및 평가지표 계산 헬퍼 함수
def get_cm(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn

def calc_metrics(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return f"{acc*100:.2f}%", f"{recall*100:.2f}%", f"{precision*100:.2f}%", f"{f1*100:.2f}%"

# 로지스틱 회귀 전후 지표
tp_lrb, fp_lrb, fn_lrb, tn_lrb = get_cm(y_test, y_pred_lr_b)
tp_lra, fp_lra, fn_lra, tn_lra = get_cm(y_test, y_pred_lr_a)
acc_lrb, rec_lrb, prec_lrb, f1_lrb = calc_metrics(tp_lrb, fp_lrb, fn_lrb, tn_lrb)
acc_lra, rec_lra, prec_lra, f1_lra = calc_metrics(tp_lra, fp_lra, fn_lra, tn_lra)

# 의사결정트리 전후 지표
tp_dtb, fp_dtb, fn_dtb, tn_dtb = get_cm(y_test, y_pred_dt_b)
tp_dta, fp_dta, fn_dta, tn_dta = get_cm(y_test, y_pred_dt_a)
acc_dtb, rec_dtb, prec_dtb, f1_dtb = calc_metrics(tp_dtb, fp_dtb, fn_dtb, tn_dtb)
acc_dta, rec_dta, prec_dta, f1_dta = calc_metrics(tp_dta, fp_dta, fn_dta, tn_dta)

st.subheader("📊 가중치 조절 전후 성능 비교 표")

col_t1, col_t2 = st.columns(2)

with col_t1:
    st.markdown("##### 1. 로지스틱 회귀(확률로 답하는 모델)")
    lr_comp_df = pd.DataFrame({
        "평가 항목": ["정확도", "재현율", "정밀도", "F1 점수", "찾은 환자(TP)", "놓친 환자(FN)", "헛짚은 사람(FP)", "정상 맞춤(TN)"],
        "가중치 조절 전": [acc_lrb, rec_lrb, prec_lrb, f1_lrb, f"{tp_lrb}명", f"{fn_lrb}명", f"{fp_lrb}명", f"{tn_lrb}명"],
        "가중치 조절 후 (Balanced)": [acc_lra, rec_lra, prec_lra, f1_lra, f"{tp_lra}명", f"{fn_lra}명", f"{fp_lra}명", f"{tn_lra}명"]
    })
    st.dataframe(lr_comp_df, use_container_width=True, hide_index=True)
    st.markdown(f"💡 **변화 요약:** 찾은 환자 수는 **{tp_lrb}명 → {tp_lra}명**, 놓친 환자는 **{fn_lrb}명 → {fn_lra}명**, 헛짚은 사람은 **{fp_lrb}명 → {fp_lra}명**으로 바뀌었습니다.")

with col_t2:
    st.markdown("##### 2. 의사결정트리(질문으로 답하는 모델)")
    dt_comp_df = pd.DataFrame({
        "평가 항목": ["정확도", "재현율", "정밀도", "F1 점수", "찾은 환자(TP)", "놓친 환자(FN)", "헛짚은 사람(FP)", "정상 맞춤(TN)"],
        "가중치 조절 전": [acc_dtb, rec_dtb, prec_dtb, f1_dtb, f"{tp_dtb}명", f"{fn_dtb}명", f"{fp_dtb}명", f"{tn_dtb}명"],
        "가중치 조절 후 (Balanced)": [acc_dta, rec_dta, prec_dta, f1_dta, f"{tp_dta}명", f"{fn_dta}명", f"{fp_dta}명", f"{tn_dta}명"]
    })
    st.dataframe(dt_comp_df, use_container_width=True, hide_index=True)
    st.markdown(f"💡 **변화 요약:** 찾은 환자 수는 **{tp_dtb}명 → {tp_dta}명**, 놓친 환자는 **{fn_dtb}명 → {fn_dta}명**, 헛짚은 사람은 **{fp_dtb}명 → {fp_dta}명**으로 바뀌었습니다.")

st.markdown("---")

# 8. 산점도 및 결정 경계 / 평면 시각화 (로지스틱 회귀 기준 전후 비교)
st.subheader("🗺️ 로지스틱 회귀 가중치 조절 전후 0.5 결정 경계(평면) 비교")

if len(selected_korean) >= 3:
    st.caption("속성을 3개 이상 선택하셨으므로 3차원 산점도를 표시합니다.")
    ax_x = st.selectbox("X축 속성", selected_korean, index=0)
    ax_y = st.selectbox("Y축 속성", selected_korean, index=1)
    ax_z = st.selectbox("Z축 속성", selected_korean, index=2)
    
    x_f, y_f, z_f = inv_feature_map[ax_x], inv_feature_map[ax_y], inv_feature_map[ax_z]
    
    test_df_vis = test_df.copy()
    test_df_vis["stroke_label"] = test_df_vis["stroke"].map({0: "정상 (0)", 1: "뇌졸중 (1)"})
    
    fig = px.scatter_3d(
        test_df_vis,
        x=x_f, y=y_f, z=z_f,
        color="stroke_label",
        labels={x_f: ax_x, y_f: ax_y, z_f: ax_z},
        title="3차원 산점도 및 가중치 조절 전후 분석"
    )
    st.info("💡 3차원 산점도가 생성되었습니다. 마우스로 회전하여 데이터를 관찰할 수 있습니다.")
    st.plotly_chart(fig, use_container_width=True)

else:
    ax_x = st.selectbox("가로축 속성", selected_korean, index=0)
    ax_y = st.selectbox("세로축 속성", selected_korean, index=1)
    
    x_f = inv_feature_map[ax_x]
    y_f = inv_feature_map[ax_y]
    
    other_feats = [f for f in selected_features if f not in [x_f, y_f]]
    other_meds = {f: test_df[f].median() for f in other_feats}
    
    if other_feats:
        m_desc = ", ".join([f"{feature_map[f]}: {other_meds[f]:.2f}" for f in other_feats])
        st.info(f"💡 두 축이 아닌 속성({m_desc})은 테스트 데이터의 중앙값에 고정하였습니다.")
        
    x_min, x_max = test_df[x_f].min(), test_df[x_f].max()
    y_min, y_max = test_df[y_f].min(), test_df[y_f].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * 0.1 if x_range > 0 else 1
    x_max += x_range * 0.1 if x_range > 0 else 1
    y_min -= y_range * 0.1 if y_range > 0 else 1
    y_max += y_range * 0.1 if y_range > 0 else 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    grid_data = {}
    for f in selected_features:
        if f == x_f:
            grid_data[f] = xx.ravel()
        elif f == y_f:
            grid_data[f] = yy.ravel()
        else:
            grid_data[f] = np.full(xx.ravel().shape, other_meds.get(f, 0))
    grid_df = pd.DataFrame(grid_data)
    grid_scaled = scaler.transform(grid_df[selected_features])
    
    z_b = lr_before.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)
    z_a = lr_after.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)
    
    test_df_vis = test_df.copy()
    test_df_vis["stroke_label"] = test_df_vis["stroke"].map({0: "정상 (0)", 1: "뇌졸중 (1)"})
    
    fig = px.scatter(
        test_df_vis,
        x=x_f, y=y_f,
        color="stroke_label",
        labels={x_f: ax_x, y_f: ax_y},
        title="로지스틱 회귀 가중치 조절 전후 0.5 결정 경계선 비교"
    )
    
    fig.add_trace(
        go.Contour(
            z=z_b, x=np.linspace(x_min, x_max, 100), y=np.linspace(y_min, y_max, 100),
            contours=dict(start=0.5, end=0.5, size=0, coloring='lines'),
            line=dict(color='blue', width=2.5, dash='dash'),
            showscale=False, name='조절 전 (p=0.5)'
        )
    )
    
    fig.add_trace(
        go.Contour(
            z=z_a, x=np.linspace(x_min, x_max, 100), y=np.linspace(y_min, y_max, 100),
            contours=dict(start=0.5, end=0.5, size=0, coloring='lines'),
            line=dict(color='red', width=2.5, dash='solid'),
            showscale=False, name='조절 후 (Balanced, p=0.5)'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if (z_b.min() > 0.5) or (z_b.max() < 0.5):
        st.warning("💡 조절 전 결정 경계선이 현재 그림 범위 밖에 있습니다.")
    if (z_a.min() > 0.5) or (z_a.max() < 0.5):
        st.warning("💡 조절 후 결정 경계선이 현재 그림 범위 밖에 있습니다.")
