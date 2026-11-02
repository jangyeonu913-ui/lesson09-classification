import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz, _tree

# 1. 페이지 설정
st.set_page_config(
    page_title="뇌졸중 분류 모델 실습실",
    page_icon="🤖",
    layout="wide"
)

# 2. 데이터 불러오기 함수
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jangyeonu913-ui/data/refs/heads/main/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

st.title("🤖 뇌졸중 분류 모델 실습실")
st.markdown("머신러닝 모델을 활용하여 뇌졸중 발생 여부를 예측하고 분류 결과를 시각화합니다.")
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
lr_train_acc = lr_model.score(X_train_scaled, y_train)
lr_test_acc = lr_model.score(X_test_scaled, y_test)

# 2) 의사결정트리 (질문을 세 번까지, 마지막 마디 5명 미만 분할 중지, 난수 고정)
dt_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_train_acc = dt_model.score(X_train, y_train)
dt_test_acc = dt_model.score(X_test, y_test)

# 3) Baseline (입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)
majority_class = y_train.mode()[0]
train_baseline_acc = (y_train == majority_class).mean()
test_baseline_acc = (y_test == majority_class).mean()

st.markdown("---")

# 7. 모델 정확도 큰 숫자 카드 비교
st.subheader("📊 모델별 정확도 비교")
c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        label="로지스틱 회귀(확률로 답하는 모델)",
        value=f"{lr_test_acc * 100:.2f}%",
        delta="테스트 정확도"
    )
    st.caption(f"훈련 정확도: {lr_train_acc * 100:.2f}% | 테스트 정확도: {lr_test_acc * 100:.2f}%")

with c2:
    st.metric(
        label="의사결정트리(질문으로 답하는 모델)",
        value=f"{dt_test_acc * 100:.2f}%",
        delta="테스트 정확도"
    )
    st.caption(f"훈련 정확도: {dt_train_acc * 100:.2f}% | 테스트 정확도: {dt_test_acc * 100:.2f}%")

with c3:
    st.metric(
        label="입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델",
        value=f"{test_baseline_acc * 100:.2f}%",
        delta="테스트 정확도"
    )
    st.caption(f"훈련 정확도: {train_baseline_acc * 100:.2f}% | 테스트 정확도: {test_baseline_acc * 100:.2f}%")

st.markdown("---")

# 8. 2차원 산점도 및 결정 경계 시각화
st.subheader("🗺️ 2차원 산점도 및 결정 경계 시각화")
st.caption("선택한 속성 중 가로축과 세로축으로 사용할 두 가지를 지정해 주세요.")

col_a, col_b = st.columns(2)
with col_a:
    axis_x_kor = st.selectbox("가로축 속성", selected_korean, index=0)
with col_b:
    axis_y_kor = st.selectbox("세로축 속성", selected_korean, index=1 if len(selected_korean) > 1 else 0)

x_feat = inv_feature_map[axis_x_kor]
y_feat = inv_feature_map[axis_y_kor]

# 두 축이 아닌 속성은 테스트 데이터 중앙값으로 고정
other_features = [f for f in selected_features if f not in [x_feat, y_feat]]
other_medians = {}
for f in other_features:
    other_medians[f] = test_df[f].median()

if other_features:
    median_desc = ", ".join([f"{feature_map[f]}: {other_medians[f]:.2f}" for f in other_features])
    st.info(f"💡 두 축이 아닌 속성({median_desc})은 테스트 데이터의 중앙값에 고정하여 계산하였습니다.")

# 격자 생성
x_min, x_max = test_df[x_feat].min(), test_df[x_feat].max()
y_min, y_max = test_df[y_feat].min(), test_df[y_feat].max()
x_range = x_max - x_min
y_range = y_max - y_min
x_min -= x_range * 0.1 if x_range > 0 else 1
x_max += x_range * 0.1 if x_range > 0 else 1
y_min -= y_range * 0.1 if y_range > 0 else 1
y_max += y_range * 0.1 if y_range > 0 else 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

grid_data = {}
for f in selected_features:
    if f == x_feat:
        grid_data[f] = xx.ravel()
    elif f == y_feat:
        grid_data[f] = yy.ravel()
    else:
        grid_data[f] = np.full(xx.ravel().shape, other_medians[f])

grid_df = pd.DataFrame(grid_data)

# 의사결정트리 영역 색상 채우기
Z_dt = dt_model.predict(grid_df[selected_features])
Z_dt = Z_dt.reshape(xx.shape)

test_df_vis = test_df.copy()
test_df_vis["stroke_label"] = test_df_vis["stroke"].map({0: "아님 (0)", 1: "뇌졸중 (1)"})

fig = px.scatter(
    test_df_vis,
    x=x_feat,
    y=y_feat,
    color="stroke_label",
    labels={x_feat: axis_x_kor, y_feat: axis_y_kor},
    title="테스트 데이터 산점도 및 결정 경계 비교"
)

# 의사결정트리 영역 (옅은 색)
fig.add_trace(
    go.Contour(
        z=Z_dt,
        x=np.linspace(x_min, x_max, 100),
        y=np.linspace(y_min, y_max, 100),
        colorscale=[[0, 'rgba(173, 216, 230, 0.15)'], [1, 'rgba(255, 182, 193, 0.15)']],
