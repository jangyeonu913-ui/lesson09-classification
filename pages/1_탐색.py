import pandas as pd
import plotly.express as px
import streamlit as st

# 1. 페이지 설정
st.set_page_config(
    page_title="뇌졸중 데이터 탐색",
    page_icon="📊",
    layout="wide"
)

# 2. 데이터 불러오기 함수
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jangyeonu913-ui/data/refs/heads/main/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

st.title("📊 뇌졸중 데이터 탐색 (EDA)")
st.markdown("데이터의 분포와 주요 위험 요인 간의 관계를 시각적으로 탐색합니다.")
st.markdown("---")

# 3. 나이와 평균 혈당의 분포 히스토그램 (나란히)
st.subheader("1. 나이(age) 및 평균 혈당(avg_glucose_level) 분포")
col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(df, x="age", nbins=30, title="나이(age) 분포", labels={"age": "나이", "count": "사람 수"})
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    fig_glucose = px.histogram(df, x="avg_glucose_level", nbins=30, title="평균 혈당(avg_glucose_level) 분포", labels={"avg_glucose_level": "평균 혈당", "count": "사람 수"})
    st.plotly_chart(fig_glucose, use_container_width=True)

st.markdown("---")

# 4. 뇌졸중 여부에 따른 나이와 평균 혈당 상자그림 및 평균값 비교 표
st.subheader("2. 뇌졸중 여부(stroke)에 따른 나이 및 평균 혈당 비교")
df_stroke_str = df.copy()
df_stroke_str["stroke_label"] = df_stroke_str["stroke"].map({0: "정상 (0)", 1: "뇌졸중 (1)"})

col3, col4 = st.columns(2)

with col3:
    fig_box_age = px.box(df_stroke_str, x="stroke_label", y="age", color="stroke_label", title="뇌졸중 여부에 따른 나이 비교", labels={"stroke_label": "뇌졸중 여부", "age": "나이"})
    st.plotly_chart(fig_box_age, use_container_width=True)

with col4:
    fig_box_glucose = px.box(df_stroke_str, x="stroke_label", y="avg_glucose_level", color="stroke_label", title="뇌졸중 여부에 따른 평균 혈당 비교", labels={"stroke_label": "뇌졸중 여부", "avg_glucose_level": "평균 혈당"})
    st.plotly_chart(fig_box_glucose, use_container_width=True)

# 두 그룹의 평균값 표
st.markdown("##### 💡 그룹별 평균값 비교 표")
mean_table = df.groupby("stroke")[["age", "avg_glucose_level"]].mean().reset_index()
mean_table["stroke"] = mean_table["stroke"].map({0: "정상 (0)", 1: "뇌졸중 (1)"})
mean_table.columns = ["뇌졸중 여부", "평균 나이", "평균 혈당"]
st.dataframe(mean_table.style.format({"평균 나이": "{:.2f}", "평균 혈당": "{:.2f}"}), use_container_width=True)

st.markdown("---")

# 5. 고혈압 및 심장병에 따른 뇌졸중 비율 막대그래프
st.subheader("3. 고혈압 및 심장병에 따른 뇌졸중 비율")
col5, col6 = st.columns(2)

with col5:
    # 고혈압
    ht_df = df.groupby("hypertension")["stroke"].mean().reset_index()
    ht_df["hypertension_label"] = ht_df["hypertension"].map({0: "고혈압 없음 (0)", 1: "고혈압 있음 (1)"})
    ht_df["stroke_pct"] = ht_df["stroke"] * 100
    fig_ht = px.bar(ht_df, x="hypertension_label", y="stroke_pct", color="hypertension_label", title="고혈압 여부에 따른 뇌졸중 비율", labels={"hypertension_label": "고혈압 여부", "stroke_pct": "뇌졸중 비율 (%)"})
    st.plotly_chart(fig_ht, use_container_width=True)

with col6:
    # 심장병
    hd_df = df.groupby("heart_disease")["stroke"].mean().reset_index()
    hd_df["heart_disease_label"] = hd_df["heart_disease"].map({0: "심장병 없음 (0)", 1: "심장병 있음 (1)"})
    hd_df["stroke_pct"] = hd_df["stroke"] * 100
    fig_hd = px.bar(hd_df, x="heart_disease_label", y="stroke_pct", color="heart_disease_label", title="심장병 여부에 따른 뇌졸중 비율", labels={"heart_disease_label": "심장병 여부", "stroke_pct": "뇌졸중 비율 (%)"})
    st.plotly_chart(fig_hd, use_container_width=True)

st.markdown("---")

# 6. 체질량지수(bmi) 빈 값 사람들 뇌졸중 비율 vs 전체 뇌졸중 비율 비교 표
st.subheader("4. 체질량지수(bmi) 결측치 그룹과 전체 뇌졸중 비율 비교")
total_stroke_ratio = (df["stroke"].sum() / len(df)) * 100

bmi_missing_df = df[df["bmi"].isnull()]
missing_count = len(bmi_missing_df)
missing_stroke_ratio = (bmi_missing_df["stroke"].sum() / missing_count * 100) if missing_count > 0 else 0.0

comparison_data = pd.DataFrame({
    "구분": ["BMI 결측치 그룹", "전체 데이터"],
    "인원 수": [missing_count, len(df)],
    "뇌졸중 환자 수": [int(bmi_missing_df["stroke"].sum()), int(df["stroke"].sum())],
    "뇌졸중 비율 (%)": [missing_stroke_ratio, total_stroke_ratio]
})

st.dataframe(comparison_data.style.format({"뇌졸중 비율 (%)": "{:.2f}%"}), use_container_width=True)

st.markdown("---")

# 7. 흡연 상태(smoking_status) 값마다 사람 수 세어 표로 보여주기
st.subheader("5. 흡연 상태(smoking_status)별 인원 수")
smoking_count_df = df["smoking_status"].value_counts().reset_index()
smoking_count_df.columns = ["흡연 상태", "사람 수"]
st.dataframe(smoking_count_df, use_container_width=True)
