import os
import json
import pandas as pd
import plotly.express as px
import streamlit as st

# 1. 페이지 및 브라우저 탭 설정 (제목 및 아이콘)
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# 2. 데이터 불러오기 함수
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jangyeonu913-ui/data/refs/heads/main/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

# 3. 사용자 기록 저장/불러오기 경로 설정 ('내_기록' 폴더 내 json 파일)
SAVE_DIR = "내_기록"
SAVE_FILE = os.path.join(SAVE_DIR, "user_records.json")

def load_records():
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"meanings": {}, "source": ""}

def save_records(records):
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)

saved_data = load_records()

# 4. 앱 헤더
st.title("🧠 뇌졸중 예측 실습실")
st.markdown("본 실습실은 뇌졸중 위험 요인을 분석하고 예측 모델을 탐색하기 위한 교육용 앱입니다.")
st.markdown("---")

# 5. 핵심 지표 (숫자 카드 4개)
total_count = len(df)
col_count = len(df.columns)
stroke_count = int(df["stroke"].sum()) if "stroke" in df.columns else 0
stroke_ratio = (stroke_count / total_count * 100) if total_count > 0 else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("전체 사람 수", f"{total_count:,} 명")
m2.metric("열 개수", f"{col_count} 개")
m3.metric("뇌졸중 환자 수 (stroke=1)", f"{stroke_count:,} 명")
m4.metric("뇌졸중 비율", f"{stroke_ratio:.2f} %")

st.markdown("---")

# 6. 열 정보 표 및 우리말 뜻 직접 입력 (편집 가능 표)
st.subheader("📋 데이터 열 (Column) 정보")
st.caption("교재를 참고하여 각 열의 우리말 뜻을 입력해 보세요. 입력 내용은 '내_기록' 폴더에 자동 저장됩니다.")

# 열 정보 데이터 생성
info_data = []
for col in df.columns:
    dtype_str = str(df[col].dtype)
    null_cnt = int(df[col].isnull().sum())
    info_data.append({
        "열 이름": col,
        "우리말 뜻": saved_data["meanings"].get(col, ""),
        "값의 종류": dtype_str,
        "빈 값 개수": null_cnt
    })

info_df = pd.DataFrame(info_data)

# st.data_editor를 통한 우리말 뜻 수정
edited_df = st.data_editor(
    info_df,
    column_config={
        "열 이름": st.column_config.Column(disabled=True),
        "우리말 뜻": st.column_config.TextColumn("우리말 뜻 (수정 가능)", help="교재를 보고 입력하세요."),
        "값의 종류": st.column_config.Column(disabled=True),
        "빈 값 개수": st.column_config.Column(disabled=True)
    },
    use_container_width=True,
    hide_index=True,
    key="data_editor"
)

# 변경사항 자동 저장
new_meanings = {}
for idx, row in edited_df.iterrows():
    col_name = row["열 이름"]
    val = row["우리말 뜻"]
    new_meanings[col_name] = val if pd.notnull(val) else ""

if new_meanings != saved_data["meanings"]:
    saved_data["meanings"] = new_meanings
    save_records(saved_data)
    st.toast("우리말 뜻이 '내_기록' 폴더에 저장되었습니다!", icon="💾")

st.markdown("---")

# 7. 데이터 상위 5줄 미리보기
st.subheader("🔍 데이터 미리보기 (첫 5줄)")
st.dataframe(df.head(5), use_container_width=True)

st.markdown("---")

# 8. 데이터 출처 입력란
st.subheader("📌 데이터 출처")
source_text = st.text_area(
    "교재에 나와 있는 데이터 출처 내용을 입력하세요:",
    value=saved_data.get("source", ""),
    height=100,
    placeholder="여기에 데이터 출처 정보를 작성하세요..."
)

if source_text != saved_data.get("source", ""):
    saved_data["source"] = source_text
    save_records(saved_data)
    st.toast("데이터 출처 정보가 '내_기록' 폴더에 저장되었습니다!", icon="💾")
