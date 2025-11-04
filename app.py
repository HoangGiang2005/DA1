import streamlit as st
import pickle
import numpy as np

# ====== LOAD MODEL ======
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# ====== UI ======
st.set_page_config(page_title="Demo Mô hình ML", page_icon="🤖")
st.title("🎯 Demo dự đoán với mô hình học máy")
st.write("Nhập các thông số để mô hình dự đoán:")

# Thay các input dưới đây bằng features của bạn
gioi_tinh = st.selectbox(
    "Giới tính",
    options=["Nữ", "Nam"],
    index=None,
    placeholder="Chọn giới tính"
)
f1 = 1 if gioi_tinh == "Nam" else (0 if gioi_tinh == "Nữ" else None)

f2 = st.number_input("Tuổi", value=0.0)
huyetAp = st.selectbox(
    "Huyết áp cao",
    options=["Không", "Có"],
    index=None,
    placeholder="Chọn tình trạng huyết áp"
)
f3 = 1 if huyetAp == "Có" else (0 if huyetAp == "Không" else None)
benhTim = st.selectbox(
    "Bệnh tim",
    options=["Không", "Có"],
    index=None,
    placeholder="Chọn tình trạng bệnh tim"
)
f4 = 1 if benhTim == "Có" else (0 if benhTim == "Không" else None)

honNhan = st.selectbox(
    "Kết hôn",
    options=["Chưa kết hôn", "Đã kết hôn"],
    index=None,
    placeholder="Chọn tình trạng hôn nhân"
)
f5 = 1 if honNhan == "Đã kết hôn" else (0 if honNhan == "Chưa kết hôn" else None)

# Nghề nghiệp
job_labels = {
    0: "Private",
    1: "Self-employed",
    2: "Children",
    3: "Govt_job",
    4: "Never_worked",
}
ngheNghiep = st.selectbox(
    "Nghề nghiệp",
    options=list(job_labels.values()),
    index=None,
    placeholder="Chọn nghề nghiệp"
)
f6 = next((k for k, v in job_labels.items() if v == ngheNghiep), None)

#Nơi sống
noiSong = st.selectbox(
    "Nơi sống",
    options=["Urban", "Rural"],
    index=None,
    placeholder="Chọn nơi sống"
)
f7 = 1 if noiSong == "Rural" else (0 if noiSong == "Urban" else None)
f8 = st.number_input("đường huyết", value=0.0)
f9 = st.number_input("bmi", value=0.0)

smoke_labels = {
    0: "Never smoked", 
    1: "Unknown" , 
    2:"Formerly smoked",
    3: "Smokes"}
smoke = st.selectbox(
    "Tình trạng hút thuốc",
    options=list(smoke_labels.values()),
    index=None,
    placeholder="Chọn tình trạng hút thuốc"
)
f10 = next((k for k, v in smoke_labels.items() if v == smoke), None)


if st.button("🧮 Dự đoán"):
    x = np.array([[f1, f2, f3, f4,f5, f6, f7, f8, f9, f10]])
    pred = model.predict(x)
    st.success(f"✅ Kết quả dự đoán: **{pred[0]}**")
    if(pred[0]==1):
        st.write("Người này có nguy cơ mắc bệnh")
    else:
        st.write("Người này không có nguy cơ mắc bệnh")

st.markdown("---")
