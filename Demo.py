import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


with open("model/xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

#Analysis Page
st.title("Phân tích bệnh đột quỵ")
data_stroke = pd.read_csv("data/data_stroke.csv")

# ======== Visualization ========
#Biểu đồ 1: 





#Model Prediction Page
st.sidebar.title("Mô hình dự đoán đột quỵ")
st.sidebar.subheader("Nhập các thông tin:")

#Gender
gioi_tinh = st.sidebar.selectbox(
     "Giới tính",
     options=["Nữ", "Nam"],
     index=None,
     placeholder="Chọn giới tính")

gender = 1 if gioi_tinh == "Nam" else (0 if gioi_tinh == "Nữ" else None)


#Age
age = st.sidebar.text_input("Tuổi:", placeholder="Nhập tuổi")
if age:
    age = int(age)

#Hypertension
huyetAp = st.sidebar.selectbox(
     "Huyết áp cao",
     options=["Không", "Có"],
     index=None,
     placeholder="Chọn tình trạng huyết áp")
hypertension = 1 if huyetAp == "Có" else (0 if huyetAp == "Không" else None)

#Heart Disease
benhTim = st.sidebar.selectbox(
    "Bệnh tim",
     options=["Không", "Có"],
     index=None,
     placeholder="Chọn tình trạng bệnh tim")
heart_disease = 1 if benhTim == "Có" else (0 if benhTim == "Không" else None)

#Married
honNhan = st.sidebar.selectbox(
     "Tình trạng hôn nhân:",
     options=["Chưa kết hôn", "Đã kết hôn"],
     index=None,
     placeholder="Chọn tình trạng hôn nhân")
married = 1 if honNhan == "Đã kết hôn" else (0 if honNhan == "Chưa kết hôn" else None)

#Occupation
job_labels = {
    0: "Private",
    1: "Self-employed",
    2: "Children",
    3: "Govt_job",
    4: "Never_worked",
}
ngheNghiep = st.sidebar.selectbox(
    "Nghề nghiệp",
    options=list(job_labels.values()),
    index=None,
    placeholder="Chọn nghề nghiệp"
)
Occupation = next((k for k, v in job_labels.items() if v == ngheNghiep), None)

#Residence Type
noiSong = st.sidebar.selectbox(
     "Nơi sống",
     options=["Urban", "Rural"],
     index=None,
     placeholder="Chọn nơi sống")
residence_type = 1 if noiSong == "Rural" else (0 if noiSong == "Urban" else None)

#Glucose Level
glucose_level = st.sidebar.text_input("Đường huyết:", placeholder="Nhập chỉ số đường huyết")
if glucose_level:
    glucose_level = float(glucose_level)

#BMI
bmi = st.sidebar.text_input("Chỉ số BMI:", placeholder="Nhập chỉ số BMI")
if bmi:
    bmi = float(bmi)

#Smoking Status
smoke_labels = {
     0: "Never smoked", 
     1: "Unknown" , 
     2:"Formerly smoked",
     3: "Smokes"}
smoke = st.sidebar.selectbox(
     "Tình trạng hút thuốc",
     options=list(smoke_labels.values()),
     index=None,
     placeholder="Chọn tình trạng hút thuốc")
smoke = next((k for k, v in smoke_labels.items() if v == smoke), None)


if st.sidebar.button("🧮 Dự đoán"):
     x = np.array([[gender,age, hypertension, heart_disease, married, Occupation, residence_type, glucose_level, bmi, smoke]])
     pred = model.predict(x)
     st.sidebar.success(f"✅ Kết quả dự đoán: **{pred[0]}**")
     if(pred[0]==1):
         st.sidebar.write("Người này có nguy cơ mắc bệnh")
     else:
         st.sidebar.write("Người này không có nguy cơ mắc bệnh")