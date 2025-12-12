import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px


with open("model/xgb_model_2.pkl", "rb") as f:    
    model = pickle.load(f)

#Analysis Page
st.title("Stroke Prediction Analysis Page")
data_stroke = pd.read_csv("data/DataStroke_OK2.csv")


st.markdown("----")

# ======== Visualization ========
#Biểu đồ 1:
st.subheader("BIểu đồ thể hiện mức độ ảnh hưởng của các thuộc tính tới việc dự đoán đột quỵ")
feature_names = data_stroke.drop(columns=['Unnamed: 0', 'stroke']).columns
feature_importances = model.feature_importances_

df_imp = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=True)



fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(df_imp["Feature"], df_imp["Importance"])
ax.grid(axis='x', linestyle='--', alpha=0.7)
ax.set_xlabel("Mức độ ảnh hưởng")
ax.set_ylabel("Thuộc tính")
ax.set_title("Feature Importance của mô hình")
st.pyplot(fig)

st.write("Tuổi tác có ảnh hưởng lớn nhất tới việc dự đoán đột quỵ.")
st.write("Tiếp theo là các yếu tố như loại hình làm việc, khu vực sinh sống,tình trạng hôn nhân.")
st.write("Các yếu tố như tiền sử bệnh tim, huyết áp cao cũng có ảnh hưởng đáng kể.")
st.write("Các yếu tố như chỉ số BMI, mức đường huyết, giới tính có ảnh hưởng thấp hơn.")



#BIểu đồ 2:
st.subheader("Phân bố các biến số theo tình trạng đột quỵ")
num_cols = ['age', 'avg_glucose_level', 'bmi']
bin_cols = ['hypertension', 'heart_disease']
cat_cols = ['gender', 'residence_type', 'work_type', 'smoking_status']

fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
colors = ['#5B8FF9', '#5AD8A6']
for i, col in enumerate(num_cols):
    sub = data_stroke.dropna(subset=[col])
    v0 = sub.loc[sub['stroke']==0, col].values
    v1 = sub.loc[sub['stroke']==1, col].values
    bins = np.histogram_bin_edges(np.concatenate([v0, v1]), bins=40)
    axes1[i].hist(v0, bins=bins, alpha=0.6, color=colors[0], label='Không đột quỵ', edgecolor='black')
    axes1[i].hist(v1, bins=bins, alpha=0.6, color=colors[1], label='Có đột quỵ', edgecolor='black')
    axes1[i].set_title(f'Phân bố {col} theo tình trạng đột quỵ')
    axes1[i].set_xlabel(col); axes1[i].set_ylabel('Tần suất')
    axes1[i].grid(axis='y', linestyle='--', alpha=0.3)
axes1[0].legend()
plt.tight_layout(); plt.savefig('stroke_fig_numeric_hist.png', bbox_inches='tight')
st.pyplot(fig1)

st.write("Nhìn chung, những người bị đột quỵ có xu hướng lớn tuổi hơn.")
st.write("Mức đường huyết trung bình của nhóm đột quỵ cũng cao hơn so với nhóm không đột quỵ.")
st.write("Chỉ số BMI của cả hai nhóm không có sự khác biệt rõ ràng.")



#Biểu đồ 4:
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
for i, col in enumerate(bin_cols):
    sub = data_stroke.dropna(subset=[col]); sub[col] = sub[col].astype(int)
    rate = (sub.groupby(col)['stroke'].mean() * 100).sort_index()
    count = sub.groupby(col)['stroke'].count().sort_index()
    xs = [0,1]
    axes2[i].bar(xs, rate.values, color=colors)
    axes2[i].set_xticks(xs); axes2[i].set_xticklabels(['Không','Có'])
    axes2[i].set_title(f'Tỷ lệ đột quỵ theo {col} (%)'); axes2[i].set_ylabel('% đột quỵ')
    axes2[i].set_ylim(0, max(rate.values)*1.25); axes2[i].grid(axis='y', linestyle='--', alpha=0.3)
    for j, v in enumerate(rate.values):
        axes2[i].text(xs[j], v + max(rate.values)*0.02, f'{v:.1f}%\\n(n={count.iloc[j]})',
                      ha='center', va='bottom', fontsize=9)
plt.tight_layout(); plt.savefig('stroke_fig_binary_rates.png', bbox_inches='tight') 
st.pyplot(fig2)

#Biểu đồ 5:
fig3, axes3 = plt.subplots(2, 2, figsize=(16, 10), dpi=150)
axes3 = axes3.ravel()
for i, col in enumerate(cat_cols):
    sub = data_stroke.dropna(subset=[col]); sub[col] = sub[col].astype(str).str.strip()
    tab = sub.groupby(['stroke', col]).size().unstack(fill_value=0)
    pct = (tab.T / tab.sum(axis=1)).T * 100
    x = np.array([0,1]); b0 = 0; b1 = 0; cats = list(pct.columns)
    for j, cat in enumerate(cats):
        c = plt.cm.Set3(j / max(1, len(cats)))
        axes3[i].bar(0, pct.loc[0, cat] if 0 in pct.index else 0, bottom=b0, color=c, edgecolor='black')
        axes3[i].bar(1, pct.loc[1, cat] if 1 in pct.index else 0, bottom=b1, color=c, edgecolor='black')
        b0 += pct.loc[0, cat] if 0 in pct.index else 0
        b1 += pct.loc[1, cat] if 1 in pct.index else 0
    axes3[i].set_xticks([0,1]); axes3[i].set_xticklabels(['Không đột quỵ','Có đột quỵ'])
    axes3[i].set_title(f'Phân bố {col} theo nhóm đột quỵ (tỷ lệ %)'); axes3[i].set_ylabel('%'); axes3[i].set_ylim(0,100)
    axes3[i].grid(axis='y', linestyle='--', alpha=0.3)
    handles = [plt.Rectangle((0,0),1,1,color=plt.cm.Set3(j / max(1, len(cats)))) for j in range(len(cats))]
    axes3[i].legend(handles, cats, title=col, bbox_to_anchor=(1.02,1), loc='upper left', fontsize=8)
plt.tight_layout(); plt.savefig('stroke_fig_categorical_stacked.png', bbox_inches='tight')
st.pyplot(fig3)

st.write("Tỷ lệ đột quỵ cao hơn ở những người có tiền sử huyết áp cao và bệnh tim.")
st.write("Những người sống ở khu vực thành thị (Urban) có tỷ lệ đột quỵ cao hơn so với khu vực nông thôn (Rural).")
st.write("Những người làm việc trong các công việc chính phủ (Govt_job) và những người tự kinh doanh (Self-employed) có tỷ lệ đột quỵ cao hơn so với những người làm việc trong khu vực tư nhân (Private).")
st.write("Những người hút thuốc hiện tại (Smokes) và những người từng hút thuốc (Formerly smoked) có tỷ lệ đột quỵ cao hơn so với những người chưa bao giờ hút thuốc (Never smoked).")











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