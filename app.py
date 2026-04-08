import gradio as gr
import pandas as pd
import joblib
import numpy as np

# Dosyaları yükle
model = joblib.load("sleep_model.joblib")
le = joblib.load("label_encoder.joblib")
model_columns = joblib.load("model_columns.joblib")

def predict(age, gender, occupation, sleep_dur, sleep_qual, phys_act, stress, bmi_cat, blood_p, heart_rate, daily_steps):
    # 1. Gelen verileri DataFrame yap
    data = pd.DataFrame([[age, gender, occupation, sleep_dur, sleep_qual, phys_act, stress, bmi_cat, blood_p, heart_rate, daily_steps]],
                        columns=['Age', 'Gender', 'Occupation', 'Sleep Duration', 'Quality of Sleep', 
                                 'Physical Activity Level', 'Stress Level', 'BMI Category', 
                                 'Blood Pressure', 'Heart Rate', 'Daily Steps'])

    # 2. SENİN YAPTIĞIN FEATURE ENGINEERING İŞLEMLERİ
    data[['Systolic', 'Diastolic']] = data['Blood Pressure'].str.split('/', expand=True).astype(int)
    data['Pulse_Pressure'] = data['Systolic'] - data['Diastolic']
    data['MAP'] = (data['Systolic'] + (2 * data['Diastolic'])) / 3
    data['RPP'] = data['Systolic'] * data['Heart Rate']
    data.drop(columns=['Blood Pressure'], inplace=True)

    data['BMI Category'] = data['BMI Category'].replace({'Normal Weight': 'Normal'})
    bmi_mapping = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}
    data['BMI_Score'] = data['BMI Category'].map(bmi_mapping)
    data.drop(columns=['BMI Category'], inplace=True)

    data['Sleep_Quality_Ratio'] = data['Quality of Sleep'] / data['Sleep Duration']
    data['Activity_Intensity'] = data['Daily Steps'] / data['Physical Activity Level']
    
    # Meslek gruplama
    keep_occupations = ['Doctor', 'Nurse', 'Engineer', 'Lawyer', 'Teacher', 'Accountant', 'Salesperson']
    data['Occupation'] = data['Occupation'].apply(lambda x: x if x in keep_occupations else 'Other')
    
    data['Risk_Factor_Apnea'] = data['Age'] * data['BMI_Score']
    data['Insomnia_Index'] = data['Stress Level'] / (data['Quality of Sleep'] + 0.1)

    # 3. Dummy Variables (Kategorik verileri 0-1 yapma)
    data_dummies = pd.get_dummies(data)
    
    # Eksik sütunları tamamla (Modelin beklediği ama o an olmayan sütunlar için)
    final_df = pd.DataFrame(columns=model_columns)
    final_df = pd.concat([final_df, data_dummies]).fillna(0)
    final_df = final_df[model_columns] # Sütun sırasını eşitle

    # 4. Tahmin
    prediction = model.predict(final_df)
    result = le.inverse_transform(prediction)[0]
    
    return result

# Arayüzü oluştur
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Yaş"),
        gr.Dropdown(["Male", "Female"], label="Cinsiyet"),
        gr.Dropdown(['Doctor', 'Nurse', 'Engineer', 'Lawyer', 'Teacher', 'Accountant', 'Salesperson', 'Other'], label="Meslek"),
        gr.Number(label="Uyku Süresi (Saat)"),
        gr.Slider(1, 10, label="Uyku Kalitesi"),
        gr.Number(label="Fiziksel Aktivite Düzeyi"),
        gr.Slider(1, 10, label="Stres Seviyesi"),
        gr.Dropdown(["Normal", "Overweight", "Obese", "Underweight"], label="BMI Kategorisi"),
        gr.Textbox(label="Kan Basıncı (Örn: 120/80)"),
        gr.Number(label="Kalp Atış Hızı"),
        gr.Number(label="Günlük Adım Sayısı")
    ],
    outputs="text",
    title="Uyku Bozukluğu Tahmin Sistemi"
)

iface.launch()