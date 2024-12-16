import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Streamlit interfeysini sozlash
st.title("Car Price Prediction App")
st.write("Ushbu ilova avtomobil narxini bashorat qilish uchun yaratilgan.")

# Datasetni yuklash va ko'rib chiqish
uploaded_file = st.file_uploader("Datasetni yuklang (CSV formatda):", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Datasetning dastlabki 5 qatori:")
    st.dataframe(data.head())

    # Ma'lumotlarning umumiy ma'lumotlari
    st.write("\nMa'lumotlarning umumiy ma'lumotlari:")
    st.write(data.info())

    # Maqsad va xususiyatlarni aniqlash
    if 'Price' in data.columns:
        y = data['Price']
        X = data.drop(columns=['Price'])

        # Kategorik ma'lumotlarni kodlash (zarurat bo'lsa)
        X = pd.get_dummies(X, drop_first=True)

        # Ma'lumotlarni mashq va test to'plamiga ajratish
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Modelni tanlash va o'rgatish
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Bashorat qilish va baholash
        y_pred = model.predict(X_test)
        st.write("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
        st.write("R-squared Score:", r2_score(y_test, y_pred))

        # Foydalanuvchidan kiritish orqali bashorat qilish
        st.write("\nYangi ma'lumotlar orqali narxni bashorat qilish:")

        # Yangi ma'lumotlar uchun inputlar
        year = st.number_input("Yilini kiriting:", min_value=1900, max_value=2024, value=2020)
        mileage = st.number_input("Yurgan masofasini (km) kiriting:", min_value=0, value=50000)
        enginesize = st.number_input("Mator hajmini kiriting:", min_value=1, value=3)

        # Foydalanuvchidan kiritilgan ma'lumotlarni DataFrame shaklida olish
        user_input = pd.DataFrame({
            'Year': [year],
            'Mileage': [mileage],
            'enginesize': [enginesize]
        })

        # Kategorik ma'lumotlarni kodlash (agar kerak bo'lsa)
        user_input = pd.get_dummies(user_input, drop_first=True)

        # Modelga moslashtirish: Kiritilgan xususiyatlar modelda mavjud xususiyatlar bilan mos kelishi kerak
        user_input = user_input.reindex(columns=X.columns, fill_value=0)

        # Modelni bashorat qilish
        prediction = model.predict(user_input)
        st.write("Bashorat qilingan narx:", prediction[0])

        # 9. Modelni saqlash (opsional)
        if st.button("Modelni saqlash"):
            joblib.dump(model, 'car_price_model.pkl')
            st.write("Model saqlandi: car_price_model.pkl")

    else:
        st.write("'Price' ustuni datasetda mavjud emas. Iltimos, to'g'ri dataset yuklang.")
else:
    st.write("Datasetni yuklang va tahlil qilishni boshlang.")
