# 🧠 Customer Segmentation using K-Means Clustering

## 📌 Project Overview

This project focuses on segmenting customers into different groups based on their purchasing behavior using the **K-Means Clustering algorithm**.

Customer segmentation helps businesses understand different types of customers and design **targeted marketing strategies** for each group.

The model groups customers based on:

- 💰 **Annual Income**
- 🛍️ **Spending Score**

The trained model is deployed as an **interactive web application using Streamlit Community Cloud**, where users can input customer information and instantly identify the customer segment.

---

## ❓ Problem Statement

Businesses often have large numbers of customers with different spending patterns. Treating all customers the same leads to inefficient marketing strategies.

Customer segmentation helps businesses:

- 🎯 Identify high-value customers
- 📊 Understand spending behavior
- 📢 Personalize marketing campaigns
- 🤝 Improve customer retention and profitability

---

## 📂 Dataset

The project uses the **Mall Customers dataset**, which contains demographic and spending information of customers.

Features used for clustering:

- 💰 Annual Income (k$)
- 🛍️ Spending Score (1–100)

These features help identify patterns in customer purchasing behavior.

---

## 🤖 Machine Learning Approach

This project uses the **K-Means Clustering algorithm**, an unsupervised machine learning technique used to group similar data points together.

Steps involved:

1️⃣ Data preprocessing  
2️⃣ Feature selection  
3️⃣ Determining optimal number of clusters  
4️⃣ Training the K-Means model  
5️⃣ Assigning cluster labels to customers  
6️⃣ Visualizing customer segments  

The model groups customers into **5 distinct segments** based on their income and spending behavior.

---

## 👥 Customer Segments Identified

The model identifies the following customer groups:

### 🔴 Standard Customers  
Moderate income and moderate spending behavior.

### 🔵 Target Customers  
High Income and High Spending customers — **priority segment for premium marketing.**

### 🟢 Careless Customers  
Low income but high spending behavior.

### 🟣 Sensible Customers  
High income but low spending behavior.

### 🟠 Careful Customers  
Low income and low spending behavior.

Each segment allows businesses to apply **different marketing strategies**.

---

## 🌐 Web Application (Streamlit)

The trained model is deployed as an **interactive Streamlit web application**.

Users can:

✔ Enter **Annual Income**  
✔ Enter **Spending Score**  
✔ Instantly identify the **customer segment**

The application also provides:

- 📊 Customer segment description
- 📈 Recommended marketing strategy
- 🧭 Customer cluster visualization
- 📋 Segment comparison insights

---

## ⭐ Key Features of the Application

- 🔍 Interactive **customer segmentation prediction**
- 📊 Visual **scatter plot of customer clusters**
- 📈 Customer **segment insights and marketing strategies**
- 📑 Comparison of all customer segments
- ⚡ Real-time prediction using trained machine learning model

---

## 🛠️ Technologies Used

- 🐍 Python  
- 📈 Pandas  
- 🔢 NumPy  
- 🤖 Scikit-learn  
- 📊 Matplotlib & Seaborn 
- 🌐 Streamlit  
- 💾 Joblib  

---

## ⚙️ Model Information

- 🤖 Algorithm: **K-Means Clustering**
- 🔢 Number of Clusters: **5**
- 📊 Features Used: **Annual Income, Spending Score**

---

## 🚀 Future Improvements

- ➕ Use additional features such as age, gender, and purchase history
- 📊 Deploy the application with a larger real-world dataset
- 🤖 Implement advanced clustering techniques such as **DBSCAN or Hierarchical Clustering**
- 📈 Build a **business dashboard for marketing teams**

---

## 📌 Conclusion

This project demonstrates how **unsupervised machine learning techniques can be used to understand customer behavior and improve marketing strategies**.

By identifying different customer segments, businesses can create **personalized marketing campaigns and improve customer engagement**.

---

## 👨‍💻 Author

**Rahul Jimmy**

Aspiring **Data Scientist**

🔗 Live App: (https://customer-segmentation--5.streamlit.app)  
💻 GitHub: (https://github.com/rahuljimmy)
