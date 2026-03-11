import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Customer Segmentation",
    layout="wide"
)

# ---------------- CSS ---------------- #
st.markdown("""
<style>
.main {padding: 0rem 1rem;}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL + DATA ---------------- #
@st.cache_resource
def load_model_and_data():
    try:

        model = joblib.load("customer_segmentation_model.pkl")

        data = pd.read_csv("Mall_Customers.csv")

        # Ensure column names are correct
        data.columns = data.columns.str.strip()

        # Prepare features
        X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

        # Predict clusters
        data['Cluster'] = model.predict(X)
        data['Cluster'] = data['Cluster'].astype(int)

        return model, data

    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        return None, None


# ---------------- CLUSTER INFO ---------------- #
cluster_info = {

    0: {
        'name': 'Standard Customers',
        'description': 'Moderate Income, Moderate Spending',
        'strategy': 'Balanced product range, loyalty programs',
        'color': '#2ecc71'
    },

    1: {
        'name': 'Target Customers',
        'description': 'High Income, High Spending',
        'strategy': 'Premium products, VIP services - PRIORITY!',
        'color': '#e74c3c'
    },

    2: {
        'name': 'Careless Customers',
        'description': 'Low Income, High Spending',
        'strategy': 'Installment plans, credit options',
        'color': '#f39c12'
    },

    3: {
        'name': 'Sensible Customers',
        'description': 'High Income, Low Spending',
        'strategy': 'Quality emphasis, investment products',
        'color': '#9b59b6'
    },

    4: {
        'name': 'Careful Customers',
        'description': 'Low Income, Low Spending',
        'strategy': 'Budget-friendly products, discount offers',
        'color': '#3498db'
    }
}


# ---------------- HEADER ---------------- #
st.title("Customer Segmentation System")
st.subheader("AI-Powered Marketing Intelligence")


# ---------------- LOAD DATA ---------------- #
model, data = load_model_and_data()

if model is None or data is None:
    st.stop()


# ---------------- SIDEBAR ---------------- #
st.sidebar.title("Customer Profile")

st.sidebar.subheader("Enter Customer Details")

annual_income = st.sidebar.slider(
    "Annual Income (k$)", 10, 150, 50
)

spending_score = st.sidebar.slider(
    "Spending Score (1-100)", 1, 100, 50
)

st.sidebar.markdown("---")

segment_btn = st.sidebar.button(
    "Find Customer Segment",
    use_container_width=True
)


# ---------------- MAIN LOGIC ---------------- #
if segment_btn:

    input_data = np.array([[annual_income, spending_score]])
    predicted_cluster = model.predict(input_data)[0]

    cluster_details = cluster_info[predicted_cluster]

    st.markdown("---")
    st.header("Segmentation Results")

    col1, col2 = st.columns([2, 1])

    with col1:

        st.markdown(f"""
        <div style='background-color:{cluster_details["color"]};
        padding:25px;border-radius:12px;color:white'>

        <h2 style='color:white'>{cluster_details["name"]}</h2>
        <p>{cluster_details["description"]}</p>

        </div>
        """, unsafe_allow_html=True)

        cluster_customers = data[data['Cluster'] == predicted_cluster]

        st.subheader("Customer Characteristics")

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Average Age",
            f"{cluster_customers['Age'].mean():.0f}"
        )

        m2.metric(
            "Average Income",
            f"${cluster_customers['Annual Income (k$)'].mean():.0f}k"
        )

        m3.metric(
            "Average Spending",
            f"{cluster_customers['Spending Score (1-100)'].mean():.0f}/100"
        )

        st.subheader("Marketing Strategy")

        st.success(cluster_details["strategy"])

    with col2:

        st.subheader("Customer Position")

        fig = go.Figure()

        for cluster_id in cluster_info:

            cluster_data = data[data['Cluster'] == cluster_id]

            fig.add_trace(go.Scatter(

                x=cluster_data['Annual Income (k$)'],
                y=cluster_data['Spending Score (1-100)'],

                mode='markers',

                marker=dict(
                    size=10,
                    color=cluster_info[cluster_id]["color"],
                    opacity=0.6
                ),

                name=cluster_info[cluster_id]["name"]

            ))

        fig.add_trace(go.Scatter(

            x=[annual_income],
            y=[spending_score],

            mode='markers',

            marker=dict(
                size=20,
                color="black",
                symbol="star"
            ),

            name="Your Customer"

        ))

        fig.update_layout(
            xaxis_title="Annual Income (k$)",
            yaxis_title="Spending Score",
            height=420
        )

        st.plotly_chart(fig, use_container_width=True)

else:

    st.markdown("---")

    st.info("Enter income and spending score from the sidebar to find the segment.")

    st.subheader("Customer Segmentation Map")

    fig = go.Figure()

    for cluster_id in cluster_info:

        cluster_data = data[data['Cluster'] == cluster_id]

        fig.add_trace(go.Scatter(

            x=cluster_data['Annual Income (k$)'],
            y=cluster_data['Spending Score (1-100)'],

            mode='markers',

            marker=dict(
                size=10,
                color=cluster_info[cluster_id]["color"],
                opacity=0.7
            ),

            name=cluster_info[cluster_id]["name"]

        ))

    fig.update_layout(
        xaxis_title="Annual Income (k$)",
        yaxis_title="Spending Score",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Model Information")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Algorithm", "K-Means")
    c2.metric("Clusters", "5")
    c3.metric("Features", "2")
    c4.metric("Dataset Size", len(data))