import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation",
    layout="wide"
)

# Load model and data
@st.cache_resource
def load_model_and_data():
    try:
        model = joblib.load('customer_segmentation_model.pkl')
        data = pd.read_csv('Mall_Customers.csv')

        X = data.iloc[:, [3, 4]].values
        data['Cluster'] = model.predict(X)

        return model, data

    except FileNotFoundError:
        return None, None


# Cluster information
cluster_info = {

    0: {
        'name': 'Standard Customers',
        'description': 'Moderate Income, Moderate Spending',
        'strategy': 'Balanced product range, loyalty programs',
        'color': '#e74c3c'                                        
    },

    1: {
        'name': 'Target Customers',
        'description': 'High Income, High Spending',
        'strategy': 'Premium products, VIP services - PRIORITY!',
        'color': '#3498db'
    },

    2: {
        'name': 'Careless Customers',                        
        'description': 'Low Income, High Spending',
        'strategy': 'Installment plans, credit options',
        'color': '#2ecc71'                                          
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
        'color': '#f39c12'
    }
}

# Header
st.title("Customer Segmentation System")
st.subheader("AI-Powered Marketing Intelligence")

model, data = load_model_and_data()

if model is None:
    st.error("Model not found.")
    st.stop()

# Sidebar
st.sidebar.title("Customer Profile")
st.sidebar.subheader("Enter Customer Details")

annual_income = st.sidebar.slider('Annual Income (in thousands $)', 10, 150, 50)
spending_score = st.sidebar.slider('Spending Score (1-100)', 1, 100, 50)

st.sidebar.markdown("---")
segment_btn = st.sidebar.button("Find Customer Segment", use_container_width=True)

# ---------------- SEGMENT PREDICTION ---------------- #

if segment_btn:

    input_data = np.array([[annual_income, spending_score]])
    predicted_cluster = model.predict(input_data)[0]

    cluster_details = cluster_info[predicted_cluster]

    st.markdown("---")
    st.header("Segmentation Results")

    col1, col2 = st.columns([2, 1])

    with col1:

        st.markdown(f"""
        <div style='background-color:{cluster_details['color']};
        padding:25px;border-radius:10px;color:white'>
        <h2>{cluster_details['name']}</h2>
        <p>{cluster_details['description']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Customer Characteristics")

        cluster_customers = data[data['Cluster'] == predicted_cluster]

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("Average Age",
                      f"{cluster_customers['Age'].mean():.0f} years")

        with m2:
            st.metric("Average Income",
                      f"${cluster_customers['Annual Income (k$)'].mean():.0f}k")

        with m3:
            st.metric("Average Spending",
                      f"{cluster_customers['Spending Score (1-100)'].mean():.0f}/100")

        st.markdown("### Marketing Strategy")

        st.success(cluster_details['strategy'])

        cluster_size = len(cluster_customers)
        total_customers = len(data)

        st.info(
            f"Segment Size: {cluster_size} customers ({cluster_size/total_customers*100:.1f}%)")

    # ---------------- MATPLOTLIB VISUALIZATION ---------------- #

    with col2:

        st.markdown("### Your Position")

        fig, ax = plt.subplots(figsize=(5,4))

        sns.scatterplot(
            data=data,
            x='Annual Income (k$)',
            y='Spending Score (1-100)',
            hue='Cluster', 
            palette='Set1',
            s=80,
            ax=ax
        )

        ax.scatter(
            annual_income,
            spending_score,
            s=250,
            c='black',
            marker='*',
            label='Your Customer'
        )

        ax.set_title("Customer Position")

        st.pyplot(fig)

    # ---------------- ALL SEGMENTS ---------------- #

    st.markdown("---")
    st.header("All Customer Segments")

    cols = st.columns(5)

    for idx, (cluster_id, info) in enumerate(cluster_info.items()):

        with cols[idx]:

            cluster_customers = data[data['Cluster'] == cluster_id]

            st.markdown(f"""
            <div style='background-color:{info['color']};
            padding:15px;border-radius:8px;color:white;text-align:center'>
            <h4>{info['name']}</h4>
            <h2>{len(cluster_customers)}</h2>
            <p>{info['description']}</p>
            </div>
            """, unsafe_allow_html=True)

    # ---------------- SEGMENT COMPARISON ---------------- #

    st.markdown("---")
    st.subheader("Segment Comparison")

    comparison_data = []

    for cluster_id in range(5):

        cluster_customers = data[data['Cluster'] == cluster_id]

        comparison_data.append({

            'Segment': cluster_info[cluster_id]['name'],
            'Count': len(cluster_customers),
            'Avg Income': f"${cluster_customers['Annual Income (k$)'].mean():.0f}k",
            'Avg Spending': f"{cluster_customers['Spending Score (1-100)'].mean():.0f}/100",
            'Avg Age': f"{cluster_customers['Age'].mean():.0f}",
            'Strategy': cluster_info[cluster_id]['strategy']

        })

    comparison_df = pd.DataFrame(comparison_data)

    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# ---------------- INITIAL PAGE ---------------- #

else:

    st.markdown("---")
    st.info("Enter income and spending score in the sidebar to find their segment")

    st.subheader("Customer Segmentation Map")

    fig, ax = plt.subplots(figsize=(10,6))

    sns.scatterplot(
        data=data,
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        hue='Cluster',             
        palette='Set1',
        s=100,
        ax=ax
    )

    st.pyplot(fig)

    # ---------------- SEGMENT CARDS ---------------- #

    st.markdown("### Five Customer Segments")

    cols = st.columns(5)

    for idx, (cluster_id, info) in enumerate(cluster_info.items()):

        with cols[idx]:

            cluster_customers = data[data['Cluster'] == cluster_id]

            st.markdown(f"""
            <div style='background-color:{info['color']};
            padding:20px;border-radius:8px;color:white'>
            <h4>{info['name']}</h4>
            <p>{info['description']}</p>
            <p>{len(cluster_customers)} customers</p>
            </div>
            """, unsafe_allow_html=True)

    # ---------------- KEY INSIGHTS ---------------- #

    st.markdown("---")
    st.subheader("Key Insights")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.info("Target Segment\n\nCluster 1 :- Customers have high value.")

    with c2:
        st.success("Growth Opportunity\n\nCluster 3 :- Customers have potential.")

    with c3:
        st.warning("Volume Sales\n\nCluster 4 :- Customers respond to discounts.")

    # ---------------- MODEL INFO ---------------- #

    st.markdown("---")
    st.subheader("Model Info")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Algorithm", "K-Means")
    m2.metric("Clusters", "5")
    m3.metric("Features", "2")
    m4.metric("Data Size", len(data))






