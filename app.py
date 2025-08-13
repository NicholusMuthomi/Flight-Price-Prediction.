import numpy as np
import streamlit as st
import pickle
from datetime import datetime, timedelta
import os
import pandas as pd
import plotly.express as px
import warnings
import base64

warnings.filterwarnings("ignore")

# ---------- Page config ----------
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Load model ----------
try:
    model = pickle.load(open('flight_price_prediction_model.pkl', 'rb'))
except Exception as e:
    st.error("Failed to load flight_price_prediction_model.pkl. Please ensure the file exists and is a valid pickle file.")
    st.stop()

# ---------- Background helper ----------
def add_bg_image(image_file):
    try:
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: linear-gradient(rgba(0, 0, 0, 0.55), rgba(0, 0, 0, 0.55)),
                                  url(data:image/jpeg;base64,{encoded});
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning(f"Background image '{image_file}' not found. Using fallback gradient.")
        st.markdown(
            """
            <style>
            .stApp {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            </style>
            """,
            unsafe_allow_html=True
        )

if os.path.exists("assets/flight.jpg"):
    add_bg_image("assets/flight.jpg")
elif os.path.exists("flight.jpg"):
    add_bg_image("flight.jpg")
else:
    add_bg_image("assets/flight.jpg")

# ---------- CSS ----------
st.markdown(
    """
    <style>
    /* Liquid-glass / glassmorphism for main card */
    .main > div {
        background: rgba(255, 255, 255, 0.08) !important;
        border-radius: 20px !important;
        backdrop-filter: blur(18px) saturate(150%) !important;
        -webkit-backdrop-filter: blur(18px) saturate(150%) !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        box-shadow:
            0 8px 32px 0 rgba(31, 38, 135, 0.30),
            inset 0 1px 0 0 rgba(255, 255, 255, 0.25) !important;
        padding: 2rem 2.5rem 2.5rem 2.5rem !important;
    }

    /* Dark inputs / selectors */
    .stSelectbox > div > div,
    .stDateInput > div > div,
    .stNumberInput > div > div {
        background-color: rgba(10, 10, 10, 0.85) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 8px;
    }

    /* Force all text white + shadow for readability */
    html, body, .stApp,
    h1, h2, h3, h4, h5, h6,
    label, .stMarkdown, .stAlert,
    .css-1d391kg, .css-1y4p8pa, .stSidebar > div {
        color: #ffffff !important;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.70) !important;
    }

    .stSidebar{
    background: linear-gradient(180deg, rgba(0, 0, 0, 0.7) 10%, transparent);
    padding: 20px 0;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 1000;
    backdrop-filter: blur(10px);
    }

    /* Button styling kept from original */
    .stButton>button {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }

    .prediction-card {
    background: linear-gradient(135deg, rgba(20, 20, 20, 0.95), rgba(40, 40, 40, 0.95));
    border-radius: 15px;
    padding: 2rem;
    margin-top: 2rem;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #ffffff;
}
    .price-display {
        font-size: 3rem;
        font-weight: bold;
        color: #e74c3c;
        text-align: center;
        margin: 1rem 0;
    }

    /* --- Cards / containers --- */
    .card {
    background-color: var(--bg-secondary);
    border: 1px solid var(--bg-tertiary);
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Dictionaries ----------
airline_dict = {'AirAsia': 0, 'Indigo': 1, 'GO_FIRST': 2, 'SpiceJet': 3, 'Air_India': 4, 'Vistara': 5}
source_dict = {'Delhi': 0, 'Hyderabad': 1, 'Bangalore': 2, 'Mumbai': 3, 'Kolkata': 4, 'Chennai': 5}
departure_dict = {'Late Night (12AM-6AM)': 0, 'Morning (6AM-12PM)': 1, 'Afternoon (12PM-6PM)': 2,
                  'Evening (6PM-9PM)': 3, 'Night (9PM-12AM)': 4, 'Early Morning (12AM-6AM)': 5,}
stops_dict = {'Non-stop': 0, '1 Stop': 1, '2+ Stops': 2}
arrival_dict = {'Early Morning (12AM-6AM)': 0, 'Morning (6AM-12PM)': 1, 'Afternoon (12PM-6PM)': 2,
                'Evening (6PM-9PM)': 3, 'Night (9PM-12AM)': 4, 'Late Night (12AM-6AM)': 5}
destination_dict = {'Hyderabad': 0, 'Delhi': 1, 'Mumbai': 2, 'Bangalore': 3, 'Chennai': 4, 'Kolkata': 5}
class_dict = {'Economy': 0, 'Business': 1}

# ---------- Main app ----------
def main():
    st.markdown('<h1 class="header"> Flight Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Predict flight prices for domestic flights in India based on various factors.</p>', unsafe_allow_html=True)

    with st.expander("ℹ️ About this app", expanded=False):
        st.write("""
        This app predicts flight prices using a machine-learning model trained on historical flight data.
        It considers factors like airline, route, departure time, stops and travel class.
        """)
        st.write("Dataset: [Kaggle Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<h3>Flight Details</h3>', unsafe_allow_html=True)
        source_city = st.selectbox("Departure City", list(source_dict.keys()))
        destination_city = st.selectbox("Destination City", list(destination_dict.keys()))
        departure_date = st.date_input(
            "Departure Date",
            min_value=datetime.today(),
            max_value=datetime.today() + timedelta(days=365),
            value=datetime.today() + timedelta(days=7)
        )
        stops = st.radio("Number of Stops", list(stops_dict.keys()), horizontal=True)
        travel_class = st.radio("Class", list(class_dict.keys()), horizontal=True)
        duration = st.number_input("Flight Duration (hours)", min_value=1.0, max_value=24.0, value=2.5, step=0.5)

    with col2:
        st.markdown('<h3>Flight Preferences</h3>', unsafe_allow_html=True)
        airline = st.selectbox("Airline", list(airline_dict.keys()))
        departure_time = st.selectbox("Departure Time", list(departure_dict.keys()))
        arrival_time = st.selectbox("Arrival Time", list(arrival_dict.keys()))

    if st.button("Predict Flight Price", use_container_width=True):
        try:
            date_diff = (departure_date - datetime.today().date()).days + 1
            features = np.array([
                airline_dict[airline],
                source_dict[source_city],
                departure_dict[departure_time],
                stops_dict[stops],
                arrival_dict[arrival_time],
                destination_dict[destination_city],
                class_dict[travel_class],
                duration
            ]).reshape(1, -1)
            prediction = round(model.predict(features)[0], 2)

            with st.container():
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.markdown('<h3>Predicted Flight Price</h3>', unsafe_allow_html=True)
                st.markdown(f'<div class="price-display">₹{prediction:,}</div>', unsafe_allow_html=True)

                details = pd.DataFrame({
                    "Airline": [airline],
                    "Route": [f"{source_city} → {destination_city}"],
                    "Departure Date": [departure_date.strftime('%Y-%m-%d')],
                    "Departure Time": [departure_time],
                    "Arrival Time": [arrival_time],
                    "Duration (h)": [duration],
                    "Stops": [stops],
                    "Class": [travel_class]
                })
                st.table(details)

                airlines = list(airline_dict.keys())
                prices = [round(model.predict(np.array([
                    [airline_dict[a], source_dict[source_city],
                    departure_dict[departure_time], stops_dict[stops],
                    arrival_dict[arrival_time], destination_dict[destination_city],
                    class_dict[travel_class], duration]  # Removed date_diff
                ]))[0], 2) for a in airlines]
                fig = px.bar(x=airlines, y=prices, labels={'x': 'Airline', 'y': 'Price (₹)'},
                            title='Price Comparison by Airline', color=airlines)
                fig.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

        st.markdown(
            """
        <div class="card" style="text-align:center;margin-top:2rem;">
        <p style="color:var(--text-muted);margin:0;font-size:clamp(0.75rem, 1.5vw, 0.875rem);">
                DISCLAIMER:
                The machine learning model for this application is over 1.4GB and exceeds GitHub’s 25MB file size limit.
                The file flight_price_prediction_model_random_forest.pkl is not included in this repository.
                To run this app locally, train your own model using the provided notebook or request the file from  [Me](https://nicholusmuthomi.website/) through an alternative transfer method.
                The app.py code is fully functional and will work once the correct model.pkl file is placed in the specified directory.
        </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
# ---------- Sidebar ----------
with st.sidebar:
    st.markdown('<h2>About</h2>', unsafe_allow_html=True)
    st.write("This flight price predictor uses machine learning to estimate ticket prices for domestic flights in India. The model was trained on historical flight data and considers various factors that influence pricing.")
    st.markdown('<h3>Tips for Cheaper Flights</h3>', unsafe_allow_html=True)
    st.write("""
    - Book 4–6 weeks in advance  
    - Consider one-stop flights  
    - Late-night flights are often cheaper  
    - Mid-week flights tend to cost less  
    """)
    st.markdown('<h3>Popular Routes</h3>', unsafe_allow_html=True)
    st.write("Delhi → Mumbai\n\nBangalore → Delhi\n\nHyderabad → Bangalore\n\nMumbai → Kolkata\n\nChennai → Delhi")

if __name__ == '__main__':
    main()
