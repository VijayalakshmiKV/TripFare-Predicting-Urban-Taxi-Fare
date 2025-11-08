# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, time
import joblib

st.set_page_config(page_title="Taxi Fare Prediction (robust loader)", layout="centered")

# -------------------------
# Load model artifact robustly
# -------------------------
st.title("🚖 Taxi Fare Prediction — Robust Model Loader")

MODEL_PATH = "best_random_forest_model.pkl"

try:
    model_artifact = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model file '{MODEL_PATH}': {e}")
    st.stop()

# Unpack artifact: support a) direct estimator, b) dict containing model + features
if isinstance(model_artifact, dict):
    # Check common keys
    model = None
    feature_order = None

    # common keys used in different workflows
    for key in ("model", "estimator", "clf", "pipeline"):
        if key in model_artifact:
            model = model_artifact[key]
            break

    # fallback: maybe the dict *is* the estimator mapping (rare)
    if model is None:
        # try to find the first value that's an estimator-like object
        for v in model_artifact.values():
            if hasattr(v, "predict"):
                model = v
                break

    # features key (optional)
    for key in ("features", "feature_order", "columns"):
        if key in model_artifact:
            feature_order = model_artifact[key]
            break

    st.info("Loaded a dict from the joblib file.")
    if model is None:
        st.error("No estimator found inside the loaded dict. Please re-save your model as either the estimator or a dict with key 'model'/'estimator'.")
        st.write("Keys in the loaded dict:", list(model_artifact.keys()))
        st.stop()
else:
    model = model_artifact
    feature_order = None
    st.info("Loaded estimator object directly from the joblib file.")

# If feature order not provided, try to obtain from the estimator
if feature_order is None:
    if hasattr(model, "feature_names_in_"):
        try:
            feature_order = list(model.feature_names_in_)
            st.write("Using feature order from model.feature_names_in_.")
        except Exception:
            feature_order = None

# Final check
if not hasattr(model, "predict"):
    st.error("Loaded object is not an estimator (no .predict method). Please save the trained model correctly.")
    st.stop()

# -------------------------
# UI (inputs)
# -------------------------
st.header("Enter trip details")

vendor_map = {"Ola": 1, "Uber": 2}
ratecode_map = {"Standard Fare": 1, "Airport Trip": 2, "Outstation": 3, "Night Rate": 4, "Festival Surge": 5, "Shared Ride": 6}
payment_map = {"Credit Card": 1, "Cash": 2, "UPI": 3, "Wallet (Paytm/PhonePe)": 4, "Corporate Account": 5, "Other": 6}

col1, col2 = st.columns(2)
with col1:
    vendor = st.selectbox("Vendor", list(vendor_map.keys()))
    ratecode = st.selectbox("Rate Type", list(ratecode_map.keys()))
    payment = st.selectbox("Payment Method", list(payment_map.keys()))
    store_flag = st.selectbox("Store & Forward Flag", ["Yes", "No"])
    passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, value=1)
with col2:
    pickup_lat = st.number_input("Pickup Latitude", value=19.0760, format="%.6f")
    pickup_long = st.number_input("Pickup Longitude", value=72.8777, format="%.6f")
    dropoff_lat = st.number_input("Dropoff Latitude", value=19.2183, format="%.6f")
    dropoff_long = st.number_input("Dropoff Longitude", value=72.9781, format="%.6f")

# persistent times so widget stays editable
if "pickup_time" not in st.session_state:
    st.session_state.pickup_time = time(10, 0)
if "dropoff_time" not in st.session_state:
    st.session_state.dropoff_time = time(10, 30)

pickup_date = st.date_input("Pickup Date", date.today())
pickup_time = st.time_input("Pickup Time", value=st.session_state.pickup_time, key="pickup_time")
dropoff_date = st.date_input("Dropoff Date", date.today())
dropoff_time = st.time_input("Dropoff Time", value=st.session_state.dropoff_time, key="dropoff_time")

pickup_dt = datetime.combine(pickup_date, pickup_time)
dropoff_dt = datetime.combine(dropoff_date, dropoff_time)
trip_duration_min = max((dropoff_dt - pickup_dt).total_seconds() / 60, 0.0)

col3, col4 = st.columns(2)
with col3:
    extra = st.number_input("Extra Charges (₹)", min_value=0.0, value=0.0)
    mta_tax = st.number_input("MTA Tax (₹)", min_value=0.0, value=0.0)
with col4:
    tip_amount = st.number_input("Tip Amount (₹)", min_value=0.0, value=0.0)
    tolls_amount = st.number_input("Tolls Amount (₹)", min_value=0.0, value=0.0)
improvement_surcharge = st.number_input("Improvement Surcharge (₹)", min_value=0.0, value=0.0)

# derived features (must match training transformations)
pickup_hour = pickup_dt.hour
am_pm = 1 if pickup_hour >= 12 else 0
is_weekend = 1 if pickup_dt.weekday() >= 5 else 0
is_night = 1 if (pickup_hour >= 22 or pickup_hour < 6) else 0

# compute trip_distance using Haversine (in km)
from math import radians, sin, cos, atan2, sqrt
R = 6371.0
lat1, lon1, lat2, lon2 = map(radians, [pickup_lat, pickup_long, dropoff_lat, dropoff_long])
dlat = lat2 - lat1
dlon = lon2 - lon1
a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
c = 2 * atan2(sqrt(a), sqrt(1-a))
trip_distance = R * c
trip_distance_log = np.log1p(trip_distance)
trip_duration_min_log = np.log1p(trip_duration_min)

# optional fare fields (if your model used them); adjust or add UI inputs if you trained with them
fare_amount = 0.0
fare_amount_log = np.log1p(fare_amount)
total_amount = fare_amount + extra + mta_tax + tip_amount + tolls_amount + improvement_surcharge
total_amount_log = np.log1p(total_amount)

# Build full candidate input dict (include everything you might have used during training)
input_row = {
    'VendorID': vendor_map[vendor],
    'RatecodeID': ratecode_map[ratecode],
    'store_and_fwd_flag': 1 if store_flag == "Yes" else 0,
    'payment_type': payment_map[payment],
    'passenger_count': passenger_count,
    'pickup_longitude': pickup_long,
    'pickup_latitude': pickup_lat,
    'dropoff_longitude': dropoff_long,
    'dropoff_latitude': dropoff_lat,
    'fare_amount': fare_amount,
    'extra': extra,
    'mta_tax': mta_tax,
    'tip_amount': tip_amount,
    'tolls_amount': tolls_amount,
    'improvement_surcharge': improvement_surcharge,
    'trip_distance': trip_distance,
    'trip_distance_log': trip_distance_log,
    'trip_duration_min': trip_duration_min,
    'trip_duration_min_log': trip_duration_min_log,
    'fare_amount_log': fare_amount_log,
    'total_amount_log': total_amount_log,
    'pickup_hour': pickup_hour,
    'am_pm': am_pm,
    'is_night': is_night,
    'is_weekend': is_weekend
}

input_df = pd.DataFrame([input_row])

# If we have a known feature_order, use it to align columns (add missing with 0)
if feature_order is not None:
    missing = [c for c in feature_order if c not in input_df.columns]
    if missing:
        for c in missing:
            input_df[c] = 0
    # reorder
    input_df = input_df[feature_order]
else:
    # fallback: if model has feature_names_in_, use that
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        missing = [c for c in expected if c not in input_df.columns]
        if missing:
            for c in missing:
                input_df[c] = 0
        input_df = input_df[expected]
    else:
        # no feature order available; proceed but warn user
        st.warning("Model feature order not available — input columns will be sent in current order. If you get a mismatch error, re-save the model with feature order or re-train.")
        # no reordering performed

st.write("### Features sent to model (preview)")
st.dataframe(input_df.T, height=300)

# Predict
if st.button("🔮 Predict Fare"):
    try:
        pred = model.predict(input_df)[0]
        st.success(f"💰 Estimated Total Fare: ₹{pred:.2f}")
    except Exception as e:
        st.error("Prediction failed — see details below.")
        st.exception(e)
        # helpful debug info
        st.write("Loaded model type:", type(model))
        if isinstance(model_artifact, dict):
            st.write("Loaded artifact keys:", list(model_artifact.keys()))
        if feature_order is not None:
            st.write("Using feature order (len={}):".format(len(feature_order)))
            st.write(feature_order)
        elif hasattr(model, "feature_names_in_"):
            st.write("Model.feature_names_in_ (len={}):".format(len(model.feature_names_in_)))
            st.write(list(model.feature_names_in_))
        st.write("Input columns:", list(input_df.columns))
