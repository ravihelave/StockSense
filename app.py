import streamlit as st
import pandas as pd
import mplfinance as mpf
from PIL import Image
from datetime import datetime, timedelta
from io import BytesIO
import requests
import os
import logging
from pathlib import Path
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Use environment variables for sensitive data
API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY', '60DQ9Y2MHHSMMEA0')
MODEL_PATH = os.environ.get('YOLO_MODEL_PATH', 'weights/custom_yolov8.pt')
LOGO_PATH = os.environ.get('LOGO_PATH', 'images/Logo1.png')

# --- User Management ---


USER_FILE = "users.csv"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if os.path.exists(USER_FILE):
        return pd.read_csv(USER_FILE)
    else:
        return pd.DataFrame(columns=["username", "password"])

def save_user(username, password):
    df = load_users()
    new_user = pd.DataFrame([[username, hash_password(password)]], columns=["username", "password"])
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(USER_FILE, index=False)

def login_user(username, password):
    df = load_users()
    hashed = hash_password(password)
    user = df[(df["username"] == username) & (df["password"] == hashed)]
    return not user.empty
# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="StockSense",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Enhanced Functions with Error Handling ---

def validate_ticker(ticker):
    """Validate ticker symbol format"""
    if not ticker:
        return False, "Ticker symbol cannot be empty"
    if not ticker.replace('.', '').replace('-', '').isalnum():
        return False, "Ticker symbol contains invalid characters"
    if len(ticker) > 10:
        return False, "Ticker symbol is too long"
    return True, "Valid"

def generate_chart(ticker, interval="1d", chunk_size=180, figsize=(18, 6.5), dpi=100):
    """
    Download and plot stock data as a candlestick chart with comprehensive error handling
    """
    try:
        # Validate inputs
        is_valid, message = validate_ticker(ticker)
        if not is_valid:
            st.error(f"Invalid ticker: {message}")
            return None
        
        if chunk_size <= 0 or chunk_size > 1000:
            st.error("Chunk size must be between 1 and 1000")
            return None

        # Define the API endpoint and parameters
        if interval == "1d":
            function = "TIME_SERIES_DAILY"
            interval_param = None
        elif interval == "1h":
            function = "TIME_SERIES_INTRADAY"
            interval_param = "60min"
        elif interval == "1wk":
            function = "TIME_SERIES_WEEKLY"
            interval_param = None
        else:
            st.error(f"Unsupported interval: {interval}")
            return None

        # Build URL
        url = f'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={API_KEY}&outputsize=full'
        if interval_param:
            url += f'&interval={interval_param}'

        # Make API request with timeout
        with st.spinner(f'Fetching data for {ticker}...'):
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise exception for bad status codes
            data = response.json()

        # Check for API errors
        if "Error Message" in data:
            st.error(f"API Error: {data['Error Message']}")
            return None
        
        if "Note" in data:
            st.warning(f"API Note: {data['Note']} - You may have exceeded your API call limit.")
            return None

        # Extract time series data
        if function == "TIME_SERIES_DAILY":
            time_series = data.get("Time Series (Daily)", {})
        elif function == "TIME_SERIES_INTRADAY":
            time_series = data.get(f"Time Series ({interval_param})", {})
        elif function == "TIME_SERIES_WEEKLY":
            time_series = data.get("Weekly Time Series", {})

        if not time_series:
            st.error(f"No data found for ticker '{ticker}' with interval '{interval}'. Please check the ticker symbol.")
            logger.warning(f"No time series data for {ticker}. Response: {data}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Check if we have enough data
        if len(df) < chunk_size:
            st.warning(f"Only {len(df)} candles available (requested {chunk_size}). Showing all available data.")
            chunk_size = len(df)

        # Limit the data to the last 'chunk_size' entries
        data_subset = df.iloc[-chunk_size:]

        # Generate the chart
        fig, ax = mpf.plot(
            data_subset,
            type="candle",
            style="yahoo",
            title=f"{ticker.upper()} Latest {len(data_subset)} Candles",
            axisoff=True,
            ylabel="",
            ylabel_lower="",
            volume=False,
            figsize=figsize,
            returnfig=True
        )

        # Save to buffer
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
        buffer.seek(0)
        
        logger.info(f"Successfully generated chart for {ticker}")
        return buffer

    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        logger.error(f"Timeout error for ticker {ticker}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {str(e)}")
        logger.error(f"Request exception for {ticker}: {e}")
        return None
    except ValueError as e:
        st.error(f"Data parsing error: {str(e)}")
        logger.error(f"ValueError for {ticker}: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error generating chart: {str(e)}")
        logger.error(f"Unexpected error for {ticker}: {e}", exc_info=True)
        return None

def load_yolo_model(model_path):
    """Load YOLO model with error handling"""
    try:
        if not Path(model_path).exists():
            st.error(f"Model file not found at: {model_path}")
            st.info("Please ensure the YOLO model file is in the correct location.")
            return None
        
        from ultralytics import YOLO
        model = YOLO(model_path)
        logger.info(f"Successfully loaded YOLO model from {model_path}")
        return model
    except ImportError:
        st.error("Ultralytics package not installed. Run: pip install ultralytics")
        logger.error("Ultralytics not installed")
        return None
    except Exception as e:
        st.error(f"Error loading YOLO model: {str(e)}")
        logger.error(f"Error loading model: {e}", exc_info=True)
        return None

def identify_patterns(boxes):
    """
    Identify and label patterns from YOLO detection boxes
    Enhanced with better error handling
    """
    if boxes is None or len(boxes) == 0:
        return []
    
    pattern_names = []
    
    try:
        for i, box in enumerate(boxes):
            try:
                # Extract coordinates safely
                if hasattr(box, 'xywh') and len(box.xywh) > 0:
                    coords = box.xywh[0].tolist()
                    if len(coords) >= 4:
                        x, y, w, h = coords[:4]
                    else:
                        pattern_names.append("Invalid Box Data")
                        continue
                else:
                    pattern_names.append("Invalid Box Format")
                    continue

                # Pattern detection logic
                if w > 100 and h > 100:
                    pattern_names.append("Bullish Engulfing")
                elif w < 50 and h > 100:
                    pattern_names.append("Bearish Engulfing")
                elif w > 50 and h < 50:
                    pattern_names.append("Hammer")
                elif w < 30 and h < 30:
                    pattern_names.append("Doji")
                elif w > 70 and h < 30:
                    pattern_names.append("Shooting Star")
                elif w < 50 and h > 50 and abs(w - h) < 10:
                    pattern_names.append("Spinning Top")
                elif w > 60 and h > 40:
                    pattern_names.append("Morning Star")
                elif w < 40 and h > 60:
                    pattern_names.append("Evening Star")
                elif abs(w - h) < 10 and w > 100:
                    pattern_names.append("Marubozu")
                elif w > 50 and h > 150:
                    pattern_names.append("Long-Legged Doji")
                elif w < 40 and h < 100:
                    pattern_names.append("Harami")
                elif w > 100 and h < 40:
                    pattern_names.append("Inverted Hammer")
                elif w > 50 and h < 100:
                    pattern_names.append("Belt Hold")
                elif w > 30 and h > 50:
                    pattern_names.append("Tweezer Top")
                elif w < 20 and h > 50:
                    pattern_names.append("Tweezer Bottom")
                else:
                    pattern_names.append("Unidentified Pattern")
                    
            except Exception as e:
                logger.error(f"Error processing box {i}: {e}")
                pattern_names.append("Error Processing Box")
                
    except Exception as e:
        logger.error(f"Error in identify_patterns: {e}")
        st.error(f"Error identifying patterns: {str(e)}")
        return []
    
    return pattern_names

# --- Initialize Session State ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'show_login' not in st.session_state:
    st.session_state.show_login = False
if 'show_signup' not in st.session_state:
    st.session_state.show_signup = False
if 'model' not in st.session_state:
    st.session_state.model = None

# --- Page Functions ---

def show_landing_page():
    """Display the landing/welcome page"""
    st.title("Welcome to StockSense! 📊")
    st.markdown("""
        Unlock the power of candlestick pattern recognition with StockSense.
        Analyze stock charts, identify key patterns, and gain insights into market trends.

        ### Features:
        - 📈 Generate candlestick charts for any stock ticker
        - 🔍 AI-powered pattern detection using YOLO
        - 📊 Support for multiple timeframes (daily, hourly, weekly)
        - 🎯 High-accuracy pattern identification

        To get started, please **Login** to access the full application features.
                            ---👨‍💻 This web is Developed by **Ravi Helave & Team**
    """)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Get Started", use_container_width=True):
            st.session_state.show_login = True
            st.rerun()

def show_login_page():
    st.title("Login to StockSense")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        col1, col2, col3 = st.columns(3)

        with col1:
            login_button = st.form_submit_button("Login")
        with col2:
            signup_button = st.form_submit_button("Signup")
        with col3:
            back_button = st.form_submit_button("Back")

        if login_button:
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.show_login = False
                st.success("Login Successful")
                st.rerun()
            else:
                st.error("Invalid username or password")

        if signup_button:
            st.session_state.show_signup = True
            st.session_state.show_login = False
            st.rerun()

        if back_button:
            st.session_state.show_login = False
            st.rerun()

def show_signup_page():
    st.title("Create New Account")

    with st.form("signup_form"):
        new_user = st.text_input("Username")
        new_pass = st.text_input("Password", type="password")
        confirm_pass = st.text_input("Confirm Password", type="password")

        col1, col2 = st.columns(2)

        with col1:
            create_btn = st.form_submit_button("Create Account")
        with col2:
            back_btn = st.form_submit_button("Back")

        if create_btn:
            if not new_user or not new_pass:
                st.error("Please fill all fields")
            elif new_pass != confirm_pass:
                st.error("Passwords do not match")
            else:
                df = load_users()
                if new_user in df["username"].values:
                    st.error("Username already exists")
                else:
                    save_user(new_user, new_pass)
                    st.success("Account created successfully! Please login")
                    st.session_state.show_signup = False
                    st.session_state.show_login = True
                    st.rerun()

        if back_btn:
            st.session_state.show_signup = False
            st.session_state.show_login = True
            st.rerun()

def main_app():
    """Main application interface"""
    
    # Load YOLO model once
    if st.session_state.model is None:
        with st.spinner("Loading YOLO model..."):
            st.session_state.model = load_yolo_model(MODEL_PATH)
    
    model = st.session_state.model
    
    # Sidebar UI elements
    with st.sidebar:
        # Display logo if available
        if Path(LOGO_PATH).exists():
            try:
                st.image(LOGO_PATH, use_container_width=True)
            except Exception as e:
                logger.warning(f"Could not load logo in sidebar: {e}")
        
        st.header("⚙️ Configurations")
        
        # Chart Generation Section
        with st.expander("📊 Generate Chart", expanded=True):
            ticker = st.text_input("Ticker Symbol (e.g., AAPL):", key="ticker_input").upper()
            interval = st.selectbox("Select Interval", ["1d", "1h", "1wk"], key="interval_select")
            chunk_size = st.slider("Number of Candles", 50, 500, 180, key="chunk_slider")
            
            if st.button("Generate Chart", use_container_width=True):
                if ticker:
                    chart_buffer = generate_chart(ticker, interval=interval, chunk_size=chunk_size)
                    if chart_buffer:
                        st.success(f"Chart generated successfully for {ticker}")
                        st.download_button(
                            label=f"📥 Download {ticker} Chart",
                            data=chart_buffer,
                            file_name=f"{ticker}_latest_{chunk_size}_candles.png",
                            mime="image/png",
                            use_container_width=True
                        )
                        st.image(chart_buffer, caption=f"{ticker} Chart", use_container_width=True)
                else:
                    st.error("Please enter a valid ticker symbol.")

        # Image Upload Section
        with st.expander("🖼️ Upload Image for Detection", expanded=True):
            source_img = st.file_uploader(
                "Upload a candlestick chart...",
                type=("jpg", "jpeg", "png", 'bmp', 'webp'),
                key="image_uploader"
            )
            confidence = float(st.slider(
                "Model Confidence (%)",
                25, 100, 60,
                help="Higher confidence = fewer but more accurate detections",
                key="confidence_slider"
            )) / 100

        # Model Status
        st.divider()
        if model is not None:
            st.success("✅ YOLO Model Loaded")
        else:
            st.error("❌ YOLO Model Not Available")
            st.info("Pattern detection will not work without the model.")

        # Logout button
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.show_login = False
            st.session_state.model = None
            logger.info("User logged out")
            st.rerun()

    # Main Page Content
    st.title("📊 StockSense - Candlestick Pattern Detection")
    
    st.markdown("""
    ### How to Use:
    
    **Option 1: Upload Your Own Image**
    1. Upload a candlestick chart image using the sidebar
    2. Adjust the confidence threshold if needed
    3. Click **Detect Patterns** to analyze
    
    **Option 2: Generate and Analyze Chart**
    1. Enter a ticker symbol and select timeframe in the sidebar
    2. Click **Generate Chart** to create and download the chart
    3. Upload the generated chart using the file uploader
    4. Click **Detect Patterns** to analyze
    
     ** ---👨‍💻 This web is Developed by **Ravi Helave & Team**
    """)

    # Image Display Columns
    col1, col2 = st.columns(2)

    # Display uploaded image
    if source_img:
        try:
            with col1:
                uploaded_image = Image.open(source_img)
                st.image(uploaded_image, caption="📤 Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            logger.error(f"Error loading uploaded image: {e}")
            source_img = None

    # Pattern Detection
    if st.sidebar.button('🔍 Detect Patterns', use_container_width=True, type="primary"):
        if model is None:
            st.error("YOLO model is not loaded. Please check the model path and try again.")
        elif not source_img:
            st.error("Please upload an image first.")
        else:
            try:
                with st.spinner('Analyzing patterns...'):
                    # Reset file pointer and load image
                    source_img.seek(0)
                    uploaded_image = Image.open(source_img)
                    
                    # Run prediction
                    res = model.predict(uploaded_image, conf=confidence)
                    
                    if not res or len(res) == 0:
                        st.warning("No results returned from model.")
                    else:
                        boxes = res[0].boxes
                        
                        # Display detected image
                        try:
                            res_plotted = res[0].plot()[:, :, ::-1]
                            with col2:
                                st.image(res_plotted, caption='🎯 Detected Patterns', use_container_width=True)
                        except Exception as e:
                            logger.error(f"Error plotting results: {e}")
                            st.error("Error displaying detection results visualization.")
                        
                        # Display pattern information
                        if boxes is not None and len(boxes) > 0:
                            pattern_names = identify_patterns(boxes)
                            
                            st.success(f"✅ Found {len(boxes)} pattern(s)")
                            
                            with st.expander("📋 Detection Results", expanded=True):
                                for i, (box, pattern_name) in enumerate(zip(boxes, pattern_names)):
                                    col_a, col_b = st.columns([1, 2])
                                    with col_a:
                                        st.metric(f"Pattern {i+1}", pattern_name)
                                    with col_b:
                                        try:
                                            coords = box.xywh[0].tolist() if hasattr(box, 'xywh') else []
                                            conf = box.conf[0].item() if hasattr(box, 'conf') else 0
                                            st.write(f"Confidence: {conf:.2%}")
                                            if len(coords) >= 4:
                                                st.write(f"Position: x={coords[0]:.1f}, y={coords[1]:.1f}")
                                        except Exception as e:
                                            logger.error(f"Error displaying box info: {e}")
                                    st.divider()
                        else:
                            st.info("No patterns detected. Try adjusting the confidence threshold.")
                            
            except Exception as e:
                st.error(f"Error during pattern detection: {str(e)}")
                logger.error(f"Error in pattern detection: {e}", exc_info=True)

# --- Main Application Router ---
if st.session_state.logged_in:
    main_app()
elif st.session_state.show_signup:
    show_signup_page()
elif st.session_state.show_login:
    show_login_page()
else:
    show_landing_page()
    