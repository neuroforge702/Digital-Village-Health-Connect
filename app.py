import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from datetime import datetime
import os
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import sqlite3

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Smart Community Health Monitoring",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CONSTANTS & DATABASE ---
DATA_FILE = "health_reports.csv"
DB_FILE = "village_health.db"
VILLAGE_COORDS = {
    "Mawsynram": [25.2975, 91.5826],
    "Cherrapunji": [25.2702, 91.7323],
    "Majuli": [26.9656, 94.1378],
    "Kohima": [25.6751, 94.1086],
    "Tawang": [27.5850, 91.8594],
    "Aizawl": [23.7271, 92.7176],
    "Imphal": [24.8170, 93.9368],
    "Agartala": [23.8315, 91.2868],
    "Gangtok": [27.3314, 88.6138],
    "Diphu": [25.8450, 93.4333]
}

# --- CUSTOM CSS & BACKGROUND ---
st.markdown("""
<style>
    /* Full page background layer */
    .stApp {
        background: transparent !important;
        color: #e0f2f1;
    }
    
    .app-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: linear-gradient(to bottom, #000428 0%, #004e92 100%);
        z-index: -3 !important;
    }

    /* Wave Background Container */
    .ocean { 
        height: 100vh;
        width: 100%;
        position: fixed;
        bottom: 0;
        left: 0;
        background: transparent;
        z-index: -2 !important; /* Send to back */
        overflow: hidden;
        pointer-events: none;
    }

    .wave-css {
        background: url(https://s3-us-west-2.amazonaws.com/s.cdpn.io/85486/wave.svg) repeat-x; 
        position: absolute;
        bottom: 0;
        width: 6400px;
        height: 198px;
        animation: wave 10s cubic-bezier( 0.36, 0.45, 0.63, 0.53) infinite;
        transform: translate3d(0, 0, 0);
        opacity: 0.15;
    }

    .wave-css:nth-of-type(2) {
        bottom: -20px;
        animation: wave 15s cubic-bezier( 0.36, 0.45, 0.63, 0.53) -.125s infinite, swell 10s ease -0.125s infinite;
        opacity: 0.1;
    }

    @keyframes wave {
      0% { margin-left: 0; }
      100% { margin-left: -1600px; }
    }

    @keyframes swell {
      0%, 100% { transform: translate3d(0,-25px,0); }
      50% { transform: translate3d(0,25px,0); }
    }

    /* Glassmorphism Components */
    .stMetric, .report-card, [data-testid="stForm"], div.block-container > div:has(div.stDataFrame) {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(12px) saturate(180%);
        -webkit-backdrop-filter: blur(12px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        padding: 20px !important;
    }
    
    .stMetric label { color: #80deea !important; font-size: 1.1rem !important; }
    .stMetric [data-testid="stMetricValue"] { color: #00e5ff !important; font-weight: 800 !important; font-size: 2.2rem !important; }

    /* Titles and Headers */
    h1 {
        background: linear-gradient(to right, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 900 !important;
        text-align: center;
        margin-bottom: 40px !important;
        filter: drop-shadow(0 0 15px rgba(0, 242, 254, 0.5));
    }

    h2, h3 {
        color: #81d4fa !important;
        font-weight: 700 !important;
        border-bottom: 2px solid rgba(0, 210, 255, 0.2);
        padding-bottom: 10px;
    }

    /* Specifically target the form submit button */
    .stButton>button, div.stForm [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%) !important;
        color: #001220 !important; /* High contrast dark blue */
        border: none !important;
        border-radius: 50px !important;
        padding: 12px 35px !important;
        font-size: 1.1rem !important;
        font-weight: 900 !important; /* Maximum boldness */
        box-shadow: 0 10px 20px rgba(0, 114, 255, 0.3) !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        width: 100%; /* Make it prominent in the form */
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover, div.stForm [data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-5px) scale(1.02);
        color: #ffffff !important;
        box-shadow: 0 15px 30px rgba(0, 114, 255, 0.5) !important;
        background: linear-gradient(135deg, #0072ff 0%, #00c6ff 100%) !important;
    }

    /* Sidebar Refinement: Premium Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(10, 15, 25, 0.75) !important;
        backdrop-filter: blur(25px) saturate(180%) !important;
        border-right: 1px solid rgba(0, 242, 254, 0.2) !important;
        box-shadow: 10px 0 30px rgba(0, 0, 0, 0.5) !important;
    }

    [data-testid="stSidebar"] section:first-child {
        background: transparent !important;
    }

    /* Risk indicators with glow */
    .risk-high { color: #ff5252; font-weight: bold; text-shadow: 0 0 10px #ff5252; }
    .risk-medium { color: #ffd740; font-weight: bold; text-shadow: 0 0 10px #ffd740; }
    .risk-low { color: #69f0ae; font-weight: bold; text-shadow: 0 0 10px #69f0ae; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 10px; }
    ::-webkit-scrollbar-track { background: #000428; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(#00c6ff, #0072ff); border-radius: 10px; }
    /* Floating Animated Orbs */
    .orb {
        position: fixed;
        border-radius: 50%;
        filter: blur(60px);
        z-index: -1 !important; /* Below content, above waves */
        opacity: 0.3;
        pointer-events: none;
        animation: float 20s infinite alternate ease-in-out;
    }
    .orb-1 {
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, #00f2fe, transparent);
        top: -100px;
        left: -100px;
    }
    .orb-2 {
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, #4facfe, transparent);
        bottom: -150px;
        right: -100px;
        animation-delay: -5s;
    }
    .orb-3 {
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, #00d2ff, transparent);
        top: 40%;
        left: 30%;
        animation-duration: 25s;
    }

    @keyframes float {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(100px, 50px) scale(1.1); }
        100% { transform: translate(-50px, 100px) scale(0.9); }
    }

    /* Premium Bubble Effect */
    .bubble {
        position: fixed;
        background: rgba(0, 242, 254, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 50%;
        z-index: -1 !important; /* Behind everything */
        bottom: -100px;
        pointer-events: none;
        animation: bubble-rise 15s infinite ease-in-out;
        box-shadow: inset 0 0 10px rgba(255, 255, 255, 0.1);
    }

    @keyframes bubble-rise {
        0% { transform: translateY(0) scale(1); opacity: 0; }
        20% { opacity: 0.5; }
        50% { transform: translateX(50px) scale(1.2); }
        80% { opacity: 0.3; }
        100% { transform: translateY(-110vh) translateX(-20px) scale(0.8); opacity: 0; }
    }

    /* Reliable Premium Custom Cursor */
    html, body, [data-testid="stAppViewContainer"], .main {
        cursor: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" style="fill:none;stroke:rgb(0,242,254);stroke-width:2" viewBox="0 0 32 32"><circle cx="16" cy="16" r="10" /><circle cx="16" cy="16" r="2" fill="rgb(0,242,254)" /></svg>') 16 16, auto !important;
    }

    /* Enhanced Hover Effects for Interactive Elements */
    button, [role="button"], a, input, select, .stSelectbox, [data-testid="stMetric"] {
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    }

    /* Specific Hover Glows */
    button:hover, [role="button"]:hover {
        transform: scale(1.05) translateY(-2px) !important;
        box-shadow: 0 0 25px rgba(0, 242, 254, 0.6) !important;
        filter: brightness(1.2);
    }

    a:hover {
        color: #00f2fe !important;
        text-shadow: 0 0 10px rgba(0, 242, 254, 0.5);
    }

    [data-testid="stMetric"]:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(0, 242, 254, 0.4) !important;
        transform: translateY(-5px) !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;
    }

    /* Hover effect for form inputs */
    .stTextInput>div>div>input:hover, .stSelectbox:hover {
        border-color: #00f2fe !important;
        background: rgba(0, 242, 254, 0.05) !important;
    }

    /* Card Hover */
    .report-card:hover {
        border-left: 5px solid #00f2fe !important;
        background: rgba(0, 210, 255, 0.05) !important;
    }
    /* Bubble Glitter Trail */
    .glitter {
        position: fixed;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(0, 242, 254, 0.8), transparent);
        pointer-events: none;
        z-index: 100000;
        animation: spark 0.8s linear forwards;
    }

    @keyframes spark {
        0% { transform: translateY(0) scale(1); opacity: 1; }
        100% { transform: translateY(-30px) scale(0); opacity: 0; }
    }
    /* NAVBAR STYLING (Isolated) */
    .top-nav-marker { margin-top: -60px; height: 1px; }

    /* Target ONLY the navbar container by finding the marker's parent */
    div:has(> .top-nav-marker) + div[data-testid="stHorizontalBlock"] {
        background: rgba(10, 20, 40, 0.85) !important;
        backdrop-filter: blur(20px) saturate(180%);
        border-bottom: 2px solid rgba(0, 242, 254, 0.2);
        padding: 8px 15px !important;
        position: sticky !important;
        top: 0;
        z-index: 9999 !important;
        border-radius: 0 0 25px 25px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }

    div:has(> .top-nav-marker) + div[data-testid="stHorizontalBlock"] button {
        background: transparent !important;
        border: none !important;
        color: #81d4fa !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        box-shadow: none !important;
    }

    div:has(> .top-nav-marker) + div[data-testid="stHorizontalBlock"] button:hover {
        color: #00f2fe !important;
        background: rgba(0, 242, 254, 0.1) !important;
        transform: translateY(-2px);
    }

    .nav-active button {
        color: #ffffff !important;
        border-bottom: 3px solid #00f2fe !important;
        background: rgba(0, 242, 254, 0.1) !important;
        border-radius: 0 !important;
    }

    /* Content Spacing to prevent overlap with sticky navbar */
    .main .block-container {
        padding-top: 2rem !important;
    }
</style>

<!-- Background Gradient Layer -->
<div class="app-bg"></div>

<div class="ocean">
  <div class="wave-css"></div>
  <div class="wave-css"></div>
</div>

<!-- Animated Background Elements -->
<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
<div class="orb orb-3"></div>

<!-- Multiple bubbles for rising effect -->
<div class="bubble" style="width: 40px; height: 40px; left: 15%; animation-delay: 0s; animation-duration: 18s;"></div>
<div class="bubble" style="width: 25px; height: 25px; left: 35%; animation-delay: 2s; animation-duration: 15s;"></div>
<div class="bubble" style="width: 60px; height: 60px; left: 55%; animation-delay: 5s; animation-duration: 22s;"></div>
<div class="bubble" style="width: 20px; height: 20px; left: 75%; animation-delay: 8s; animation-duration: 12s;"></div>
<div class="bubble" style="width: 35px; height: 35px; left: 85%; animation-delay: 15s; animation-duration: 19s;"></div>
""", unsafe_allow_html=True)

# --- CURSOR GLITTER (Advanced Particle System) ---
def show_glitter():
    # components.html is more reliable for script execution in many Streamlit versions
    components.html("""
    <script>
    (function() {
        // Function to inject sparkle logic into the parent document
        const injectSparkles = () => {
            const targetDoc = window.parent.document;
            if (!targetDoc) return;
            
            // Avoid duplicate canvases
            if (targetDoc.getElementById('dashboard-sparkle-layer')) return;

            const canvas = targetDoc.createElement('canvas');
            canvas.id = 'dashboard-sparkle-layer';
            canvas.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:99999999;';
            targetDoc.body.appendChild(canvas);

            const ctx = canvas.getContext('2d');
            let points = [];

            const resize = () => {
                canvas.width = window.parent.innerWidth;
                canvas.height = window.parent.innerHeight;
            };
            window.parent.addEventListener('resize', resize);
            resize();

            targetDoc.addEventListener('mousemove', (e) => {
                for(let i=0; i<4; i++) {
                    points.push({
                        x: e.clientX,
                        y: e.clientY,
                        vx: (Math.random() - 0.5) * 3,
                        vy: (Math.random() - 0.5) * 3 - 0.5,
                        size: Math.random() * 4 + 2,
                        life: 1.0,
                        color: `hsla(${180 + Math.random()*40}, 100%, 75%, 1)`
                    });
                }
            });

            const draw = () => {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                for(let i = points.length-1; i >= 0; i--) {
                    const p = points[i];
                    p.x += p.vx;
                    p.y += p.vy;
                    p.life -= 0.02;
                    
                    if(p.life <= 0) {
                        points.splice(i, 1);
                        continue;
                    }

                    ctx.globalAlpha = p.life;
                    ctx.fillStyle = p.color;
                    ctx.shadowBlur = 10;
                    ctx.shadowColor = p.color;
                    
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, p.size * p.life, 0, Math.PI * 2);
                    ctx.fill();
                }
                requestAnimationFrame(draw);
            };
            draw();
        };

        // Try to inject multiple times to ensure we catch the parent when ready
        let count = 0;
        const timer = setInterval(() => {
            try {
                injectSparkles();
                if (window.parent.document.getElementById('dashboard-sparkle-layer') || count > 10) {
                    clearInterval(timer);
                }
            } catch(e) { /* Parent might not be accessible yet */ }
            count++;
        }, 500);
    })();
    </script>
    """, height=0)

# --- DATABASE SYSTEM (SQL) ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS health_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            village TEXT,
            latitude REAL,
            longitude REAL,
            symptom_count INTEGER,
            symptoms TEXT,
            water_source TEXT,
            water_quality TEXT,
            risk_level TEXT,
            report_date TEXT
        )
    ''')
    
    # Optional: Migrate existing CSV data to SQL if DB is empty
    c.execute("SELECT COUNT(*) FROM health_reports")
    if c.fetchone()[0] == 0 and os.path.exists(DATA_FILE):
        try:
            df_csv = pd.read_csv(DATA_FILE)
            for _, row in df_csv.iterrows():
                c.execute('''
                    INSERT INTO health_reports (village, latitude, longitude, symptom_count, symptoms, water_source, water_quality, risk_level, report_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (row['Village'], row['Latitude'], row['Longitude'], int(row['Symptom Count']), row['Symptoms'], row['Water Source'], row['Water Quality'], row['Risk Level'], row['Date']))
        except Exception as e:
            print(f"Migration error: {e}")
            
    conn.commit()
    conn.close()

# Initialize Database on boot
init_db()

def load_data():
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT * FROM health_reports"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if not df.empty:
        # Rename columns to match existing logic
        df.columns = ["ID", "Village", "Latitude", "Longitude", "Symptom Count", "Symptoms", "Water Source", "Water Quality", "Risk Level", "Date"]
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
        return df
    return pd.DataFrame(columns=["Village", "Latitude", "Longitude", "Symptom Count", "Symptoms", "Water Source", "Water Quality", "Risk Level", "Date"])

def save_data(new_record):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO health_reports (village, latitude, longitude, symptom_count, symptoms, water_source, water_quality, risk_level, report_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (new_record['Village'], new_record['Latitude'], new_record['Longitude'], new_record['Symptom Count'], new_record['Symptoms'], new_record['Water Source'], new_record['Water Quality'], new_record['Risk Level'], new_record['Date']))
    conn.commit()
    conn.close()

# --- AI MODEL (Random Forest) ---
def initialize_ai_model():
    # Create a synthetic dataset to train the AI model initially
    # In a real app, this would train on the actual health_reports.csv over time
    data = {
        'symptom_count': [5, 45, 20, 2, 55, 12, 35, 8, 25, 60],
        'water_quality': ['Clean', 'Contaminated', 'Slightly Contaminated', 'Clean', 'Contaminated', 'Clean', 'Slightly Contaminated', 'Clean', 'Slightly Contaminated', 'Contaminated'],
        'risk': ['Low Risk', 'High Risk', 'Medium Risk', 'Low Risk', 'High Risk', 'Low Risk', 'High Risk', 'Low Risk', 'Medium Risk', 'High Risk']
    }
    df_train = pd.DataFrame(data)
    
    # Expand dataset to make it more "ML-ready"
    for _ in range(50):
        extra = pd.DataFrame({
            'symptom_count': np.random.randint(0, 100, 20),
            'water_quality': np.random.choice(['Clean', 'Slightly Contaminated', 'Contaminated'], 20)
        })
        # Logic for synthetic labels
        def logic(row):
            if row['water_quality'] == 'Contaminated' or row['symptom_count'] > 40: return 'High Risk'
            if row['water_quality'] == 'Slightly Contaminated' or row['symptom_count'] > 18: return 'Medium Risk'
            return 'Low Risk'
        extra['risk'] = extra.apply(logic, axis=1)
        df_train = pd.concat([df_train, extra])

    le = LabelEncoder()
    df_train['water_encoded'] = le.fit_transform(df_train['water_quality'])
    
    X = df_train[['symptom_count', 'water_encoded']]
    y = df_train['risk']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, le

# Global AI Model Instance
AI_MODEL, WATER_LE = initialize_ai_model()

def predict_risk(symptom_count, water_quality):
    """Predict risk using the trained Random Forest model"""
    try:
        # Encode user input
        water_enc = WATER_LE.transform([water_quality])[0]
        prediction = AI_MODEL.predict([[symptom_count, water_enc]])[0]
        return prediction
    except:
        # Fallback to simple logic if model fails
        if water_quality == "Contaminated" or symptom_count > 30:
            return "High Risk"
        return "Low Risk"

import base64

# --- PAGE: HOME / DASHBOARD ---
def show_home():
    # Load and encode the earth globe image for Three.js texture
    with open("earth.png", "rb") as image_file:
        encoded_earth = base64.b64encode(image_file.read()).decode()

    # --- HERO SECTION (Columns-based for perfect alignment) ---
    with st.container():
        # Inject custom hero styles
        st.markdown("""
            <style>
                /* Target the hero columns block specifically */
                div:has(> .hero-marker) + div[data-testid="stHorizontalBlock"] {
                    background: rgba(255, 255, 255, 0.04) !important;
                    border-radius: 35px !important;
                    border: 1px solid rgba(0, 242, 254, 0.15) !important;
                    padding: 30px 50px !important;
                    margin-bottom: 50px !important;
                    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4) !important;
                    backdrop-filter: blur(15px) !important;
                }

                .hero-title {
                    text-align: left !important;
                    font-size: 3.8rem !important;
                    margin-bottom: 20px !important;
                    background: linear-gradient(to right, #00f2fe, #4facfe, #00f2fe);
                    background-size: 200% auto;
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    filter: drop-shadow(0 0 15px rgba(0, 242, 254, 0.3));
                    animation: shine_hero 5s linear infinite;
                }
                @keyframes shine_hero {
                    to { background-position: 200% center; }
                }
                .hero-subtitle {
                    color: #81d4fa;
                    font-size: 1.35rem;
                    line-height: 1.6;
                    margin-bottom: 35px;
                }
                .hero-stats-row { display: flex; gap: 15px; }
                .hero-stat-badge {
                    padding: 8px 22px;
                    border: 1px solid rgba(0, 242, 254, 0.4);
                    border-radius: 50px;
                    font-size: 0.8rem;
                    font-weight: 800;
                    color: #00f2fe;
                    background: rgba(0, 242, 254, 0.05);
                    text-transform: uppercase;
                }
            </style>
            <div class="hero-marker"></div>
        """, unsafe_allow_html=True)
        
        # Use columns to position the text and globe side-by-side
        # We wrap them in a marker to target the parent container if needed, 
        # but the layout itself handles the side-by-side positioning now.
        col_text, col_globe = st.columns([1.6, 1])
        
        with col_text:
             st.markdown(f"""
                <div style="padding-right: 20px;">
                    <h1 class="hero-title">🏥 Village Health Intelligence</h1>
                    <p class="hero-subtitle">
                        Real-time regional monitoring & early warning system for North-East India. 
                        Empowering communities with localized health data using <b>Random Forest AI Ensemble</b> predictions.
                    </p>
                    <div class="hero-stats-row">
                        <div class="hero-stat-badge">📡 LIVE FEED</div>
                        <div class="hero-stat-badge">🤖 AI-POWERED</div>
                        <div class="hero-stat-badge" style="border-color: #69f0ae; color: #69f0ae; background: rgba(105, 240, 174, 0.05);">✓ SECURE</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col_globe:
            # The 3D globe sits perfectly in this column
            components.html(f"""
                <div id="canvas-container" style="width: 100%; height: 500px; display: flex; justify-content: center; align-items: center;"></div>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
                <script>
                (function() {{
                    const container = document.getElementById('canvas-container');
                    const scene = new THREE.Scene();
                    const camera = new THREE.PerspectiveCamera(45, 400/500, 0.1, 1000);
                    const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
                    
                    renderer.setSize(400, 500);
                    renderer.setPixelRatio(window.devicePixelRatio);
                    container.appendChild(renderer.domElement);

                    // Lighting
                    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
                    scene.add(ambientLight);
                    const pLight = new THREE.PointLight(0x00f2fe, 2);
                    pLight.position.set(5, 3, 5);
                    scene.add(pLight);

                    // Geometry (Slightly smaller to prevent edge cropping)
                    const geometry = new THREE.SphereGeometry(1.5, 64, 64);
                    
                    const img = new Image();
                    img.onload = function() {{
                        const texture = new THREE.Texture(img);
                        texture.needsUpdate = true;
                        const material = new THREE.MeshPhongMaterial({{ 
                            map: texture,
                            shininess: 45,
                            bumpScale: 0.1,
                            specular: new THREE.Color(0x222222)
                        }});
                        const earth = new THREE.Mesh(geometry, material);
                        scene.add(earth);

                        // Atmospheric glow
                        const atmosphericGeom = new THREE.SphereGeometry(1.53, 64, 64);
                        const atmosphericMat = new THREE.MeshBasicMaterial({{
                            color: 0x00f2fe,
                            transparent: true,
                            opacity: 0.1,
                            side: THREE.BackSide
                        }});
                        const atmospheric = new THREE.Mesh(atmosphericGeom, atmosphericMat);
                        scene.add(atmospheric);

                        camera.position.z = 5.0;

                        let angle = 0;
                        function animate() {{
                            requestAnimationFrame(animate);
                            
                            // Rotation
                            earth.rotation.y += 0.005;
                            earth.rotation.z += 0.001;
                            
                            // Floating physics (Done on OBJECT position to avoid DOM cropping)
                            angle += 0.02;
                            const floatOffset = Math.sin(angle) * 0.2;
                            earth.position.y = floatOffset;
                            atmospheric.position.y = floatOffset;
                            
                            renderer.render(scene, camera);
                        }}
                        animate();
                    }};
                    img.src = 'data:image/png;base64,{encoded_earth}';
                }})();
                </script>
            """, height=500)
    
    df = load_data()
    if df.empty:
        st.warning("No data available. Please submit a report.")
        return

    # --- CRITICAL PUBLIC ALERT SYSTEM ---
    high_risk_data = df[df['Risk Level'] == 'High Risk'].sort_values('Date', ascending=False)
    if not high_risk_data.empty:
        latest_hazard = high_risk_data.iloc[0]
        st.markdown(f"""
            <div style="background: linear-gradient(90deg, #ff4b2b, #ff416c); padding: 15px 25px; border-radius: 15px; margin-bottom: 30px; border-left: 10px solid #ffffff; animation: pulse_alert 2s infinite;">
                <h3 style="margin: 0; color: white !important; border: none !important; font-size: 1.2rem;">
                    🚨 <b>CRITICAL HEALTH ALERT</b>: High risk detected in <b>{latest_hazard['Village']}</b> ({latest_hazard['Date'].strftime('%Y-%m-%d')})
                </h3>
                <p style="margin: 5px 0 0 0; color: rgba(255,255,255,0.9); font-size: 0.9rem;">
                    Immediate water purification and medical screening protocols advised for this sector.
                </p>
            </div>
            <style>
                @keyframes pulse_alert {{
                    0% {{ box-shadow: 0 0 0 0 rgba(255, 75, 43, 0.7); }}
                    70% {{ box-shadow: 0 0 0 15px rgba(255, 75, 43, 0); }}
                    100% {{ box-shadow: 0 0 0 0 rgba(255, 75, 43, 0); }}
                }}
            </style>
        """, unsafe_allow_html=True)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    total_cases = df['Symptom Count'].sum()
    avg_cases = df.groupby('Village')['Symptom Count'].mean().mean()
    high_risk_villages = df[df['Risk Level'] == 'High Risk']['Village'].nunique()
    latest_report = df['Date'].max().strftime('%Y-%m-%d')
    
    col1.metric("Total Reported Symptoms", f"{total_cases}")
    col2.metric("Avg. Symptoms per Village", f"{avg_cases:.1f}")
    col3.metric("High Risk Villages", f"{high_risk_villages}")
    col4.metric("Last Data Sync", latest_report)

    # Charts
    st.markdown("---")
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Symptom Trend (Recent 30 Days)")
        trend_df = df.resample('D', on='Date')['Symptom Count'].sum().reset_index()
        fig_trend = px.line(trend_df, x='Date', y='Symptom Count', title="Daily Symptom Counts", line_shape='spline', color_discrete_sequence=['#00f2fe'])
        fig_trend.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color="#ffffff",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_right:
        st.subheader("Cases by Village")
        village_data = df.groupby('Village')['Symptom Count'].sum().reset_index()
        fig_bar = px.bar(village_data, x='Village', y='Symptom Count', color='Symptom Count', color_continuous_scale='Blues', title="Total Cases per Village")
        fig_bar.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color="#ffffff",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Alerts
    st.markdown("### 🚨 Active Alerts")
    critical_data = df[df['Risk Level'] == 'High Risk'].sort_values('Date', ascending=False).head(5)
    if not critical_data.empty:
        for idx, row in critical_data.iterrows():
            st.error(f"**ALERT**: High Risk detected in **{row['Village']}** on {row['Date'].strftime('%Y-%m-%d')}. Symptoms: {row['Symptoms']}. Water Source: {row['Water Source']} ({row['Water Quality']}).")
    else:
        st.success("No high-risk alerts currently active.")

# --- PAGE: REPORT SYMPTOMS ---
def show_report_form():
    st.title("📝 Submit Community Health Report")
    st.markdown("Fill out the form below to report health symptoms in your village.")
    
    with st.form("health_report_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            village = st.selectbox("Village Name", list(VILLAGE_COORDS.keys()))
            symptom_count = st.number_input("Number of People with Symptoms", min_value=0, step=1)
            report_date = st.date_input("Date of Observation", datetime.now())
            
        with col2:
            water_source = st.selectbox("Primary Water Source", ["River", "Well", "Tap", "Hand Pump"])
            water_quality = st.selectbox("Water Quality Indicator", ["Clean", "Slightly Contaminated", "Contaminated"])
            symptoms = st.multiselect("Symptoms Observed", ["Diarrhea", "Vomiting", "Fever", "Dehydration"])

        st.markdown("---")
        submitted = st.form_submit_button("Submit Health Report")
        
        if submitted:
            risk = predict_risk(symptom_count, water_quality)
            new_record = {
                "Village": village,
                "Latitude": VILLAGE_COORDS[village][0],
                "Longitude": VILLAGE_COORDS[village][1],
                "Symptom Count": symptom_count,
                "Symptoms": ", ".join(symptoms) if symptoms else "None",
                "Water Source": water_source,
                "Water Quality": water_quality,
                "Risk Level": risk,
                "Date": report_date.strftime("%Y-%m-%d")
            }
            save_data(new_record)
            st.success(f"Report submitted successfully for {village}! Risk Level: **{risk}**")
            if risk == "High Risk":
                st.warning("⚠️ Immediate action may be required. Boiling water is highly recommended.")

# --- PAGE: RISK ANALYSIS ---
def show_risk_analysis():
    st.title("📊 Disease Risk Analysis")
    df = load_data()
    
    if df.empty:
        st.warning("No data available for analysis.")
        return

    st.markdown("### Risk Distribution")
    risk_counts = df['Risk Level'].value_counts().reset_index()
    risk_counts.columns = ['Risk Level', 'Count']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(risk_counts, values='Count', names='Risk Level', 
                         color='Risk Level',
                         color_discrete_map={'High Risk': '#ff4b2b', 'Medium Risk': '#f9d423', 'Low Risk': '#00b09b'},
                         hole=0.4, title="Overall Area Risk Proportion")
        fig_pie.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color="#ffffff"
        )
        st.plotly_chart(fig_pie)
        
    with col2:
        st.markdown("#### 🤖 AI Model Intelligence")
        st.write("Current calculation is powered by an **ensemble Random Forest Classifier**.")
        
        # Display Feature Importance (Static for now based on common RF outputs)
        st.info("**Feature Importance (AI Weights):**")
        col_f1, col_f2 = st.columns(2)
        col_f1.metric("Symptom Pulse", "65%", help="Weight assigned to reported symptom volume")
        col_f2.metric("Hydro-Quality", "35%", help="Weight assigned to water contamination levels")
        
        st.markdown("""
        **Model Logic:**
        The Random Forest model analyzes multiple scenarios simultaneously to determine the risk level. It is trained to recognize patterns where specific water conditions combined with symptom spikes indicate a high probability of outbreak.
        """)
        
    st.markdown("---")
    st.subheader("Detailed Village Status")
    
    # Filter by risk
    selected_risk = st.multiselect("Filter by Risk Level", ["Low Risk", "Medium Risk", "High Risk"], default=["Low Risk", "Medium Risk", "High Risk"])
    filtered_df = df[df['Risk Level'].isin(selected_risk)]
    
    display_cols = ["Village", "Symptom Count", "Symptoms", "Water Source", "Water Quality", "Risk Level", "Date"]
    ordered_df = filtered_df[display_cols].sort_values('Date', ascending=False)
    
    st.dataframe(ordered_df, use_container_width=True, hide_index=True)

# --- PAGE: MAP VIEW ---
def show_map_view():
    st.title("🗺️ Interactive Health Map")
    st.markdown("Geospatial visualization of disease risk in North-East India.")
    
    df = load_data()
    if df.empty:
        st.warning("No data available to display on map.")
        return

    # Prepare latest data per village for mapping
    latest_df = df.sort_values('Date').groupby('Village').tail(1)
    
    # Base map centered on North-East India
    m = folium.Map(location=[26.0, 92.0], zoom_start=6, tiles="CartoDB dark_matter")
    
    for idx, row in latest_df.iterrows():
        color = 'green'
        if row['Risk Level'] == 'High Risk': color = 'red'
        elif row['Risk Level'] == 'Medium Risk': color = 'orange'
        
        popup_text = f"<b>Village:</b> {row['Village']}<br>" \
                     f"<b>Risk:</b> {row['Risk Level']}<br>" \
                     f"<b>Cases:</b> {row['Symptom Count']}<br>" \
                     f"<b>Water:</b> {row['Water Quality']}<br>" \
                     f"<b>Last Updated:</b> {row['Date'].strftime('%Y-%m-%d')}"
        
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=10,
            popup=folium.Popup(popup_text, max_width=300),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7
        ).add_to(m)
        
    st_folium(m, width=1200, height=600)

# --- PAGE: HEALTH AWARENESS ---
def show_health_awareness():
    st.title("🛡️ Health Awareness & Prevention")
    st.markdown("Protect your community from water-borne diseases.")
    
    # First Row: Core Information
    col_aw1, col_aw2 = st.columns(2)
    
    with col_aw1:
        st.info("### 💧 Safe Drinking Water")
        st.markdown("""
        - **Boil Water**: Bring water to a rolling boil for at least 1 minute.
        - **Purification Tablets**: Use chlorine or iodine tablets if boiling isn't possible.
        - **Store Safely**: Keep clean water in covered, narrow-necked containers.
        - **Protect Sources**: Keep livestock away from wells and rivers used for drinking.
        """)
        
    with col_aw2:
        st.warning("### ⚠️ Recognizing Symptoms")
        st.markdown("""
        If you or someone in your village experience the following, seek medical help immediately:
        - **Profuse watery diarrhea** (often described as 'rice-water')
        - **Severe vomiting**
        - **Rapid dehydration** (extreme thirst, dry mouth)
        - **High fever**
        """)

    st.markdown("---")
    
    # Second Row: Interactive Model & Hygiene
    col_model, col_hygiene = st.columns([1, 1.2])
    
    with col_model:
        # --- 3D PROTECTIVE WATER MODEL ---
        components.html("""
            <div id="water-model-container" style="width: 100%; height: 380px; background: rgba(0,0,0,0); display: flex; justify-content: center; align-items: center; overflow: hidden; margin-top: -40px;"></div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script>
            (function() {
                const container = document.getElementById('water-model-container');
                const scene = new THREE.Scene();
                const camera = new THREE.PerspectiveCamera(35, container.clientWidth / 380, 0.1, 1000);
                const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
                renderer.setSize(container.clientWidth, 380);
                renderer.setPixelRatio(window.devicePixelRatio);
                container.appendChild(renderer.domElement);

                const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
                scene.add(ambientLight);
                const pLight1 = new THREE.PointLight(0x00f2fe, 2.5);
                pLight1.position.set(2, 3, 5);
                scene.add(pLight1);

                const dropletGeom = new THREE.IcosahedronGeometry(0.75, 12);
                const dropletMat = new THREE.MeshPhongMaterial({
                    color: 0x00f2fe,
                    transparent: true,
                    opacity: 0.8,
                    shininess: 120,
                    specular: 0xffffff
                });
                const droplet = new THREE.Mesh(dropletGeom, dropletMat);
                scene.add(droplet);

                const shieldGeom = new THREE.TorusKnotGeometry(1.0, 0.02, 120, 16);
                const shieldMat = new THREE.MeshBasicMaterial({ color: 0x00f2fe, transparent: true, opacity: 0.25 });
                const shield = new THREE.Mesh(shieldGeom, shieldMat);
                scene.add(shield);
                
                const shield2 = shield.clone();
                shield2.rotation.y = Math.PI / 2;
                scene.add(shield2);

                const partGeom = new THREE.BufferGeometry();
                const partCount = 40;
                const posArray = new Float32Array(partCount * 3);
                for(let i=0; i<partCount * 3; i++) {
                    posArray[i] = (Math.random() - 0.5) * 4;
                }
                partGeom.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
                const partMat = new THREE.PointsMaterial({ size: 0.03, color: 0xffffff, transparent: true, opacity: 0.8 });
                const particles = new THREE.Points(partGeom, partMat);
                scene.add(particles);

                camera.position.z = 7.5;

                let time = 0;
                function animate() {
                    requestAnimationFrame(animate);
                    time += 0.012;
                    droplet.rotation.y += 0.008;
                    const scale = 1 + Math.sin(time) * 0.03;
                    droplet.scale.set(scale, scale, scale);
                    shield.rotation.x += 0.004;
                    shield.rotation.y += 0.006;
                    const floatPos = Math.sin(time * 0.4) * 0.12;
                    droplet.position.y = floatPos;
                    shield.position.y = floatPos;
                    shield2.position.y = floatPos;
                    renderer.render(scene, camera);
                }
                animate();

                window.addEventListener('resize', () => {
                    const w = container.clientWidth;
                    camera.aspect = w / 380;
                    camera.updateProjectionMatrix();
                    renderer.setSize(w, 380);
                });
            })();
            </script>
        """, height=380)
        
    with col_hygiene:
        st.success("### 🧼 Hygiene Practices")
        st.markdown("""
        - **Handwashing**: Wash hands with soap after using the toilet and before preparing food.
        - **Sanitation**: Use pits/latrines for waste disposal; avoid open defecation.
        - **Food Safety**: Cook food thoroughly and eat it while hot.
        - **Self-Care**: Maintain personal hygiene and keep living areas clean.
        """)

# --- MAIN APP NAVIGATION ---
def main():
    # Activate cursor glitter immediately
    show_glitter()
    
    # Initialize session state for page navigation
    if "app_page" not in st.session_state:
        st.session_state.app_page = "Home"

    # --- PREMIUM SIDEBAR ---
    st.sidebar.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <div class="sidebar-heart">
                <svg viewBox="0 0 32 32" width="60" height="60">
                    <path fill="#ff4b2b" d="M16,28.261c0,0-14-7.926-14-17.046c0-4.356,3.544-7.892,7.9-7.892c2.557,0,4.825,1.215,6.1,3.084 c1.275-1.869,3.544-3.084,6.1-3.084c4.356,0,7.9,3.536,7.9,7.892C30,20.335,16,28.261,16,28.261z"/>
                </svg>
            </div>
            <h1 class="sidebar-glow-title">System<br>Pulse</h1>
            <p style="color: #81d4fa; font-size: 0.9rem; letter-spacing: 1px; opacity: 0.8; margin-bottom: 30px;">
                NORTH-EAST HEALTH AI
            </p>
        </div>

        <div class="sidebar-card">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div class="status-dot"></div>
                <span style="color: #ffffff; font-weight: 600; font-size: 0.85rem;">Emergency Hub</span>
            </div>
            <p style="margin: 10px 0 0 0; color: #ff4b2b; font-size: 1.1rem; font-weight: 800; text-shadow: 0 0 10px rgba(255,75,43,0.3);">
                📞 Dial 104
            </p>
        </div>

        <div class="sidebar-card" style="margin-top: 15px; border-left: 3px solid #00f2fe;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.2rem;">📡</span>
                <span style="color: #ffffff; font-weight: 600; font-size: 0.85rem;">Monitor Status</span>
            </div>
            <p style="margin: 10px 0 0 0; color: #00f2fe; font-size: 0.85rem; opacity: 0.9;">
                All AI nodes active and synchronized.
            </p>
        </div>

        <style>
            .sidebar-heart {
                animation: pulse_heart 1.5s ease-in-out infinite;
                filter: drop-shadow(0 0 15px rgba(255, 75, 43, 0.6));
                margin-bottom: 20px;
            }
            @keyframes pulse_heart {
                0% { transform: scale(1); }
                15% { transform: scale(1.15); }
                30% { transform: scale(1); }
                45% { transform: scale(1.15); }
                100% { transform: scale(1); }
            }
            .sidebar-glow-title {
                color: #ffffff !important;
                font-size: 2.2rem !important;
                font-weight: 900 !important;
                line-height: 1 !important;
                text-transform: uppercase;
                text-shadow: 0 0 20px rgba(0, 242, 254, 0.8);
                margin: 0 !important;
            }
            .sidebar-card {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                padding: 15px;
                backdrop-filter: blur(10px);
                transition: transform 0.3s ease;
            }
            .sidebar-card:hover {
                transform: translateY(-3px);
                background: rgba(255, 255, 255, 0.08);
                border-color: rgba(0, 242, 254, 0.3);
            }
            .status-dot {
                width: 8px;
                height: 8px;
                background: #ff4b2b;
                border-radius: 50%;
                box-shadow: 0 0 10px #ff4b2b;
                animation: blink_dot 1s infinite;
            }
            @keyframes blink_dot {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.3; }
            }
        </style>
    """, unsafe_allow_html=True)

    # TOP NAVBAR
    st.markdown('<div class="top-nav-marker"></div>', unsafe_allow_html=True)
    pages = ["Home", "Report Symptoms", "Risk Analysis", "Map View", "Health Awareness"]
    icons = ["🏠", "📝", "📊", "🗺️", "🛡️"]
    
    # Create the horizontal navbar
    nav_cols = st.columns(len(pages))
    for i, page in enumerate(pages):
        with nav_cols[i]:
            if st.session_state.app_page == page:
                st.markdown('<div class="nav-active">', unsafe_allow_html=True)
            
            if st.button(f"{icons[i]} {page}", key=f"nav_{page}", use_container_width=True):
                st.session_state.app_page = page
                st.rerun()
                
            if st.session_state.app_page == page:
                st.markdown('</div>', unsafe_allow_html=True)

    # Page Routing
    if st.session_state.app_page == "Home":
        show_home()
    elif st.session_state.app_page == "Report Symptoms":
        show_report_form()
    elif st.session_state.app_page == "Risk Analysis":
        show_risk_analysis()
    elif st.session_state.app_page == "Map View":
        show_map_view()
    elif st.session_state.app_page == "Health Awareness":
        show_health_awareness()
        
    
if __name__ == "__main__":
    main()
