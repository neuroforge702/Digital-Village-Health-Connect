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
import bcrypt

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
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

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
            report_date TEXT,
            location TEXT,
            image_path TEXT,
            progress TEXT DEFAULT 'Unresolved',
            reporter TEXT
        )
    ''')
    
    # Add missing columns if they don't exist (Migration)
    try:
        c.execute("ALTER TABLE health_reports ADD COLUMN location TEXT")
    except:
        pass
    try:
        c.execute("ALTER TABLE health_reports ADD COLUMN image_path TEXT")
    except:
        pass
    try:
        c.execute("ALTER TABLE health_reports ADD COLUMN progress TEXT DEFAULT 'Unresolved'")
    except:
        pass
    try:
        c.execute("ALTER TABLE health_reports ADD COLUMN reporter TEXT")
    except:
        pass

    # Create report_updates table
    c.execute('''
        CREATE TABLE IF NOT EXISTS report_updates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id INTEGER,
            update_text TEXT,
            update_date TEXT,
            updated_by TEXT,
            update_photo_path TEXT,
            update_status TEXT DEFAULT 'Pending',
            admin_comment TEXT,
            FOREIGN KEY(report_id) REFERENCES health_reports(id)
        )
    ''')

    try:
        c.execute("ALTER TABLE report_updates ADD COLUMN update_photo_path TEXT")
    except: pass
    try:
        c.execute("ALTER TABLE report_updates ADD COLUMN update_status TEXT DEFAULT 'Pending'")
    except: pass
    try:
        c.execute("ALTER TABLE report_updates ADD COLUMN admin_comment TEXT")
    except: pass

    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            role TEXT, -- 'super_admin', 'admin', 'field_team', 'surveyor'
            created_by INTEGER,
            FOREIGN KEY (created_by) REFERENCES users (id)
        )
    ''')

    # Add a default super admin if no users exist
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        password = "superadmin123"
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", 
                  ("superadmin", hashed, "super_admin"))
    
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
        # Rename columns to match existing logic (Added Location and Image URL)
        df.columns = ["ID", "Village", "Latitude", "Longitude", "Symptom Count", "Symptoms", "Water Source", "Water Quality", "Risk Level", "Date", "Location", "Image Path", "Progress", "Reporter"]
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
        return df
    return pd.DataFrame(columns=["ID", "Village", "Latitude", "Longitude", "Symptom Count", "Symptoms", "Water Source", "Water Quality", "Risk Level", "Date", "Location", "Image Path", "Progress", "Reporter"])

def save_data(new_record):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO health_reports (village, latitude, longitude, symptom_count, symptoms, water_source, water_quality, risk_level, report_date, location, image_path, progress, reporter)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (new_record['Village'], new_record['Latitude'], new_record['Longitude'], new_record['Symptom Count'], new_record['Symptoms'], new_record['Water Source'], new_record['Water Quality'], new_record['Risk Level'], new_record['Date'], new_record['Location'], new_record['Image Path'], new_record['Progress'], new_record['Reporter']))
    conn.commit()
    conn.close()

def get_report_updates(report_id):
    conn = sqlite3.connect(DB_FILE)
    query = f"SELECT * FROM report_updates WHERE report_id = {report_id} ORDER BY id ASC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def add_report_update(report_id, update_text, updated_by, custom_date, update_photo_path):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO report_updates (report_id, update_text, update_date, updated_by, update_photo_path, update_status) VALUES (?, ?, ?, ?, ?, 'Pending')",
              (report_id, update_text, custom_date, updated_by, update_photo_path))
    conn.commit()
    conn.close()

def resolve_report_update(update_id, action, admin_comment):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE report_updates SET update_status = ?, admin_comment = ? WHERE id = ?", (action, admin_comment, update_id))
    
    if action == 'Accepted':
        c.execute("SELECT report_id FROM report_updates WHERE id = ?", (update_id,))
        row = c.fetchone()
        if row:
            c.execute("UPDATE health_reports SET progress = 'Resolved' WHERE id = ?", (row[0],))
            
    conn.commit()
    conn.close()

# --- AUTHENTICATION HELPERS ---
def hash_password(password):
    # Use bcrypt for hashing
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed):
    # Verify password with bcrypt
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def add_user(username, password, role, created_by=None):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    hashed = hash_password(password)
    try:
        c.execute("INSERT INTO users (username, password_hash, role, created_by) VALUES (?, ?, ?, ?)", 
                  (username, hashed, role, created_by))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    except Exception as e:
        st.error(f"Error adding user: {e}")
        return False
    finally:
        conn.close()

def get_all_users(creator_id=None, role=None):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    if role == 'super_admin': 
        # Superadmin can see everyone
        c.execute("SELECT id, username, role FROM users")
    elif role == 'admin': 
        # Admins can see teams and surveyors they created
        # Note: Super Admin might have created some as well, but usually admins manage their own
        c.execute("SELECT id, username, role FROM users WHERE created_by = ? OR role IN ('field_team', 'surveyor')", (creator_id,))
    else:
        conn.close()
        return []
    users = c.fetchall()
    conn.close()
    return users

def verify_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, username, password_hash, role FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    if user and check_password(password, user[2]):
        return {'id': user[0], 'username': user[1], 'role': user[3]}
    return None

# --- SESSION STATE INIT ---
if 'user' not in st.session_state:
    st.session_state.user = None

if 'app_page' not in st.session_state:
    st.session_state.app_page = "Home"

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
    st.title("Community Health Reports & Forum")
    tab1, tab2 = st.tabs(["📝 Submit New Report", "💬 Submitted Reports & Forum (Updates)"])

    with tab1:
        st.markdown("Fill out the form below to report health symptoms in your village.")
        
        with st.form("health_report_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                village = st.selectbox("Village Name", list(VILLAGE_COORDS.keys()))
                location_text = st.text_input("Specific Location / Landmark", placeholder="e.g. Near Primary School")
                symptom_count = st.number_input("Number of People with Symptoms", min_value=0, step=1)
                report_date = st.date_input("Date of Observation", datetime.now())
                
            with col2:
                water_source = st.selectbox("Primary Water Source", ["River", "Well", "Tap", "Hand Pump"])
                water_quality = st.selectbox("Water Quality Indicator", ["Clean", "Slightly Contaminated", "Contaminated"])
                symptoms = st.multiselect("Symptoms Observed", ["Diarrhea", "Vomiting", "Fever", "Dehydration"])
                uploaded_file = st.file_uploader("Upload Evidence Photo", type=['png', 'jpg', 'jpeg'])

            st.text_input("Progress", value="Unresolved", disabled=True, help="New reports are Unresolved by default. Update them in the Forum tab.")

            st.markdown("---")
            submitted = st.form_submit_button("Submit Health Report")
            
            if submitted:
                # Save image if exists
                image_path = ""
                if uploaded_file is not None:
                    file_path = os.path.join(UPLOAD_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}")
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    image_path = file_path

                risk = predict_risk(symptom_count, water_quality)
                reporter_name = st.session_state.user['username'] if st.session_state.user else "Anonymous"
                new_record = {
                    "Village": village,
                    "Latitude": VILLAGE_COORDS[village][0],
                    "Longitude": VILLAGE_COORDS[village][1],
                    "Symptom Count": symptom_count,
                    "Symptoms": ", ".join(symptoms) if symptoms else "None",
                    "Water Source": water_source,
                    "Water Quality": water_quality,
                    "Risk Level": risk,
                    "Date": report_date.strftime("%Y-%m-%d"),
                    "Location": location_text,
                    "Image Path": image_path,
                    "Progress": "Unresolved",
                    "Reporter": reporter_name
                }
                save_data(new_record)
                st.success(f"Report submitted successfully for {village}! Risk Level: **{risk}**")
                st.balloons()
                if risk == "High Risk":
                    st.warning("⚠️ Immediate action may be required. Boiling water is highly recommended.")

    with tab2:
        st.markdown("### Community Health Forum")
        st.write("View past reports and provide live updates as interventions and resolutions take place.")
        
        df = load_data()
        
        if df.empty:
            st.info("No reports have been submitted yet.")
        else:
            # Sort by newest first
            df_display = df.sort_values(by='ID', ascending=False)
            
            for index, row in df_display.iterrows():
                report_id = row['ID']
                with st.container():
                    st.markdown(f"#### 📌 Report #{report_id} - {row['Village']} ({row['Date']})")
                    progress_color = 'red' if row['Progress'] == 'Unresolved' else 'green'
                    st.markdown(f"**Reporter:** {row['Reporter']} | **Progress:** <strong style='color:{progress_color};'>{row['Progress']}</strong>", unsafe_allow_html=True)
                    st.markdown(f"- **Symptoms:** {row['Symptom Count']} cases ({row['Symptoms']})")
                    st.markdown(f"- **Water Source:** {row['Water Source']} ({row['Water Quality']})")
                    if row['Location']:
                        st.markdown(f"- **Location:** {row['Location']}")
                    if row['Image Path'] and os.path.exists(row['Image Path']):
                        st.image(row['Image Path'], width=300)

                    # Display forum thread (updates)
                    updates_df = get_report_updates(report_id)
                    current_user_obj = st.session_state.user
                    is_admin = current_user_obj and current_user_obj.get('role') in ['super_admin', 'admin']

                    if not updates_df.empty:
                        for u_idx, u_row in updates_df.iterrows():
                            update_id = int(u_row['id'])
                            st.markdown(f"> 💬 **{u_row['updated_by']}** ({u_row['update_date']}): {u_row['update_text']}")
                            
                            update_photo = u_row.get('update_photo_path', None)
                            if pd.notna(update_photo) and update_photo and os.path.exists(update_photo):
                                st.image(update_photo, width=200)

                            status = u_row['update_status'] if 'update_status' in u_row and pd.notna(u_row['update_status']) else 'Pending'
                            existing_comment = u_row['admin_comment'] if 'admin_comment' in u_row and pd.notna(u_row['admin_comment']) else ''
                            
                            status_color = "orange" if status == "Pending" else "green" if status == "Accepted" else "red"
                            st.markdown(f"> *Status: <strong style='color:{status_color}'>{status}</strong>*", unsafe_allow_html=True)
                            
                            if status in ["Accepted", "Rejected"] and existing_comment:
                                st.markdown(f"> *Admin Comment:* {existing_comment}")

                            # Admin / Super Admin action controls for pending updates
                            if status == "Pending" and is_admin and row['Progress'] != 'Resolved':
                                with st.form(f"forum_admin_action_{update_id}"):
                                    admin_comment = st.text_area(
                                        "💬 Comment",
                                        placeholder="Provide feedback or comments on this update...",
                                        key=f"forum_comment_{update_id}"
                                    )
                                    btn_col1, btn_col2 = st.columns(2)
                                    with btn_col1:
                                        accept_btn = st.form_submit_button("✅ Accept & Resolve")
                                    with btn_col2:
                                        reject_btn = st.form_submit_button("❌ Reject")

                                    if accept_btn:
                                        resolve_report_update(update_id, "Accepted", admin_comment)
                                        st.success("Update accepted — report marked as Resolved.")
                                        st.rerun()
                                    elif reject_btn:
                                        resolve_report_update(update_id, "Rejected", admin_comment)
                                        st.warning("Update rejected — report remains Unresolved.")
                                        st.rerun()

                    # Bottom action area — different for admins vs workers
                    if row['Progress'] != 'Resolved':
                        if is_admin:
                            # Admins see Comment + Accept/Reject controls on the report
                            with st.expander(f"🛡️ Admin Action — Report #{report_id}", expanded=False):
                                with st.form(f"admin_report_action_{report_id}"):
                                    admin_comment = st.text_area(
                                        "💬 Comment",
                                        placeholder="Provide your feedback or decision notes...",
                                        key=f"admin_report_comment_{report_id}"
                                    )
                                    btn_col1, btn_col2 = st.columns(2)
                                    with btn_col1:
                                        accept_btn = st.form_submit_button("✅ Accept & Resolve")
                                    with btn_col2:
                                        reject_btn = st.form_submit_button("❌ Reject")

                                    if accept_btn:
                                        # Add the admin comment as an update and mark as accepted/resolved
                                        current_admin = st.session_state.user['username']
                                        add_report_update(report_id, admin_comment or "Accepted by admin.", current_admin, datetime.now().strftime("%Y-%m-%d"), "")
                                        # Get the latest update id and resolve it
                                        latest_updates = get_report_updates(report_id)
                                        if not latest_updates.empty:
                                            latest_update_id = int(latest_updates.iloc[-1]['id'])
                                            resolve_report_update(latest_update_id, "Accepted", admin_comment)
                                        st.success("Report accepted — marked as Resolved.")
                                        st.rerun()
                                    elif reject_btn:
                                        current_admin = st.session_state.user['username']
                                        add_report_update(report_id, admin_comment or "Rejected by admin.", current_admin, datetime.now().strftime("%Y-%m-%d"), "")
                                        latest_updates = get_report_updates(report_id)
                                        if not latest_updates.empty:
                                            latest_update_id = int(latest_updates.iloc[-1]['id'])
                                            resolve_report_update(latest_update_id, "Rejected", admin_comment)
                                        st.warning("Report rejected — remains Unresolved.")
                                        st.rerun()
                        else:
                            # Non-admin users can add updates
                            with st.expander(f"➕ Add Update to Issue #{report_id}"):
                                with st.form(f"update_form_{report_id}"):
                                    update_date = st.date_input("Update Date", datetime.now(), key=f"date_{report_id}")
                                    update_notes = st.text_area("Update Details / Actions Taken")
                                    update_photo_upload = st.file_uploader("Upload Updated Photo (Optional)", type=['png', 'jpg', 'jpeg'], key=f"photo_{report_id}")
                                    update_submitted = st.form_submit_button("Post Update")
                                    
                                    if update_submitted:
                                        update_photo_path = ""
                                        if update_photo_upload is not None:
                                            file_path = os.path.join(UPLOAD_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_update_{update_photo_upload.name}")
                                            with open(file_path, "wb") as f:
                                                f.write(update_photo_upload.getbuffer())
                                            update_photo_path = file_path
                                            
                                        current_user = st.session_state.user['username'] if st.session_state.user else "Anonymous"
                                        add_report_update(report_id, update_notes, current_user, update_date.strftime("%Y-%m-%d"), update_photo_path)
                                        st.success("Update posted! Waiting for admin approval.")
                                        st.rerun()
                    else:
                        st.info("✅ This issue has been Resolved and is now closed. No further updates can be made.")
                    st.divider()

# --- PAGE: RISK ANALYSIS ---
def show_risk_analysis():
    st.title("📊 Disease Risk Analysis")
    df = load_data()
    
    if df.empty:
        st.warning("No data available for analysis.")
        return

    # --- COMPREHENSIVE FILTER PANEL ---
    st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(0, 210, 255, 0.08), rgba(0, 114, 255, 0.08));
                    border: 1px solid rgba(0, 242, 254, 0.2); border-radius: 16px;
                    padding: 8px 18px; margin-bottom: 18px;">
            <span style="color: #00f2fe; font-weight: 700; font-size: 1.1rem;">🔍 Data Filters</span>
            <span style="color: #81d4fa; font-size: 0.85rem; margin-left: 10px;">
                Refine the risk analysis charts and table below
            </span>
        </div>
    """, unsafe_allow_html=True)

    with st.expander("⚙️ **Filter Options** — Click to expand / collapse", expanded=True):
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            all_villages = sorted(df['Village'].dropna().unique().tolist())
            select_all_villages = st.checkbox("Select All Villages", value=True, key="risk_select_all_villages")
            selected_villages = st.multiselect(
                "🏘️ Village / Area",
                options=all_villages,
                default=all_villages if select_all_villages else [],
                key="risk_filter_village"
            )
        with fcol2:
            min_date = df['Date'].min().date() if not df['Date'].isna().all() else datetime.now().date()
            max_date = df['Date'].max().date() if not df['Date'].isna().all() else datetime.now().date()
            date_start = st.date_input("📅 Date From", value=min_date, min_value=min_date, max_value=max_date, key="risk_filter_date_start")
            date_end = st.date_input("📅 Date To", value=max_date, min_value=min_date, max_value=max_date, key="risk_filter_date_end")
        with fcol3:
            all_reporters = sorted(df['Reporter'].dropna().unique().tolist())
            select_all_reporters = st.checkbox("Select All Workers", value=True, key="risk_select_all_reporters")
            selected_reporters = st.multiselect(
                "👷 Reporter / Worker",
                options=all_reporters,
                default=all_reporters if select_all_reporters else [],
                key="risk_filter_reporter"
            )

        fcol4, fcol5, fcol6, fcol7 = st.columns(4)
        with fcol4:
            all_risks = sorted(df['Risk Level'].dropna().unique().tolist())
            selected_risks = st.multiselect(
                "⚠️ Risk Level",
                options=all_risks,
                default=all_risks,
                key="risk_filter_risk"
            )
        with fcol5:
            all_water_sources = sorted(df['Water Source'].dropna().unique().tolist())
            selected_water_sources = st.multiselect(
                "💧 Water Source",
                options=all_water_sources,
                default=all_water_sources,
                key="risk_filter_water_source"
            )
        with fcol6:
            all_water_quality = sorted(df['Water Quality'].dropna().unique().tolist())
            selected_water_quality = st.multiselect(
                "🧪 Water Quality",
                options=all_water_quality,
                default=all_water_quality,
                key="risk_filter_water_quality"
            )
        with fcol7:
            all_progress = sorted(df['Progress'].dropna().unique().tolist())
            selected_progress = st.multiselect(
                "🔄 Progress",
                options=all_progress,
                default=all_progress,
                key="risk_filter_progress"
            )

    # --- APPLY FILTERS ---
    filtered_df = df.copy()
    if selected_villages:
        filtered_df = filtered_df[filtered_df['Village'].isin(selected_villages)]
    else:
        filtered_df = filtered_df.iloc[0:0]  # empty if nothing selected
    if selected_reporters:
        filtered_df = filtered_df[filtered_df['Reporter'].isin(selected_reporters)]
    else:
        filtered_df = filtered_df.iloc[0:0]
    if selected_risks:
        filtered_df = filtered_df[filtered_df['Risk Level'].isin(selected_risks)]
    else:
        filtered_df = filtered_df.iloc[0:0]
    if selected_water_sources:
        filtered_df = filtered_df[filtered_df['Water Source'].isin(selected_water_sources)]
    else:
        filtered_df = filtered_df.iloc[0:0]
    if selected_water_quality:
        filtered_df = filtered_df[filtered_df['Water Quality'].isin(selected_water_quality)]
    else:
        filtered_df = filtered_df.iloc[0:0]
    if selected_progress:
        filtered_df = filtered_df[filtered_df['Progress'].isin(selected_progress)]
    else:
        filtered_df = filtered_df.iloc[0:0]
    # Date range filter
    filtered_df = filtered_df[
        (filtered_df['Date'].dt.date >= date_start) &
        (filtered_df['Date'].dt.date <= date_end)
    ]

    # --- FILTER SUMMARY ---
    active_filters = []
    if len(selected_villages) < len(all_villages):
        active_filters.append(f"**Villages:** {len(selected_villages)}/{len(all_villages)}")
    if date_start != min_date or date_end != max_date:
        active_filters.append(f"**Date:** {date_start} → {date_end}")
    if len(selected_reporters) < len(all_reporters):
        active_filters.append(f"**Reporters:** {len(selected_reporters)}/{len(all_reporters)}")
    if len(selected_risks) < len(all_risks):
        active_filters.append(f"**Risk Levels:** {', '.join(selected_risks)}")
    if len(selected_water_sources) < len(all_water_sources):
        active_filters.append(f"**Water Sources:** {', '.join(selected_water_sources)}")
    if len(selected_water_quality) < len(all_water_quality):
        active_filters.append(f"**Water Quality:** {', '.join(selected_water_quality)}")
    if len(selected_progress) < len(all_progress):
        active_filters.append(f"**Progress:** {', '.join(selected_progress)}")

    if active_filters:
        st.markdown(
            f"<div style='background: rgba(0, 242, 254, 0.06); border-left: 4px solid #00f2fe; "
            f"padding: 10px 16px; border-radius: 8px; margin-bottom: 16px; font-size: 0.9rem;'>"
            f"🎯 <b>Active Filters:</b> {' &nbsp;|&nbsp; '.join(active_filters)} "
            f"&nbsp;— Showing <b>{len(filtered_df)}</b> of <b>{len(df)}</b> records</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='background: rgba(105, 240, 174, 0.06); border-left: 4px solid #69f0ae; "
            f"padding: 10px 16px; border-radius: 8px; margin-bottom: 16px; font-size: 0.9rem;'>"
            f"✅ Showing all <b>{len(filtered_df)}</b> records (no filters active)</div>",
            unsafe_allow_html=True
        )

    if filtered_df.empty:
        st.warning("No data matches the selected filters. Adjust your filters above.")
        return

    # --- RISK DISTRIBUTION (FILTERED) ---
    st.markdown("### Risk Distribution")
    risk_counts = filtered_df['Risk Level'].value_counts().reset_index()
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
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col2:
        st.markdown("#### 🤖 AI Model Intelligence")
        st.write("Current calculation is powered by an **ensemble Random Forest Classifier**.")
        # Dynamically evaluate Symptom Pulse and Hydro-Quality based on filtered data severity
        if not filtered_df.empty:
            total_filtered = len(filtered_df)
            bad_water_count = len(filtered_df[filtered_df['Water Quality'] != 'Clean'])
            # Calculate a representative hydro quality factor
            hydro_pct = (bad_water_count / total_filtered) * 100 if total_filtered > 0 else 35.0
            
            # For symptom pulse, let's normalize the severity based on average symptoms
            avg_symp = filtered_df['Symptom Count'].mean()
            # If avg symptoms > 10, it's very high. Scale it.
            symp_pct = min(100.0, max(0.0, avg_symp * 5))
            
            # Normalize them to sum to 100% just like weights
            total_weight = hydro_pct + symp_pct
            if total_weight > 0:
                hydro_weight = (hydro_pct / total_weight) * 100
                symp_weight = (symp_pct / total_weight) * 100
            else:
                hydro_weight = 35.0
                symp_weight = 65.0
        else:
            symp_weight = 65.0
            hydro_weight = 35.0
            
        st.info("**Feature Analytics (Dynamic Indicators):**")
        col_f1, col_f2 = st.columns(2)
        col_f1.metric("Symptom Pulse", f"{symp_weight:.1f}%", help="Calculated pulse from filtered symptom volume")
        col_f2.metric("Hydro-Quality", f"{hydro_weight:.1f}%", help="Calculated from water contamination levels in filtered data")
        
        st.markdown("""
        **Model Logic:**
        The Random Forest model analyzes multiple scenarios simultaneously to determine the risk level. It is trained to recognize patterns where specific water conditions combined with symptom spikes indicate a high probability of outbreak.
        """)
        
    st.markdown("---")
    st.subheader("Detailed Village Status")
    
    display_cols = ["Village", "Location", "Symptom Count", "Symptoms", "Water Source", "Water Quality", "Risk Level", "Date", "Reporter", "Image Path"]
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

    # --- COMPREHENSIVE FILTER PANEL ---
    st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(0, 210, 255, 0.08), rgba(0, 114, 255, 0.08));
                    border: 1px solid rgba(0, 242, 254, 0.2); border-radius: 16px;
                    padding: 8px 18px; margin-bottom: 18px;">
            <span style="color: #00f2fe; font-weight: 700; font-size: 1.1rem;">🔍 Map Filters</span>
            <span style="color: #81d4fa; font-size: 0.85rem; margin-left: 10px;">
                Filter which data points appear on the map
            </span>
        </div>
    """, unsafe_allow_html=True)

    with st.expander("⚙️ **Filter Options** — Click to expand / collapse", expanded=True):
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            all_villages = sorted(df['Village'].dropna().unique().tolist())
            select_all_villages = st.checkbox("Select All Villages", value=True, key="map_select_all_villages")
            selected_villages = st.multiselect(
                "🏘️ Village / Area",
                options=all_villages,
                default=all_villages if select_all_villages else [],
                key="map_filter_village"
            )
        with fcol2:
            min_date = df['Date'].min().date() if not df['Date'].isna().all() else datetime.now().date()
            max_date = df['Date'].max().date() if not df['Date'].isna().all() else datetime.now().date()
            date_start = st.date_input("📅 Date From", value=min_date, min_value=min_date, max_value=max_date, key="map_filter_date_start")
            date_end = st.date_input("📅 Date To", value=max_date, min_value=min_date, max_value=max_date, key="map_filter_date_end")
        with fcol3:
            all_reporters = sorted(df['Reporter'].dropna().unique().tolist())
            select_all_reporters = st.checkbox("Select All Workers", value=True, key="map_select_all_reporters")
            selected_reporters = st.multiselect(
                "👷 Reporter / Worker",
                options=all_reporters,
                default=all_reporters if select_all_reporters else [],
                key="map_filter_reporter"
            )

        fcol4, fcol5, fcol6, fcol7 = st.columns(4)
        with fcol4:
            all_risks = sorted(df['Risk Level'].dropna().unique().tolist())
            selected_risks = st.multiselect(
                "⚠️ Risk Level",
                options=all_risks,
                default=all_risks,
                key="map_filter_risk"
            )
        with fcol5:
            all_water_sources = sorted(df['Water Source'].dropna().unique().tolist())
            selected_water_sources = st.multiselect(
                "💧 Water Source",
                options=all_water_sources,
                default=all_water_sources,
                key="map_filter_water_source"
            )
        with fcol6:
            all_water_quality = sorted(df['Water Quality'].dropna().unique().tolist())
            selected_water_quality = st.multiselect(
                "🧪 Water Quality",
                options=all_water_quality,
                default=all_water_quality,
                key="map_filter_water_quality"
            )
        with fcol7:
            all_progress = sorted(df['Progress'].dropna().unique().tolist())
            selected_progress = st.multiselect(
                "🔄 Progress",
                options=all_progress,
                default=all_progress,
                key="map_filter_progress"
            )

    # --- APPLY FILTERS ---
    filtered_df = df.copy()
    if selected_villages:
        filtered_df = filtered_df[filtered_df['Village'].isin(selected_villages)]
    else:
        filtered_df = filtered_df.iloc[0:0]
    if selected_reporters:
        filtered_df = filtered_df[filtered_df['Reporter'].isin(selected_reporters)]
    else:
        filtered_df = filtered_df.iloc[0:0]
    if selected_risks:
        filtered_df = filtered_df[filtered_df['Risk Level'].isin(selected_risks)]
    else:
        filtered_df = filtered_df.iloc[0:0]
    if selected_water_sources:
        filtered_df = filtered_df[filtered_df['Water Source'].isin(selected_water_sources)]
    else:
        filtered_df = filtered_df.iloc[0:0]
    if selected_water_quality:
        filtered_df = filtered_df[filtered_df['Water Quality'].isin(selected_water_quality)]
    else:
        filtered_df = filtered_df.iloc[0:0]
    # Date range filter
    if not filtered_df.empty:
        filtered_df = filtered_df[
            (filtered_df['Date'].dt.date >= date_start) &
            (filtered_df['Date'].dt.date <= date_end)
        ]

    # --- FILTER SUMMARY ---
    active_filters = []
    if len(selected_villages) < len(all_villages):
        active_filters.append(f"**Villages:** {len(selected_villages)}/{len(all_villages)}")
    if date_start != min_date or date_end != max_date:
        active_filters.append(f"**Date:** {date_start} → {date_end}")
    if len(selected_reporters) < len(all_reporters):
        active_filters.append(f"**Reporters:** {len(selected_reporters)}/{len(all_reporters)}")
    if len(selected_risks) < len(all_risks):
        active_filters.append(f"**Risk Levels:** {', '.join(selected_risks)}")
    if len(selected_water_sources) < len(all_water_sources):
        active_filters.append(f"**Water Sources:** {', '.join(selected_water_sources)}")
    if len(selected_water_quality) < len(all_water_quality):
        active_filters.append(f"**Water Quality:** {', '.join(selected_water_quality)}")

    if active_filters:
        st.markdown(
            f"<div style='background: rgba(0, 242, 254, 0.06); border-left: 4px solid #00f2fe; "
            f"padding: 10px 16px; border-radius: 8px; margin-bottom: 16px; font-size: 0.9rem;'>"
            f"🎯 <b>Active Filters:</b> {' &nbsp;|&nbsp; '.join(active_filters)} "
            f"&nbsp;— Showing <b>{len(filtered_df)}</b> of <b>{len(df)}</b> records on map</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='background: rgba(105, 240, 174, 0.06); border-left: 4px solid #69f0ae; "
            f"padding: 10px 16px; border-radius: 8px; margin-bottom: 16px; font-size: 0.9rem;'>"
            f"✅ Showing all <b>{len(filtered_df)}</b> records on map (no filters active)</div>",
            unsafe_allow_html=True
        )

    if filtered_df.empty:
        st.warning("No data matches the selected filters. Adjust your filters above.")
        # Still show an empty map
        m = folium.Map(location=[26.0, 92.0], zoom_start=6, tiles="CartoDB dark_matter")
        st_folium(m, width=1200, height=600)
        return

    # Prepare latest data per village from FILTERED data for mapping
    latest_df = filtered_df.sort_values('Date').groupby('Village').tail(1)
    
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
                     f"<b>Reporter:</b> {row['Reporter']}<br>" \
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

def show_logs():
    st.title("🛡️ Admin Logs — Community Health Reports")

    df = load_data()
    if df.empty:
        st.info("No reports have been submitted yet.")
        return

    # --- Notification Banner ---
    conn = sqlite3.connect(DB_FILE)
    pending_updates = pd.read_sql_query(
        "SELECT ru.*, hr.village FROM report_updates ru "
        "JOIN health_reports hr ON ru.report_id = hr.id "
        "WHERE ru.update_status = 'Pending' ORDER BY ru.id DESC",
        conn
    )
    conn.close()

    if not pending_updates.empty:
        st.markdown(
            f"""<div style="background: linear-gradient(90deg, #ff6200, #ffb347); padding: 14px 20px;
            border-radius: 12px; margin-bottom: 20px; display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.6rem;">🔔</span>
            <span style="color: white; font-weight: 700; font-size: 1rem;">
                {len(pending_updates)} pending update(s) awaiting your review
            </span></div>""",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Filter controls
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        filter_progress = st.multiselect(
            "Filter by Progress", ["Unresolved", "Resolved"],
            default=["Unresolved", "Resolved"]
        )
    with filter_col2:
        filter_village = st.multiselect(
            "Filter by Village", df['Village'].unique().tolist(),
            default=df['Village'].unique().tolist()
        )

    df_filtered = df[df['Progress'].isin(filter_progress) & df['Village'].isin(filter_village)]
    df_sorted = df_filtered.sort_values('ID', ascending=False)

    for _, row in df_sorted.iterrows():
        report_id = int(row['ID'])
        progress_color = "#ff4b2b" if row['Progress'] == "Unresolved" else "#69f0ae"

        with st.container():
            header_col, badge_col = st.columns([5, 1])
            with header_col:
                st.markdown(
                    f"#### 📌 Report #{report_id} — {row['Village']} "
                    f"<span style='color:gray; font-size:0.85rem;'>{str(row['Date'])[:10]}</span>",
                    unsafe_allow_html=True
                )
            with badge_col:
                st.markdown(
                    f"<div style='background:{progress_color}; color:white; border-radius:8px; "
                    f"padding:4px 10px; font-size:0.8rem; font-weight:700; text-align:center; margin-top:8px;'>"
                    f"{row['Progress']}</div>",
                    unsafe_allow_html=True
                )

            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.markdown(f"- **Reporter:** {row['Reporter']}")
                st.markdown(f"- **Location:** {row['Location'] or 'N/A'}")
                st.markdown(f"- **Symptoms:** {row['Symptom Count']} cases — {row['Symptoms']}")
            with info_col2:
                st.markdown(f"- **Water Source:** {row['Water Source']} ({row['Water Quality']})")
                st.markdown(f"- **AI Risk Level:** {row['Risk Level']}")

            if row['Image Path'] and pd.notna(row['Image Path']) and os.path.exists(str(row['Image Path'])):
                st.image(row['Image Path'], caption="Original Evidence Photo", width=280)

            # Thread of updates
            updates_df = get_report_updates(report_id)
            if not updates_df.empty:
                st.markdown("**Update Thread:**")
                for _, u_row in updates_df.iterrows():
                    update_id = int(u_row['id'])
                    u_status = u_row.get('update_status', 'Pending') or 'Pending'
                    u_comment = u_row.get('admin_comment', '') or ''
                    u_photo = u_row.get('update_photo_path', '') or ''

                    s_color = "orange" if u_status == "Pending" else "green" if u_status == "Accepted" else "red"

                    with st.expander(
                        f"Update by {u_row['updated_by']} on {u_row['update_date']} — Status: {u_status}",
                        expanded=(u_status == "Pending")
                    ):
                        st.markdown(f"**Details:** {u_row['update_text']}")
                        if u_photo and os.path.exists(u_photo):
                            st.image(u_photo, caption="Update Photo", width=250)
                        st.markdown(
                            f"<span style='color:{s_color}; font-weight:700;'>Status: {u_status}</span>",
                            unsafe_allow_html=True
                        )
                        if u_comment:
                            st.info(f"**Your Response:** {u_comment}")

                        # Admin action form — only for pending updates on unresolved reports
                        if u_status == "Pending" and row['Progress'] != "Resolved":
                            with st.form(f"admin_action_{update_id}"):
                                admin_comment = st.text_area(
                                    "Admin Comment / Feedback",
                                    placeholder="Provide feedback to the worker..."
                                )
                                a_col1, a_col2 = st.columns(2)
                                with a_col1:
                                    accept_btn = st.form_submit_button("Accept and Resolve")
                                with a_col2:
                                    reject_btn = st.form_submit_button("Reject")

                                if accept_btn:
                                    resolve_report_update(update_id, "Accepted", admin_comment)
                                    st.success("Update accepted. Report marked as Resolved.")
                                    st.rerun()
                                elif reject_btn:
                                    resolve_report_update(update_id, "Rejected", admin_comment)
                                    st.warning("Update rejected. Worker will be notified.")
                                    st.rerun()
            else:
                st.caption("No updates yet for this report.")

            st.divider()

def show_login():

    st.markdown("""
        <div style="max-width: 400px; margin: 0 auto; padding-top: 50px;">
            <h2 style="text-align: center; color: #00f2fe;">🔐 Secure Portal</h2>
            <p style="text-align: center; color: #81d4fa; margin-bottom: 30px;">Authorized Health Personnel Only</p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("LOGIN")
                
                if submit:
                    user = verify_user(username, password)
                    if user:
                        st.session_state.user = user
                        st.session_state.app_page = "Home"
                        st.success(f"Welcome back, {user['username']}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")

def show_user_management():
    st.title("👤 User Management")
    user = st.session_state.user
    
    if user['role'] not in ['super_admin', 'admin']:
        st.error("Unauthorized access.")
        return

    tab1, tab2 = st.tabs(["Create User", "Existing Users"])
    
    with tab1:
        st.subheader("Add New Personnel")
        with st.form("add_user_form"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            
            roles = []
            if user['role'] == 'super_admin':
                roles = ['admin', 'worker']
            elif user['role'] == 'admin':
                roles = ['worker']
                
            new_role = st.selectbox("Role", roles)
            add_submit = st.form_submit_button("CREATE USER")
            
            if add_submit:
                if new_username and new_password:
                    if add_user(new_username, new_password, new_role, user['id']):
                        st.success(f"User {new_username} created successfully as {new_role}.")
                    else:
                        st.error("Username already exists or error occurred.")
                else:
                    st.warning("Please fill all fields.")
                    
    with tab2:
        st.subheader("Manage Personnel")
        users_list = get_all_users(user['id'], user['role'])
        if users_list:
            user_df = pd.DataFrame(users_list, columns=["ID", "Username", "Role"])
            st.dataframe(user_df, use_container_width=True)
        else:
            st.info("No users found.")

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

    # AUTH CHECK (Critical: Show login if not authenticated)
    if not st.session_state.user:
        show_login()
        return

    # USER CONTEXT IN SIDEBAR
    with st.sidebar:
        st.markdown(f"""
            <div class="sidebar-card" style="margin-top: 15px; border-left: 3px solid #69f0ae;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.2rem;">👤</span>
                    <div style="display: flex; flex-direction: column;">
                        <span style="color: #ffffff; font-weight: 600; font-size: 0.85rem;">{st.session_state.user['username']}</span>
                        <span style="color: #69f0ae; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px;">{st.session_state.user['role'].replace('_', ' ')}</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚪 LOGOUT", use_container_width=True):
            logout_user()

    # TOP NAVBAR
    st.markdown('<div class="top-nav-marker"></div>', unsafe_allow_html=True)
    pages = ["Home", "Report Symptoms", "Risk Analysis", "Map View", "Health Awareness"]
    icons = ["🏠", "📝", "📊", "🗺️", "🛡️"]
    
    # Dynamic Navbar for Admins
    if st.session_state.user['role'] in ['super_admin', 'admin']:
        pages.append("Logs")
        icons.append("🛡️")
        pages.append("User Management")
        icons.append("👥")
    
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
    elif st.session_state.app_page == "Logs" and st.session_state.user['role'] in ['super_admin', 'admin']:
        show_logs()
    elif st.session_state.app_page == "User Management" and "User Management" in pages:
        show_user_management()

def logout_user():
    st.session_state.user = None
    st.session_state.app_page = "Home"
    st.rerun()
        
    
if __name__ == "__main__":
    main()
