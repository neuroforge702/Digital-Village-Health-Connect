# 🏥 Village Health Intelligence & Early Warning System

> **Empowering North-East India with AI-Driven Geospatial Health Analytics.**

![Project Banner](https://img.shields.io/badge/Status-Live%20Monitor-00f2fe?style=for-the-badge&logo=ai)
![Database](https://img.shields.io/badge/Backend-SQLite%20SQL-ff416c?style=for-the-badge&logo=sqlite)
![UI](https://img.shields.io/badge/UI-Three.js%203D-4facfe?style=for-the-badge&logo=three.js)

---

## 🌟 Overview
**Village Health Intelligence** is a high-performance, interactive dashboard designed to monitor and predict water-borne disease outbreaks in remote villages. It bridges the gap between local communities and health officials by using **Random Forest AI ensembles** and immersive **3D visualizations** to turn symptom reports into life-saving early warnings.

## 🔥 Key Features
*   **🤖 Random Forest AI Engine**: Analyzes multi-dimensional health reports (symptom counts + water quality) to predict risk probabilities in real-time.
*   **🌍 Interactive 3D Geospatial Intelligence**: A realistic, rotating Three.js Earth globe on the landing page representing global health connectivity.
*   **🚨 Critical Alert System**: Dynamic neon pulsing banners that automatically flag high-risk zones directly from the database to ensure immediate response.
*   **🗺️ Interactive Risk Map**: Folium-powered geospatial mapping with color-coded risk markers (Red/Orange/Green) for all monitored villages.
*   **🗄️ SQL Backend Infrastructure**: Robust SQLite relational database storage replacing legacy CSV files for atomic data integrity and high-speed queries.
*   **💧 3D Healthy Awareness Model**: Custom-engineered refractive 3D water droplet with a molecular shield, promoting purification protocols through immersive design.
*   **✨ Premium UI/UX**: Futuristic glassmorphism design, bioluminescent sidebar pulse, and high-performance cursor glitter effects.

## 🛠️ Technology Stack
- **Frontend**: [Streamlit](https://streamlit.io/) (Python Framework)
- **3D Graphics**: [Three.js](https://threejs.org/) (JavaScript WebGL)
- **AI/ML**: [Scikit-Learn](https://scikit-learn.org/) (Random Forest Ensemble)
- **Database**: SQL (SQLite3)
- **Data Visualization**: [Plotly Express](https://plotly.com/python/), [Pandas](https://pandas.pydata.org/)
- **Geospatial**: [Folium](https://python-visualization.github.io/folium/)

## 🚀 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/village-health-intelligence.git
cd village-health-intelligence
```

### 2. Install Dependencies
Ensure you have Python 3.9+ installed.
```bash
pip install streamlit pandas numpy plotly folium streamlit-folium scikit-learn
```

### 3. Launch the Intelligence Center
```bash
streamlit run app.py
```
The app will automatically initialize the SQL database and start the AI engine.

## 📝 How to Test the Intelligence
1.  **Navigate** to the `📝 Report Symptoms` tab.
2.  **Submit** a high-risk report (e.g., 50 cases + Contaminated Water).
3.  **Return** to the `🏠 Home` page to see the **🚨 Critical Health Alert** trigger instantly.
4.  **Explore** the `📊 Risk Analysis` tab to see the Random Forest feature importance weights and the chronological data ledger.

## 📡 Deployment Note
This application is designed for **Streamlit Community Cloud**. It includes a self-healing database initialization logic that builds the SQL structure on first boot.

---

### 🛡️ Developed by Team North-East Health AI
*Dedicated to improving community resilience through technology.*
