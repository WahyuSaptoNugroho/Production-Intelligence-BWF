import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from datetime import timedelta, datetime
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import tempfile
import os

# ==========================================
# 1. SETUP UI
# ==========================================
st.set_page_config(
    page_title="Production Intelligence BWF", 
    page_icon="🏭", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; }
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    h1 { color: #1E3D59; font-family: 'Helvetica', sans-serif; }
    .stDownloadButton button {
        background-color: #1E3D59;
        color: white;
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
    }
    .stDownloadButton button:hover {
        background-color: #162c40;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. PDF GENERATOR (COLOR CHARTS + STANDARD NOTE)
# ==========================================
def create_pdf(df_forecast, fig_forecast, fig_mp, fig_time, df_pareto, fig_pareto_bar, fig_pareto_pie):
    pdf = FPDF()
    
    # --- HALAMAN 1: TABEL FORECAST ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(190, 10, txt="Laporan Produksi & Pareto BWF", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(190, 10, txt=f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}", ln=True, align='C')
    pdf.ln(5)
    
    # Tabel Forecast Header
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, txt="1. Rencana Produksi Mingguan", ln=True, align='L')
    
    pdf.set_font("Arial", 'B', 9)
    pdf.set_fill_color(240, 240, 240)
    cols = ['Tanggal', 'Target (Pcs)', 'Durasi (Jam)', 'MP Needed', 'Status']
    col_widths = [35, 30, 30, 30, 30] 
    start_x = (210 - sum(col_widths)) / 2
    
    pdf.set_x(start_x)
    for i, col in enumerate(cols):
        pdf.cell(col_widths[i], 8, col, 1, 0, 'C', fill=True)
    pdf.ln()
    
    # Tabel Forecast Data
    pdf.set_font("Arial", '', 9)
    for _, row in df_forecast.iterrows():
        pdf.set_x(start_x)
        pdf.set_text_color(0, 0, 0) # Reset Hitam
        
        pdf.cell(col_widths[0], 8, str(row['Tgl']), 1, 0, 'C')
        pdf.cell(col_widths[1], 8, str(f"{row['Target']:,}"), 1, 0, 'C')
        pdf.cell(col_widths[2], 8, str(row['Durasi']), 1, 0, 'C')
        pdf.cell(col_widths[3], 8, str(row['MP']), 1, 0, 'C')
        
        # Color Coding Status
        status = str(row['Status'])
        if status == 'LEMBUR': pdf.set_text_color(231, 76, 60) # Merah
        else: pdf.set_text_color(39, 174, 96) # Hijau
        
        pdf.cell(col_widths[4], 8, status, 1, 1, 'C')

    # --- FOOTER NOTE (KEMBALI KE STANDARD) ---
    pdf.ln(10)
    pdf.set_x(10)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(100, 100, 100) 
    note = "Catatan: Laporan ini digenerate otomatis oleh sistem AI Production Intelligence BWF. Angka target mempertimbangkan history produksi dan forecast marketing."
    pdf.multi_cell(0, 5, txt=note, align='C')

    # --- HALAMAN 2: GRAFIK FORECAST ---
    pdf.add_page()
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, txt="2. Analisa Visual Forecast", ln=True, align='L')
    
    try:
        def save_img(fig, w, h):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                fig.write_image(tmp.name, width=w, height=h)
                return tmp.name
        
        img_forecast = save_img(fig_forecast, 1000, 400)
        pdf.image(img_forecast, x=10, y=30, w=190)
        
        img_mp = save_img(fig_mp, 800, 500)
        pdf.image(img_mp, x=10, y=100, w=90)
        
        img_time = save_img(fig_time, 800, 500)
        pdf.image(img_time, x=105, y=100, w=90)
        
        os.remove(img_forecast)
        os.remove(img_mp)
        os.remove(img_time)
    except:
        pdf.cell(190, 10, txt="Error loading images. Please ensure 'kaleido' is installed.", ln=True)

    # --- HALAMAN 3: PARETO ANALYSIS ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, txt="3. Analisa Pareto (Produk & Size)", ln=True, align='L')
    
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(190, 8, txt="Top 10 Hose Terbanyak:", ln=True)
    
    pareto_cols = ['Hose Type', 'Size Category', 'Total Qty']
    pareto_widths = [80, 40, 40]
    p_start_x = (210 - sum(pareto_widths)) / 2
    
    pdf.set_x(p_start_x)
    pdf.set_fill_color(240, 240, 240)
    for i, col in enumerate(pareto_cols):
        pdf.cell(pareto_widths[i], 8, col, 1, 0, 'C', fill=True)
    pdf.ln()
    
    pdf.set_font("Arial", 'B', 9)
    for _, row in df_pareto.iterrows():
        pdf.set_x(p_start_x)
        
        pdf.set_text_color(0, 0, 0)
        pdf.cell(pareto_widths[0], 8, str(row['Hose']), 1, 0, 'L')
        
        # Color Coding Size (TETAP ADA)
        size_cat = str(row['Size_Cat'])
        if 'Small' in size_cat: pdf.set_text_color(46, 204, 113) 
        elif 'Big' in size_cat: pdf.set_text_color(231, 76, 60) 
        else: pdf.set_text_color(243, 156, 18) 
            
        pdf.cell(pareto_widths[1], 8, size_cat, 1, 0, 'C')
        
        pdf.set_text_color(0, 0, 0)
        pdf.cell(pareto_widths[2], 8, str(f"{row['Output_Qty']:,.0f}"), 1, 1, 'C')
        
    pdf.ln(10)
    try:
        img_bar = save_img(fig_pareto_bar, 1000, 500)
        pdf.image(img_bar, x=10, y=pdf.get_y(), w=190)
        os.remove(img_bar)
    except: pass

    # --- HALAMAN 4: PIE CHART (COLOR FIXED) ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, txt="4. Proporsi Ukuran (Product Mix)", ln=True, align='L')
    try:
        img_pie = save_img(fig_pareto_pie, 900, 700)
        pdf.image(img_pie, x=35, y=40, w=140)
        os.remove(img_pie)
    except: pass
    
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 3. HEADER & UPLOAD
# ==========================================
st.title("Production Intelligence BWF")
st.markdown("**BWF: Predicting the Future of Production**")

with st.expander("📂 KLIK UNTUK UPLOAD DATA (Log & Forecast)", expanded=True):
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        st.info("1. Data History (Wajib)")
        file_history = st.file_uploader("Upload Log Produksi", type=['csv', 'xlsx'], key="hist")
    with col_up2:
        st.success("2. Data Target (Opsional)")
        file_plan = st.file_uploader("Upload Forecast Marketing", type=['csv', 'xlsx'], key="plan")

# ==========================================
# 4. SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3061/3061341.png", width=80)
    st.title("🎛️ Control Panel")
    st.divider()
    
    st.markdown("### ⚙️ Parameter BWF")
    current_mp = st.number_input("Manpower Saat Ini", value=17)
    work_hours = st.number_input("Jam Kerja Normal", value=6.0, step=0.5)
    
    st.markdown("### ⏱️ Cycle Time (Menit)")
    c1, c2 = st.columns(2)
    ct_small = c1.number_input("Small", value=3.0)
    ct_medium = c2.number_input("Med", value=4.0)
    ct_big = st.number_input("Big", value=15.0)
    
    st.divider()
    st.caption("Settings")
    remove_dot = st.checkbox("Hapus Titik Ribuan", value=False)
    comma_decimal = st.checkbox("Koma = Desimal", value=True)

# ==========================================
# 5. ENGINE & LOGIC
# ==========================================
def clean_number(x):
    if isinstance(x, (int, float)): return x
    x = str(x).strip()
    x = x.replace('.', '').replace(',', '.')
    try: return float(x)
    except: return 0

@st.cache_data
def load_data_all(f_hist, f_plan):
    # Load History
    if f_hist.name.endswith('.csv'): df = pd.read_csv(f_hist, header=2, encoding='ISO-8859-1')
    else: df = pd.read_excel(f_hist, header=2)
    
    if 'Tanggal_Produksi' not in df.columns: return None, {}
    df['Tanggal_Produksi'] = pd.to_datetime(df['Tanggal_Produksi'], errors='coerce')
    df = df.dropna(subset=['Tanggal_Produksi'])
    df['Output_Qty'] = df['Output_Qty'].apply(clean_number)
    df = df[df['Output_Qty'] > 0]
    
    def get_cat(x):
        s = str(x).lower()
        if 'big' in s or 'skive' in s: return 'Big'
        if 'small' in s: return 'Small'
        return 'Medium'
    df['Size_Cat'] = df['Size'].apply(get_cat) if 'Size' in df.columns else 'Medium'
    
    # Load Plan
    target_dict = {}
    if f_plan:
        try:
            if f_plan.name.endswith('.csv'): df_p = pd.read_csv(f_plan, header=2)
            else: df_p = pd.read_excel(f_plan, header=2)
            if 'New Assy' in df_p.columns:
                df_p = df_p[~df_p['New Assy'].astype(str).str.contains('TOTAL', case=False, na=False)]
            month_map = {'Mei': 5, 'JUNI': 6, 'JULI': 7, 'AGUST': 8, 'SEPT': 9, 'OKT': 10, 'NOV': 11, 'DES': 12}
            for col in df_p.columns:
                if col in month_map: target_dict[month_map[col]] = df_p[col].sum()
        except: pass
        
    return df, target_dict

def create_features_hybrid(df_daily, target_dict=None):
    df = df_daily.copy().sort_values('ds')
    if df.empty: return df
    full_idx = pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq='D')
    df = df.set_index('ds').reindex(full_idx, fill_value=0).reset_index()
    df.columns = ['ds', 'y']
    df['DayOfWeek'] = df['ds'].dt.dayofweek
    df['Day'] = df['ds'].dt.day
    df['Month'] = df['ds'].dt.month
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    df['Is_End_Month'] = df['Day'].apply(lambda x: 1 if x > 25 else 0)
    df['Quarter'] = df['ds'].dt.quarter
    if target_dict: df['Monthly_Load'] = df['Month'].map(target_dict).fillna(0)
    else: df['Monthly_Load'] = df.groupby(['Month'])['y'].transform('sum')
    avg_load = df[df['Monthly_Load'] > 0]['Monthly_Load'].mean()
    if pd.isna(avg_load): avg_load = 5000 
    df['Monthly_Load'] = df['Monthly_Load'].replace(0, avg_load)
    df['Lag_7'] = df['y'].shift(7).fillna(0)
    return df

# --- MAIN LOGIC ---
if file_history:
    df_raw, target_data = load_data_all(file_history, file_plan)
    
    if df_raw is not None and not df_raw.empty:
        st.divider()
        
        # 1. HITUNG FORECAST
        daily = df_raw.groupby('Tanggal_Produksi')['Output_Qty'].sum().reset_index()
        daily.columns = ['ds', 'y']
        df_train = create_features_hybrid(daily, target_dict=None)
        
        df_forecast_res = pd.DataFrame()
        fig_forecast, fig_mp, fig_time = go.Figure(), go.Figure(), go.Figure()
        
        # WARNA HARUS SAMA DI GRAFIK & PDF
        COLOR_MAP = {'Small': '#2ecc71', 'Medium': '#f1c40f', 'Big': '#e74c3c'}
        
        if len(df_train) > 5:
            features = ['DayOfWeek', 'Day', 'IsWeekend', 'Is_End_Month', 'Quarter', 'Monthly_Load', 'Lag_7']
            model_mid = GradientBoostingRegressor(loss='quantile', alpha=0.5, n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42).fit(df_train[features], df_train['y'])
            model_low = GradientBoostingRegressor(loss='quantile', alpha=0.1, n_estimators=300, learning_rate=0.05, max_depth=4).fit(df_train[features], df_train['y'])
            model_high = GradientBoostingRegressor(loss='quantile', alpha=0.9, n_estimators=300, learning_rate=0.05, max_depth=4).fit(df_train[features], df_train['y'])
            
            last_date = daily['ds'].max()
            forecast_res = []
            history_df = df_train[['ds', 'y']].copy()
            
            for i in range(1, 8):
                next_date = last_date + timedelta(days=i)
                m_load = target_data.get(next_date.month, 0)
                if m_load == 0: m_load = df_train['Monthly_Load'].mean()
                val_7 = history_df[history_df['ds'] == (next_date - timedelta(days=7))]['y'].values
                lag_7 = val_7[0] if len(val_7) > 0 else 0
                
                input_feat = pd.DataFrame([{
                    'DayOfWeek': next_date.dayofweek, 'Day': next_date.day,
                    'IsWeekend': 1 if next_date.dayofweek >= 5 else 0, 'Is_End_Month': 1 if next_date.day > 25 else 0,
                    'Quarter': (next_date.month - 1) // 3 + 1, 'Monthly_Load': m_load, 'Lag_7': lag_7
                }])
                
                if next_date.dayofweek >= 5: p_mid, p_low, p_high = 0, 0, 0
                else:
                    p_mid = int(model_mid.predict(input_feat)[0])
                    p_low = int(model_low.predict(input_feat)[0])
                    p_high = int(model_high.predict(input_feat)[0])
                    if p_mid < 0: p_mid = 0
                    if p_low < 0: p_low = 0
                    if p_high < p_mid: p_high = p_mid
                
                new_row = pd.DataFrame({'ds': [next_date], 'y': [p_mid]})
                history_df = pd.concat([history_df, new_row], ignore_index=True)
                
                if next_date.dayofweek < 5:
                    mix = df_raw['Size_Cat'].value_counts(normalize=True)
                    avg_ct = (mix.get('Small', 0)*ct_small) + (mix.get('Medium', 0)*ct_medium) + (mix.get('Big', 0)*ct_big)
                    total_man_hours = (p_mid * avg_ct) / 60
                    mp_needed = total_man_hours / work_hours
                    team_dur = total_man_hours / current_mp
                    status = "AMAN" if mp_needed <= current_mp else "LEMBUR"
                    forecast_res.append({
                        "Tgl": next_date.strftime('%Y-%m-%d'),
                        "Target": p_mid, "Low": p_low, "High": p_high,
                        "Durasi": round(team_dur, 1), "MP": round(mp_needed, 1), "Status": status
                    })
            
            df_forecast_res = pd.DataFrame(forecast_res)
            
            # --- FIGURES ---
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=df_forecast_res['Tgl'], y=df_forecast_res['High'], mode='lines', line=dict(width=0), showlegend=False))
            fig_forecast.add_trace(go.Scatter(x=df_forecast_res['Tgl'], y=df_forecast_res['Low'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(46, 204, 113, 0.2)', name='Zona Aman'))
            fig_forecast.add_trace(go.Scatter(x=df_forecast_res['Tgl'], y=df_forecast_res['Target'], mode='lines+markers+text', text=df_forecast_res['Target'], textposition='top center', line=dict(color='#1E3D59', width=3), name='Target'))
            fig_forecast.update_layout(title="Forecast & Risiko", height=400, width=800)
            
            fig_mp = go.Figure()
            fig_mp.add_trace(go.Bar(x=df_forecast_res['Tgl'], y=df_forecast_res['MP'], name='Orang', marker_color=['#2ecc71' if s=='AMAN' else '#e74c3c' for s in df_forecast_res['Status']]))
            fig_mp.add_trace(go.Scatter(x=df_forecast_res['Tgl'], y=[current_mp]*len(df_forecast_res), mode='lines', name=f'Tim ({int(current_mp)})', line=dict(color='black', dash='dash')))
            fig_mp.update_layout(title="Kebutuhan Orang", height=400, width=600)
            
            fig_time = go.Figure()
            fig_time.add_trace(go.Bar(x=df_forecast_res['Tgl'], y=df_forecast_res['Durasi'], name='Jam', marker_color=['#3498db' if s=='AMAN' else '#f39c12' for s in df_forecast_res['Status']]))
            fig_time.add_trace(go.Scatter(x=df_forecast_res['Tgl'], y=[work_hours]*len(df_forecast_res), mode='lines', name=f'Limit ({work_hours}h)', line=dict(color='red', dash='dot')))
            fig_time.update_layout(title="Estimasi Jam", height=400, width=600)

        # 2. HITUNG PARETO
        df_pareto_top = pd.DataFrame()
        fig_pareto_bar, fig_pareto_pie = go.Figure(), go.Figure()
        size_summary = pd.DataFrame()
        
        if 'Size_Cat' in df_raw.columns:
            total_prod = df_raw['Output_Qty'].sum()
            size_summary = df_raw.groupby('Size_Cat')['Output_Qty'].sum().reset_index()
            size_summary['Persen'] = (size_summary['Output_Qty'] / total_prod) * 100
            
            df_pareto_top = df_raw.groupby(['Hose', 'Size_Cat'])['Output_Qty'].sum().reset_index().sort_values('Output_Qty', ascending=False).head(10)
            
            fig_pareto_bar = px.bar(df_pareto_top, x='Output_Qty', y='Hose', color='Size_Cat', orientation='h', text_auto='.2s', title="Top 10 Hose", color_discrete_map=COLOR_MAP)
            fig_pareto_bar.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=150), width=800, height=500)
            
            fig_pareto_pie = px.pie(
                size_summary, 
                values='Output_Qty', 
                names='Size_Cat', 
                color='Size_Cat',
                title="Proporsi Ukuran", 
                hole=0.4,
                color_discrete_map=COLOR_MAP
            )
            fig_pareto_pie.update_layout(width=600, height=400)

        # --- UI DISPLAY ---
        col_title, col_download = st.columns([3, 1])
        with col_title: st.markdown(f"### 📅 Dashboard & Reporting")
        with col_download:
            if not df_forecast_res.empty:
                # TIDAK LAGI MENGIRIM ARGUMEN 'acc_text' KARENA SUDAH DI-HARDCODE DI DALAM FUNGSI
                pdf_bytes = create_pdf(df_forecast_res, fig_forecast, fig_mp, fig_time, df_pareto_top, fig_pareto_bar, fig_pareto_pie)
                st.download_button(
                    label="📄 Download Full PDF Report",
                    data=pdf_bytes,
                    file_name=f"Report_BWF_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        
        st.divider()
        tab1, tab2 = st.tabs(["🚀 Forecast & Planning", "📊 Analisa Pareto"])
        
        with tab1:
            if not df_forecast_res.empty:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total Target", f"{df_forecast_res['Target'].sum():,} Pcs")
                k2.metric("Target Bln", f"{target_data.get((datetime.now().month), 'N/A')} Pcs")
                k3.metric("Rata-rata Durasi", f"{df_forecast_res['Durasi'].mean():.1f} Jam")
                k4.metric("Max MP", f"{df_forecast_res['MP'].max()} Org")
                
                st.plotly_chart(fig_forecast, use_container_width=True)
                c1, c2 = st.columns(2)
                with c1: st.plotly_chart(fig_mp, use_container_width=True)
                with c2: st.plotly_chart(fig_time, use_container_width=True)
                st.dataframe(df_forecast_res, use_container_width=True, hide_index=True)
            else: st.warning("Data history kurang.")

        with tab2:
            if not df_pareto_top.empty:
                c_small, c_med, c_big = st.columns(3)
                def get_pct(cat):
                    val = size_summary[size_summary['Size_Cat'] == cat]['Persen'].values
                    return val[0] if len(val) > 0 else 0
                c_small.metric("Small", f"{get_pct('Small'):.1f}%")
                c_med.metric("Medium", f"{get_pct('Medium'):.1f}%")
                c_big.metric("Big", f"{get_pct('Big'):.1f}%")
                
                c1, c2 = st.columns([2, 1])
                with c1: st.plotly_chart(fig_pareto_bar, use_container_width=True)
                with c2: st.plotly_chart(fig_pareto_pie, use_container_width=True)
                st.dataframe(df_pareto_top, use_container_width=True)
            else: st.error("Gagal memuat pareto.")
else:
    st.info("👋 Silakan upload file Log Produksi di atas.")