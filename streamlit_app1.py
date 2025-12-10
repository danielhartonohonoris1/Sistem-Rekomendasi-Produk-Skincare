# Sistem Rekomendasi Produk Skincare Sociolla (IBCF) 


# --- Import packages  ---
import streamlit as st 
import pandas as pd 
import numpy as np 
import re 
import plotly.express as px 
import plotly.graph_objects as go 
from sklearn.preprocessing import MinMaxScaler 

st.set_page_config(page_title="Sistem Rekomendasi Skincare", layout="wide")

# -------------------------------------------------------------------
# BAGIAN 1: FUNGSI PREPROCESSING DATA
# -------------------------------------------------------------------
@st.cache_data 
def preprocess_data(path):
    try:
        # 'path' sekarang bisa berupa nama file ATAU file yang diunggah
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError: 
        df = pd.read_csv(path, encoding="latin1")
    except Exception as e:
        st.error(f"Gagal membaca dataset: {e}") 
        st.stop() 
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

    

    
    def clean_price(price):
        if isinstance(price, str):
            match = re.search(r'(?:Rp\s?)?(\d{1,3}(?:\.\d{3})*|\d+)', price)
            if match:
                number_str = match.group(1) 
                cleaned_number_str = number_str.replace('.', '') 
                try:
                    return float(cleaned_number_str) 
                except ValueError:
                    return 0 
        if isinstance(price, (int, float)) and not pd.isna(price):
            return float(price)
        return 0 

    def clean_rating(rating):
        
        if pd.isna(rating): return 0.0 
        num = pd.to_numeric(rating, errors='coerce') 
        if pd.isna(num): return 0.0 
        return num if num <= 5.0 else num / 10.0

    def parse_repurchase(value):
    
        if isinstance(value, str):
            numbers = re.findall(r'\((\d+)\)', value)
            if numbers:
                return int(numbers[0]) 
        return 0 

    
    def classify_category(name):
        """Mengklasifikasikan produk ke kategori berdasarkan kata kunci di nama produk."""
        if pd.isna(name): return "Lainnya" 
        n = str(name).lower() 
        if "sunscreen" in n or "sunblock" in n: return "Sunscreen"
        if "serum" in n: return "Serum"
        if "toner" in n: return "Toner"
        if "cleanser" in n or "wash" in n or "foam" in n: return "Skincare" 
        if "scrub" in n: return "Scrub"
        if "moisturizer" in n: return "Moisturizer"
        if "cream" in n: return "Cream"
        if "body" in n or "bath" in n or "lotion" in n: return "Bath & Body"
        return "Skincare" 

   
    df.dropna(subset=['product_name'], inplace=True)
    df = df[df['product_name'].str.len() > 1].copy() 
    df['price_clean'] = df['price'].apply(clean_price)
    df['rating_clean'] = df['rating'].apply(clean_rating)
    df['repurchase_yes'] = df['repurchase_yes'].apply(parse_repurchase)
    df['repurchase_no'] = df['repurchase_no'].apply(parse_repurchase)
    df['repurchase_maybe'] = df['repurchase_maybe'].apply(parse_repurchase)
    df['category_classified'] = df['product_name'].apply(classify_category)

    
    reviews_numeric = pd.to_numeric(df['number_of_reviews'], errors='coerce')
    reviews_filled = reviews_numeric.fillna(0)
    reviews_abs = reviews_filled.abs()
    df['reviews_clean'] = reviews_abs.astype(int)
    df = df[df['price_clean'] > 0].copy()
    df = df.sort_values('rating_clean', ascending=False).drop_duplicates('product_name').sort_index()

    if df.empty:
        st.error("Tidak ada produk yang tersisa setelah pemfilteran. Periksa dataset Anda.")
        st.stop()
    return df

# -------------------------------------------------------------------
# BAGIAN 2: FUNGSI PERHITUNGAN SKOR
# -------------------------------------------------------------------
def calculate_scores(df):
    df_scored = df.copy() 
    scaler = MinMaxScaler() 
    if len(df_scored) == 1:
        
        df_scored['rating_score'] = 5.0 
        df_scored['review_score'] = 5.0
    elif len(df_scored) > 1:
        df_scored['rating_score'] = scaler.fit_transform(df_scored[['rating_clean']])
        max_reviews = 2000 
        log_max_reviews = np.log1p(max_reviews) 
        if log_max_reviews == 0: 
             df_scored['review_score'] = 0.0
        else:
             df_scored['review_score'] = df_scored['reviews_clean'].apply(
                  lambda x: np.log1p(min(x, max_reviews)) / log_max_reviews
             )
    else: 
        df_scored['rating_score'] = 0.0
        df_scored['review_score'] = 0.0

    df_scored['final_score'] = (0.5 * df_scored['rating_score']) + (0.5 * df_scored['review_score'])

    df_scored['is_relevant'] = df_scored['repurchase_yes'] > df_scored['repurchase_no']

    return df_scored

# -------------------------------------------------------------------
# BAGIAN 2.1: FUNGSI PENGAMBILAN REKOMENDASI
def get_recommendations(df, product_name, top_n=10):
    if product_name not in df['product_name'].values:
        return pd.DataFrame() 
    selected_product_list = df[df['product_name'] == product_name]
    if selected_product_list.empty:
         return pd.DataFrame()
    selected_product = selected_product_list.iloc[0]

    product_category = selected_product['category_classified']
    recommendations = df[(df['category_classified'] == product_category) & (df['product_name'] != product_name)]
    recommendations = recommendations.sort_values(by='final_score', ascending=False).head(top_n)

    return recommendations

# -------------------------------------------------------------------
# BAGIAN 2.5: FUNGSI EVALUASI SISTEM
# -------------------------------------------------------------------
@st.cache_data 
def run_evaluation(_df, category):
    category_products = _df[_df['category_classified'] == category]
    product_names = category_products['product_name'].tolist() 

    if not product_names:
        st.warning(f"Tidak ada produk dalam kategori '{category}' untuk dievaluasi.")
        return 0, 0, 0, pd.DataFrame(columns=['product_name', 'precision', 'recall', 'f1_score'])

    total_relevant_in_category = category_products['is_relevant'].sum()

    if total_relevant_in_category == 0:
        st.warning(f"Tidak ada produk yang dianggap 'relevan' (Repurchase Yes > No) dalam kategori '{category}'. Recall akan selalu 0.")

    evaluation_results = [] 
    progress_bar = st.progress(0, text="Mengevaluasi produk...") 
    num_products = len(product_names)

    for i, product_name in enumerate(product_names):
        recommendations_df = get_recommendations(_df, product_name, top_n=10)
        precision_at_10 = 0.0
        recall_at_10 = 0.0
        f1_at_10 = 0.0

        if not recommendations_df.empty:
            relevant_found = recommendations_df['is_relevant'].sum()
            precision_at_10 = relevant_found / 10.0
            if total_relevant_in_category > 0:
                recall_at_10 = relevant_found / total_relevant_in_category
            if (precision_at_10 + recall_at_10) > 0:
                f1_at_10 = 2 * (precision_at_10 * recall_at_10) / (precision_at_10 + recall_at_10)

        evaluation_results.append({
            'product_name': product_name,
            'precision': precision_at_10,
            'recall': recall_at_10,
            'f1_score': f1_at_10
        })

        if num_products > 0:
             progress_bar.progress((i + 1) / num_products, text=f"Mengevaluasi produk {i+1}/{num_products}")

    progress_bar.empty()

    df_eval_results = pd.DataFrame(evaluation_results)

    if not df_eval_results.empty:
        avg_precision = df_eval_results['precision'].mean()
        avg_recall = df_eval_results['recall'].mean()
        avg_f1_score = df_eval_results['f1_score'].mean()
    else: 
        avg_precision, avg_recall, avg_f1_score = 0, 0, 0

    return avg_precision, avg_recall, avg_f1_score, df_eval_results

# -------------------------------------------------------------------
# BAGIAN 3: TAMPILAN APLIKASI STREAMLIT
# -------------------------------------------------------------------
st.title("Sistem Rekomendasi Produk Skincare 💄")
st.markdown("Temukan produk skincare yang sesuai dengan kebutuhan anda:")

try:

    st.image("banner.png", use_container_width=True)
except Exception:
    pass 

st.markdown("---")

# --- PERUBAHAN DIMULAI DI SINI ---
# 1. Tambahkan file uploader
uploaded_file = st.file_uploader("Unggah dataset CSV Anda di sini:", type=["csv"])

# 2. Hanya jalankan sisa aplikasi JIKA file sudah diunggah
if uploaded_file is not None:
    try:
        # 3. Ganti nama file hardcode dengan file yang diunggah
        df_processed = preprocess_data(uploaded_file)
        
        # Cek apakah hasil preprocessing kosong.
        if df_processed.empty:
            st.error("Gagal memproses data atau tidak ada data yang valid setelah preprocessing.")
            st.stop()
        # Panggil fungsi perhitungan skor.
        df_final = calculate_scores(df_processed)
        # Cek apakah hasil perhitungan skor kosong.
        if df_final.empty:
            st.error("Gagal menghitung skor atau tidak ada data yang valid setelah perhitungan skor.")
            st.stop()
        # Buat daftar unik nama produk untuk dropdown.
        product_list = sorted(df_final['product_name'].unique().tolist())
        # Cek apakah daftar produk kosong.
        if not product_list:
            st.error("Tidak ada nama produk unik yang ditemukan setelah pemrosesan data.")
            st.stop()

    # Menangani error umum lainnya saat pemrosesan data.
    except Exception as e:
        st.error(f"Terjadi error saat memproses data awal: {e}")
        st.exception(e) # Menampilkan detail error (traceback) untuk debugging.
        st.stop()

    # Membuat 4 Tab utama pada aplikasi.
    tab1, tab2, tab3, tab4 = st.tabs([
        "**▶️ Rekomendasi Produk**",
        "**📖 Daftar Produk**",
        "**📊 Analisis Data**",
        "**🧪 Evaluasi Sistem**"
    ])

    # --- Konten untuk Tab 1: Rekomendasi Produk ---
    with tab1:
        st.header("Temukan Produk Serupa Berdasarkan Pilihan Anda")

        # Cek jika daftar produk kosong.
        if not product_list:
             st.warning("Tidak ada produk tersedia untuk dipilih.")
             selected_product_name = None # Jika kosong, tidak ada produk yang dipilih.
        else:
            # Membuat dropdown (selectbox) untuk memilih produk.
            selected_product_name = st.selectbox(
                "Pilih satu produk yang Anda sukai:",
                options=product_list, # Pilihan berasal dari daftar produk unik.
                index=0 # Default ke produk pertama dalam daftar.
            )

        # Hanya jalankan jika ada produk yang dipilih.
        if selected_product_name:
            st.markdown("---") # Garis pemisah.
            # Panggil fungsi untuk mendapatkan 10 rekomendasi.
            recommendations_df = get_recommendations(df_final, selected_product_name, top_n=10)
            # Ambil detail lengkap dari produk yang dipilih pengguna.
            selected_product_details_list = df_final[df_final['product_name'] == selected_product_name]

            # Cek apakah detail produk ditemukan.
            if selected_product_details_list.empty:
                st.warning(f"Detail untuk produk '{selected_product_name}' tidak ditemukan.")
            else:
                selected_product_details = selected_product_details_list.iloc[0] # Ambil baris pertama.
                # Bagi layout menjadi 2 kolom: col1 (kecil) untuk detail, col2 (besar) untuk rekomendasi.
                col1, col2 = st.columns([1, 2])

                # Konten Kolom Kiri: Menampilkan detail produk yang dipilih.
                with col1:
                    st.subheader("Produk yang Dipilih:")
                    # .get(..., 'N/A') digunakan untuk menghindari error jika kolom tidak ada.
                    st.markdown(f"**Nama:** {selected_product_details.get('product_name', 'N/A')}")
                    st.markdown(f"**Brand:** {selected_product_details.get('brand', 'N/A')}")
                    st.markdown(f"**Kategori:** {selected_product_details.get('category_classified', 'N/A')}")
                    # Format harga agar menggunakan titik sebagai pemisah ribuan.
                    price_val = selected_product_details.get('price_clean', 0)
                    st.markdown(f"**Harga:** Rp {price_val:,.0f}".replace(",", "."))
                    st.markdown(f"**Rating:** {selected_product_details.get('rating_clean', 0.0):.1f} ⭐")
                    # Format jumlah review agar menggunakan titik sebagai pemisah ribuan.
                    review_val = selected_product_details.get('reviews_clean', 0)
                    st.markdown(f"**Jumlah Review:** {review_val:,}".replace(",", "."))

                # Konten Kolom Kanan: Menampilkan 10 rekomendasi dalam tabel.
                with col2:
                    st.subheader(f"Top 10 Rekomendasi Produk Serupa (Kategori: {selected_product_details.get('category_classified', 'N/A')})")
                    # Cek jika ada rekomendasi yang dihasilkan.
                    if recommendations_df.empty:
                        st.info("Tidak ada rekomendasi serupa yang ditemukan untuk produk ini.")
                    else:
                        # Buat salinan DataFrame rekomendasi untuk diformat tampilannya.
                        recommendations_display = recommendations_df.copy()
                        # Format kolom harga, review, dan skor agar lebih rapi di tabel.
                        if 'price_clean' in recommendations_display.columns:
                            recommendations_display['price_clean'] = recommendations_display['price_clean'].apply(lambda x: f"Rp {x:,.0f}".replace(",", "."))
                        if 'reviews_clean' in recommendations_display.columns:
                            recommendations_display['reviews_clean'] = recommendations_display['reviews_clean'].apply(lambda x: f"{x:,}".replace(",", "."))
                        if 'final_score' in recommendations_display.columns:
                            recommendations_display['final_score'] = recommendations_display['final_score'].round(3)

                        # Tampilkan DataFrame rekomendasi yang sudah diformat.
                        st.dataframe(recommendations_display[[
                            'product_name', 'brand', 'rating_clean', 'price_clean', 'reviews_clean', 'final_score'
                        ]].rename(columns={ # Ganti nama kolom agar lebih user-friendly.
                            'product_name': 'Nama Produk', 'brand': 'Brand', 'rating_clean': 'Rating',
                            'price_clean': 'Harga', 'reviews_clean': 'Jml Review', 'final_score': 'Skor Rekomendasi'
                        }))

                # Tampilkan tabel detail perhitungan skor jika ada rekomendasi.
                if not recommendations_df.empty:
                    st.markdown("---")
                    st.subheader("🔍 Detail Perhitungan Skor Rekomendasi (Top 10)")
                    # Deskripsi singkat tentang bagaimana skor dihitung.
                    st.markdown("""
            Skor Final dihitung dengan bobot **50% dari Skor Rating Ternormalisasi** dan **50% dari Skor Review Ternormalisasi**.
            - **Skor Rating**: Dihitung menggunakan **Normalisasi Min-Max** pada kolom rating (`rating_clean`).
            - **Skor Review**: Dihitung menggunakan **Normalisasi Logaritmik** pada kolom jumlah review (`reviews_clean`) dengan formula: `log(1 + jumlah_review) / log(1 + 2000)`.
            """)
                    # Siapkan data untuk tabel detail skor.
                    detail_data = []
                    # Loop melalui 10 rekomendasi.
                    for index, row in recommendations_df.iterrows():
                        detail_data.append({ # Tambahkan data setiap produk ke list.
                            "Peringkat": len(detail_data) + 1,
                            "Nama Produk": row.get('product_name', 'N/A'),
                            "Skor Rating (Normalisasi)": f"{row.get('rating_score', 0.0):.2f}", # Format 2 desimal.
                            "Skor Review (Normalisasi Log)": f"{row.get('review_score', 0.0):.2f}", # Format 2 desimal.
                            "Skor Final": f"{row.get('final_score', 0.0):.3f}" # Format 3 desimal.
                        })
                    # Tampilkan list data sebagai tabel.
                    st.table(pd.DataFrame(detail_data).set_index("Peringkat"))

    # --- Konten untuk Tab 2: Daftar Produk ---
    with tab2:
        st.header("Daftar Lengkap Produk Skincare")
        st.markdown("Berikut adalah seluruh data produk yang telah melalui tahap preprocessing (harga > 0).")
        # Dapatkan daftar kategori unik dari data final.
        available_categories = sorted(df_final['category_classified'].unique())

        # Cek jika ada kategori yang tersedia.
        if not available_categories:
            st.warning("Tidak ada kategori produk yang tersedia.")
            selected_category_filter = [] # Jika tidak ada, filter kosong.
        else:
            # Membuat filter multiselect untuk kategori.
            selected_category_filter = st.multiselect(
                "Filter berdasarkan kategori:",
                options=available_categories, # Pilihan berasal dari kategori unik.
                default=available_categories # Defaultnya, semua kategori terpilih.
            )

        # Filter DataFrame utama berdasarkan kategori yang dipilih pengguna.
        if selected_category_filter:
            filtered_df = df_final[df_final['category_classified'].isin(selected_category_filter)]
        else:
            # Jika tidak ada filter dipilih (atau tidak ada kategori), tampilkan semua.
            filtered_df = df_final

        # Cek jika hasil filter kosong.
        if filtered_df.empty:
            st.info("Tidak ada produk yang cocok dengan filter yang dipilih.")
        else:
            # Buat salinan DataFrame hasil filter untuk diformat tampilannya.
            filtered_display = filtered_df.copy()
            # Format kolom harga dan review.
            if 'price_clean' in filtered_display.columns:
                filtered_display['price_clean'] = filtered_display['price_clean'].apply(lambda x: f"Rp {x:,.0f}".replace(",", "."))
            if 'reviews_clean' in filtered_display.columns:
                filtered_display['reviews_clean'] = filtered_display['reviews_clean'].apply(lambda x: f"{x:,}".replace(",", "."))

            # Tampilkan DataFrame yang sudah difilter dan diformat.
            st.dataframe(filtered_display[[
                'product_name', 'brand', 'category_classified', 'price_clean', 'rating_clean', 'reviews_clean',
                'repurchase_yes', 'repurchase_no', 'repurchase_maybe'
            ]].rename(columns={ # Ganti nama kolom.
                'product_name': 'Nama Produk', 'brand': 'Brand', 'category_classified': 'Kategori',
                'price_clean': 'Harga', 'rating_clean': 'Rating', 'reviews_clean': 'Jml Review',
                'repurchase_yes': 'Beli Lagi (Ya)', 'repurchase_no': 'Beli Lagi (Tidak)', 'repurchase_maybe': 'Beli Lagi (Mungkin)'
            }))

    # --- Konten untuk Tab 3: Analisis Data ---
    with tab3:
        st.header("Analisis Data Produk Skincare")

        # Cek jika data final kosong.
        if df_final.empty:
            st.warning("Tidak ada data produk untuk dianalisis.")
        else:
            # Analisis 1: Top 10 Brand Terpopuler.
            st.subheader("Top 10 Brand Terpopuler")
            st.markdown("Berdasarkan skor final gabungan (rating dan review).")
            # Cek apakah kolom 'brand' ada dan tidak kosong semua.
            if 'brand' in df_final.columns and not df_final['brand'].isnull().all():
                 # Hapus baris dengan brand kosong sebelum mengelompokkan.
                 df_brand_analysis = df_final.dropna(subset=['brand'])
                 if not df_brand_analysis.empty:
                      # Kelompokkan berdasarkan brand, hitung rata-rata final_score, urutkan, ambil 10 teratas.
                      brand_popularity = df_brand_analysis.groupby('brand')['final_score'].mean().sort_values(ascending=False).head(10)
                      if not brand_popularity.empty:
                           # Buat grafik batang menggunakan Plotly Express.
                           fig_brand = px.bar(
                                brand_popularity,
                                x=brand_popularity.index, # Sumbu X: Nama Brand.
                                y='final_score', # Sumbu Y: Rata-rata Skor Final.
                                labels={'x': 'Brand', 'final_score': 'Rata-rata Skor Final'},
                                title="Peringkat Brand Berdasarkan Skor",
                                color=brand_popularity.index, # Beri warna berbeda untuk setiap brand.
                                text_auto='.2f' # Tampilkan nilai di atas batang (format 2 desimal).
                           )
                           fig_brand.update_layout(yaxis_title="Rata-rata Skor Final") # Perjelas label sumbu Y.
                           st.plotly_chart(fig_brand, use_container_width=True) # Tampilkan grafik.
                      else:
                           st.info("Tidak cukup data brand untuk menampilkan grafik popularitas.")
                 else:
                      st.info("Tidak ada data brand yang valid setelah menghapus nilai kosong.")
            else:
                 st.info("Kolom 'brand' tidak ditemukan atau kosong.")

            st.markdown("---") # Garis pemisah.

            # Analisis 2: Top 10 Produk Rating Tertinggi.
            st.subheader("Top 10 Produk dengan Rating Tertinggi")
            # Ambil 10 produk dengan 'rating_clean' tertinggi.
            top_10_rating = df_final.nlargest(10, 'rating_clean')
            if not top_10_rating.empty:
                 # Buat grafik batang.
                 fig_rating = px.bar(
                      top_10_rating,
                      x='product_name', # Sumbu X: Nama Produk.
                      y='rating_clean', # Sumbu Y: Rating.
                      labels={'product_name': 'Nama Produk', 'rating_clean': 'Rating'},
                      title="Peringkat Produk Berdasarkan Rating",
                      color='rating_clean', # Warna batang berdasarkan nilai rating.
                      color_continuous_scale='Viridis', # Skala warna.
                      text_auto='.1f' # Tampilkan nilai rating di atas batang (format 1 desimal).
                 )
                 # Urutkan sumbu X berdasarkan nilai Y (rating) secara menurun.
                 fig_rating.update_layout(xaxis_title="Nama Produk", yaxis_title="Rating", xaxis={'categoryorder':'total descending'})
                 st.plotly_chart(fig_rating, use_container_width=True) # Tampilkan grafik.
            else:
                 st.info("Tidak cukup data untuk menampilkan top 10 produk berdasarkan rating.")

            st.markdown("---") # Garis pemisah.

            # Analisis 3: Pembelian Ulang per Kategori.
            st.subheader("Analisis Pembelian Ulang per Kategori Produk")
            st.markdown("Persentase keputusan konsumen untuk membeli kembali produk dalam setiap kategori.")
            # Cek apakah kolom kategori ada.
            if 'category_classified' in df_final.columns and not df_final['category_classified'].isnull().all():
                 # Cek apakah kolom repurchase ada.
                 rep_cols = ['repurchase_yes', 'repurchase_no', 'repurchase_maybe']
                 if all(col in df_final.columns for col in rep_cols):
                      # Kelompokkan berdasarkan kategori, jumlahkan nilai repurchase.
                      rep_df = df_final.groupby('category_classified')[rep_cols].sum().reset_index()
                      # Cek apakah ada data dan total repurchase > 0.
                      if not rep_df.empty and rep_df[rep_cols].sum().sum() > 0:
                           # Hitung persentase.
                           rep_perc = rep_df.set_index('category_classified').copy()
                           rep_perc['total'] = rep_perc.sum(axis=1) # Hitung total per kategori.
                           rep_perc = rep_perc[rep_perc['total'] > 0] # Hapus kategori dengan total 0.

                           if not rep_perc.empty:
                                # Hitung persentase untuk Yes, No, Maybe.
                                for col in rep_cols:
                                     rep_perc[col+'_pct'] = (rep_perc[col] / rep_perc['total']).fillna(0) * 100

                                # Buat grafik batang tumpuk (stacked bar) menggunakan Plotly Graph Objects.
                                fig_repurchase = go.Figure()
                                # Tambahkan trace (lapisan) untuk setiap status repurchase.
                                fig_repurchase.add_trace(go.Bar(x=rep_perc.index, y=rep_perc['repurchase_yes_pct'], name="Ya, Beli Lagi", marker_color="green", hovertemplate='%{x}<br>Ya: %{y:.1f}%'))
                                fig_repurchase.add_trace(go.Bar(x=rep_perc.index, y=rep_perc['repurchase_maybe_pct'], name="Mungkin", marker_color="gold", hovertemplate='%{x}<br>Mungkin: %{y:.1f}%'))
                                fig_repurchase.add_trace(go.Bar(x=rep_perc.index, y=rep_perc['repurchase_no_pct'], name="Tidak", marker_color="crimson", hovertemplate='%{x}<br>Tidak: %{y:.1f}%'))
                                # Atur layout grafik.
                                fig_repurchase.update_layout(barmode='stack', title='Distribusi Keputusan Pembelian Ulang', xaxis_title='Kategori Produk', yaxis_title='Persentase (%)')
                                st.plotly_chart(fig_repurchase, use_container_width=True) # Tampilkan grafik.
                           else:
                                st.info("Tidak ada data pembelian ulang yang valid untuk divisualisasikan setelah filter.")
                      else:
                           st.info("Tidak ada data pembelian ulang untuk dianalisis.")
                 else:
                      st.info("Kolom data pembelian ulang (repurchase_yes/no/maybe) tidak ditemukan.")
            else:
                 st.info("Kolom 'category_classified' tidak ditemukan atau kosong.")

    # --- Konten untuk Tab 4: Evaluasi Sistem ---
    with tab4:
        st.header("Evaluasi Kinerja Sistem Rekomendasi")
        # Penjelasan metrik evaluasi yang digunakan.
        st.markdown("""
        Evaluasi ini menggunakan metrik **Precision@10**, **Recall@10**, dan **F1-Score**.
        - **Ground Truth (Relevan)**: Didefinisikan sebagai produk dimana `Repurchase (Yes) > Repurchase (No)`.
        - **Precision@10**: Dari 10 item yang direkomendasikan, berapa persen yang relevan.
        - **Recall@10**: Dari semua item relevan di kategori ini, berapa persen yang berhasil direkomendasikan.
        - **F1-Score**: Rata-rata harmonik dari Precision dan Recall, untuk menyeimbangkan keduanya.
        """)
        st.markdown("---")

        # Cek apakah data final valid untuk evaluasi.
        if 'df_final' in locals() and not df_final.empty and 'category_classified' in df_final.columns:
             # Dapatkan daftar kategori unik untuk dropdown evaluasi.
             available_categories = sorted(df_final['category_classified'].unique())

             if not available_categories:
                  st.warning("Tidak ada kategori yang tersedia untuk evaluasi.")
             else:
                  # Dropdown untuk memilih kategori yang akan dievaluasi.
                  selected_category_eval = st.selectbox(
                       "Pilih Kategori untuk Evaluasi:",
                       options=available_categories,
                       # Coba set default ke 'Skincare', jika tidak ada, gunakan kategori pertama.
                       index=available_categories.index("Skincare") if "Skincare" in available_categories else 0
                  )

                  # Tombol untuk memulai proses evaluasi.
                  if st.button("Jalankan Evaluasi"):
                       # Panggil fungsi evaluasi.
                       avg_p, avg_r, avg_f1, df_eval_results = run_evaluation(df_final, selected_category_eval)

                       st.subheader(f"Hasil Evaluasi untuk Kategori: {selected_category_eval}")

                       # 1. Tampilkan Metrik Rata-rata dalam 3 kolom.
                       col1, col2, col3 = st.columns(3)
                       col1.metric(label="Rata-rata Precision@10", value=f"{avg_p:.2%}") # Format persentase 2 desimal.
                       col2.metric(label="Rata-rata Recall@10", value=f"{avg_r:.2%}")
                       col3.metric(label="Rata-rata F1-Score", value=f"{avg_f1:.2%}")

                       st.markdown("---")

                       # 2. Tampilkan Grafik Rangkuman Metrik Rata-rata.
                       st.subheader("Grafik Rangkuman Metrik")
                       metrics_data = { # Siapkan data untuk grafik.
                            'Metrik': ['Precision@10', 'Recall@10', 'F1-Score'],
                            'Nilai': [avg_p, avg_r, avg_f1]
                       }
                       df_metrics = pd.DataFrame(metrics_data)
                       # Hanya tampilkan grafik jika ada nilai > 0.
                       if df_metrics['Nilai'].sum() > 0:
                            # Buat grafik batang rangkuman.
                            fig_summary = px.bar(
                                 df_metrics,
                                 x='Metrik',
                                 y='Nilai',
                                 color='Metrik',
                                 title="Rangkuman Skor Rata-rata Evaluasi",
                                 text_auto='.2%' # Tampilkan nilai persentase di batang.
                            )
                            fig_summary.update_yaxes(range=[0, 1]) # Atur sumbu Y dari 0% - 100%.
                            st.plotly_chart(fig_summary, use_container_width=True)
                       else:
                            st.info("Semua skor rata-rata adalah 0, grafik rangkuman tidak ditampilkan.")

                       st.markdown("---")

                       # 3. Tampilkan Grafik Perbandingan Top Presisi vs Top Recall.
                       st.subheader("Analisis Performa Produk (Presisi vs Recall)")
                       st.markdown("Grafik ini menunjukkan perbandingan produk dengan skor Presisi tertinggi dan skor Recall tertinggi.")

                       # Cek apakah ada hasil evaluasi per produk.
                       if not df_eval_results.empty:
                            # Bagi layout jadi 2 kolom.
                            col_precision, col_recall = st.columns(2)

                            # Kolom Kiri: Top 10 Presisi.
                            with col_precision:
                                 st.markdown("##### Top 10 Produk dengan Presisi Tertinggi")
                                 # Urutkan DataFrame hasil evaluasi berdasarkan 'precision' menurun, ambil 10 teratas.
                                 df_best_p = df_eval_results.sort_values(by='precision', ascending=False).head(10)

                                 # Hanya tampilkan grafik jika ada data dan skor tertinggi > 0.
                                 if not df_best_p.empty and df_best_p['precision'].iloc[0] > 0:
                                      # Buat grafik batang horizontal.
                                      fig_best_p = px.bar(
                                           df_best_p,
                                           x='precision', # Sumbu X: Skor Presisi.
                                           y='product_name', # Sumbu Y: Nama Produk.
                                           orientation='h', # Orientasi horizontal.
                                           title='Top 10 Performa Presisi Terbaik',
                                           text='precision' # Tampilkan skor di batang.
                                      )
                                      # Atur layout grafik.
                                      fig_best_p.update_layout(
                                           xaxis_title="Precision@10",
                                           yaxis_title="Nama Produk",
                                           yaxis=dict(autorange="reversed"), # Urutkan dari atas ke bawah.
                                           xaxis=dict(range=[0, 1]) # Paksa sumbu X dari 0-1.
                                      )
                                      # Format teks di batang menjadi 2 desimal.
                                      fig_best_p.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                                      st.plotly_chart(fig_best_p, use_container_width=True)
                                 else: # Jika tidak ada produk dengan skor > 0.
                                      st.info("Tidak ada produk dengan skor presisi > 0 untuk ditampilkan.")

                            # Kolom Kanan: Top 10 Recall.
                            with col_recall:
                                 st.markdown("##### Top 10 Produk dengan Recall Tertinggi")
                                 # Urutkan DataFrame hasil evaluasi berdasarkan 'recall' menurun, ambil 10 teratas.
                                 df_best_r = df_eval_results.sort_values(by='recall', ascending=False).head(10)

                                 # Hanya tampilkan grafik jika ada data dan skor tertinggi > 0.
                                 if not df_best_r.empty and df_best_r['recall'].iloc[0] > 0:
                                      # Buat grafik batang horizontal.
                                      fig_best_r = px.bar(
                                           df_best_r,
                                           x='recall', # Sumbu X: Skor Recall.
                                           y='product_name', # Sumbu Y: Nama Produk.
                                           orientation='h',
                                           title='Top 10 Performa Recall Terbaik',
                                           text='recall',
                                           color_discrete_sequence=['red'] # Beri warna berbeda.
                                      )
                                      # Atur layout grafik.
                                      fig_best_r.update_layout(
                                           xaxis_title="Recall@10",
                                           yaxis_title="Nama Produk",
                                           yaxis=dict(autorange="reversed"),
                                           xaxis=dict(range=[0, 1])
                                      )
                                      # Format teks di batang menjadi 2 desimal.
                                      fig_best_r.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                                      st.plotly_chart(fig_best_r, use_container_width=True)
                                 else: # Jika tidak ada produk dengan skor > 0.
                                      st.info("Tidak ada produk dengan skor recall > 0 untuk ditampilkan.")
                       else: # Jika DataFrame hasil evaluasi kosong.
                            st.warning("Tidak ada hasil evaluasi per produk untuk ditampilkan.")
        else: # Jika data final tidak valid di awal.
             st.info("Tidak ada data produk yang valid untuk dievaluasi. Silakan periksa file dataset Anda.")

# --- PERUBAHAN BERAKHIR DI SINI ---
else:
    # 4. Tampilkan pesan jika belum ada file yang diunggah
    st.info("Silakan unggah file CSV untuk memulai.")