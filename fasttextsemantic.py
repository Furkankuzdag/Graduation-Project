import streamlit as st
import PyPDF2
import numpy as np
import re
import google.generativeai as genai
import os

from gensim.models import KeyedVectors, FastText
from gensim.models.fasttext import load_facebook_vectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# --- AYARLAR ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "fasttext_enwiki_20260401_cbow_epoch1.bin")
PDF_PATH   = os.path.join(BASE_DIR, "Mohamed Elgendy - Deep Learning for Vision Systems MEAP V06 (2019, Manning Publications Co.) - libgen.li.pdf")
LOG_PATH   = os.path.join(BASE_DIR, "rag_log.txt")

vector_size = 200
top_k = 5

# YENİ EKLENEN SEMANTIC CHUNKING AYARLARI
semantic_threshold = 0.55  # Peş peşe gelen iki cümle arasındaki benzerlik bunun altına düşerse metni kes
max_sentences = 9  # Bir chunk çok uzamasın diye konulacak maksimum sınır


# --- YARDIMCI FONKSİYONLAR ---
def log_yaz(mesaj):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{mesaj}\n")


def log_temizle():
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("")


@st.cache_resource
def model_yukle(model_path):
    lower_path = model_path.lower()
    if lower_path.endswith(".bin"):
        return load_facebook_vectors(model_path)
    if lower_path.endswith(".vec") or lower_path.endswith(".txt"):
        return KeyedVectors.load_word2vec_format(model_path)
    try:
        return FastText.load(model_path).wv
    except Exception:
        return KeyedVectors.load(model_path)


def metni_temizle(metin):
    metin = metin.lower()
    metin = re.sub(r"[^\w\s]", " ", metin)
    metin = re.sub(r"\s+", " ", metin).strip()
    return metin


def kelime_filtrele(kelime):
    if kelime in ENGLISH_STOP_WORDS:
        return False
    if len(kelime) <= 2:
        return False
    if kelime.isdigit():
        return False
    return True


def idf_hesapla(metin_listesi):
    dokuman_sayisi = len(metin_listesi)
    df = {}

    for metin in metin_listesi:
        kelimeler = set(k for k in metni_temizle(metin).split() if kelime_filtrele(k))
        for k in kelimeler:
            df[k] = df.get(k, 0) + 1

    idf = {
        k: np.log((dokuman_sayisi + 1) / (v + 1)) + 1
        for k, v in df.items()
    }
    return idf


def metin_vektorle(metin, model, idf=None):
    temiz = metni_temizle(metin)
    kelimeler = temiz.split()
    vektorler = []
    agirliklar = []

    for k in kelimeler:
        if not kelime_filtrele(k):
            continue
        try:
            vektor = model[k]
        except KeyError:
            continue

        agirlik = idf.get(k, 1.0) if idf else 1.0
        vektorler.append(vektor * agirlik)
        agirliklar.append(agirlik)

    if not vektorler:
        return np.zeros(vector_size)

    sonuc = np.sum(vektorler, axis=0) / np.sum(agirliklar)
    norm = np.linalg.norm(sonuc)
    if norm > 0:
        sonuc = sonuc / norm

    return sonuc


# --- PDF & SEMANTIC CHUNK ---
def pdf_okuma(pdf_yolu):
    sayfalar = []
    with open(pdf_yolu, "rb") as f:
        pdf = PyPDF2.PdfReader(f)
        for i in range(19 , len(pdf.pages)):
            metin = pdf.pages[i].extract_text()
            if metin:
                sayfalar.append(metin.replace("\n", " "))
    return sayfalar


def cumlelere_ayir(sayfalar):
    tum = re.sub(r"\s+", " ", " ".join(sayfalar)).strip()
    # Cümleleri ayır
    return [c for c in re.split(r"(?<=[.!?]) +", tum) if len(c) > 20]


def semantic_chunk_olustur(cumleler, model, idf):
    """
    Cümleleri alır, aralarındaki anlamsal benzerliğe bakarak gruplar.
    Konu değiştiğinde (benzerlik eşiğin altına düştüğünde) metni keser.
    """
    chunks = []
    guncel_chunk = []

    # 1. Her cümlenin anlamsal vektörünü önceden hesapla
    cumle_vektorleri = []
    for c in cumleler:
        cumle_vektorleri.append(metin_vektorle(c, model, idf))

    # İlk cümleyi bloğa ekleyerek başla
    guncel_chunk.append(cumleler[0])

    for i in range(1, len(cumleler)):
        v1 = cumle_vektorleri[i - 1]
        v2 = cumle_vektorleri[i]

        # Eğer cümleler boşsa veya kelime filtrelendiyse (vektör 0 ise)
        if np.all(v1 == 0) or np.all(v2 == 0):
            sim = 0.0
        else:
            # İki cümle arasındaki kosinüs benzerliğini hesapla
            sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]

        # Eğer konu aynıysa (eşikten yüksekse) ve max sınırı aşmadıysa aynı chunk'a ekle
        if sim >= semantic_threshold and len(guncel_chunk) < max_sentences:
            guncel_chunk.append(cumleler[i])
        else:
            # Konu değişti! Mevcut bloğu birleştirip kaydet, yeni bloğa geç.
            chunks.append(" ".join(guncel_chunk))
            guncel_chunk = [cumleler[i]]

    # Son kalan bloğu eklemeyi unutma
    if guncel_chunk:
        chunks.append(" ".join(guncel_chunk))

    # Kalite Filtresi: Çok anlamsız/kısa veya sırf rakamdan oluşan blokları temizle
    filtrelenmis_chunks = []
    for chunk in chunks:
        kelime_sayisi = len(chunk.split())
        if kelime_sayisi < 25 or kelime_sayisi > 300:
            continue

        rakamlar = [k for k in chunk.split() if k.isdigit()]
        if kelime_sayisi > 0 and len(rakamlar) / kelime_sayisi > 0.15:
            continue

        filtrelenmis_chunks.append(chunk)

    return filtrelenmis_chunks


# --- LOG (INDEX OLUŞTURMA) ---
def log_olustur():
    st.info("PDF okunuyor...")
    sayfalar = pdf_okuma(PDF_PATH)

    st.info("Cümlelere ayrılıyor...")
    cumleler = cumlelere_ayir(sayfalar)

    st.info("FastText modeli yükleniyor...")
    model = model_yukle(MODEL_PATH)

    if model.vector_size != vector_size:
        st.error(f"Model vector size {model.vector_size}, fakat kodda vector_size={vector_size}.")
        st.stop()

    st.info("Cümleler arası karşılaştırma için IDF hesaplanıyor...")
    # Semantic chunking yapabilmek için önce cümle bazında idf çıkarıyoruz
    idf_cumleler = idf_hesapla(cumleler)

    st.info("Anlamsal (Semantic) Chunk'lar oluşturuluyor...")
    chunks = semantic_chunk_olustur(cumleler, model, idf_cumleler)

    st.info("Oluşturulan Chunk'lar için son IDF ağırlıkları hesaplanıyor...")
    # Gruplanmış metinler için ana idf haritamızı oluşturuyoruz
    idf_chunks = idf_hesapla(chunks)

    st.info("Chunk vektörleri log'a yazılıyor...")
    log_temizle()
    progress = st.progress(0)

    for i, chunk in enumerate(chunks):
        vektor = metin_vektorle(chunk, model, idf_chunks)
        vektor_str = " ".join([f"{v:.6f}" for v in vektor])

        log_yaz(f"CHUNK #{i + 1}")
        log_yaz("=" * 80)
        log_yaz("METIN:")
        log_yaz(chunk)
        log_yaz("")
        log_yaz("VEKTOR:")
        log_yaz(vektor_str)
        log_yaz("")
        log_yaz("-" * 80)
        log_yaz("")

        progress.progress((i + 1) / len(chunks))

    return len(chunks)


@st.cache_data
def log_oku(log_path):
    chunks_data = []
    try:
        with open(log_path, "r", encoding="utf-8", newline="") as f:
            icerik = f.read()
    except FileNotFoundError:
        return []

    icerik = icerik.replace("\r\n", "\n").replace("\r", "\n")
    if not icerik.strip():
        return []

    bloklar = re.split(r"CHUNK #\d+\n=+\n", icerik)

    for blok in bloklar:
        blok = blok.strip()
        if not blok:
            continue
        try:
            metin_b = re.search(r"METIN:\n(.*?)\n\s*\nVEKTOR:", blok, re.DOTALL)
            if not metin_b:
                metin_b = re.search(r"METIN:\n(.*?)VEKTOR:", blok, re.DOTALL)
            vektor_b = re.search(r"VEKTOR:\n([\d.\- \n\r]+)", blok)

            if not metin_b or not vektor_b:
                continue

            metin = metin_b.group(1).strip()
            sayilar = re.findall(r"-?[\d]+\.[\d]+", vektor_b.group(1))
            vektor = np.array([float(v) for v in sayilar])

            if len(vektor) == vector_size:
                chunks_data.append((metin, vektor))
        except Exception:
            continue

    return chunks_data


# --- ARAMA ---
def kelime_ortusme_skoru(soru, chunk):
    soru_kelimeleri = {k for k in metni_temizle(soru).split() if kelime_filtrele(k)}
    chunk_kelimeleri = {k for k in metni_temizle(chunk).split() if kelime_filtrele(k)}

    if not soru_kelimeleri:
        return 0.0

    ortak = soru_kelimeleri & chunk_kelimeleri
    return len(ortak) / len(soru_kelimeleri)


def benzer_chunk_bul(soru, model, chunks_data, k=3):
    chunk_metinleri = [metin for metin, _ in chunks_data]
    idf = idf_hesapla(chunk_metinleri)
    soru_vektor = metin_vektorle(soru, model, idf)

    if np.all(soru_vektor == 0):
        return []

    chunk_vektorleri = np.array([v for _, v in chunks_data])

    kosinuslar = cosine_similarity(soru_vektor.reshape(1, -1), chunk_vektorleri)[0]
    sonuclar = []

    for i, (metin, _) in enumerate(chunks_data):
        overlap = kelime_ortusme_skoru(soru, metin)
        final_skor = (0.75 * float(kosinuslar[i]) + 0.25 * float(overlap))

        sonuclar.append({
            "metin": metin,
            "skor": final_skor,
            "cosine": float(kosinuslar[i]),
            "overlap": float(overlap),
            "indeks": int(i)
        })

    sonuclar = sorted(sonuclar, key=lambda x: x["skor"], reverse=True)
    return sonuclar[:k]


# --- GEMINI ---
GEMINI_MODELLER = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
]

SISTEM_PROMPTU = (
    "You are a helpful assistant that answers questions about a book. "
    "Use the provided context passages to answer the question. "
    "Reply in the same language as the question, Turkish or English. "
    "Be clear, concise, and base your answer only on the context. "
    "If the answer is not in the context, say so explicitly."
)


def gemini_test(api_key):
    genai.configure(api_key=api_key)
    for m in GEMINI_MODELLER:
        try:
            genai.GenerativeModel(model_name=m, system_instruction=SISTEM_PROMPTU).generate_content("Hi")
            return m
        except Exception:
            continue
    return None


def gemini_cevapla(soru, benzer_chunks, api_key, model_adi=None):
    genai.configure(api_key=api_key)
    baglam = "".join(
        f"[Context {i} - Final Score: {c['skor']:.3f}, Cosine: {c['cosine']:.3f}, Keyword: {c['overlap']:.3f}]\n"
        f"{c['metin']}\n\n"
        for i, c in enumerate(benzer_chunks, 1)
    )

    kullanici_mesaji = (
        f"Use the following context to answer the question:\n\n{baglam}\nQuestion: {soru}"
    )

    denemeler = [model_adi] if model_adi else GEMINI_MODELLER
    son_hata = None

    for m in denemeler:
        try:
            yanit = genai.GenerativeModel(model_name=m, system_instruction=SISTEM_PROMPTU).generate_content(
                kullanici_mesaji)
            return yanit.text, m
        except Exception as e:
            son_hata = e

    raise Exception(f"No model worked. Last error: {son_hata}")


def ortusme_hesapla(cevap_metni, chunk_metni):
    cevap_kelimeleri = {k for k in metni_temizle(cevap_metni).split() if kelime_filtrele(k)}
    chunk_kelimeleri = {k for k in metni_temizle(chunk_metni).split() if kelime_filtrele(k)}
    if not chunk_kelimeleri:
        return 0.0, set()
    ortak = cevap_kelimeleri & chunk_kelimeleri
    oran = len(ortak) / len(chunk_kelimeleri)
    return oran, ortak


# --- STREAMLIT ARAYÜZ ---
def main():
    st.set_page_config(page_title="PDF RAG - Semantic", page_icon="📄", layout="wide")
    st.title("PDF RAG - FastText (Semantic Chunking)")
    st.caption("Smart grouping by sentence context similarity")

    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("Google Gemini API Key", type="password", placeholder="AIza...")
        st.markdown("[Get Free Key](https://aistudio.google.com/app/apikey)")
        st.divider()

        st.write("FastText model:")
        st.code(MODEL_PATH, language="text")

        st.write(f"Vector size: `{vector_size}`")
        st.write(f"Semantic Threshold: `{semantic_threshold}`")
        st.write(f"Max Sentences / Chunk: `{max_sentences}`")

        if api_key:
            if st.button("Test Model"):
                with st.spinner("Testing Gemini models..."):
                    bulunan = gemini_test(api_key)
                if bulunan:
                    st.success(f"Working model: `{bulunan}`")
                    st.session_state["aktif_model"] = bulunan
                else:
                    st.error("No model worked. Try VPN or check your key.")
            st.success("API key entered.")
        else:
            st.warning("No API key.")

    tab1, tab2 = st.tabs(["Build Log", "Ask Question"])

    with tab1:
        st.write(f"**PDF:** `{PDF_PATH}`")
        st.write(f"**Log file:** `{LOG_PATH}`")
        if st.button("Build Log (Semantic Chunking)"):
            toplam = log_olustur()
            st.success(f"Done! {toplam} anlamsal (semantic) chunk processed.")
            log_oku.clear()

    with tab2:
        st.subheader("Ask a Question")
        col_r, col_d = st.columns(2)
        with col_r:
            if st.button("Reload Log"):
                log_oku.clear()
                st.rerun()
        with col_d:
            debug = st.checkbox("Debug mode")

        if debug:
            import os
            if os.path.exists(LOG_PATH):
                boyut = os.path.getsize(LOG_PATH)
                st.write(f"Log found - {boyut:,} bytes")
                with open(LOG_PATH, "r", encoding="utf-8", newline="") as f:
                    st.code(repr("".join(f.readlines()[:30])), language="text")
            else:
                st.error(f"Not found: `{LOG_PATH}`")

        chunks_data = log_oku(LOG_PATH)

        if not chunks_data:
            import os
            if not os.path.exists(LOG_PATH):
                st.error(f"Log not found: `{LOG_PATH}`")
            else:
                st.warning("Log exists but no chunks parsed. Rebuild log.")
            return

        st.success(f"{len(chunks_data)} semantic chunks loaded.")

        col1, col2 = st.columns([3, 1])
        with col2:
            k_deger = st.slider("Chunks to retrieve", 1, 5, 3)
        with col1:
            soru = st.text_input("Your question:", placeholder="e.g. What is the role of max pooling?")

        ara_tiklandi = st.button("Search & Answer", type="primary", key="ara_butonu")

        if ara_tiklandi and soru.strip():
            with st.spinner("Loading FastText model..."):
                model = model_yukle(MODEL_PATH)

            if model.vector_size != vector_size:
                st.error(f"Model vector size mismatch.")
                st.stop()

            with st.spinner("Finding similar chunks..."):
                benzer_chunks = benzer_chunk_bul(soru, model, chunks_data, k=k_deger)

            if not benzer_chunks:
                st.error("Could not vectorize question.")
                return

            st.markdown("---")
            st.subheader("Most Similar Semantic Chunks")

            for i, chunk in enumerate(benzer_chunks, 1):
                with st.expander(f"Chunk #{chunk['indeks'] + 1} - Final: {chunk['skor']:.4f}", expanded=(i == 1)):
                    st.write(chunk["metin"])
                    st.progress(
                        min(max(chunk["skor"], 0.0), 1.0),
                        text=f"Final score: {chunk['skor']:.4f} | Cosine: {chunk['cosine']:.4f}"
                    )

            st.markdown("---")
            st.subheader("Gemini Answer")

            with st.spinner("Generating answer..."):
                try:
                    if not api_key:
                        st.error("Enter API key.")
                        st.stop()

                    cevap, kullanilan_model = gemini_cevapla(soru, benzer_chunks, api_key,
                                                             st.session_state.get("aktif_model"))
                    st.caption(f"Model: `{kullanilan_model}`")
                    st.success(cevap)

                except Exception as e:
                    st.error(f"Gemini API error: {e}")
                    st.stop()

        elif ara_tiklandi:
            st.warning("Please enter a question.")


if __name__ == "__main__":
    main()