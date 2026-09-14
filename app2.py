import io
import os
import re
import zipfile
import google.generativeai as genai
import numpy as np
import PyPDF2
import streamlit as st
from gensim.models import FastText, KeyedVectors, Word2Vec
from gensim.models.fasttext import load_facebook_vectors
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CONFIGS = {
    "word2vec": {
        "label": "Word2Vec",
        "vector_size": 200,
        "model_path": os.path.join(BASE_DIR, "wiki_word2vec.model"),
        "log_path": os.path.join(BASE_DIR, "rag_log_word2vec.txt"),
    },
    "fasttext": {
        "label": "FastText",
        "vector_size": 200,
        "model_path": os.path.join(BASE_DIR, "fasttext_enwiki_20260401_cbow_epoch1.bin"),
        "log_path": os.path.join(BASE_DIR, "rag_log_fasttext.txt"),
    },
    "glove": {
        "label": "GloVe",
        "vector_size": 300,
        "model_path": os.path.join(BASE_DIR, "glove.2024.dolma.300d.zip"),
        "log_path": os.path.join(BASE_DIR, "rag_log_glove.txt"),
    },
}
SLIDING_DEFAULTS = {
    "word2vec": (5, 2),
    "fasttext": (6, 3),
    "glove": (5, 2),
}

top_k = 5

semantic_threshold_default = 0.55
max_sentences_default = 9
def log_yaz(log_path, mesaj):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{mesaj}\n")


def log_temizle(log_path):
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("")


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

    return {
        k: np.log((dokuman_sayisi + 1) / (v + 1)) + 1
        for k, v in df.items()
    }


def metin_vektorle(metin, model, vector_size, idf=None):
    temiz = metni_temizle(metin)
    kelimeler = temiz.split()
    vektorler = []
    agirliklar = []

    for k in kelimeler:
        if not kelime_filtrele(k) or k not in model:
            continue

        agirlik = idf.get(k, 1.0) if idf else 1.0
        vektorler.append(model[k] * agirlik)
        agirliklar.append(agirlik)

    if not vektorler:
        return np.zeros(vector_size, dtype=np.float32)

    sonuc = np.sum(vektorler, axis=0) / np.sum(agirliklar)
    norm = np.linalg.norm(sonuc)
    if norm > 0:
        sonuc = sonuc / norm

    return sonuc
def _glove_text_stream(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        txt_files = [name for name in zf.namelist() if name.lower().endswith(".txt")]
        if not txt_files:
            raise ValueError("No .txt GloVe vector file was found in the ZIP archive.")

        with zf.open(txt_files[0], "r") as raw:
            yield from io.TextIOWrapper(raw, encoding="utf-8", errors="ignore")


@st.cache_resource(show_spinner=False)
def glove_yukle(zip_path, vector_size, max_vectors=None):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"GloVe zip bulunamadı: {zip_path}")

    kv = KeyedVectors(vector_size=vector_size)
    words, vectors = [], []
    batch_size = 50_000
    loaded = 0

    for line in _glove_text_stream(zip_path):
        parts = line.rstrip().split(" ")
        if len(parts) != vector_size + 1:
            continue

        word = parts[0]
        vector = np.fromstring(" ".join(parts[1:]), sep=" ", dtype=np.float32)
        if vector.shape[0] != vector_size:
            continue

        words.append(word)
        vectors.append(vector)
        loaded += 1

        if len(words) >= batch_size:
            kv.add_vectors(words, np.vstack(vectors))
            words, vectors = [], []

        if max_vectors is not None and loaded >= max_vectors:
            break

    if words:
        kv.add_vectors(words, np.vstack(vectors))

    if len(kv) == 0:
        raise ValueError("The GloVe model could not be loaded; the vector file is not in the expected format.")

    kv.fill_norms()
    return kv


@st.cache_resource(show_spinner=False)
def word2vec_yukle(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Word2Vec model not found: {model_path}")
    return Word2Vec.load(model_path).wv


@st.cache_resource(show_spinner=False)
def fasttext_yukle(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"FastText model not found: {model_path}")

    lower_path = model_path.lower()

    if lower_path.endswith(".bin"):
        return load_facebook_vectors(model_path)

    if lower_path.endswith(".vec") or lower_path.endswith(".txt"):
        return KeyedVectors.load_word2vec_format(model_path)

    try:
        return FastText.load(model_path).wv
    except Exception:
        return KeyedVectors.load(model_path)


def model_yukle(secim, max_glove_vectors=None):
    cfg = MODEL_CONFIGS[secim]
    if secim == "word2vec":
        return word2vec_yukle(cfg["model_path"])
    if secim == "fasttext":
        return fasttext_yukle(cfg["model_path"])
    if secim == "glove":
        return glove_yukle(cfg["model_path"], cfg["vector_size"], max_glove_vectors)
    raise ValueError(f"Unknown embedding model: {secim}")
def pdf_okuma(pdf_kaynagi, baslangic_sayfa=19):
    sayfalar = []
    pdf = PyPDF2.PdfReader(pdf_kaynagi)
    for i in range(min(baslangic_sayfa, len(pdf.pages)), len(pdf.pages)):
        metin = pdf.pages[i].extract_text()
        if metin:
            sayfalar.append(metin.replace("\n", " "))
    return sayfalar


def cumlelere_ayir(sayfalar):
    tum = re.sub(r"\s+", " ", " ".join(sayfalar)).strip()
    return [c for c in re.split(r"(?<=[.!?]) +", tum) if len(c) > 20]


def _chunk_filtrele(chunks):
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


def semantic_chunk_olustur(cumleler, model, vector_size, idf, threshold, max_sentences):
    if not cumleler:
        return []

    chunks = []
    guncel_chunk = [cumleler[0]]
    cumle_vektorleri = [metin_vektorle(c, model, vector_size, idf) for c in cumleler]

    for i in range(1, len(cumleler)):
        v1 = cumle_vektorleri[i - 1]
        v2 = cumle_vektorleri[i]

        if np.all(v1 == 0) or np.all(v2 == 0):
            sim = 0.0
        else:
            sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]

        if sim >= threshold and len(guncel_chunk) < max_sentences:
            guncel_chunk.append(cumleler[i])
        else:
            chunks.append(" ".join(guncel_chunk))
            guncel_chunk = [cumleler[i]]

    if guncel_chunk:
        chunks.append(" ".join(guncel_chunk))

    return _chunk_filtrele(chunks)


def sliding_window_chunk_olustur(cumleler, pencere_boyutu=5, ortusme=2):
    if not cumleler:
        return []

    pencere_boyutu = max(1, int(pencere_boyutu))
    ortusme = max(0, int(ortusme))
    if ortusme >= pencere_boyutu:
        ortusme = pencere_boyutu - 1

    adim = pencere_boyutu - ortusme
    chunks = []
    i = 0

    while i < len(cumleler):
        parca = cumleler[i:i + pencere_boyutu]
        if parca:
            chunks.append(" ".join(parca))

        if i + pencere_boyutu >= len(cumleler):
            break

        i += adim

    return _chunk_filtrele(chunks)


CHUNK_TURLERI = {
    "semantic": "Semantic Chunking",
    "sliding": "Sliding Window",
}
def log_olustur(secim, pdf_kaynagi, baslangic_sayfa, chunk_turu,
                 pencere_boyutu, ortusme, semantic_threshold, max_sentences,
                 max_glove_vectors=None):
    cfg = MODEL_CONFIGS[secim]
    log_path = cfg["log_path"]
    vektor_boyutu = cfg["vector_size"]

    st.info("Reading PDF...")
    sayfalar = pdf_okuma(pdf_kaynagi, baslangic_sayfa)

    if not sayfalar:
        st.error("No text could be extracted from the PDF. The file may be scanned or corrupted.")
        st.stop()

    st.info("Splitting into sentences...")
    cumleler = cumlelere_ayir(sayfalar)

    st.info(f"{cfg['label']} model is loading...")
    model = model_yukle(secim, max_glove_vectors)

    model_vector_size = getattr(model, "vector_size", vektor_boyutu)
    if model_vector_size != vektor_boyutu:
        st.error(
            f"Model vector size is {model_vector_size}, but "
            f"vector_size={vektor_boyutu} is configured for '{secim}' in MODEL_CONFIGS. Update the value."
        )
        st.stop()

    if chunk_turu == "semantic":
        st.info("Calculating IDF for sentence comparison...")
        idf_cumleler = idf_hesapla(cumleler)

        st.info("Creating semantic chunks...")
        chunks = semantic_chunk_olustur(
            cumleler, model, vektor_boyutu, idf_cumleler, semantic_threshold, max_sentences
        )
    else:
        st.info(f"Creating sliding-window chunks... (window={pencere_boyutu}, overlap={ortusme})")
        chunks = sliding_window_chunk_olustur(cumleler, pencere_boyutu, ortusme)

    if not chunks:
        st.error("No chunks were created. Check the threshold or window-size parameters.")
        st.stop()

    st.info("Calculating final IDF weights for the generated chunks...")
    idf_chunks = idf_hesapla(chunks)

    st.info("Writing chunk vectors to the log...")
    log_temizle(log_path)
    progress = st.progress(0)

    for i, chunk in enumerate(chunks):
        vektor = metin_vektorle(chunk, model, vektor_boyutu, idf_chunks)
        vektor_str = " ".join([f"{v:.6f}" for v in vektor])

        log_yaz(log_path, f"CHUNK #{i + 1}")
        log_yaz(log_path, "=" * 80)
        log_yaz(log_path, "METIN:")
        log_yaz(log_path, chunk)
        log_yaz(log_path, "")
        log_yaz(log_path, "VEKTOR:")
        log_yaz(log_path, vektor_str)
        log_yaz(log_path, "")
        log_yaz(log_path, "-" * 80)
        log_yaz(log_path, "")

        progress.progress((i + 1) / len(chunks))

    kelime_sayisi = len(model) if hasattr(model, "__len__") else None
    return len(chunks), kelime_sayisi


@st.cache_data
def log_oku(log_path, vector_size):
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
            vektor = np.array([float(v) for v in sayilar], dtype=np.float32)

            if len(vektor) == vector_size:
                chunks_data.append((metin, vektor))
        except Exception:
            continue

    return chunks_data
def kelime_ortusme_skoru(soru, chunk):
    soru_kelimeleri = {k for k in metni_temizle(soru).split() if kelime_filtrele(k)}
    chunk_kelimeleri = {k for k in metni_temizle(chunk).split() if kelime_filtrele(k)}

    if not soru_kelimeleri:
        return 0.0

    ortak = soru_kelimeleri & chunk_kelimeleri
    return len(ortak) / len(soru_kelimeleri)


def benzer_chunk_bul(soru, model, vector_size, chunks_data, k=3):
    chunk_metinleri = [metin for metin, _ in chunks_data]
    idf = idf_hesapla(chunk_metinleri)
    soru_vektor = metin_vektorle(soru, model, vector_size, idf)

    if np.all(soru_vektor == 0):
        return []

    chunk_vektorleri = np.array([v for _, v in chunks_data])
    kosinuslar = cosine_similarity(soru_vektor.reshape(1, -1), chunk_vektorleri)[0]
    sonuclar = []

    for i, (metin, _) in enumerate(chunks_data):
        overlap = kelime_ortusme_skoru(soru, metin)
        final_skor = 0.75 * float(kosinuslar[i]) + 0.25 * float(overlap)

        sonuclar.append({
            "metin": metin,
            "skor": final_skor,
            "cosine": float(kosinuslar[i]),
            "overlap": float(overlap),
            "indeks": int(i),
        })

    sonuclar = sorted(sonuclar, key=lambda x: x["skor"], reverse=True)
    return sonuclar[:k]
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
                kullanici_mesaji
            )
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
def main():
    st.set_page_config(page_title="PDF RAG", layout="wide")
    st.title("PDF RAG - Word2Vec / FastText / GloVe")
    st.caption("Choose your embedding model, chunking strategy, and Gemini model")

    with st.sidebar:
        st.header("Settings")

        st.write("**Embedding Model**")
        embedding_secim = st.selectbox(
            "Select an embedding model",
            options=list(MODEL_CONFIGS.keys()),
            format_func=lambda k: MODEL_CONFIGS[k]["label"],
            key="embedding_secim",
            help="Her modelin kendi ayrı log dosyası vardır; model değiştirdiğinizde "
                 "o modele ait log'u görürsünüz (yeniden Build Log gerekebilir).",
        )
        cfg = MODEL_CONFIGS[embedding_secim]

        st.write("Model file:")
        st.code(cfg["model_path"], language="text")
        st.write(f"Vector size: `{cfg['vector_size']}`")
        st.write(f"Log file: `{cfg['log_path']}`")

        max_glove_vectors = None
        if embedding_secim == "glove":
            max_glove_vectors_input = st.number_input(
                "Max GloVe vectors (0 = full model)",
                min_value=0,
                value=0,
                step=50_000,
                help="If memory is limited, try a value such as 200000 or 300000.",
            )
            max_glove_vectors = None if max_glove_vectors_input == 0 else int(max_glove_vectors_input)

        st.divider()
        api_key = st.text_input("Google Gemini API Key", type="default", placeholder="AIza...")
        st.markdown("[Get Free Key](https://aistudio.google.com/app/apikey)")

        st.divider()
        st.write("**Gemini Model**")
        model_secim_secenekleri = ["Automatic"] + GEMINI_MODELLER
        secilen_model_label = st.selectbox(
            "Gemini model",
            options=model_secim_secenekleri,
            index=0,
            help="When Automatic is selected, the available models are tried in order and the first working model is used.",
        )
        secili_gemini_model = None if secilen_model_label == "Automatic" else secilen_model_label
        st.session_state["secili_gemini_model"] = secili_gemini_model

        if api_key:
            if st.button("Test Model"):
                with st.spinner("Testing Gemini models..."):
                    bulunan = gemini_test(api_key)
                if bulunan:
                    st.success(f"Working model: `{bulunan}`")
                else:
                    st.error("No model worked. Try VPN or check your key.")
            st.success("API key entered.")
        else:
            st.warning("No API key.")

    tab1, tab2 = st.tabs(["Build Log", "Ask Question"])

    with tab1:
        st.write(f"**Active embedding model:** {cfg['label']}")

        st.write("**Select PDF File**")
        yuklenen_pdf = st.file_uploader(
            "Upload a PDF file",
            type=["pdf"],
            accept_multiple_files=False,
            key="pdf_uploader",
        )

        baslangic_sayfa = st.number_input(
            "Start page (0 = beginning)",
            min_value=0,
            value=19,
            step=1,
            help="Use this to skip pages such as the cover or table of contents.",
        )

        st.markdown("---")
        st.write("**Select Chunking Method**")

        chunk_turu_secim = st.radio(
            "Chunking method",
            options=list(CHUNK_TURLERI.keys()),
            format_func=lambda k: CHUNK_TURLERI[k],
            horizontal=True,
            key="chunk_turu_secim",
        )

        varsayilan_pencere, varsayilan_ortusme = SLIDING_DEFAULTS[embedding_secim]

        if chunk_turu_secim == "semantic":
            col_t, col_m = st.columns(2)
            with col_t:
                semantic_threshold_ui = st.slider(
                    "Semantic Threshold", min_value=0.0, max_value=1.0,
                    value=semantic_threshold_default, step=0.01,
                )
            with col_m:
                max_sentences_ui = st.number_input(
                    "Max Sentences / Chunk", min_value=1, value=max_sentences_default, step=1,
                )
            pencere_boyutu, ortusme = varsayilan_pencere, varsayilan_ortusme
        else:
            semantic_threshold_ui, max_sentences_ui = semantic_threshold_default, max_sentences_default
            col_w, col_o = st.columns(2)
            with col_w:
                pencere_boyutu = st.number_input(
                    "Window Size (sentences)", min_value=1,
                    value=varsayilan_pencere, step=1,
                )
            with col_o:
                ortusme = st.number_input(
                    "Overlap (sentences)", min_value=0,
                    value=varsayilan_ortusme, step=1,
                    help="Number of sentences shared between consecutive chunks. Must be smaller than the window size.",
                )

        if yuklenen_pdf is not None:
            st.success(f"Selected file: `{yuklenen_pdf.name}` ({yuklenen_pdf.size:,} bytes)")
        else:
            st.info("Select a PDF file to continue.")

        build_disabled = yuklenen_pdf is None
        if st.button(f"Build Log ({cfg['label']} · {CHUNK_TURLERI[chunk_turu_secim]})", disabled=build_disabled):
            yuklenen_pdf.seek(0)
            pdf_bytes = io.BytesIO(yuklenen_pdf.read())

            toplam, model_kelime_sayisi = log_olustur(
                embedding_secim, pdf_bytes, baslangic_sayfa,
                chunk_turu=chunk_turu_secim,
                pencere_boyutu=pencere_boyutu,
                ortusme=ortusme,
                semantic_threshold=semantic_threshold_ui,
                max_sentences=max_sentences_ui,
                max_glove_vectors=max_glove_vectors,
            )
            kelime_bilgisi = f"{model_kelime_sayisi:,}" if model_kelime_sayisi is not None else "?"
            st.success(
                f"Done! {toplam} chunk processed with {cfg['label']} · {CHUNK_TURLERI[chunk_turu_secim]}. "
                f"Model vocabulary loaded: {kelime_bilgisi}"
            )
            log_oku.clear()

    with tab2:
        st.subheader("Ask a Question")
        st.write(f"**Active embedding model:** {cfg['label']} (log: `{os.path.basename(cfg['log_path'])}`)")

        col_r, col_d = st.columns(2)
        with col_r:
            if st.button("Reload Log"):
                log_oku.clear()
                st.rerun()
        with col_d:
            debug = st.checkbox("Debug mode")

        secili_gemini_model_gosterim = st.session_state.get("secili_gemini_model")
        if secili_gemini_model_gosterim:
            st.caption(f"Selected Gemini model: `{secili_gemini_model_gosterim}`")
        else:
            st.caption("Selected Gemini model: Automatic")

        if debug:
            if os.path.exists(cfg["log_path"]):
                boyut = os.path.getsize(cfg["log_path"])
                st.write(f"Log found - {boyut:,} bytes")
                with open(cfg["log_path"], "r", encoding="utf-8", newline="") as f:
                    st.code(repr("".join(f.readlines()[:30])), language="text")
            else:
                st.error(f"Not found: `{cfg['log_path']}`")

        chunks_data = log_oku(cfg["log_path"], cfg["vector_size"])

        if not chunks_data:
            if not os.path.exists(cfg["log_path"]):
                st.error(
                    f"Log not found: `{cfg['log_path']}`. First use the 'Build Log' tab to "
                    f"{cfg['label']} upload a PDF and build a log."
                )
            else:
                st.warning("Log exists but no chunks parsed. Rebuild log.")
            return

        st.success(f"{len(chunks_data)} chunks loaded.")

        col1, col2 = st.columns([3, 1])
        with col2:
            k_deger = st.slider("Chunks to retrieve", 1, 5, 3)
        with col1:
            soru = st.text_input("Your question:", placeholder="e.g. What is the role of max pooling?")

        ara_tiklandi = st.button("Search & Answer", type="primary", key="ara_butonu")

        if ara_tiklandi and soru.strip():
            with st.spinner(f"Loading {cfg['label']} model..."):
                model = model_yukle(embedding_secim, max_glove_vectors)

            model_vector_size = getattr(model, "vector_size", cfg["vector_size"])
            if model_vector_size != cfg["vector_size"]:
                st.error("Model vector size mismatch.")
                st.stop()

            with st.spinner("Finding similar chunks..."):
                benzer_chunks = benzer_chunk_bul(soru, model, cfg["vector_size"], chunks_data, k=k_deger)

            if not benzer_chunks:
                st.error("Could not vectorize question. The question words may be outside the loaded vocabulary.")
                return

            st.markdown("---")
            st.subheader("Most Similar Chunks")

            for i, chunk in enumerate(benzer_chunks, 1):
                with st.expander(f"Chunk #{chunk['indeks'] + 1} - Final: {chunk['skor']:.4f}", expanded=(i == 1)):
                    st.write(chunk["metin"])
                    st.progress(
                        min(max(chunk["skor"], 0.0), 1.0),
                        text=f"Final score: {chunk['skor']:.4f} | Cosine: {chunk['cosine']:.4f}",
                    )

            st.markdown("---")
            st.subheader("Gemini Answer")

            with st.spinner("Generating answer..."):
                try:
                    if not api_key:
                        st.error("Enter API key.")
                        st.stop()

                    cevap, kullanilan_model = gemini_cevapla(
                        soru,
                        benzer_chunks,
                        api_key,
                        st.session_state.get("secili_gemini_model"),
                    )
                    st.caption(f"Model: `{kullanilan_model}`")
                    st.success(cevap)

                except Exception as e:
                    st.error(f"Gemini API error: {e}")
                    st.stop()

        elif ara_tiklandi:
            st.warning("Please enter a question.")


if __name__ == "__main__":
    main()