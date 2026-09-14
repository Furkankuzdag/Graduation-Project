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


# AYARLAR


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "fasttext_enwiki_20260401_cbow_epoch1.bin")
PDF_PATH   = os.path.join(BASE_DIR, "Mohamed Elgendy - Deep Learning for Vision Systems MEAP V06 (2019, Manning Publications Co.) - libgen.li.pdf")
LOG_PATH   = os.path.join(BASE_DIR, "rag_log.txt")

vector_size = 200
top_k       = 5
chunk_size  = 6
chunk_step  = 3


# YARDIMCI

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


def idf_hesapla(chunks):
    dokuman_sayisi = len(chunks)
    df = {}

    for chunk in chunks:
        kelimeler = set(
            k for k in metni_temizle(chunk).split()
            if kelime_filtrele(k)
        )

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


# PDF & CHUNK

def pdf_okuma(pdf_yolu):
    sayfalar = []

    with open(pdf_yolu, "rb") as f:
        pdf = PyPDF2.PdfReader(f)

        for i in range(15, len(pdf.pages)):
            metin = pdf.pages[i].extract_text()

            if metin:
                sayfalar.append(metin.replace("\n", " "))

    return sayfalar


def cumlelere_ayir(sayfalar):
    tum = re.sub(r"\s+", " ", " ".join(sayfalar)).strip()
    return [c for c in re.split(r"(?<=[.!?]) +", tum) if len(c) > 20]


def chunk_olustur(cumleler):
    chunks = []

    for i in range(0, len(cumleler), chunk_step):
        if i + chunk_size > len(cumleler):
            break

        chunk = " ".join(cumleler[i:i + chunk_size])
        kelime_sayisi = len(chunk.split())

        if kelime_sayisi < 25 or kelime_sayisi > 220:
            continue

        rakamlar = [k for k in chunk.split() if k.isdigit()]

        if kelime_sayisi > 0 and len(rakamlar) / kelime_sayisi > 0.15:
            continue

        chunks.append(chunk)

    return chunks


# LOG

def log_olustur():
    st.info("PDF okunuyor...")
    sayfalar = pdf_okuma(PDF_PATH)

    st.info("Cumlelere ayriliyor...")
    cumleler = cumlelere_ayir(sayfalar)

    st.info("Chunk'lar olusturuluyor...")
    chunks = chunk_olustur(cumleler)

    st.info("FastText modeli yukleniyor...")
    model = model_yukle(MODEL_PATH)

    if model.vector_size != vector_size:
        st.error(
            f"Model vector size {model.vector_size}, fakat kodda vector_size={vector_size}. "
            f"vector_size degerini {model.vector_size} yapmalisin."
        )
        st.stop()

    st.info("IDF agirliklari hesaplanıyor...")
    idf = idf_hesapla(chunks)

    st.info("Vektorler hesaplanıyor...")
    log_temizle()

    progress = st.progress(0)

    for i, chunk in enumerate(chunks):
        vektor = metin_vektorle(chunk, model, idf)
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


# ARAMA

def kelime_ortusme_skoru(soru, chunk):
    soru_kelimeleri = {
        k for k in metni_temizle(soru).split()
        if kelime_filtrele(k)
    }

    chunk_kelimeleri = {
        k for k in metni_temizle(chunk).split()
        if kelime_filtrele(k)
    }

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

    kosinuslar = cosine_similarity(
        soru_vektor.reshape(1, -1),
        chunk_vektorleri
    )[0]

    sonuclar = []

    for i, (metin, _) in enumerate(chunks_data):
        overlap = kelime_ortusme_skoru(soru, metin)

        final_skor = (
            0.75 * float(kosinuslar[i]) +
            0.25 * float(overlap)
        )

        sonuclar.append({
            "metin": metin,
            "skor": final_skor,
            "cosine": float(kosinuslar[i]),
            "overlap": float(overlap),
            "indeks": int(i)
        })

    sonuclar = sorted(
        sonuclar,
        key=lambda x: x["skor"],
        reverse=True
    )

    return sonuclar[:k]


# GEMINI

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
            genai.GenerativeModel(
                model_name=m,
                system_instruction=SISTEM_PROMPTU
            ).generate_content("Hi")

            return m
        except Exception:
            continue

    return None


def gemini_cevapla(soru, benzer_chunks, api_key, model_adi=None):
    genai.configure(api_key=api_key)

    baglam = "".join(
        f"[Context {i} - Final Score: {c['skor']:.3f}, "
        f"Cosine: {c['cosine']:.3f}, Keyword: {c['overlap']:.3f}]\n"
        f"{c['metin']}\n\n"
        for i, c in enumerate(benzer_chunks, 1)
    )

    kullanici_mesaji = (
        f"Use the following context to answer the question:\n\n"
        f"{baglam}\n"
        f"Question: {soru}"
    )

    denemeler = [model_adi] if model_adi else GEMINI_MODELLER
    son_hata = None

    for m in denemeler:
        try:
            yanit = genai.GenerativeModel(
                model_name=m,
                system_instruction=SISTEM_PROMPTU
            ).generate_content(kullanici_mesaji)

            return yanit.text, m

        except Exception as e:
            son_hata = e

    raise Exception(f"No model worked. Last error: {son_hata}")


# CEVAP-KAYNAK ORTUSME

def ortusme_hesapla(cevap_metni, chunk_metni):
    cevap_kelimeleri = {
        k for k in metni_temizle(cevap_metni).split()
        if kelime_filtrele(k)
    }

    chunk_kelimeleri = {
        k for k in metni_temizle(chunk_metni).split()
        if kelime_filtrele(k)
    }

    if not chunk_kelimeleri:
        return 0.0, set()

    ortak = cevap_kelimeleri & chunk_kelimeleri
    oran = len(ortak) / len(chunk_kelimeleri)

    return oran, ortak


# STREAMLIT

def main():
    st.set_page_config(
        page_title="PDF RAG",
        page_icon="📄",
        layout="wide"
    )

    st.title("PDF RAG - FastText")
    st.caption("FastText weighted retrieval · Keyword overlap reranking · Gemini generation")

    with st.sidebar:
        st.header("Settings")

        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            placeholder="AIza...",
            help="Get a free key at aistudio.google.com"
        )

        st.markdown("[Get Free Key](https://aistudio.google.com/app/apikey)")

        st.divider()

        st.write("FastText model:")
        st.code(MODEL_PATH, language="text")

        st.write(f"Vector size: `{vector_size}`")
        st.write(f"Chunk size: `{chunk_size}`")
        st.write(f"Chunk step: `{chunk_step}`")

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
        st.write(f"**FastText model:** `{MODEL_PATH}`")

        if st.button("Build Log"):
            toplam = log_olustur()
            st.success(f"Done! {toplam} chunks processed.")
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

        st.success(f"{len(chunks_data)} chunks loaded.")

        col1, col2 = st.columns([3, 1])

        with col2:
            k_deger = st.slider("Chunks to retrieve", 1, 5, 3)

        with col1:
            soru = st.text_input(
                "Your question, English or Turkish:",
                placeholder="e.g. What is a convolutional neural network?"
            )

        ara_tiklandi = st.button(
            "Search & Answer",
            type="primary",
            key="ara_butonu"
        )

        if ara_tiklandi and soru.strip():
            with st.spinner("Loading FastText model..."):
                model = model_yukle(MODEL_PATH)

            if model.vector_size != vector_size:
                st.error(
                    f"Model vector size {model.vector_size}, fakat kodda vector_size={vector_size}. "
                    f"vector_size degerini {model.vector_size} yap."
                )
                st.stop()

            with st.spinner("Finding similar chunks..."):
                benzer_chunks = benzer_chunk_bul(
                    soru,
                    model,
                    chunks_data,
                    k=k_deger
                )

            if not benzer_chunks:
                st.error("Could not vectorize question. Try a more technical query.")
                return

            st.markdown("---")
            st.subheader("Most Similar Chunks")

            for i, chunk in enumerate(benzer_chunks, 1):
                with st.expander(
                    f"Chunk #{chunk['indeks'] + 1} - "
                    f"Final: {chunk['skor']:.4f} | "
                    f"Cosine: {chunk['cosine']:.4f} | "
                    f"Keyword: {chunk['overlap']:.4f}",
                    expanded=(i == 1)
                ):
                    st.write(chunk["metin"])

                    progress_value = min(max(chunk["skor"], 0.0), 1.0)

                    st.progress(
                        progress_value,
                        text=(
                            f"Final score: {chunk['skor']:.4f} | "
                            f"FastText cosine: {chunk['cosine']:.4f} | "
                            f"Keyword overlap: {chunk['overlap']:.4f}"
                        )
                    )

            st.markdown("---")
            st.subheader("Gemini Answer")

            with st.spinner("Generating answer..."):
                try:
                    if not api_key:
                        st.error("Enter your Gemini API key in the sidebar.")
                        st.stop()

                    cevap, kullanilan_model = gemini_cevapla(
                        soru,
                        benzer_chunks,
                        api_key,
                        st.session_state.get("aktif_model")
                    )

                    st.caption(f"Model: `{kullanilan_model}`")
                    st.success(cevap)

                except Exception as e:
                    st.error(f"Gemini API error: {e}")
                    st.stop()

            st.markdown("---")
            st.subheader("Answer Source Analysis")

            st.info(
                "**Retrieval method:**\n\n"
                "1. PDF is split into sentence-based chunks\n"
                "2. Each chunk is embedded with IDF-weighted FastText vectors\n"
                "3. Question is embedded with the same FastText + IDF method\n"
                "4. Final retrieval score combines FastText cosine similarity and keyword overlap\n\n"
                "`final_score = 0.75 * cosine + 0.25 * keyword_overlap`"
            )

            st.markdown("**Technical term overlap, chunk to answer:**")

            for i, chunk in enumerate(benzer_chunks, 1):
                oran, ortak = ortusme_hesapla(cevap, chunk["metin"])
                ortak_liste = ", ".join(sorted(ortak)[:10]) if ortak else "-"

                st.progress(
                    min(max(oran, 0.0), 1.0),
                    text=(
                        f"Chunk #{chunk['indeks'] + 1} -> "
                        f"{oran * 100:.0f}% of chunk keywords appear in answer "
                        f"({len(ortak)} shared: {ortak_liste})"
                    )
                )

            max_oran = max(
                ortusme_hesapla(cevap, c["metin"])[0]
                for c in benzer_chunks
            )

            st.markdown("---")

            if max_oran >= 0.15:
                st.success(f"Good retrieval - best chunk overlap: {max_oran * 100:.0f}%")
            elif max_oran >= 0.05:
                st.warning(
                    f"Low overlap ({max_oran * 100:.0f}%). "
                    "Retrieved chunks may be weak."
                )
            else:
                st.error(
                    f"Very low overlap ({max_oran * 100:.0f}%). "
                    "Try asking with more exact technical terms from the book."
                )

        elif ara_tiklandi:
            st.warning("Please enter a question.")


if __name__ == "__main__":
    main()
