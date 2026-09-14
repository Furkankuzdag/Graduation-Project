import streamlit as st
import PyPDF2
import numpy as np
import re
import google.generativeai as genai
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import os

# ─── AYARLAR ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "wiki_word2vec.model")
PDF_PATH   = os.path.join(BASE_DIR, "Mohamed Elgendy - Deep Learning for Vision Systems MEAP V06 (2019, Manning Publications Co.) - libgen.li.pdf")
LOG_PATH   = os.path.join(BASE_DIR, "rag_log.txt")

vector_size = 200
top_k       = 5
chunk_size  = 5
chunk_step  = 3

# ─── YARDIMCI ─────────────────────────────────────────────────────────────────

def log_yaz(mesaj):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{mesaj}\n")

def log_temizle():
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("")

@st.cache_resource
def model_yukle(model_path):
    return Word2Vec.load(model_path)

def metni_temizle(metin):
    metin = metin.lower()
    metin = re.sub(r'[^\w\s]', '', metin)
    return metin

def kelime_filtrele(kelime):
    """İngilizce stop word ve çok kısa kelimeleri ele."""
    if kelime in ENGLISH_STOP_WORDS:
        return False
    if len(kelime) <= 2:
        return False
    return True

def metin_vektorle(metin, model):
    temiz = metni_temizle(metin)
    kelimeler = temiz.split()
    vektorler = [
        model.wv[k]
        for k in kelimeler
        if kelime_filtrele(k) and k in model.wv.key_to_index
    ]
    if not vektorler:
        return np.zeros(vector_size)
    return np.mean(vektorler, axis=0)


# ─── PDF & CHUNK ──────────────────────────────────────────────────────────────

def pdf_okuma(pdf_yolu):
    sayfalar = []
    with open(pdf_yolu, "rb") as f:
        pdf = PyPDF2.PdfReader(f)
        for i in range(15, len(pdf.pages)):
            metin = pdf.pages[i].extract_text()
            if metin:
                sayfalar.append(metin.replace('\n', ' '))
    return sayfalar

def cumlelere_ayir(sayfalar):
    tum = re.sub(r'\s+', ' ', " ".join(sayfalar)).strip()
    return [c for c in re.split(r'(?<=[.!?]) +', tum) if len(c) > 20]

def chunk_olustur(cumleler):
    chunks = []
    for i in range(0, len(cumleler), chunk_step):
        if i + chunk_size > len(cumleler):
            break
        chunk = " ".join(cumleler[i:i+chunk_size])
        kelime_sayisi = len(chunk.split())
        if kelime_sayisi < 30 or kelime_sayisi > 300:
            continue
        rakamlar = [k for k in chunk.split() if k.isdigit()]
        if len(rakamlar) / kelime_sayisi > 0.15:
            continue
        chunks.append(chunk)
    return chunks


# ─── LOG ──────────────────────────────────────────────────────────────────────

def log_olustur():
    st.info("📄 PDF okunuyor...")
    sayfalar = pdf_okuma(PDF_PATH)
    st.info("✂️ Cümlelere ayrılıyor...")
    cumleler = cumlelere_ayir(sayfalar)
    st.info("🧩 Chunk'lar oluşturuluyor...")
    chunks = chunk_olustur(cumleler)
    st.info("🤖 Word2Vec modeli yükleniyor...")
    model = Word2Vec.load(MODEL_PATH)
    st.info("📐 Vektörler hesaplanıyor...")
    log_temizle()
    progress = st.progress(0)
    for i, chunk in enumerate(chunks):
        vektor = metin_vektorle(chunk, model)
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
        with open(log_path, "r", encoding="utf-8", newline='') as f:
            icerik = f.read()
    except FileNotFoundError:
        return []
    icerik = icerik.replace('\r\n', '\n').replace('\r', '\n')
    if not icerik.strip():
        return []

    # ASCII etiket kullan (METİN → METIN, VEKTÖR → VEKTOR)
    bloklar = re.split(r'CHUNK #\d+\n=+\n', icerik)
    for blok in bloklar:
        blok = blok.strip()
        if not blok:
            continue
        try:
            metin_b = re.search(r'METIN:\n(.*?)\n\s*\nVEKTOR:', blok, re.DOTALL)
            if not metin_b:
                metin_b = re.search(r'METIN:\n(.*?)VEKTOR:', blok, re.DOTALL)
            vektor_b = re.search(r'VEKTOR:\n([\d.\- \n\r]+)', blok)
            if not metin_b or not vektor_b:
                # Eski log formatı (Türkçe etiket) için de dene
                metin_b = re.search(r'MET[İI]N:\n(.*?)\n\s*\nVEKT[ÖO]R:', blok, re.DOTALL)
                if not metin_b:
                    metin_b = re.search(r'MET[İI]N:\n(.*?)VEKT[ÖO]R:', blok, re.DOTALL)
                vektor_b = re.search(r'VEKT[ÖO]R:\n([\d.\- \n\r]+)', blok)
            if not metin_b or not vektor_b:
                continue
            metin = metin_b.group(1).strip()
            sayilar = re.findall(r'-?[\d]+\.[\d]+', vektor_b.group(1))
            vektor = np.array([float(v) for v in sayilar])
            if len(vektor) == vector_size:
                chunks_data.append((metin, vektor))
        except Exception:
            continue
    return chunks_data


# ─── ARAMA ────────────────────────────────────────────────────────────────────

def benzer_chunk_bul(soru, model, chunks_data, k=3):
    soru_vektor = metin_vektorle(soru, model)
    if np.all(soru_vektor == 0):
        return []
    chunk_vektorleri = np.array([v for _, v in chunks_data])
    benzerlikler = cosine_similarity(soru_vektor.reshape(1, -1), chunk_vektorleri)[0]
    indeksler = np.argsort(benzerlikler)[::-1][:k]
    return [
        {"metin": chunks_data[i][0], "skor": float(benzerlikler[i]), "indeks": int(i)}
        for i in indeksler
    ]


# ─── GEMİNİ ───────────────────────────────────────────────────────────────────

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
    "Reply in the same language as the question (Turkish or English). "
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
        f"[Context {i} – Similarity: {c['skor']:.3f}]\n{c['metin']}\n\n"
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
                model_name=m, system_instruction=SISTEM_PROMPTU
            ).generate_content(kullanici_mesaji)
            return yanit.text, m
        except Exception as e:
            son_hata = e
    raise Exception(f"No model worked. Last error: {son_hata}")


# ─── KELIME ÖRTÜŞME METRİĞİ (DÜZELTİLDİ) ────────────────────────────────────

def ortusme_hesapla(cevap_metni, chunk_metni):
    """
    Her iki metin de İngilizce stop-word filtresinden geçirilir.
    Cevap Türkçe olsa bile teknik terimler (perceptron, cnn, relu …)
    her iki dilde aynı yazıldığından doğru eşleşir.
    Oran: chunk'taki filtrelenmiş kelimelerin kaçı cevap metninde geçiyor.
    """
    cevap_kelimeleri  = {k for k in metni_temizle(cevap_metni).split()  if kelime_filtrele(k)}
    chunk_kelimeleri  = {k for k in metni_temizle(chunk_metni).split()  if kelime_filtrele(k)}

    if not chunk_kelimeleri:
        return 0.0, set()

    ortak = cevap_kelimeleri & chunk_kelimeleri
    # Oran: "chunk'taki teknik kelimelerin kaçı cevaba yansıdı?"
    oran  = len(ortak) / len(chunk_kelimeleri)
    return oran, ortak


# ─── STREAMLIT ────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="PDF RAG", page_icon="🗂️", layout="wide")
    st.title("🗂️ PDF RAG — Deep Learning for Vision Systems")
    st.caption("Word2Vec retrieval  ·  Gemini generation  ·  English PDF support")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            placeholder="AIza...",
            help="Get a free key at aistudio.google.com"
        )
        st.markdown("[🔑 Get Free Key](https://aistudio.google.com/app/apikey)")

        if api_key:
            if st.button("🧪 Test Model"):
                with st.spinner("Testing models..."):
                    bulunan = gemini_test(api_key)
                if bulunan:
                    st.success(f"✅ Working model: `{bulunan}`")
                    st.session_state["aktif_model"] = bulunan
                else:
                    st.error("❌ No model worked. Try VPN or check your key.")
            st.success("✅ API key entered.")
        else:
            st.warning("⚠️ No API key.")

    tab1, tab2 = st.tabs(["📋 Build Log", "🔍 Ask Question"])

    # Tab 1
    with tab1:
        st.write(f"**PDF:** `{PDF_PATH}`")
        st.write(f"**Log file:** `{LOG_PATH}`")
        if st.button("📋 Build Log"):
            toplam = log_olustur()
            st.success(f"✅ Done! {toplam} chunks processed.")
            log_oku.clear()

    # Tab 2
    with tab2:
        st.subheader("🔍 Ask a Question")

        col_r, col_d = st.columns(2)
        with col_r:
            if st.button("🔄 Reload Log"):
                log_oku.clear()
                st.rerun()
        with col_d:
            debug = st.checkbox("🐛 Debug mode")

        if debug:
            import os
            if os.path.exists(LOG_PATH):
                boyut = os.path.getsize(LOG_PATH)
                st.write(f"📁 Log found — {boyut:,} bytes")
                with open(LOG_PATH, "r", encoding="utf-8", newline='') as f:
                    st.code(repr("".join(f.readlines()[:30])), language="text")
            else:
                st.error(f"❌ Not found: `{LOG_PATH}`")

        chunks_data = log_oku(LOG_PATH)
        if not chunks_data:
            import os
            if not os.path.exists(LOG_PATH):
                st.error(f"❌ Log not found: `{LOG_PATH}`")
            else:
                st.warning("⚠️ Log exists but no chunks parsed. Use debug mode or rebuild log.")
            return

        st.success(f"✅ {len(chunks_data)} chunks loaded.")

        col1, col2 = st.columns([3, 1])
        with col2:
            k_deger = st.slider("Chunks to retrieve", 1, 5, 3)
        with col1:
            soru = st.text_input(
                "Your question (English or Turkish):",
                placeholder="e.g. What is a convolutional neural network?"
            )

        ara_tiklandi = st.button("🔎 Search & Answer", type="primary", key="ara_butonu")

        if ara_tiklandi and soru.strip():
            with st.spinner("Loading Word2Vec model..."):
                model = model_yukle(MODEL_PATH)

            with st.spinner("Finding similar chunks..."):
                benzer_chunks = benzer_chunk_bul(soru, model, chunks_data, k=k_deger)

            if not benzer_chunks:
                st.error("❌ Could not vectorize question. Try a more descriptive query.")
                return

            # En iyi chunk'lar
            st.markdown("---")
            st.subheader("📌 Most Similar Chunks")
            for i, chunk in enumerate(benzer_chunks, 1):
                with st.expander(
                    f"Chunk #{chunk['indeks']+1} — Cosine Similarity: {chunk['skor']:.4f}",
                    expanded=(i == 1)
                ):
                    st.write(chunk["metin"])
                    st.progress(min(chunk["skor"], 1.0), text=f"Similarity: {chunk['skor']:.4f}")

            # Gemini cevabı
            st.markdown("---")
            st.subheader("🤖 Gemini Answer")
            with st.spinner("Generating answer..."):
                try:
                    if not api_key:
                        st.error("❌ Enter your Gemini API key in the sidebar.")
                        st.stop()
                    cevap, kullanilan_model = gemini_cevapla(
                        soru, benzer_chunks, api_key,
                        st.session_state.get("aktif_model")
                    )
                    st.caption(f"✅ Model: `{kullanilan_model}`")
                    st.success(cevap)
                except Exception as e:
                    st.error(f"Gemini API error: {e}")
                    st.stop()

            # ── DÜZELTİLMİŞ ÖRTÜŞME METRİĞİ ──────────────────────────────
            st.markdown("---")
            st.subheader("🔎 Answer Source Analysis")

            st.info(
                "**How it works:**\n\n"
                "1. Question is vectorized with Word2Vec\n"
                "2. Most similar chunks found via cosine similarity\n"
                "3. Chunks are given as **context** to Gemini\n"
                "4. Gemini reads the context and answers in its own words\n\n"
                "**Overlap metric:** counts technical terms (e.g. *perceptron*, *relu*, *cnn*) "
                "that appear in **both** the chunk and the answer — these language-neutral "
                "terms work correctly even when the answer is in Turkish."
            )

            st.markdown("**📊 Technical term overlap (chunk → answer):**")

            for i, chunk in enumerate(benzer_chunks, 1):
                oran, ortak = ortusme_hesapla(cevap, chunk["metin"])
                ortak_liste = ", ".join(sorted(ortak)[:10]) if ortak else "—"
                st.progress(
                    min(oran, 1.0),
                    text=(
                        f"Chunk #{chunk['indeks']+1}  →  "
                        f"{oran*100:.0f}% of chunk's keywords appear in answer  "
                        f"({len(ortak)} shared: {ortak_liste})"
                    )
                )

            # Genel değerlendirme
            max_oran = max(ortusme_hesapla(cevap, c["metin"])[0] for c in benzer_chunks)
            st.markdown("---")
            if max_oran >= 0.15:
                st.success(f"✅ Good retrieval — best chunk overlap: {max_oran*100:.0f}%")
            elif max_oran >= 0.05:
                st.warning(
                    f"⚠️ Low overlap ({max_oran*100:.0f}%). "
                    "Gemini may be using general knowledge rather than the book."
                )
            else:
                st.error(
                    f"❌ Very low overlap ({max_oran*100:.0f}%). "
                    "Retrieved chunks are likely off-topic. "
                    "Try rephrasing the question with more technical terms."
                )

        elif ara_tiklandi:
            st.warning("Please enter a question.")


if __name__ == "__main__":
    main()