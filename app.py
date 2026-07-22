"""
تطبيق أسوان RAG — شات بوت Streamlit
نظام استرجاع وتوليد معزز بالمعرفة (RAG) عن ثقافة وسياحة أسوان،
مع دعم توليد الإجابة عبر Ollama (نموذج محلي) أو Anthropic API (سحابي).

للتشغيل محلياً:
    streamlit run app.py
"""

import json
import os
import re

import numpy as np
import pandas as pd
import requests
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_arabic(text):
    """توحيد شكل الكلمات العربية قبل الاسترجاع اللفظي (TF-IDF / BM25)."""
    text = re.sub(r"[\u064B-\u065F]", "", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# إعداد الصفحة + الهوية البصرية
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="أسوان AI - دليلك الثقافي الذكي",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PRIMARY = "#0E5E66"       # تركواز نيلي غامق
PRIMARY_DARK = "#0A454B"
ACCENT = "#C98A2E"        # ذهبي/رملي (مستوحى من رمال أسوان)
ACCENT_DEEP = "#7A3B2E"   # طيني نوبي
BG = "#FBF6EC"            # عاجي دافئ
CARD = "#FFFFFF"
TEXT = "#2B2420"
MUTED = "#7A7264"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@500;700;800&family=Tajawal:wght@400;500;700&display=swap');

    html, body, .stApp {{
        direction: rtl;
        text-align: right;
        background: {BG};
        font-family: 'Tajawal', sans-serif;
        color: {TEXT};
    }}
    h1, h2, h3, .display-font {{ font-family: 'Cairo', sans-serif; }}

    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding-top: 1.5rem; max-width: 900px; }}

    /* شريط الهيدر */
    .app-header {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 0.6rem 0 1.2rem 0; border-bottom: 1px solid #E9E0CC; margin-bottom: 1.5rem;
    }}
    .app-brand {{ display: flex; align-items: center; gap: 0.7rem; }}
    .app-logo {{
        width: 46px; height: 46px; border-radius: 12px; background: {PRIMARY};
        display: flex; align-items: center; justify-content: center; font-size: 22px;
    }}
    .app-title {{ font-family: 'Cairo', sans-serif; font-weight: 800; font-size: 1.15rem; color: {TEXT}; margin: 0; }}
    .app-subtitle {{ font-size: 0.8rem; color: {MUTED}; margin: 0; }}

    /* بادج الوضع الحالي */
    .mode-badge {{
        display: inline-block; background: {PRIMARY}1A; color: {PRIMARY_DARK};
        border: 1px solid {PRIMARY}55; padding: 0.5rem 1rem; border-radius: 999px;
        font-size: 0.85rem; margin: 0.8rem 0;
    }}

    /* هيرو الترحيب */
    .hero-title {{ text-align: center; font-family: 'Cairo', sans-serif; font-weight: 800; font-size: 1.9rem; margin-top: 1rem; }}
    .hero-desc {{ text-align: center; color: {MUTED}; font-size: 1.02rem; max-width: 560px; margin: 0.6rem auto 0 auto; line-height: 1.9; }}
    .hero-lang {{ text-align: center; color: {PRIMARY_DARK}; font-weight: 700; margin-top: 0.6rem; }}

    .try-label {{
        text-align: center; letter-spacing: 2px; color: {MUTED}; font-size: 0.75rem;
        font-weight: 700; margin: 2rem 0 0.9rem 0; text-transform: uppercase;
    }}

    div[data-testid="stButton"] > button {{
        border-radius: 14px; border: 1px solid #E9E0CC; background: {CARD};
        color: {TEXT}; padding: 0.9rem 1rem; font-family: 'Tajawal', sans-serif;
        box-shadow: 0 1px 2px rgba(43,36,32,0.04); transition: all 0.15s ease;
    }}
    div[data-testid="stButton"] > button:hover {{
        border-color: {PRIMARY}; color: {PRIMARY_DARK}; transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(14,94,102,0.12);
    }}
    .mode-btn button {{ border-radius: 999px !important; padding: 0.45rem 1.1rem !important; font-size: 0.85rem !important; }}

    .footer-note {{ text-align: center; color: {MUTED}; font-size: 0.78rem; margin-top: 2.2rem; }}
    .footer-note b {{ color: {ACCENT_DEEP}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

MODES = {
    "دقيق": {"icon": "🎯", "retriever": "هجين (موصى به)", "k": 4, "alpha": 0.7,
             "desc": "إجابات دقيقة معتمدة فقط على مصادر موثقة عن أسوان — الأنسب للمعلومات الصحيحة."},
    "متوازن": {"icon": "⚖️", "retriever": "هجين (موصى به)", "k": 3, "alpha": 0.6,
               "desc": "توازن بين السرعة والدقة — الوضع الافتراضي المناسب لمعظم الأسئلة."},
    "سريع": {"icon": "⚡", "retriever": "BM25", "k": 2, "alpha": 0.5,
             "desc": "استرجاع سريع بالكلمات المفتاحية بدون توليد إجابة طويلة."},
}

EXAMPLES_AR = [
    "عادات وتقاليد أهل النوبة في الاحتفالات",
    "ما هي ظاهرة تعامد الشمس في أبو سمبل؟",
    "أين يمكنني التنزه بجانب النيل مساءً؟",
    "مراسم ليلة الحنة عند النوبيين",
    "الحديقة النباتية جزيرة كتشنر",
    "أصل تسمية النوبة ومعناها",
]


# ---------------------------------------------------------------------------
# تحميل البيانات والنماذج
# ---------------------------------------------------------------------------
@st.cache_data
def load_corpus():
    with open(os.path.join(DATA_DIR, "aswan_corpus.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_resource
def build_indexes(_documents):
    documents_clean = [preprocess_arabic(doc) for doc in _documents]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents_clean)
    tokenized_corpus = [doc.split() for doc in documents_clean]
    bm25 = BM25Okapi(tokenized_corpus)
    model = load_embedding_model()
    doc_embeddings = model.encode(_documents, convert_to_numpy=True, normalize_embeddings=True)
    return tfidf_vectorizer, tfidf_matrix, bm25, model, doc_embeddings


documents_df = load_corpus()
documents = documents_df["document"].tolist()
tfidf_vectorizer, tfidf_matrix, bm25, embedding_model, doc_embeddings = build_indexes(documents)


# ---------------------------------------------------------------------------
# دوال الاسترجاع
# ---------------------------------------------------------------------------
def normalize(scores):
    scores = np.array(scores, dtype=float)
    if scores.max() - scores.min() == 0:
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


def tfidf_scores(query):
    query_vec = tfidf_vectorizer.transform([preprocess_arabic(query)])
    return cosine_similarity(query_vec, tfidf_matrix)[0]


def bm25_scores(query):
    return np.array(bm25.get_scores(preprocess_arabic(query).split()))


def semantic_scores(query):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return cosine_similarity(query_embedding, doc_embeddings)[0]


def hybrid_scores(query, alpha=0.6):
    lex = normalize(bm25_scores(query))
    sem = normalize(semantic_scores(query))
    return alpha * sem + (1 - alpha) * lex


RETRIEVERS = {
    "هجين (موصى به)": lambda q, alpha: hybrid_scores(q, alpha),
    "دلالي (Embeddings)": lambda q, alpha: semantic_scores(q),
    "BM25": lambda q, alpha: bm25_scores(q),
    "TF-IDF": lambda q, alpha: tfidf_scores(q),
}


def retrieve(query, method, k=3, alpha=0.6):
    scores = RETRIEVERS[method](query, alpha)
    ranked_ids = np.argsort(scores)[::-1][:k]
    return [
        {
            "document_id": int(doc_id),
            "category": documents_df.loc[doc_id, "category"],
            "text": documents_df.loc[doc_id, "document"],
            "score": float(scores[doc_id]),
        }
        for doc_id in ranked_ids
    ]


def build_prompt(query, context, history_text=""):
    context_text = "\n".join(f"[{i+1}] ({c['category']}) {c['text']}" for i, c in enumerate(context))
    history_part = f"\nسياق المحادثة السابقة:\n{history_text}\n" if history_text else ""
    return f"""أنت مرشد سياحي وثقافي متخصص في مدينة أسوان، تتحدث بأسلوب ودود وواضح بالعربية.
أجب عن سؤال المستخدم بالاعتماد فقط على المعلومات الموجودة في السياق أدناه.
إذا لم تكفِ المعلومات للإجابة، صرّح بذلك بوضوح ولا تختلق معلومات من عندك.
{history_part}
السياق المسترجع من قاعدة المعرفة:
{context_text}

سؤال المستخدم الحالي: {query}

الإجابة:"""


def generate_with_ollama(prompt, base_url, model):
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/chat",
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "").strip() or "لم يصل رد من Ollama."
    except requests.exceptions.ConnectionError:
        return (
            "❌ تعذّر الاتصال بـ Ollama. تأكد إنه شغال على جهازك (`ollama serve`) "
            f"وإن الموديل `{model}` متاح (`ollama pull {model}`)."
        )
    except Exception as e:
        return f"❌ خطأ أثناء توليد الإجابة عبر Ollama: {e}"


def generate_with_anthropic(prompt, api_key):
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=600, messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"❌ خطأ أثناء توليد الإجابة عبر Anthropic API: {e}"


# ---------------------------------------------------------------------------
# الحالة (state)
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "متوازن"
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# ---------------------------------------------------------------------------
# الهيدر
# ---------------------------------------------------------------------------
header_col1, header_col2 = st.columns([2, 2])
with header_col1:
    st.markdown(
        """
        <div class="app-brand">
            <div class="app-logo">🏛️</div>
            <div>
                <p class="app-title">أسوان AI</p>
                <p class="app-subtitle">دليلك الثقافي الذكي</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_col2:
    m1, m2, m3 = st.columns(3)
    for col, mode_name in zip([m1, m2, m3], MODES.keys()):
        with col:
            st.markdown('<div class="mode-btn">', unsafe_allow_html=True)
            is_active = st.session_state.mode == mode_name
            if st.button(
                f"{MODES[mode_name]['icon']} {mode_name}",
                key=f"mode_{mode_name}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                st.session_state.mode = mode_name
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

with st.popover("⚙️ إعدادات متقدمة", use_container_width=False):
    st.caption("مصدر توليد الإجابة النهائية")
    generation_source = st.radio(
        "مصدر التوليد", ["Ollama (محلي)", "Anthropic API (سحابي)", "بدون توليد"],
        index=0, label_visibility="collapsed",
    )
    ollama_base_url, ollama_model, anthropic_api_key = None, None, None
    if generation_source == "Ollama (محلي)":
        ollama_base_url = st.text_input("رابط خادم Ollama", value="http://localhost:11434")
        ollama_model = st.text_input("اسم الموديل", value="llama3.1")
        st.caption("محتاج Ollama شغال على جهازك محلياً.")
    elif generation_source == "Anthropic API (سحابي)":
        anthropic_api_key = st.text_input("مفتاح Anthropic API", type="password")
    if st.button("🗑️ مسح المحادثة", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

active_mode = MODES[st.session_state.mode]
st.markdown(
    f'<div style="text-align:center;"><span class="mode-badge">{active_mode["icon"]} '
    f'وضع {st.session_state.mode}: {active_mode["desc"]}</span></div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# شاشة الترحيب (تظهر فقط قبل أول سؤال)
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown('<p class="hero-title">👋 أهلاً! أنا دليل أسوان الذكي</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-desc">اسألني عن أي حاجة تخص أسوان: المعابد، الحدائق، النزهات النيلية، '
        'وعادات وتقاليد النوبة — وأنا هجاوبك بالاعتماد على قاعدة معرفة موثقة.</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="hero-lang">🌍 تقدر تسأل بالعربي الفصحى أو العامية المصرية</p>', unsafe_allow_html=True)

    st.markdown('<p class="try-label">جرب تسأل</p>', unsafe_allow_html=True)
    ex_cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES_AR):
        with ex_cols[i % 2]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state.pending_query = ex
                st.rerun()

    st.markdown(
        '<p class="footer-note">🕌 الإجابات مبنية على مصادر ثقافية موثقة · '
        f'الوضع الحالي <b>{st.session_state.mode}</b> · استشر مرشداً سياحياً محلياً للتفاصيل الدقيقة.</p>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# عرض المحادثة
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("context"):
            with st.expander("📄 المستندات المسترجعة"):
                for c in msg["context"]:
                    st.markdown(f"**#{c['document_id']} — {c['category']}** (score: {c['score']:.3f})")
                    st.write(c["text"])

typed_query = st.chat_input("اسأل عن أي شيء يخص أسوان...")
query = st.session_state.pending_query or typed_query
st.session_state.pending_query = None

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("جاري الاسترجاع..."):
            context = retrieve(query, active_mode["retriever"], k=active_mode["k"], alpha=active_mode["alpha"])

        history_text = ""
        prior = st.session_state.messages[-5:-1]
        if prior:
            history_text = "\n".join(f"{m['role']}: {m['content']}" for m in prior)

        prompt = build_prompt(query, context, history_text)

        if generation_source == "Ollama (محلي)":
            with st.spinner("جاري توليد الإجابة عبر Ollama..."):
                answer = generate_with_ollama(prompt, ollama_base_url, ollama_model)
        elif generation_source == "Anthropic API (سحابي)":
            if not anthropic_api_key:
                answer = "⚠️ أدخل مفتاح Anthropic API من ⚙️ الإعدادات المتقدمة أولاً."
            else:
                with st.spinner("جاري توليد الإجابة..."):
                    answer = generate_with_anthropic(prompt, anthropic_api_key)
        else:
            answer = "تم عرض أقرب المستندات المطابقة أدناه (بدون توليد إجابة نهائية)."

        st.markdown(answer)
        with st.expander("📄 المستندات المسترجعة"):
            for c in context:
                st.markdown(f"**#{c['document_id']} — {c['category']}** (score: {c['score']:.3f})")
                st.write(c["text"])

    st.session_state.messages.append({"role": "assistant", "content": answer, "context": context})
