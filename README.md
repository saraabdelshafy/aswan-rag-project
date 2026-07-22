# 🏛️ أسوان RAG — دليل ذكي لثقافة ومعالم أسوان

مشروع أكاديمي لبناء نظام **RAG (Retrieval-Augmented Generation)** كامل، مبني بنفس منهجية
معامل: `TF-IDF/BM25` → `Sentence Embeddings` → `Hybrid Retrieval` → `RAG Pipeline`،
لكن مطبَّق على قاعدة معرفة أصلية عن **أسوان**: المعابد الأثرية (فيلة، أبو سمبل، كلابشة)،
السد العالي، الحدائق (جزيرة كتشنر)، النزهات النيلية (الفلوكة)، المتاحف، عادات وتقاليد
النوبة، الحرف اليدوية، المطبخ المحلي، والمهرجانات.

## محتويات المشروع

```
aswan_rag_project/
├── Aswan_RAG_Project.ipynb   # دفتر الملاحظات الكامل: بناء وتقييم النظام خطوة بخطوة
├── app.py                    # تطبيق Streamlit التفاعلي (الواجهة النهائية للمستخدم)
├── requirements.txt          # المكتبات المطلوبة
├── data/
│   ├── aswan_corpus.json     # قاعدة المعرفة (34 مستنداً عن أسوان)
│   └── aswan_queries.json    # استعلامات تقييم + الحقيقة المرجعية (Ground Truth)
└── README.md
```

## 1) التشغيل محلياً

```bash
git clone <رابط-المستودع-بعد-رفعه-على-GitHub>
cd aswan_rag_project
pip install -r requirements.txt
streamlit run app.py
```

سيفتح التطبيق تلقائياً على `http://localhost:8501` كـ **شات بوت** — بتسأل وبيرد عليك مع الاحتفاظ
بتاريخ المحادثة، وتحت كل رد فيه قائمة قابلة للطي بالمستندات اللي اعتمد عليها.

### توليد الإجابة: Ollama (محلي ومجاني) أو Anthropic API (سحابي)

من الشريط الجانبي تقدر تختار مصدر توليد الإجابة النهائية:

**أ) Ollama — يشتغل بالكامل على جهازك بدون إنترنت بعد التحميل، ومجاني:**
```bash
# 1) نزّل Ollama من https://ollama.com (يدعم Windows / Mac / Linux)

# 2) حمّل موديل (مرة واحدة فقط) — أي موديل من الموجودين على ollama.com/library
ollama pull llama3.1
# أو موديل أخف وأسرع:
# ollama pull qwen2.5:7b

# 3) شغّل خادم Ollama (لو مش شغال تلقائي في الخلفية)
ollama serve

# 4) شغّل تطبيق Streamlit في تيرمينال تاني، واختر "Ollama (محلي)" من الشريط الجانبي
streamlit run app.py
```
تأكد إن اسم الموديل في خانة "اسم الموديل" بالتطبيق مطابق تماماً لاسم اللي حمّلته (`ollama list` يوريك الموديلات المتاحة عندك).

**ب) Anthropic API — يشتغل من أي مكان، مناسب لو التطبيق هيتنشر على الإنترنت:**
اختر "Anthropic API (سحابي)" من الشريط الجانبي وحط مفتاح الـ API بتاعك في الخانة.

> ⚠️ **ملاحظة مهمة:** Ollama محتاج جهاز حقيقي يشغّل الموديل، فـ Streamlit Community Cloud (الاستضافة
> المجانية السحابية) **مش هيقدر يشغّل Ollama**. لو هتنشر التطبيق على الإنترنت للدكتور يشوفه، استخدم
> "Anthropic API" كمصدر التوليد. Ollama يبقى الخيار المثالي وانت شغّال محلياً على جهازك أو أثناء العرض المباشر.

## 2) رفع المشروع على GitHub

```bash
cd aswan_rag_project
git init
git add .
git commit -m "Aswan RAG project"
git branch -M main
git remote add origin https://github.com/<اسم-المستخدم>/aswan-rag-project.git
git push -u origin main
```

إذا لم يكن لديك مستودع بعد: افتح GitHub → **New repository** → اختر اسماً مثل
`aswan-rag-project` → اتركه فارغاً بدون README (لأنه موجود بالفعل هنا) → أنشئه، ثم نفّذ
الأوامر أعلاه.

## 3) نشر التطبيق على Streamlit Community Cloud (رابط عام تشاركه مع الدكتور)

1. ادخل إلى https://share.streamlit.io وسجّل الدخول بحساب GitHub.
2. اضغط **New app**.
3. اختر المستودع الذي رفعته (`aswan-rag-project`) والفرع `main`.
4. في خانة **Main file path** اكتب: `app.py`.
5. (اختياري) لإضافة مفتاح Anthropic API بشكل آمن بدل كتابته يدوياً كل مرة، من إعدادات
   التطبيق افتح **Advanced settings → Secrets** وأضف:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-...."
   ```
   ثم عدّل في `app.py` سطر قراءة المفتاح ليأخذه تلقائياً من `st.secrets["ANTHROPIC_API_KEY"]`
   بدل حقل الإدخال اليدوي، إن رغبت في ذلك.
6. اضغط **Deploy**. بعد دقيقة أو دقيقتين سيصبح لديك رابط عام بالشكل:
   `https://<اسم-التطبيق>.streamlit.app`
   وهذا هو الرابط الذي تسلّمه للدكتور مع رابط مستودع GitHub.

## 4) ملخص المنهجية المستخدمة

- **الاسترجاع اللفظي:** TF-IDF و BM25 يعتمدان على تطابق الكلمات الحرفي.
- **الاسترجاع الدلالي:** نموذج `paraphrase-multilingual-MiniLM-L12-v2` يحوّل كل جملة
  إلى متجه يمثل معناها، فيتفوق عندما تختلف صياغة السؤال عن صياغة المستند.
- **الاسترجاع الهجين:** دمج الطريقتين بوزن `alpha` قابل للتعديل.
- **التقييم:** Precision@K, Recall@K, Hit Rate@K, MRR على 14 استعلاماً مصمماً يدوياً
  (منها استعلامات كلمات مفتاحية، وأسئلة طبيعية، واستعلامات بها عدم تطابق مفردات عمداً).
- **RAG:** بعد الاسترجاع، تُبنى حزمة سياق (Context Package) من أفضل النتائج، ثم تُصاغ
  كـ Prompt يُرسل إلى Claude لتوليد إجابة نهائية مقيّدة بالسياق فقط (لتقليل الاختلاق/Hallucination).

## المصادر

- scikit-learn, rank_bm25, sentence-transformers, Streamlit, Anthropic API — روابط التوثيق
  الكاملة موجودة في نهاية دفتر الملاحظات `Aswan_RAG_Project.ipynb`.
