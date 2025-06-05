from google import genai
import google.genai.types as genai_types
import requests
import urllib3
import os
import ssl
from telebot import telebot, types
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

login(token=os.getenv('HUGGINGFACE_TOKEN'))

# Configuration and Environment Variables
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
SUPADATA_API_KEY = os.getenv('SUPADATA_API_KEY')
GEMINI_API_TOKEN = os.getenv('GEMINI_API_TOKEN')


bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, threaded=True, num_threads=4)
chroma_client = chromadb.Client()

labse_embedding_function  = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='sentence-transformers/LaBSE'
)
labse_model = SentenceTransformer('sentence-transformers/LaBSE')

collection = chroma_client.get_or_create_collection(
    name="youtube_transcripts",
    embedding_function=labse_embedding_function
)


# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# proxies = {
#     'http': os.getenv('HTTP_PROXY'),
#     'https': os.getenv('HTTP_PROXY')
# }

# os.environ['HTTP_PROXY'] = proxies['http']
# os.environ['HTTPS_PROXY'] = proxies['https']

# def _create_unverified_https_context():
#     try:
#         _create_unverified_https_context = ssl._create_unverified_context
#     except AttributeError:
#         pass
#     else:
#         ssl._create_default_https_context = _create_unverified_https_context

# def configure_requests_with_proxy():
#     requests.Session().verify = False
#     requests.Session().proxies.update(proxies)


# configure_requests_with_proxy()
# _create_unverified_https_context()


client = genai.Client(api_key=GEMINI_API_TOKEN)


st.title("🎥 YouTube AI Assistant")

st.markdown("""
    <style>
    body {
        direction: rtl;
        text-align: right;
        font-family: "Vazirmatn", "Tahoma", sans-serif;
    }
    .stTextInput > div > div > input {
        direction: rtl;
        text-align: right;
    }
    .stTextArea > div > textarea {
        direction: rtl;
        text-align: right;
    }
    .stSelectbox > div > div {
        direction: rtl;
        text-align: right;
    }
    </style>
""", unsafe_allow_html=True)

action = st.selectbox("چه کاری می‌خواهید انجام دهید؟", ["خلاصه ویدئو", "ترجمه به فارسی"])
youtube_url = st.text_input("لینک ویدئو یوتیوب را وارد کنید:")

if st.button("پردازش ویدئو"):
    if youtube_url:
        st.info("در حال استخراج متن ویدئو ...")
        api_url = f'https://api.supadata.ai/v1/youtube/transcript?url={youtube_url}&text=true'
        headers = {'x-api-key': SUPADATA_API_KEY}

        try:
            response = requests.get(api_url, headers=headers, verify=False)
            response.raise_for_status()
            content = response.json()['content']

            if not content:
                st.error("متاسفانه متن ویدئو پیدا نشد.")
            else:
                sys_instruct = 'این متن را به فارسی روان ترجمه کن.' if action == 'ترجمه به فارسی' else 'این متن را به فارسی خلاصه کن.'
                gemini_response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    config=genai_types.GenerateContentConfig(system_instruction=sys_instruct),
                    contents=[content]
                )
                result_text = gemini_response.text

                # Save to knowledge base
                embedding = labse_model.encode(result_text).tolist()
                doc_id = str(hash(result_text))[:10]
                collection.add(
                    documents=[result_text],
                    embeddings=[embedding],
                    metadatas=[{"video_url": youtube_url}],
                    ids=[doc_id]
                )
                st.success("✅ ویدئو پردازش شد و به پایگاه دانش اضافه شد!")
                st.write(result_text)
        except Exception as e:
            st.error(f"خطا: {str(e)}")
    else:
        st.warning("لطفاً لینک را وارد کنید.")

# Query section
st.header("❓ سوال از پایگاه دانش")
query = st.text_input("سوال خود را اینجا بنویسید:")

if st.button("جستجو و پاسخ"):
    if query:
        st.info("در حال جستجو ...")
        try:
            query_embedding = labse_model.encode(query).tolist()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=2,
                include=["documents", "metadatas"]
            )

            if not results["documents"][0]:
                st.warning("اطلاعاتی یافت نشد.")
            else:
                context = "\n\n".join([doc for doc in results["documents"][0]])
                prompt = f'''
                با استفاده از اطلاعات زیر به سوال پاسخ دهید:

                {context}

                سوال: {query}

                پاسخ را حداکثر در 3 جمله به فارسی ارائه دهید.
                '''
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[prompt]
                )
                st.success("✅ پاسخ:")
                st.write(response.text)
        except Exception as e:
            st.error(f"خطا: {str(e)}")
    else:
        st.warning("لطفاً یک سوال وارد کنید.")
        

