import os
import streamlit as st
import pickle
from io import BytesIO
from gtts import gTTS
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# ✅ ✅ ✅ Text-to-Speech (gTTS + Download Button)
def text_to_speech_gtts(text):
    tts = gTTS(text, lang='en')
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp

# ✅ Load environment variables (.env)
load_dotenv()

# ✅ Initialize LLM (Groq API)
llm = ChatOpenAI(
    temperature=0.9,
    max_tokens=500,
    model="llama3-70b-8192",
    openai_api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# ✅ Streamlit UI Setup
st.set_page_config(page_title="📰 News Research Tool", layout="wide")
st.title("📰 News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Sidebar Inputs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("🚀 Process URLs")
file_path = "faiss_store.pkl"
main_placeholder = st.empty()

# ✅ Process URLs
if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.info("🚀 Loading articles...")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
    main_placeholder.info("🔗 Splitting text into chunks...")
    docs = text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    main_placeholder.info("🔎 Building vector embeddings...")

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

    st.success("✅ Articles processed and stored successfully!")

# ✅ Search Query
query = st.text_input("🔍 Enter your research question:")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            st.header("📋 Answer:")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("📚 Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
    else:
        st.error("❌ No processed articles found. Please process URLs first.")

# ✅ ✅ ✅ Summarization Feature (with Downloadable Audio)
# ✅ Initialize Session State Safely
if "summary_text" not in st.session_state:
    st.session_state["summary_text"] = None
if "audio_file" not in st.session_state:
    st.session_state["audio_file"] = None

st.markdown("---")
st.subheader("📝 Summarize All Loaded Articles")

if st.button("Generate Summary"):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            docs = vectorstore.similarity_search("summary", k=10)
            combined_text = "\n\n".join([doc.page_content for doc in docs])

            prompt = PromptTemplate(
                input_variables=["text"],
                template="""
You are a financial news analyst. Please summarize the following news articles in **5 bullet points**,
focusing on **finance, stocks, and market trends**:
{text}
"""
            )
            final_prompt = prompt.format(text=combined_text)

            with st.spinner("Generating summary..."):
                summary = llm.invoke(final_prompt)

            st.session_state["summary_text"] = summary.content
            st.session_state["audio_file"] = text_to_speech_gtts(summary.content)
    else:
        st.error("❌ Please process URLs before summarization.")

# ✅ Show Summary & Download Only If Available
if st.session_state["summary_text"] and st.session_state["audio_file"]:
    st.subheader("📋 Summary of Articles:")
    st.write(st.session_state["summary_text"])

    st.download_button(
        label="⬇️ Download Summary Audio (MP3)",
        data=st.session_state["audio_file"],
        file_name="summary.mp3",
        mime="audio/mpeg"
    )


# ✅ Footer
st.markdown("---")
st.caption("🚀 Built with ❤️ using Groq, FAISS, HuggingFace, gTTS & Streamlit")