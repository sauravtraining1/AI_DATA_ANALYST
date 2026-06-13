from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from main import get_sql_query_from_user_input
import whisper
import tempfile
import ssl
import certifi


ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)


st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

model = load_whisper()

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_model()


if "cache" not in st.session_state:
    st.session_state.cache = []


def save_to_cache(user_query, result):
    embedding = embed_model.encode(user_query)

    st.session_state.cache.append({
        "question": user_query,
        "embedding": embedding,
        "result": result
    })

def get_cached_result(user_input,threshold=0.90):
    query_embedding = embed_model.encode(user_input)

    for item in st.session_state.cache:

        similarity = cosine_similarity(
            [query_embedding],
            [item["embedding"]]
        )[0][0]

        if similarity >= threshold:
            return item["result"]

    return None

def speech_to_text(audio):

    audio_bytes = audio.get("bytes")

    if not audio_bytes:
        return ""

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as tmp:

        tmp.write(audio_bytes)
        audio_path = tmp.name

    result = model.transcribe(audio_path)

    return result["text"].strip()

st.title("📊 AI Data Analyst")
st.caption("Ask questions about your database using text or voice")


st.markdown("""
<style>
div[data-testid="stHorizontalBlock"] {
    align-items: center;
}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([10,1])

with col1:
    user_query = st.chat_input("Ask anything about your database...")

with col2:
    audio = mic_recorder(
        start_prompt="🎤",
        stop_prompt="⏹️",
        key="mic"
    )


voice_query = None

if audio:

    try:
        voice_query = speech_to_text(audio)
        print(voice_query)

        if voice_query is not None:

            with st.spinner("Analyzing your query..."):
                cached = get_cached_result(voice_query)
                print("test1")

                if cached:
                    print(cached)
                    st.success(f"Loaded from cache (similar to: {cached['question']})")
            
                    st.dataframe(cached["result"],use_container_width=True)
            

                else:
                    database_response = get_sql_query_from_user_input(voice_query)
            
                    save_to_cache(voice_query, database_response)

                    for item in st.session_state.cache:
                        st.markdown(f"**Question:** {item['question']}")
                        st.dataframe(item["result"], use_container_width=True)

            

    except Exception as e:
        st.error(f"Speech recognition failed: {e}")



if user_query:
    with st.spinner("Analyzing your query..."):

        cached = get_cached_result(user_query)

        if cached:
            st.success(
            f"Loaded from cache (similar to: {cached['question']})"
            )
            
            st.dataframe(
            cached["result"],
            use_container_width=True
            )
        else:
            database_response = get_sql_query_from_user_input(user_query)
            
            save_to_cache(user_query, database_response)

            for item in st.session_state.cache:
                st.markdown(f"**Question:** {item['question']}")
                st.dataframe(item["result"], use_container_width=True)

   # st.dataframe(database_response, use_container_width=True