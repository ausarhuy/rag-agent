# Text processing config
CACHE_THRESHOLD = 0.7
CHUNK_SIZE = 450
CHUNK_OVERLAP = 50

# Backend (Langchain)

EMBEDDING_MODEL = "models/vietnamese-embedding"

GEMINI_MODEL = "gemini-1.5-flash"
GGUF_MODEL = "models/llms/ggml-llms-7B-chat-q8.gguf"

LLM_CONFIG = {
    "context_length": "4096",
    "max_new_tokens": "512",
    "gpu_layers": 50,
    "temperature": 0.5,
    "repetition_penalty": 1.4,
    "reset": False,
    "stream": True
}

GEMINI_MENU = "Google / Gemini 1.5"
VISTRAL_MENU = "Vietnamese / Vistral"

DEFAULT_MODEL = GEMINI_MENU
DEFAULT_MENU_CHOICE = 0
DEFAULT_TEMPERATURE = 0.2

COLLECTION_NAME = "tdtu"  # Name of the collection in the vector DB
# SQLite utils
CONNECTION_STRING = "data.sql"

VECTORDB_MAX_RESULTS = 10
BM25_MAX_RESULTS = 10

MAX_MESSAGES_IN_MEMORY = 5

CONTEXTUALIZE_PROMPT = """Dựa vào lịch sử trò chuyện và câu hỏi mới nhất của người dùng \
có thể tham chiếu ngữ cảnh trong lịch sử trò chuyện, tạo thành một câu hỏi độc lập có thể là \
được hiểu mà không cần lịch sử trò chuyện. KHÔNG trả lời câu hỏi, chỉ sửa lại câu hỏi nếu cần \
và nếu không thì trả lại như cũ.

Lịch sử trò chuyện:

{chat_history}"""

# This system prompt is used with models other than OpenAI
SYSTEM_PROMPT = """Bạn là trợ thủ đắc lực của Đại học Tôn Đức Thắng, nhiệm vụ của bạn là trả lời những câu hỏi của 
sinh viên. Bạn phải trả lời bằng ngôn ngữ giống như câu hỏi. Đầu tiên hãy xác định ngôn ngữ nào là câu hỏi. Trước 
tiên bạn phải tìm kiếm câu trả lời trong "Cơ sở kiến thức". Nếu không tìm thấy câu trả lời trong phần "Cơ sở kiến 
thức”, sau đó trả lời bằng kiến thức của mình.

Cơ sở kiến thức:

{context}

Lịch sử trò chuyện:

{chat_history}
"""

# Frontend (Streamlit)

LOGO_PATH = "./images/tdtu_logo.png"
ASSISTANT_ICON = "👑"
ASSISTANT_NAME = "TDTU Chatbot"

HELLO_MESSAGE = "Hello! Xin chào! 👋"
NEW_CHAT_MESSAGE = "New chat / Cuộc trò chuyện mới"
USER_PROMPT = "Enter your question / Nhập câu hỏi"

ABOUT_TEXT = """
### About this assistant


### Về chatbot


#### Examples of questions you can ask

- When did King Leopold I die? Do you have any images of the funeral?
- Do you have any images of Queen Elizabeth during the First World War?
- Can you show me the canvas "The school review"? *And then you can ask the question:*
- Who painted this canvas? *And then again:*
- What is the size of the canvas? *And then again:*

"""

SIDEBAR_FOOTER = """
_________
Hybrid RAG with memory powered by Langchain. Web interface powered by Streamlit.
"""
