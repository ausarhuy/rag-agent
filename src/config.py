# Text processing config
CHUNK_SIZE = 450
CHUNK_OVERLAP = 50

# Backend (Langchain)

EMBEDDING_MODEL = "models/vietnamese-embedding"

GEMINI_MODEL = "gemini-1.5-flash"
GGUF_MODEL = "models/llms/ggml-vistral-7B-chat-q4_1.gguf"


GEMINI_MENU = "Google / Gemini 1.5"
VISTRAL_MENU = "Vietnamese / Vistral"

DEFAULT_MODEL = GEMINI_MENU
DEFAULT_MENU_CHOICE = 0
DEFAULT_TEMPERATURE = 0.2

LLM_CONFIG = {
    "context_length": "8000",
    "max_new_tokens": "512",
    "temperature": DEFAULT_TEMPERATURE,
    "repetition_penalty": 1.4,
    "reset": False,
    "stream": True
}

COLLECTION_NAME = "tdtu"  # Name of the collection in the vector DB
DOCSTORE_PATH = "./docstore"
CHROMA_PATH = "./chromadb"
DATA_PATH = "./data"

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
sinh viên liên quan đến quy chế tổ chức và quản lý đào tạo trình độ đại học của khoá tuyển sinh K21 trở về sau. 
Bạn phải trả lời bằng ngôn ngữ giống như câu hỏi với phong cách học đường. Đầu tiên hãy xác định ngôn ngữ nào là câu hỏi. 
Trước tiên bạn phải tìm kiếm câu trả lời trong "Cơ sở kiến thức". Nếu không tìm thấy câu trả lời trong phần "Cơ sở kiến 
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

HELLO_MESSAGE = "Xin chào 👋"
NEW_CHAT_MESSAGE = "Cuộc trò chuyện mới"
USER_PROMPT = "Nhập câu hỏi"

ABOUT_TEXT = """
### Thông tin về chatbot
Đây là chatbot hỏi đáp về các vấn đề liên quan đến quy chế tổ chức và quản lý đào tạo trình độ đại học Tôn Đức Thắng dành 
cho khoá tuyển sinh K21 trở về sau. 

Nguồn: [QUY CHẾ TỔ CHỨC VÀ QUẢN LÝ ĐÀO TẠO TRÌNH ĐỘ ĐẠI HỌC](https://undergrad.tdtu.edu.vn/hoc-vu/quy-che-chuc-va-quan-ly-dao-tao-trinh-do-dai-hoc-khoa-ts2021-tro-ve-sau)
#### Một số câu hỏi
- Tóm tắt Chương I về quy chế của trường.
- Điểm M là gì?
- Làm sao để nghỉ học tạm thời?
- Học bổng tại trường?

...
"""

SIDEBAR_FOOTER = """
_________
Hybrid RAG with memory powered by Langchain. Web interface powered by Streamlit.
"""
