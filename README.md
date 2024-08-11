## Create Python 3.10.x env
```
conda create -n rag-agent python=3.10 # For conda venv
python -m venv rag-agent # Python venv
```
# Install library
```
pip install -r requirements.txt
```
## Download Embedding model
```
huggingface-cli download dangvantuan/vietnamese-embedding --exclude "*.bin" --local-dir ./models/vietnamese-embedding
```
### Optional: Download vistral gguf quantized model (currently not work in this app) and save it in /models/llms/gguf/
[Vistral GGLM hf space](https://huggingface.co/uonlp/Vistral-7B-Chat-gguf/tree/main)
## Get Google Gemini API key: [Google AI Studio](https://aistudio.google.com/app/apikey)
## Create .env file
```
GOOGLE_API_KEY="<GOOGLE_API_KEY>"
ADMIN_PASSWORD="<Enter Admin page password>"
```
## Run app
```
bash app.sh start|stop|restart
```