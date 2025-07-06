# 🏦 Swiss Bank AI/NLP Job Q&A

Q&A chatbot based on Swiss bank AI/NLP scientist experience

## Features
- Answers questions about AI/NLP scientist positions at Swiss banks
- Provides career guidance and job preparation tips
- Shows source pages for verification

## 🚀 Get Started

**🔗 [Swiss Bank AI/NLP Job Q&A](https://swiss-bank-ai-nlp-job-assistant.streamlit.app)**

1. Click the link above
2. Enter your OpenAI API key
3. Ask questions and get answers

---

## 💻 For Developers

### Local Setup
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Streamlit Cloud Deployment
This app is optimized for Streamlit Cloud deployment with the following improvements:
- Optimized PDF processing with smaller chunk sizes
- Error handling and progress indicators
- Memory-efficient document processing
- Cached resource loading

### Troubleshooting Streamlit Cloud Issues

If your app shows "your app is in the oven" for extended periods:

1. **Check requirements.txt**: Ensure all dependencies have specific versions
2. **PDF size**: Large PDFs (>5MB) may cause timeout issues
3. **Memory limits**: The app is optimized to handle memory constraints
4. **Dependencies**: All llama-index packages have specific versions to avoid conflicts

## Environment Variables
- OpenAI API key required at runtime

## Notes
- Please respect copyright for PDF files
- Answer quality depends on model and embedding quality
- For large documents, processing may take time on first load