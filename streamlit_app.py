import streamlit as st
import torch
import google.generativeai as genai
from transformers import BertTokenizer, BertForSequenceClassification
from streamlit_chat import message

# Load API key from secrets (set in Streamlit secrets.toml or environment)
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except KeyError:
    st.error("Please set GOOGLE_API_KEY in Streamlit secrets or environment.")
    st.stop()

# Load Legal-BERT model
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("RahulMandadi/fine-tuned-legal-bert")
    tokenizer = BertTokenizer.from_pretrained("RahulMandadi/fine-tuned-legal-bert")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_model()

# Class labels
labels = ["Cap on Liability", "Audit Rights", "Insurance", "None"]

# Legal-BERT classification
def classify_clause_legal_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    prediction = torch.argmax(logits, dim=-1).item()
    return labels[prediction], probs[prediction]

# Gemini risk analysis
def run_risk_analysis_gemini(clause):
    try:
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')  # Verify model name
        prompt = f"You are a legal advisor. Identify 2-3 key risks and 1 mitigation for this clause: '{clause}'"
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error analyzing clause: {str(e)}"

# Combined analysis
def classify_and_analyze_clause(clause):
    # Classify with Legal-BERT
    bert_label, bert_conf = classify_clause_legal_bert(clause)
    
    # Risk analysis
    risk_analysis = run_risk_analysis_gemini(clause)
    
    # Format response
    response = (
        f"### Clause Analysis\n\n"
        f"**Input Clause**: '{clause}'\n\n"
        f"#### Classification\n"
        f"- **Legal-BERT**: {bert_label} (Confidence: {bert_conf:.2%})\n\n"
        f"#### Risk Analysis (Gemini)\n{risk_analysis}"
    )
    return response

# Streamlit app
st.title("LexCounsel: Contract Clause Analysis")

# Instructions
st.markdown("Enter a contract clause to classify it and assess risks using Legal-BERT and Gemini.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
if st.session_state.messages:
    for i, chat in enumerate(st.session_state.messages):
        message(chat['question'], is_user=True, key=f"user_{i}", avatar_style="big-smile")
        message(chat['answer'], key=f"bot_{i}")
else:
    st.markdown("No chat history yet. Start by entering a clause below.")

# User input
user_input = st.chat_input(placeholder="Enter a contract clause...")

if user_input:
    with st.spinner('Analyzing your clause...'):
        response = classify_and_analyze_clause(user_input)
    st.session_state.messages.append({"question": user_input, "answer": response})
    st.rerun()