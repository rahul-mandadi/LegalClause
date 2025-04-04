# LexCounsel: Intelligent Legal Clause Analyzer

## Overview

LexCounsel is an NLP-based tool designed to automate the understanding and risk assessment of contract clauses. It utilizes a fine-tuned Legal-BERT model for clause classification and leverages Gemini 2.0 Flash for detailed risk analysis. This repository serves as a starter project for the LexCounsel team to collaboratively integrate and enhance these components into a Streamlit-based application.

## Use Case and Importance

Legal professionals often need to review contract clauses for potential risks and implications. LexCounsel automates this process by classifying clauses (e.g., identifying audit clauses) and analyzing their risks, thereby enhancing productivity for lawyers, legal analysts, and compliance officers.

**Audit Clause**:  
An audit clause grants one party the right to inspect the other party’s records and operations to ensure compliance with contract terms. It is vital for transparency, risk management, and regulatory adherence.

**Importance**:  
- **Transparency**: Ensures adherence to contract terms.  
- **Risk Management**: Identifies discrepancies or risks early.  
- **Compliance**: Supports legal and regulatory standards.

## Technologies Used

- **Python**: Core programming language.
- **Hugging Face Transformers**: For fine-tuning Legal-BERT (`nlpaueb/legal-bert-base-uncased`).
- **Google Gemini 2.0 Flash**: For generating risk analysis.
- **Streamlit**: For the interactive web application.
- **Scikit-Learn**: For TF-IDF and SVM baseline modeling.
- **Matplotlib & Seaborn**: For evaluation visualizations.

## Project Structure

This starter project includes:  
- **`notebooks/BaselineModel.ipynb`**: Implements a TF-IDF + SVM baseline for clause classification.  
- **`notebooks/LegalBERTFineTuning.ipynb`**: Fine-tunes Legal-BERT for clause classification.  
- **`notebooks/RiskAnalysis.ipynb`**: Uses Gemini 2.0 Flash for risk analysis.  
- **`streamlit_app.py`**: Streamlit app integrating classification and risk analysis (to be updated with Gemini).

## Setup

1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/rahul-mandadi/LexCounsel.git
   cd LexCounsel
# Install Dependencies

```bash
pip install -r requirements.txt
```

# Set Up Gemini API Key

Create a `.env` file in the root directory:

```ini
GEMINI_API_KEY=your_api_key_here
```

> **Note:** The `.env` file is ignored by `.gitignore` to protect sensitive data.

# Running the Application

To launch the Streamlit app, run:

```bash
streamlit run streamlit_app.py
```

Enter a contract clause to receive classification and risk analysis outputs. *(Remember to update `streamlit_app.py` to integrate Gemini 2.0 Flash after testing in `RiskAnalysis.ipynb`.)*

# Model Development

- **Baseline Model:** Uses TF-IDF vectorization with an SVM, trained on the LegalBench CUAD dataset.
- **Legal-BERT:** Fine-tuned using `nlpaueb/legal-bert-base-uncased` for binary classification (e.g., audit clause: Yes/No).
- **Risk Analysis:** Leverages Gemini 2.0 Flash to analyze risks based on the clause text.

> **Note:** The current setup uses a simplified 5-fold cross-validation from previous experiments. Further refinements may be applied.

# Benchmark Results

| Metric     | Score (%) |
|------------|----------|
| Accuracy   | 99.23    |
| Precision  | 98.49    |
| Recall     | 100.00   |
| F1 Score   | 99.24    |

These results provide a baseline; actual outcomes may vary with updates to the SVM baseline and Gemini integration.

# Demo

In the Streamlit app, input a contract clause to view:

- **Clause Classification:** For example, *“Audit Clause: Yes.”*
- **Risk Assessment:** For example, *“Potential ambiguity in scope.”*


# Team

- **Raunaksingh Khalsa:** Data & Baseline Models *(refining `BaselineModel.ipynb`)*
- **Vishak Nair:** Transformer Fine-Tuning & Evaluation *(refining `LegalBERTFineTuning.ipynb`)*
- **Rahul Reddy Mandadi:** Risk Analysis & System Integration *(refining `RiskAnalysis.ipynb` and updating `streamlit_app.py`)*

# Contributing

1. Fork the repository.
2. Create a branch for your task *(e.g., `feature/baseline-model`)*.
3. Make your changes and test locally.
4. Submit a pull request to `main`.



# References

- **[LegalBench Dataset]**
- **[LegalBench Research Paper]**
- **[Hugging Face Transformers]**
- **[Google Gemini API]**
- **[Fine-Tuned Legal-BERT]**
