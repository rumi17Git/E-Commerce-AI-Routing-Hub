# E-Commerce AI Routing Hub

A multilingual customer support routing system combining a fine-tuned DistilBERT intent classifier with MarianMT translation middleware, built on Amazon customer support interactions filtered from the [Twitter Customer Support Dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).

**[Live Interactive Demo on Hugging Face Spaces](https://huggingface.co/spaces/RummanJ17/ecommerce-ai-routing-hub)**

---

## Results

Fine-tuned on 80% of the dataset, evaluated on the held-out 20% (320 samples, balanced across 4 classes).

| Intent Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| WHERE_IS_MY_ORDER | 0.97 | 0.90 | 0.94 | 83 |
| REFUND_REQUEST | 0.94 | 0.98 | 0.96 | 84 |
| PRODUCT_FEEDBACK | 0.93 | 0.97 | 0.95 | 64 |
| CANCEL_ORDER | 1.00 | 1.00 | 1.00 | 89 |
| **Overall (weighted avg)** | **0.96** | **0.96** | **0.96** | **320** |

Training converged cleanly over 3 epochs with no signs of overfitting (validation loss: 0.65 → 0.19 → 0.14).

---

## System Architecture

Incoming customer tickets are handled asynchronously across language boundaries. The system decides whether to serve a deterministic templated response or escalate to a human reviewer based on the classifier's confidence score.

```text
          [ User Query (EN or FR) ]
                       │
                       ▼
          [ Language Detection Layer ]
                       │
        ┌──────────────┴──────────────┐
 Detected: English             Detected: French
        │                             │
        │                             ▼
        │                  [ MarianMT Translation ]
        │                        (FR → EN)
        ▼                             │
 ┌──────────────────────────────────────┐
 │   DistilBERT Intent Classification   │
 └──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
 Confidence ≥ 72%?             Confidence < 72%?
        │                             │
        ▼                             ▼
 [ Static Response ]         [ Fallback GenAI Draft ]
(Deterministic Match)       (Human-in-the-Loop Flagged)
        │                             │
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
 Original Query French?      Original Query English?
        │                             │
        ▼                             ▼
 [ MarianMT Translation ]     [ Final Response ]
       (EN → FR)
        │
        ▼
 [ Final Response ]
```

---

## Core Features

- **Fine-tuned intent classifier:** DistilBERT fine-tuned for sequence classification across four e-commerce intents, achieving 96% accuracy and 0.96 weighted F1 on the held-out test set.
- **Multilingual middleware:** MarianMT (Helsinki-NLP) with SentencePiece tokenization allows the English-trained classifier to handle French-language input and return responses in the original query language.
- **Confidence-based routing:** A calibrated 72% confidence threshold routes high-confidence queries to deterministic templated responses and flags low-confidence or edge-case queries for human review.
- **Dual-pane telemetry dashboard:** A Gradio interface separates customer-facing query simulation from agent-side telemetry, including confidence scores and routing decisions.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-green)
![Gradio](https://img.shields.io/badge/Gradio-5.x-purple)

---

## Project Structure

```text
├── app/
│   ├── main.py        # FastAPI backend, translation tensors, confidence thresholding
│   └── app.py         # Gradio dashboard layout and alert renderers
├── requirements.txt   # Core dependencies
└── .gitignore
```

---

## How to Run Locally

```bash
git clone https://github.com/RummanJ17/ecommerce-ai-routing-hub
cd ecommerce-ai-routing-hub
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open `http://localhost:8000` in your browser, or launch the Gradio interface directly:

```bash
python app/app.py
```

---

## Limitations

- **Domain specificity:** The classifier was fine-tuned on Twitter-style customer support text (short, informal messages). Performance may degrade on formal, technical, or highly domain-specific language outside the training distribution.
- **Language support:** The translation middleware currently supports English and French only. Extending to additional languages would require additional MarianMT model pairs.
- **Confidence threshold:** The 72% routing threshold was tuned empirically on this dataset. Deployment in a different customer support context would require recalibration against domain-specific validation data.
- **Intent coverage:** The system handles four intents. Queries outside these categories (billing disputes, account access, general enquiries) will be routed to the human-review fallback regardless of confidence score.

---

## Dataset

[Twitter Customer Support Dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) — filtered to Amazon customer support interactions for e-commerce relevance.
