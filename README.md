#  Emotion Prediction using BiGRU

An end-to-end Deep Learning project that classifies text input into fine-grained emotional categories using a Bidirectional Gated Recurrent Unit (BiGRU) architecture built with TensorFlow/Keras.

---

##  Project Overview
This project uses Natural Language Processing (NLP) to process text sequences and analyze emotional sentiment. The model leverages bidirectional temporal feature extraction to understand complex textual semantics effectively.

* **Model Architecture:** Bidirectional GRU (BiGRU)
* **Frameworks:** TensorFlow, Keras, NumPy, Scikit-learn
* **Preprocessing:** Keras Tokenizer, Sequence Padding, Class Weight Balancing

---

##  Repository Structure
```text
├── Artifacts/
│   ├── BiGRU_model.keras   # Trained Keras model
│   └── tokenizer.pkl       # Fitted Tokenizer instance
├── main.py                  # Main inference script
├── requirements.txt        # Python dependencies
└── README.md                # Project documentation

 Getting Started
1. Prerequisites

Ensure you have Python installed on your system. Install the required dependencies:

Bash
pip install tensorflow numpy scikit-learn