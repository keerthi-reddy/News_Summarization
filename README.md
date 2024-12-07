# News Summarization and Simplification

Karthik Reddy Musku - G01446785
Keerthi Ramireddy - G01450961
Diwita Banerjee - G01455461

## NEWS SUMMARIZATION
This project focuses on developing a model to summarize news articles using Natural Language Processing (NLP) techniques. The goal is to generate concise, informative summaries that capture the most important aspects of the original article.


### Steps to run the code for summarization:
1. Install the packages using pip install -r requirements.txt
2. Run the preprocess data using python src/preprocess.py
3. Run the model building using python src/model.py
4. Run the summarize model using python src/main.py

## News Simplification Project

This project simplifies news articles to make them more accessible and readable for a broader audience, including non-native English speakers and individuals who prefer simplified text. The process uses a fine-tuned T5 model trained on the WikiAuto dataset.

### Features
- **Dataset**: Processed WikiAuto dataset for complex-to-simple sentence mappings.
- **Model**: Fine-tuned T5 model for effective text simplification.
- **Evaluation**: Measured performance using BLEU and Flesch-Kincaid Grade Level metrics.
- **Robustness Testing**: Evaluated model performance against noisy inputs, such as typos and lexical variations.
- **Error Analysis**: Logged and analyzed common errors for further optimization.

### Workflow
1. **Preprocessing**: Prepared and tokenized the WikiAuto dataset.
2. **Fine-Tuning**: Trained the T5 model with adjusted parameters and batch sizes.
3. **Evaluation**: Tested the model for BLEU scores and readability levels.
4. **Robustness Studies**: Evaluated model resilience under challenging scenarios.

### Dependencies
Install dependencies:
pip install -r requirements.txt

## Future Scope
- Integrate summarization and simplification into a unified pipeline.
- Extend multilingual support for broader usability.
- Expand robustness studies to cover domain-specific jargon and complex inputs.

## Usage
1. Clone the repository:
   ```bash
   git clone <repository-url>
