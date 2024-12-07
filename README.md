# News Summarization and Simplification

Karthik Reddy Musku - G01446785
Keerthi Ramireddy - G01450961
Diwita Banerjee - G01455461

## NEWS SUMMARIZATION
This project focuses on developing a model to summarize news articles using Natural Language Processing (NLP) techniques. The goal is to generate concise, informative summaries that capture the most important aspects of the original article.

## DATASET - CNN/DailyMail 
To download the CNN/Daily Mail dataset for newspaper text summarization, you can use the following script:

```bash
#!/bin/bash
curl -L -o ~/Downloads/newspaper-text-summarization-cnn-dailymail.zip\
  https://www.kaggle.com/api/v1/datasets/download/gowrishankarp/newspaper-text-summarization-cnn-dailymail
```


### Steps to run the code for summarization:
1. Install the packages using pip install -r News_Summarization/requirements.txt
2. Run the preprocess data using python News_Summarization/preprocess.py
3. Run the model building using python News_Summarization/model.py
4. Run the summarize model using python News_Summarization/main.py

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
```bash
pip install -r requirements.txt
```

## Future Scope
- Integrate summarization and simplification into a unified pipeline.
- Extend multilingual support for broader usability.
- Expand robustness studies to cover domain-specific jargon and complex inputs.

## Usage
Clone the repository:
```bash
git clone <repository-url>
```
