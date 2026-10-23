Sentiment Analysis of Internship Feedback

An end-to-end Natural Language Processing (NLP) pipeline designed to analyze intern feedback reviews, classify sentiments (Positive, Neutral, Negative), and uncover actionable insights to improve overall intern satisfaction, onboarding, and mentorship structures.

📌 Project Overview

Internship programs generate a wealth of qualitative feedback through reviews and evaluations. Manually reading through hundreds of feedback submissions is inefficient. This project automates the sentiment analysis process using two approaches:

Baseline Machine Learning: TF-IDF Vectorization combined with a multi-class Logistic Regression classifier.

Deep Learning / Transformer: Pre-trained Hugging Face transformer models (pipeline('sentiment-analysis')) for zero-shot and advanced contextual sentiment scoring.

🚀 Getting Started & Installation

Prerequisites

Make sure you have Python 3.8+ installed along with Jupyter Notebook.

1. Install Dependencies

Run the following command in your terminal or inside a notebook cell to install all required libraries (including scikit-learn, transformers, and PyTorch):

pip install pandas numpy matplotlib seaborn scikit-learn transformers torch

2. Running the Notebook

Open Jupyter Notebook in your project directory:

jupyter notebook


Open sentiment_analysis_internship.ipynb.

Run the cells sequentially from top to bottom.

📊 Pipeline Steps

Data Loading & Cleaning: Loads intern_feedback_data.csv, removes null values and duplicate records, and normalizes review text.

Exploratory Data Analysis (EDA): Generates count plots visualizing the distribution of Positive, Neutral, and Negative intern sentiments.

Text Vectorization: Converts raw textual feedback into numerical feature matrices using TfidfVectorizer with English stop-word removal.

Model Training & Evaluation: Trains a Logistic Regression model and evaluates performance using accuracy, precision, recall, and classification reports.

Transformer Integration: Utilizes Hugging Face pipelines to classify reviews using state-of-the-art deep learning.

Actionable Insights Extraction: Filters negative feedback entries to highlight pain points in mentorship, task distribution, and workload management for program coordinators.

📈 Evaluation Summary

Classification Performance: The TF-IDF + Logistic Regression model achieves robust accuracy and high precision in distinguishing between positive praise and constructive/negative feedback.

Key Areas for Improvement: Isolating negative sentiments allows HR and internship managers to quickly identify systemic issues (e.g., unclear expectations, lack of technical support) and implement corrective measures.
