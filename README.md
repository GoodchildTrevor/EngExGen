# English_Exercise_Generator

## Overview
English Exercise Generator is a Python-based tool that generates English language exercises based on user input text. Using AI and NLP libraries, it can generate a variety of exercises such as sentence selection, word completion, noun phrase identification, word selection, and verb form selection. The aim of the tool is to make learning English more interactive and personalized.

## Features
* Supports different types of exercises for comprehensive learning.
* The input text can be any English paragraph or article, provided it does not exceed 30 sentences.
* Exercises are randomly generated to ensure variety in learning.
* Includes checks for the language of input text, ensuring exercises are only generated from English sentences.
* Outputs include the exercise question, multiple options for answers (where applicable), and the correct answer.

# Dependencies
* Gensim
* Pandas
* NumPy
* re
* Streamlit
* Sentence_Splitter
* langdetect
* Random
* Spacy
* en_core_web_sm
* PyInflect
  
# How to use
* Run the application. A text box will be displayed in the Streamlit application interface.
* Enter your text (paragraph or article) into the text box and submit.
* The application will generate a series of exercises based on your input text.
* Answer the exercises. Immediate feedback on your answers will be provided.
* Continue learning!

# Notes
* Make sure the text you input is in English and doesn't exceed 30 sentences.
* Ensure all the required libraries are installed and imported.
