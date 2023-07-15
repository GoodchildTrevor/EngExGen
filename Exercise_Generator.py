import gensim.downloader as api
import pandas as pd
import numpy as np
import re
import streamlit as st
from sentence_splitter import SentenceSplitter
from langdetect import detect
import random
import spacy
import en_core_web_sm
import pyinflect

random.seed(12345)

types = ['select_sent', 'missing_word', 'noun_phrases',  'select_word', 'verb_form']
description = ['Какое предложение верно?', 'Какое слово пропущено?', 'Чем является выделенная фраза/слово?',
               'Выберете слово', 'Выберите верную форму глагола']

dependencies = ['predet', 'ROOT', 'amod', 'nsubj', 'pobj', 'dobj', 'ccomp']
dependencies_full = ['subject', 'predicate', 'adjectival modifier', 'nominal subject',
                     'object of a preposition', 'direct object', 'clausal complement']

model = api.load("glove-wiki-gigaword-100")

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print('Downloading language model for the first time.')
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

st.header('Генератор упражнений по английскому')
st.subheader('Вставьте текст для создания упражнений (не больше 30 предложений)')

splitter = SentenceSplitter(language='en')


def check_language(sentence):
    try:
        if detect(sentence) == 'en':
            return True
    except:
        return False


def process_text(raw_text):
    proc_sentences = splitter.split(text=raw_text)
    if len(proc_sentences) > 30:
        st.warning(f'В вашем тексте не должно быть больше 30 предложений. Сейчас их {len(proc_sentences)}, введите текст снова.')
        return
    else:
        return proc_sentences


def language_checking(raw_sentences):
    has_english = False
    for sentence in raw_sentences:
        if check_language(sentence) == True:
            has_english = True
    if has_english == False:
        st.warning('Ваш текст должен содержать хотя бы одно предложение на английском языке.')
    return has_english


def random_sentence(input_sentence):
    similar_sentences = []
    similar_sentences.append(input_sentence)
    for snt in range(0, 3):
        tokens = input_sentence
        similar_sentence = ''
        for token in tokens:
            if token.text in model.key_to_index:
                if token.pos_ in ['NOUN', 'VERB', 'ADV', 'ADJ']:
                    sim_words = model.similar_by_word(token.text)
                    similar_sentence += random.choice(sim_words[:5])[0] + token.whitespace_
                else:
                    similar_sentence += token.text + token.whitespace_
            else:
                similar_sentence += token.text + token.whitespace_

        similar_sentence = re.sub(r'\s*,', ',', similar_sentence)
        similar_sentences.append(similar_sentence)
        random.shuffle(similar_sentences)

    return similar_sentences


def random_words(original_word):

    similar_words = model.similar_by_word(original_word)
    selected_words = [word[0] for word in similar_words[:5] if word[0][0].isalpha()]
    selected_words = random.sample(selected_words, 3)
    selected_words.append(original_word)
    random.shuffle(selected_words)

    return selected_words


def deps(input_sentence):

    tokens_in_dependencies = []
    words_in_dependencies = []

    for token in input_sentence:
        if token.dep_ in dependencies:
            if token.text not in words_in_dependencies:
                words_in_dependencies.append(token.text)
                tokens_in_dependencies.append(token.dep_)

    random_wrd_cnstr = random.choice(words_in_dependencies)
    random_dep = tokens_in_dependencies[words_in_dependencies.index(random_wrd_cnstr)]
    random_right_dep = dependencies_full[dependencies.index(random_dep)]

    options = random.sample([dep for dep in dependencies_full if dep != random_right_dep], 3)

    return random_wrd_cnstr, random_right_dep, options


def define_random_word(words):

    while True:
        word = random.choice(words)
        if word in model and len(word) > 3 and word not in df['answer']:
            break
    return word


def replace_random_word(r_word, sentence):
    replaced_sentence = sentence.replace(f' {r_word}', " _____ ")
    return replaced_sentence


text = st.text_area('Введите текст', value='', key='text_input', label_visibility="hidden")
sentences = splitter.split(text=text)
if text:
    checking = language_checking(sentences)

    if len(sentences) > 30 or checking == False:
        process_text(text)

    else:
        df = pd.DataFrame(sentences, columns=['sentence'])
        df = df[df['sentence'].str.strip() != ''].reset_index()
        df['type'] = np.nan
        df['description'] = np.nan
        df['object'] = np.nan
        df['options'] = np.nan
        df['answer'] = np.nan

        for index, row in df.iterrows():
            filtered_words = []
            list_of_words = []
            random_word = ''
            doc = nlp(str(row['sentence']))
            if len(doc) > 8 and check_language(row['sentence']) == True:
                if len(doc) < 25:
                    task_type = random.randint(0, 3)
                else:
                    task_type = random.randint(1, 3)
                filtered_words = [token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADV', 'ADJ', 'AUX']]
                if task_type == 3:
                    random_word = define_random_word(filtered_words)
                    token = nlp(random_word)[0]
                    if token.pos_ == 'VERB':
                        list_of_words.extend([token._.inflect('VBP'), token._.inflect('VBZ'), token._.inflect('VBG'),
                                              token._.inflect('VBD')])
                        df.loc[index, 'sentence'] = replace_random_word(random_word, row['sentence'])
                        df.loc[index, 'type'] = types[task_type + 1]
                        df.loc[index, 'description'] = description[task_type + 1]
                        df.loc[index, 'object'] = random_word
                        df.loc[index, 'options'] = '//'.join(str(word) for word in list_of_words)
                        df.loc[index, 'answer'] = token._.inflect(token.tag_)
                    else:
                        list_of_words = random_words(random_word)
                        df.loc[index, 'sentence'] = replace_random_word(random_word, row['sentence'])
                        df.loc[index, 'type'] = types[task_type]
                        df.loc[index, 'description'] = description[task_type]
                        df.loc[index, 'object'] = random_word
                        df.loc[index, 'options'] = '//'.join(list_of_words)
                        df.loc[index, 'answer'] = random_word
                if task_type == 0:
                    sentences = random_sentence(doc)
                    df.loc[index, 'sentence'] = '__________'
                    df.loc[index, 'type'] = types[task_type]
                    df.loc[index, 'description'] = description[task_type]
                    df.loc[index, 'object'] = row['sentence']
                    df.loc[index, 'options'] = '//'.join(str(sentence) for sentence in sentences)
                    df.loc[index, 'answer'] = row['sentence']
                if task_type == 1:
                    random_word = define_random_word(filtered_words)
                    df.loc[index, 'sentence'] = replace_random_word(random_word, row['sentence'])
                    df.loc[index, 'type'] = types[task_type]
                    df.loc[index, 'description'] = description[task_type]
                    df.loc[index, 'object'] = random_word
                    df.loc[index, 'options'] = np.nan
                    df.loc[index, 'answer'] = random_word
                if task_type == 2:
                    wrd_cnstr, right_dep, other_deps = deps(doc)
                    other_deps.append(right_dep)
                    random.shuffle(other_deps)
                    df.loc[index, 'sentence'] = df.loc[index, 'sentence'].replace(str(wrd_cnstr), f' [{wrd_cnstr}] ')
                    df.loc[index, 'type'] = types[task_type]
                    df.loc[index, 'description'] = description[task_type]
                    df.loc[index, 'object'] = wrd_cnstr
                    df.loc[index, 'options'] = '//'.join(other_deps)
                    df.loc[index, 'answer'] = right_dep

        df['result'] = np.nan

        col = st.columns(1)

        counter = 1

        with col[0]:
            for i in range(len(df)):
                st.write(
                    f"<h3 style='font-size: 22px; text-align: left; margin-left: 0; font-weight: normal;'>{df.loc[i, 'sentence']}</h3>",
                    unsafe_allow_html=True)
                if isinstance(df.loc[i, 'answer'], str) and df.loc[i, 'answer'] != np.nan:
                    st.write(f"Задание {counter}")
                    counter += 1
                    st.markdown(
                        f"<h2 style='font-size: 16px; text-align: left; margin-left: 0;'>{df.loc[i, 'description']}</h2>",
                        unsafe_allow_html=True)
                option = str(df.loc[i, 'options'])
                values = option.split('//')

                if isinstance(df.loc[i, 'options'], str) and not df.loc[i, 'type'] == 'missing_word':
                    df.loc[i, 'result'] = st.selectbox(label='Вопрос', options=['–––'] + values, key=f"selectbox_{i}",
                                                       label_visibility="hidden")

                elif df.loc[i, 'type'] == 'missing_word':
                    df.loc[i, 'result'] = st.text_area(label='Впишите слово в поле ниже',
                                                       key=f"selectbox_{i}", height=30, value='–––')
                else:
                    df.loc[i, 'result'] = '–––'

                if df.loc[i, 'result'] == '–––':
                    pass
                elif df.loc[i, 'result'] == df.loc[i, 'answer']:
                    st.success('', icon="✅")
                else:
                    st.error('', icon="😟")

        df['total'] = df['result'] == df['answer']

        if len(text) > 0:
            if df['total'].all():
                st.success('Успех!')
                st.balloons()

