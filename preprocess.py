"""
브런치 데이터 전처리

https://github.com/MrBananaHuman/KorNlpTutorial/blob/main/0_%ED%95%9C%EA%B5%AD%EC%96%B4_%EC%A0%84%EC%B2%98%EB%A6%AC.ipynb 를 참고함
"""

import json
import re
import os

from pykospacing import spacing
from hanspell import spell_checker
from soynlp.normalizer import *

# 문단 단위로 분리
def paragraph_tokenize(text):
    paragraphs = text.split('\n')
    return paragraphs

# 기호 전처리
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

def clean_punc(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f'{p}')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text.strip()

# 링크 주소 등 제거
def clean_text(texts):
    corpus = []
    for i in range(0, len(texts)):
        review = re.sub(r'[@%\\*=/~#&\+á?\xc3\xa1\-\|\:\;\-\,\_\~\$\'\"]', '',str(texts[i])) #remove punctuation
#         review = re.sub(r'\d+','', str(texts[i]))# remove number
        review = review.lower() #lower case
        review = re.sub(r'\s+', ' ', review) #remove extra space
        review = re.sub(r'<[^>]+>','',review) #remove Html tags
        review = re.sub(r'\s+', ' ', review) #remove spaces
        review = re.sub(r"^\s+", '', review) #remove space from start
        review = re.sub(r'\s+$', '', review) #remove space from the end
        corpus.append(review)
    return corpus

# 외래어 사전 
# !curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1RNYpLE-xbMCGtiEHIoNsCmfcyJP3kLYn" > /dev/null
# !curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1RNYpLE-xbMCGtiEHIoNsCmfcyJP3kLYn" -o confused_loanwords.txt
lownword_map = {}
lownword_data = open('confused_loanwords.txt', 'r', encoding='utf-8')
lines = lownword_data.readlines()

for line in lines:
    line = line.strip()
    miss_spell = line.split('\t')[0]
    ori_word = line.split('\t')[1]
    lownword_map[miss_spell] = ori_word

# 전처리 함수
def text_preprocessor(text):
    corpus = []

    p_text = paragraph_tokenize(text)
    pp_text = [clean_punc(text, punct, punct_mapping) for text in p_text]
    ppc_text = [sents for sents in clean_text(pp_text) if sents != '']

    for sent in ppc_text:
        spaced_text = spacing(sent)
        spelled_sent = spell_checker.check(sent)
        checked_sent = spelled_sent.checked
        normalized_sent = repeat_normalize(checked_sent)
        for lownword in lownword_map:
            normalized_sent = normalized_sent.replace(lownword, lownword_map[lownword])
        corpus.append(normalized_sent)
    return corpus

if __name__ == '__main__':

    # 데이터 로드
    print("-"*40)
    print("Data Loading...")
    org_data_fname = 'Brunch_accm_20210328.json' 
    with open(os.path.join('data', org_data_fname), 'r') as f:
        data = json.load(f)
    # data = data[:10]
    print(f"The number of original data is {len(data)}.")
    
    # 인쇄
    print(f"The first data is as below:")
    print(data[0])

    # 전처리
    print("-"*40)
    print("Data is being preprocessed...")
    data_preprocessed = []
    for i, dat in enumerate(data):
        try:
            new_d = {}
            new_d['text'] = ' '.join(text_preprocessor(dat['text']))
            new_d['tag'] = dat['tag'].split(',')
            data_preprocessed.append(new_d)
        except:
            print(f"Error occured at {i}-th passage :")
            pass

        if i % 100 == 0 and i > 0:
            print(f"{i}-th data is processed.")


    # 전처리 완료
    print("-"*40)
    print("Data preprocessing is finished.")
    print(f"The number of processed data is {len(data_preprocessed)}")
    new_data_fname = 'Brunch_accm_20210328_preprocessed.json' 
    with open(os.path.join('data', new_data_fname), 'w') as f:
        json.dump(data_preprocessed, f)
    print(f"The data is saved as {new_data_fname}.")
    

