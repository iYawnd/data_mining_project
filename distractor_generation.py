import nltk
from nltk.corpus import wordnet as wn
import requests
import torch
from transformers import BertTokenizer, BertForMaskedLM
import re
from helpers import is_same_pos

nltk.download('wordnet')

# Input: keyword from keyword extraction and used in question generation
# Output: list containing 3 distractor words

# load BERT
# save model to cache to prevent reloading
bert_cache = {}
def load_bert_model():
    if "tokenizer" not in bert_cache:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        bert_cache["tokenizer"] = tokenizer
        bert_cache["model"] = model

    return bert_cache["tokenizer"], bert_cache["model"]

# Method 1: Wordnet
# baseline model
def wordnet_generate_distractors(word):
    distractors = set()
    word = word.lower()

    # get synsets
    synsets = wn.synsets(word)
    if not synsets:
        return []
    
    # get hypernym (parent category)
    hypernyms = synsets[0].hypernyms()
    if not hypernyms: 
        return []
    
    # get hyponyms (children of hypernym)
    for hypernym in hypernyms:
        for hyponym in hypernym.hyponyms():
            # first word associated with concept
            name = hyponym.lemmas()[0].name()
            # replaces _ with spaces (WordNet stores multi-word terms with underscores)
            name = name.replace("_", " ")

            # both words have same pos and tag
            if is_same_pos(word, name):
                # make sure distractor is not the correct answer
                if name.lower() != word and name not in distractors:
                    distractors.add(name.title())
    
    return list(distractors)[:3]


# Method 2: ConceptNet
# Semantic Knowledge Graph
def conceptnet_generate_distractors(word):
    # use ConceptNet API
    word = word.lower()
    url = f"http://api.conceptnet.io/c/en/{word}"

    try:
        obj = requests.get(url).json()
        distractors = set()

        for edge in obj['edges']:   # for each arrow on list of arrows (arrow = relationship)
            link = edge['rel']['label']   # the label of the arrow
            start = edge['start']['label']   # word at start of arrow
            end = edge['end']['label']   # word at end of arrow

            # only want related different words
            if link in ['DistinctFrom', 'SimilarTo', 'Hyponym']:
                # search both sides of the arrow
                if start.lower() == word:
                    candidate = end
                else:
                    candidate = start

                # distractor cannot be answer
                if is_same_pos(word, candidate):
                    if candidate.lower() != word and candidate not in distractors:
                        distractors.add(candidate.title())
        
        return list(distractors)[:3]
    
    except:
        return []
    

# Method 3: BERT
# Deep learning, Context-aware
def bert_generate_distractors(context, answer, top_k=3):
    tokenizer, model = load_bert_model()

    # create masked sentence (replace answer with [mask])
    mask_token = tokenizer.mask_token

    # replace whole words only
    masked_context = re.sub(rf'\b{re.escape(answer)}\b', mask_token, context, flags=re.IGNORECASE)
    # if regex fails, use replace
    if mask_token not in masked_context:
        masked_context = context.replace(answer, mask_token)

    # convert text to numbers
    inputs = tokenizer(masked_context, return_tensors="pt", truncation=True, max_length=512)
    # feed numbers into BERT neural networks, get list of scores for every word in the English language
    # score rates how likely that word is to fill the blank
    with torch.no_grad():
        logits = model(**inputs).logits

    try:
        # get index of masked word
        mask_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        # get list of 30 most likely words, ordered from most likely to least likely
        predicted_token_ids = logits[0, mask_index].argsort(descending=True)[0][:30]
        # convert words to ids
        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_ids)
    except IndexError:
        return []
    
    # filter
    distractors = []
    answer_lower = answer.lower()

    for token in predicted_tokens:
        # remove '##' symbols
        word = token.replace("##", "").lower().strip()
        if len(word) < 2: continue   # cannot be 1 letter word
        if word == answer_lower: continue # Can't be answer
        if word in answer_lower or answer_lower in word: continue # Can't be part of answer
        
        if is_same_pos(answer, word):
            word_display = word.title()
            if word_display not in distractors:
                distractors.append(word_display)
                
        if len(distractors) >= top_k:
            break

    return distractors
