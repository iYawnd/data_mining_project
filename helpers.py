import spacy

spacy.cli.download("en_core_web_sm")   # spacy: https://spacy.io/models/en#en_core_web_sm
pos_spacy = spacy.load("en_core_web_sm")

# Helper functions

# get POS and tag (singular/pluraL) for a word
def get_pos_tag(word):
    doc = pos_spacy(word)
    pos = doc[0].pos_
    tag = doc[0].tag_
    return pos, tag

def is_valid_keyword(word):
    pos, tag = get_pos_tag(word)
    # word must be a noun, proper noun, or number
    return pos in ["NOUN", "PROPN", "NUM"]

# check if two words have same POS and tag
def is_same_pos(word1, word2):
    pos1, tag1 = get_pos_tag(word1)
    pos2, tag2 = get_pos_tag(word2)
    
    if pos1 != pos2: return False # Must match Noun/Verb
    if tag1 != tag2: return False # Must match Singular/Plural
    
    return True

# checks if two words (answer and distractor) have the same POS and tag (singular/plural)
def is_same_pos(word1, word2):
    # converts word to Spacy Doc object
    doc1 = pos_spacy(word1)
    doc2 = pos_spacy(word2)
    # grab word
    token1 = doc1[0]
    token2 = doc2[0]
    
    # check if pos of both words are the same
    if token1.pos_ != token2.pos_:
        return False

    # check if tag (singular/plural)of both words are the same
    if token1.tag_ != token2.tag_:
        return False
    
    return True