from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import re

# Input: context = string of preprocessed text, answer = keyword
# Output: question (in string format)

# cache to store model in RAM to prevent reloading
qg_model_cache = {}

# load model once and save to memory
# prevents needing to reload model
def load_t5_model():
    if "tokenizer" not in qg_model_cache:
        # tool to convert text to vectors
        tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
        # neural network weights
        model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
        # load in cache
        qg_model_cache["tokenizer"] = tokenizer
        qg_model_cache["model"] = model

    return qg_model_cache["tokenizer"], qg_model_cache["model"]

# T5
# Deep Learning model
# for multiple choice questions
def t5_question_generate(context, answer):
    # load from cache
    tokenizer, model = load_t5_model()

    # format input for model
    input_text = f"answer: {answer}  context: {context}"

    # tokenize
    # text to context vector
    features = tokenizer(
        [input_text],
        return_tensors='pt',
        max_length=512,
        truncation=True
    )

    # generate 
    output = model.generate(
        input_ids=features['input_ids'],
        attention_mask=features['attention_mask'],
        max_length=64,
        num_beams=4   # beam search
    )

    # decode
    # context vector to text
    question = tokenizer.decode(output[0], skip_special_tokens=True)

    # if question contains answer, skip question
    # warning: this essentially deletes the question. If the user requests 5 questions, make sure 5 is returned
    # recommend over-fetching keyword extraction (if request=5, get 10 keywords, then loop through until 5 questions are generated)
    if answer.lower() in question.lower():
        return None

    return question


# Cloze/Fill-in-the-blank
# Rule-based model
# for cloze/fill-in-the-blank questions
def cloze_generate_question(context, answer):
    # split into list of sentences
    sentences = nltk.sent_tokenize(context)   # sent_tokenize: https://www.nltk.org/api/nltk.tokenize.sent_tokenize.html

    # find sentence that contains the answer
    target_sentence = None
    for sentence in sentences:
        if answer in sentence:
            target_sentence = sentence
            break
    
    if not target_sentence:
        return None
    
    # replace answer with blank
    question = re.sub(re.escape(answer), "__________", target_sentence, flags=re.IGNORECASE)

    return f"Fill in the blank: {question}"