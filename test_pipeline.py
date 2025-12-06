import preprocessing
import keyword_extraction
import question_generation
import distractor_generation
import time

# ==============================================================================
# TEST DATA
# ==============================================================================
# A text with Nouns, Proper Nouns, and Numbers to test your filters
raw_text = """
Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. 
Python is dynamically-typed and garbage-collected. It supports multiple programming paradigms, including structured (particularly procedural), object-oriented and functional programming. 
It is often described as a "batteries included" language due to its comprehensive standard library.
Guido van Rossum began working on Python in the late 1980s as a successor to the ABC programming language and first released it in 1991 as Python 0.9.0.
"""

def print_section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")

def test_deep_learning_pipeline():
    print_section("TESTING PATH A: DEEP LEARNING (KeyBERT + T5 + BERT)")
    
    # 1. PREPROCESSING (Light)
    print("1. Running Light Cleaning...")
    clean_text = preprocessing.light_clean(raw_text)
    print(f"   Sample: {clean_text[:100]}...")

    # 2. KEYWORD EXTRACTION (KeyBERT)
    print("\n2. Extracting Keywords (KeyBERT)...")
    try:
        keywords = keyword_extraction.keybert_extract_keywords(clean_text, top_n=3, diversity=0.7)
        print(f"   Keywords Found: {keywords}")
    except Exception as e:
        print(f"   CRITICAL ERROR in KeyBERT: {e}")
        return

    # 3. GENERATION LOOP
    print("\n3. Generating Questions & Distractors...")
    
    for kw in keywords:
        print(f"\n   --- Processing Keyword: '{kw}' ---")
        
        # A. Question (T5)
        print("   Generating Question (T5)...")
        question = question_generation.t5_question_generate(clean_text, kw)
        
        if question is None:
            print("   -> SKIPPED (Spoiler Filter triggered)")
            continue
        
        print(f"   Q: {question}")
        
        # B. Distractors (BERT)
        print("   Generating Distractors (BERT)...")
        distractors = distractor_generation.bert_generate_distractors(clean_text, kw)
        print(f"   Distractors: {distractors}")
        
        if len(distractors) < 3:
            print("   -> WARNING: Less than 3 distractors found.")

def test_baseline_pipeline():
    print_section("TESTING PATH B: BASELINE (TF-IDF + Cloze + WordNet)")
    
    # 1. PREPROCESSING (Deep)
    print("1. Running Deep Cleaning...")
    clean_text = preprocessing.deep_clean(raw_text)
    print(f"   Sample: {clean_text[:100]}...")
    
    # Note: For Cloze generation, we still need the original sentences (light clean),
    # but we use deep clean for TF-IDF extraction.
    context_for_cloze = preprocessing.light_clean(raw_text)

    # 2. KEYWORD EXTRACTION (TF-IDF)
    print("\n2. Extracting Keywords (TF-IDF)...")
    try:
        keywords = keyword_extraction.tfidf_extract_keywords(clean_text, top_n=3)
        print(f"   Keywords Found: {keywords}")
    except Exception as e:
        print(f"   CRITICAL ERROR in TF-IDF: {e}")
        return

    # 3. GENERATION LOOP
    print("\n3. Generating Questions & Distractors...")
    
    for kw in keywords:
        print(f"\n   --- Processing Keyword: '{kw}' ---")
        
        # A. Question (Cloze)
        # Note: Cloze needs the full context to find sentences
        question = question_generation.cloze_generate_question(context_for_cloze, kw)
        
        if question is None:
            print(f"   -> SKIPPED (Could not find keyword '{kw}' in context)")
            continue
            
        print(f"   Q: {question}")
        
        # B. Distractors (WordNet)
        print("   Generating Distractors (WordNet)...")
        distractors = distractor_generation.wordnet_generate_distractors(kw)
        print(f"   Distractors: {distractors}")

if __name__ == "__main__":
    start_time = time.time()
    
    # Run Path A
    test_deep_learning_pipeline()
    
    # Run Path B
    test_baseline_pipeline()
    
    print_section("TEST COMPLETE")
    print(f"Total Time: {round(time.time() - start_time, 2)} seconds")