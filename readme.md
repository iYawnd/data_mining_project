File information:  
helpers.py: helpful functions, mainly for filtering (completed)
text_input.py: functions to take in user input from webpage (not completed)  
preprocessing.py: text preprocessing (completed)  
keyword_extraction.py: extracts important keywords from text to be used as answers (and to generate the questions) (completed, potentially add more models)  
question_generation.py: uses keywords and context to generate questions (completed, potentially add more models)  
distractor_generation.py: generates distractors (wrong answers) for the questions (incomplete, need to implement LLM, potentially add more models)

readme.md: important info  
requirements.txt: contains all packages needed to run the code. Recommended to create a virtual environment (venv) and then running: pip install -r requirements.txt (windows)  
test_pipeline.py: test if models and pipeline code are working correctly. Purely for testing and debugging, will not be used in final project  

Notes:  
Currently only supports English. Chinese is a bit tricky since there are not very many fine-tuned Chinese question generating models.  

Tasks:  
Implement LLM (at minimum for distractor generation and as a complete replacement of the pipeline)  
Create webpage (perhaps use Streamlit?) to allow user to input data and receive output, including completing text_input.py and pipeline file
Analyze results of different combinations of models  
Write report + PPT  
Improve code and comments, think of ways to improve results (Everyone should help with this, as everyone should know how the pipeline works)  
Potentially add more models, if time allows implement Chinese support  