from langchain.evaluation import load_evaluator
from langchain_openai import ChatOpenAI

# 1. Load an LLM-based evaluator (requires an LLM, often GPT-4 for high quality)
evaluator = load_evaluator("criteria", llm=ChatOpenAI(model="gpt-4o-mini"), criteria="conciseness")

# 2. Define the inputs and output
input_text = "The capital of Australia is Canberra."
prediction = "Canberra is the capital."
reference = "The capital city of Australia is Canberra." # Ground Truth

# 3. Evaluate the output based on the 'conciseness' criteria
evaluation_result = evaluator.evaluate_strings(
    input=input_text,
    prediction=prediction,
    reference=reference, # Often needed for accuracy checks
)

print(evaluation_result)
# The result will contain a score and a reasoning from the LLM evaluator