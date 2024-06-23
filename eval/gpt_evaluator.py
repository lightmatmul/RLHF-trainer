import openai
import json
from tqdm import tqdm 

'''Feed our model outputs based on the 100 prompts and determine the preferred responses'''

# Load answers from the two models
with open("rlhf_responses.json", "r") as f1, open("sft_responses.json", "r") as f2:
    model1_responses = json.load(f1)
    model2_responses = json.load(f2)

# Set up OpenAI API key
openai.api_key = 'OPENAI_API_KEY'

evaluations = []
# Iterate over prompts and their responses from the two models
for r1, r2 in tqdm(zip(model1_responses, model2_responses), total=len(model1_responses), desc="Evaluating"):
    prompt = r1['prompt']
    answer1 = r1['response']
    answer2 = r2['response']

    # Compose the message format for GPT Chat mode
    messages = [
        {"role": "user", "content": f"Given the following two responses to a specific prompt, please select the one that models most human cognitive skills and provides step-by-step reasoning, and offers a comprehensive justification: \nPrompt: {prompt}\n\nRLHF answer: {answer1}\nSFT: {answer2}\n\nWhich is the better answer? Respond by either RLHF, SFT or Both"}
    ]

    # Query GPT
    response = openai.ChatCompletion.create(model="gpt-4", messages=messages, max_tokens=256)
    gpt4_evaluation_content = response.choices[0].message['content']

    # Determine the choice
    choice = "draw"
    if "RLHF" in gpt4_evaluation_content:
        choice = 0
    elif "SFT" in gpt4_evaluation_content:
        choice = 1
    elif "both" in gpt4_evaluation_content:
        choice = "draw"

    # Store evaluation
    evaluations.append({
        'prompt': prompt,
        'RLHF': answer1,
        'SFT': answer2,
        'gpt4_evaluation': gpt4_evaluation_content,
        'choice': choice
    })
    
# Save evaluations
with open('evaluations.json', 'w') as eval_file:
    json.dump(evaluations, eval_file, indent=4)
