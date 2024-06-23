import re
import transformers
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

'''Use the following script to infer with our models
You can interact via CLI and exit by either typing: quit, bye, exit'''

class ModelHandler:
    def __init__(self, model_name: str, template: str):
        # Initialize pre-training models and tokenizers
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, load_in_8bit=True, device_map='auto')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_bos_token = False
        self.tokenizer.add_eos_token = True
        
        # Setup generation pipeline
        self.generate_text = transformers.pipeline(model=self.model, 
                                                   tokenizer=self.tokenizer,
                                                   do_sample=True, 
                                                   task='text-generation',
                                                   temperature=0.7, 
                                                   max_new_tokens=256,
                                                   repetition_penalty=1.12,
                                                   top_p=0.1,
                                                   top_k=40,
                                                  )
        
        # Setup prompt template to feed models for inference
        self.prompt = PromptTemplate(template=template, input_variables=["query"])
        # Setup langchain to manage conversations
        self.llm_chain = LLMChain(
            prompt=self.prompt, 
            llm=HuggingFacePipeline(pipeline=self.generate_text), 
            verbose=False
        )
    
    def get_response(self, query: str):
        # Fetch the model's response for the given query
        output = self.llm_chain.predict(query=query)
        # Return the response based on the output's format
        if isinstance(output, dict) and "response" in output:
            return output["response"]
        elif isinstance(output, str):
            return output
        else:
            raise ValueError("Unexpected output type or format")

# Clean response
def format_response(output: str) -> str:
    output = output.replace('</ASSISTANT>', '')
    formatted_output = re.sub(r'\n\n', '\n', output)
    return formatted_output + '\n'

# Setup  template for models' input
template="""<s>[INST] <<SYS>>\n You are an AI assistant that helps people find information. User will you give you a question. Your task is to answer as faithfully as you can. While answering think step-bystep and justify your answer.<</SYS>>\n\n
User {query}\n[/INST]\n
\nAssistant """

# Initialize PPO and SFT models..
sft_handler = ModelHandler('HumanDynamics/sft_model', template)
ppo_handler = ModelHandler('HumanDynamics/ppo_model', template)

if __name__ == "__main__":
    # CLI app for inference
    print("\nWelcome to the AI Assistant CLI. Type your instruction below.")
    print("Type 'quit', 'exit', or 'bye' to leave the application. \n")
    while True:
        user_query = input("User: ")
        if user_query.lower() in ["quit", "exit", "bye"]:
            print("Goodbye!")
            break
        sft_response = sft_handler.get_response(user_query)
        ppo_response = ppo_handler.get_response(user_query)
        print(f"\nSFT_Assistant: {format_response(sft_response)}")
        print(f"PPO_Assistant: {format_response(ppo_response)}")