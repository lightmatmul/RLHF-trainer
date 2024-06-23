import json

def count_model_choices(file_path):
    # Load evaluations
    with open(file_path, 'r') as f:
        evaluations = json.load(f)

    # Extract choices from evaluations
    choices = [evaluation['choice'] for evaluation in evaluations]

    # Count the number of 0s, 1s, and draws
    count_0s = choices.count(0)
    count_1s = choices.count(1)
    count_draw = choices.count("draw")

    print(f"Model 1 (0s) won {count_0s} times.")
    print(f"Model 2 (1s) won {count_1s} times.")
    print(f"There were {count_draw} draws.")

if __name__ == "__main__":
    file_path = "evaluations.json"
    
    count_model_choices(file_path)
