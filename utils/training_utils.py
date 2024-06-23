import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def save_ckpt(path, model, prefix, epoch, step):
    my_path = path + prefix + "_epoch_{}_step_{}".format(epoch, step)
    model.save_pretrained(my_path)
    print("Checkpoint saved as {}".format(my_path))

def smoothed_plot_reward_loss(y, title, sigma_):
    y_smoothed = gaussian_filter1d(y, sigma=sigma_)
    plt.figure()
    plt.plot(y_smoothed)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.show()

def smoothed_plot_avg_scores(y1, y2, title, sigma_):
    y1_smoothed = gaussian_filter1d(y1, sigma=sigma_)
    y2_smoothed = gaussian_filter1d(y2, sigma=sigma_)
    plt.figure()
    plt.plot(y1_smoothed, label='Avg Score (Chosen)')
    plt.plot(y2_smoothed, label='Avg Score (Rejected)')
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Average Score")
    plt.legend()
    plt.show()

def print_eval_results(input_prompts, sentences_policy, sentences_baseline, epoch, step, log_file="evaluation_log.txt"):
    # Print the results of the evaluation
    with open(log_file, "a") as f:
        f.write(f"EVALUATION START: [Epoch: {epoch}, Step: {step}]\n")
        for question, answer1, answer2 in zip(input_prompts, sentences_policy, sentences_baseline):
            answer1 = answer1.split('### Response: ')[-1].strip() if '### Response: ' in answer1 else answer1
            answer1 = answer1.replace('$}}%', '').replace('\n', ' ').replace('%}', '')  
            answer2 = answer2.split('### Response: ')[-1].strip() if '### Response: ' in answer2 else answer2
            answer2 = answer2.replace('$}}%', '').replace('\n', ' ').replace('%}', '') 
            f.write(f"Instruction: {question}\n")
            f.write(f"Answer Policy   : {answer1}\n")
            f.write(f"Answer Baseline : {answer2}\n")
            f.write("-----------------------------\n")
        f.write("EVALUATION COMPLETE\n")
        f.write("=============================\n")
    print(f"Evaluation for epoch {epoch} and step {step} has been saved in {log_file}")
