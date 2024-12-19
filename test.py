# inspect_checkpoint.py
import torch
import pprint

def inspect_checkpoint(file_path):
    checkpoint = torch.load(file_path)
    pp = pprint.PrettyPrinter(indent=4)
    print("Contents of checkpoint:")
    pp.pprint(checkpoint)

if __name__ == "__main__":
    file_path = "/home/sina/projects/VLM-Uncertainty-Bench/trained_policies/llava-v1.5-7b/llava-v1.5-7b_scienceqa_policy.pth"
    inspect_checkpoint(file_path)