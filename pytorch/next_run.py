import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Next.")
    parser.add_argument("--start_from", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    
    q = open("experiments.txt").read().splitlines()

    with open("run_experiments.sh", "w") as outfile:
        outfile.write("#!/bin/bash\n\n")
        for i in q[args.start_from:args.start_from+args.step]:
                outfile.write(i + "\n")
