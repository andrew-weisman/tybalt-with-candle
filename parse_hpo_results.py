#!/usr/bin/env python
import os, sys, re
import json                                            

def compare(item):
    return item["score"]

if __name__ == "__main__":

    if len(sys.argv) == 2:
        ktuner_outdir = sys.argv[1]
    else:
        sys.exit("\nusage: parse_ktuner_results.py <ktuner_output_dir> \n")

    trial_results = []
    for trial_dir in os.listdir(ktuner_outdir):
        if re.search("trial_", trial_dir):
            trial_json = os.path.join(ktuner_outdir, trial_dir, "trial.json")
            with open(trial_json) as f:
                data = json.load(f) 
                trial_results.append({"score":data["score"], 
                                      "hyperparameters":data["hyperparameters"]["values"]})
    sorted_results = sorted([r for r in trial_results \
                             if r["score"] is not None], key=compare)
    my_format="score= %5.3g depth=%2d hidden_dim=%2d kappa=%.3f batch_size=%3d " + \
              "num_epochs=%4d learning_rate=%.3g"
    for i in range(len(sorted_results)):
        print(my_format % \
              (float(sorted_results[i]["score"]), 
               int(sorted_results[i]["hyperparameters"]["depth"]), 
               int(sorted_results[i]["hyperparameters"]["hidden_dim"]),
               float(sorted_results[i]["hyperparameters"]["kappa"]),
               int(sorted_results[i]["hyperparameters"]["batch_size"]),
               int(sorted_results[i]["hyperparameters"]["num_epochs"]),
               float(sorted_results[i]["hyperparameters"]["learning_rate"])))
           
