import os
import argparse
import pandas as pd

def collect_checkpoints(directory, write, model, layout, probability, exp):
    for folder in os.listdir(directory):
        f = os.path.join(directory, folder)
        if os.path.isdir(f):

            assert "info.csv" in os.listdir(f)
            info = pd.read_csv(os.path.join(f, "info.csv"), index_col=0)['info']
            if info['status'] != "success": continue
            
            assert "args.csv" in os.listdir(f)
            args = pd.read_csv(os.path.join(f, "args.csv"), index_col=0)['arguments']

        
            model_f = args['model']
            layout_f = args['layout']
            probability_f = args['probability_bound']
            seed_f = args['seed']
            exp_f = args['exp_certificate']
            lip_f = args['weighted']
        
            if model_f == model and layout_f == layout and probability_f == probability and exp_f == exp and lip_f == "True":
                os.system("cp -r "+os.path.join(f)+" "+os.path.join(write, model+layout+"_"+exp_f+"_"+probability+"_"+seed_f))
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="LinearSystem")
    parser.add_argument('--layout', type=str, default="0")
    parser.add_argument('--probability_bound', type=str)
    parser.add_argument('--exp_certificate', type=str, default="True")
    parser.add_argument('--read_directory', type=str, default="./main")
    parser.add_argument('--write_directory', type=str, default="./ckpt_lipbab")
    args = parser.parse_args()
    collect_checkpoints(args.read_directory, args.write_directory, args.model, args.layout, args.probability_bound, args.exp_certificate)
