Need "pytorch", "Biopython", "argparse", "itertools"

1. Z_Unit.py
This file can using z-curve encode method to coding DNA sequence to vector
Note: python Z_Unit.py -fasta "fasta file path" -out "output file path" -z "z-curve parameters numbers: 1, 2, 3" -phase "True/False"

2. OriFinderH.py
This is our predict tool, you can using this tool to predict ori. This python file need using with "ori_finder_h.pkl", this two file better save in same fold.
Note: python OriFinderH.py -fasta "fasta file path" -out "output file path"  -model_path "path of ori_finder_h.pkl" (if ori_finder_h.pkl and OriFinderH.py in same fold this item dont need add.)

3. Model_Unit.py
This file save our model structure code. 

4. Train_Unit.py
This file mainly our model train process code.
