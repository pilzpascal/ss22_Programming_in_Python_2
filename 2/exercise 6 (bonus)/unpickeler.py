import os
import pickle as pkl

with open(os.path.join("unittest", "unittest_ex6_data.pkl"), "rb") as ufh:
    all_inputs_outputs = pkl.load(ufh)

print(all_inputs_outputs[-1][0][1])
