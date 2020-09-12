from glob import glob
from pprint import pprint

vec = []
for name in glob("results/*.cl"):
    x = float(name.replace(".cl", "").replace("results/", ""))
    vec.append(x)
vec = sorted(vec)[::-1]
pprint(vec)
