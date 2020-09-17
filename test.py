"""tylko aby znalesc wielomian do step_function"""
import math
from pprint import pprint
oo = 1111111111111

shape = (512, 512)
num = 1000

##########################################################################


def step_function_default(shape, num=None, config=None):
    steps = []
    for factor in range(1, +oo, 1):
        f = math.ceil(max(shape) / (2**(factor)))
        steps.append(f)
        if f <= 1:
            break
    return steps


def mod_step_function__special(shape, num=None, config=None):
    # [EXAMPLE]
    # Special(1.51/0.92/0.92/1.08/0.42)
    # ------- A -- B -- C -- D -- X --- 

    A = config["A"]  # <1,   2>
    B = config["B"]  # <0,   1>
    C = config["C"]  # <0,   1>
    D = config["D"]  # <1,   2>
    X = config["X"]  # <0.2, 1>

    q = num / (shape[0] * shape[1])
    qm = ((shape[0] + shape[1]) / 2) * q**(1 / 2)
    S = B * qm + (1 - B) * (max(shape) / 2)
    St = math.log2(S)

    steps = []
    for i in range(1, int(X * St * 2), 1):
        f = round(1 / (D**(i**A) + i % max(1, int(C * St))), 4)
        ffm = int(f * S)
        if ffm >= 1:
            steps.append(ffm)
    if len(steps) == 0:
        return [1]

    return steps

##########################################################################


config = {
    "A": 1.2,
    "B": 0,
    "C": 0.5,
    "D": 1.5,
    "X": 0.6
}

print("== JFA ====================")
v1 = step_function_default(shape, num, config)
pprint(v1)

print("== SPECIAL ================")
v2 = mod_step_function__special(shape, num, config)
pprint([102, 56, 33, 30, 14, 7, 3, 1])
pprint(v2)
