"""tylko aby znalesc wielomian do step_function"""
import math
from pprint import pprint
oo = 1111111111111

shape = (512, 512)
num = 1000

################################################################################

def step_function_default(shape, num=None, config=None):
    steps = []
    for factor in range(1, +oo, 1):
        f = math.ceil(max(shape) / (2**(factor)))
        steps.append(f)
        if f <= 1:
            break
    return steps

def step_function_special(shape, num=None, config=None):
    A = 1.2 # <1, 2>
    B = 0   # <0, 1>
    C = 0.5 # <0, 1> FIXME: ile maksymalnie bedzie?
    D = 1.5 # <1, 2>
    X = 0.6 # <0.2, 1>

    steps = []
    q = num/(shape[0]*shape[1])
    qm = ((shape[0]+shape[1])/2) * q**(1/2)
    print(f"q={q} --> qm={qm}")
    # qm <-----> max(shape)
    # jakis x?
    S = B*qm + (1-B)*(max(shape)/2)
    St = math.log2(S)
    print(f"====> S={S} | {St}")

    print()
    for i in range(1, int(X*St*2), 1):
        f = round(1/(D**(i**A) + i%(C*St)), 4)
        fm = f * S
        ffm = int(fm)
        print(f"--------> {i} f={f:10} fm={fm} | {ffm}")
        #f = math.ceil(max(shape) / (2**(i)))
        if ffm >= 1:
            steps.append(ffm)
        
    print()

    return steps

################################################################################

print("== JFA ====================")
v1 = step_function_default(shape, num)
pprint(v1)

print("== SPECIAL ================")
v2 = step_function_special(shape, num)
pprint(v2)
