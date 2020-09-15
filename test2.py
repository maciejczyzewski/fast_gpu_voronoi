import math
from pprint import pprint
oo = 1111111111111

def step_function_special(shape, num=None, config=None):
    A = config["A"] #1.2 # <1, 2>
    B = config["B"] #0   # <0, 1>
    C = config["C"] #0.5 # <0, 1> FIXME: ile maksymalnie bedzie?
    D = config["D"] #1.5 # <1, 2>
    X = config["X"] #0.6 # <0.2, 1>

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
        f = round(1/(D**(i**A) + i%max(1, int(C*St))), 4)
        fm = f * S
        ffm = int(fm)
        print(f"--------> {i} f={f:10} fm={fm} | {ffm}")
        #f = math.ceil(max(shape) / (2**(i)))
        if ffm >= 1:
            steps.append(ffm)

    print()
    if len(steps) == 0:
        return [1]
    return steps

################################################################################

def step_function_special_rambo(shape, num=None, config=None):
    A = config["A"] #1.2 # <1, 2>
    B = config["B"] #0   # <0, 1>
    C = config["C"] #0.5 # <0, 1> FIXME: ile maksymalnie bedzie?
    D = config["D"] #1.5 # <1, 2>
    X = config["X"] #0.6 # <0.2, 1>

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
        f = round(1/(D**(i**A) + i%max(1, int(C*St))), 4)
        fm = f * S
        ffm = int(fm)
        print(f"--------> {i} f={f:10} fm={fm} | {ffm}")
        #f = math.ceil(max(shape) / (2**(i)))
        if ffm >= 1:
            steps.append(ffm)

    print()
    if len(steps) == 0:
        return [1]
    return steps

################################################################################

print("=== 256x256 ==========================================")

shape = (256, 256) # 384x384
num = 3276

config = {
    "A": 1.42,
    "B": 0.99,
    "C": 0.94,
    "D": 1.29,
    "X": 0.67,
}

vec256 = step_function_special(shape, num, config)
print(f"\033[92m {vec256} \033[0m")

################################################################################

print("=== 384x384 ==========================================")

shape = (384, 384) # 384x384
num = 7372

config = {
    "A": 1.67,
    "B": 0.99,
    "C": 0.94,
    "D": 1.2,
    "X": 0.91,
}

vec384 = step_function_special(shape, num, config)
print(f"\033[92m {vec384} \033[0m")

################################################################################

print("=== CROSS ============================================")

config = {
    "A": 1.67,
    "B": 0.99,
    "C": 0.94,
    "D": 1.2,
    "X": 0.91,
}

vec1 = step_function_special_rambo((256, 256), 3276, config)
vec2 = step_function_special_rambo((384, 384), 7372, config)

print("256x256")
print(f"\033[92m {vec256} \033[0m")
print(f"\033[93m {vec1} \033[0m")

print("384x384")
print(f"\033[92m {vec384} \033[0m")
print(f"\033[93m {vec2} \033[0m")
