import os

# "14lap" "14res" "15res" "16res"

def get_ret(filepath):
    with open(filepath, 'r') as inf:
        lines = list(inf.readlines())
        return float(lines[-1].rstrip('\n').split()[-5].rstrip(','))*100.0, float(lines[-1].rstrip('\n').split()[-3].rstrip(','))*100.0, float(lines[-1].rstrip('\n').split()[-1])*100.0

p_list, r_list, f1_list = [], [], []
for j in range(1,6,1):
    path = 'random%s_woA_wopiece_wmask_16res/log.txt' % (str(j))
    p,r,f1 = get_ret(path)
    p_list.append(p)
    r_list.append(r)
    f1_list.append(f1)

print("p: ", sum(p_list)/len(p_list))
print("r: ", sum(r_list)/len(r_list))
print("f1: ", sum(f1_list)/len(f1_list))
print()
