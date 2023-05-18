import os
dataset = ['14lap', '14res', '15res', '16res']

for d in dataset:
    for r in [0,1,2,3,4]:
        save_dir = "random{}_{}".format(str(r), d)
        print('python train.py --dataset {} --save_dir {}'.format(str(d), save_dir))
        os.system('python train.py --dataset {} --save_dir {}'.format(str(d), save_dir))
