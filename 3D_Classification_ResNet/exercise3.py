

import torch

TAGET_LIST = [6, 5, 4, 3]
list1 = 'm, b, sct'
list2 = 'so, x'

print([x.strip() for x in list1.split(",")])


target_tensor = ['m', 'a', 'e', 'sct', 'z']
for idx, target_elmt in enumerate(target_tensor):
    if target_elmt in list1:
        target_tensor[idx] = 1

#print(target_tensor)
