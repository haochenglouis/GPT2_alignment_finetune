import json
from collections import defaultdict
import random
import pickle
import numpy as np
random.seed(0)

data = []
with open('align_data_safety_align_v4.jsonl',"rb") as f:
    for line in f:
        data.append(json.loads(line))


question_label_dict={}
for line in data:
    q = line['question']
    if q not in question_label_dict:
        question_label_dict[q] = defaultdict(list)
        question_label_dict[q][line['label']].append(line['answer'])
    else:
        question_label_dict[q][line['label']].append(line['answer'])


two_label_question_label_dict={}
one_label_question_label_dict={}
for q in question_label_dict:
    if len(question_label_dict[q])>1:
        two_label_question_label_dict[q] = question_label_dict[q]
    elif len(question_label_dict[q][1])>0:
        one_label_question_label_dict[q] = question_label_dict[q]
print(len(two_label_question_label_dict))
print(len(one_label_question_label_dict))

gpt2_step1 = {}
gpt2_step1["train"] = []
gpt2_step1["test"] = []
gpt2_step2 = {}
gpt2_step2["train"] = []
gpt2_step2["test"] = []
gpt2_step3_q_in_step1 = []
gpt2_step3_q_in_step2 = []
gpt2_eval = []
for i,q in enumerate(one_label_question_label_dict):
    if i<383:
        gpt2_step3_q_in_step1.append('Human: '+q)
        for a in one_label_question_label_dict[q][1]:
            gpt2_step1["train"].append('Human: '+q+' Assistant: '+a)
    elif i<433:
        gpt2_step3_q_in_step1.append('Human: '+q)
        for a in one_label_question_label_dict[q][1]:
            gpt2_step1["test"].append('Human: '+q+' Assistant: '+a)
    else:
        gpt2_eval.append('Human: '+q)

num_pairs_each_prompt = 3
for i,q in enumerate(two_label_question_label_dict):
    gpt2_step3_q_in_step2.append('Human: '+q)
    if i < 343:
        prompt = q
        for j in range(num_pairs_each_prompt):
            positive_ans = random.choice(two_label_question_label_dict[prompt][1])
            negative_ans = random.choice(two_label_question_label_dict[prompt][0])
            gpt2_step2["train"].append({'prompt':'Human: '+q,'chosen':'Assistant: '+positive_ans,'rejected':'Assistant: '+negative_ans})
    else:
        prompt = q
        positive_ans = random.choice(two_label_question_label_dict[prompt][1])
        negative_ans = random.choice(two_label_question_label_dict[prompt][0])
        gpt2_step2["test"].append({'prompt':'Human: '+q,'chosen':'Assistant: '+positive_ans,'rejected':'Assistant: '+negative_ans})

random.shuffle(gpt2_step1["train"])
random.shuffle(gpt2_step1["test"])
random.shuffle(gpt2_eval)        
random.shuffle(gpt2_step2["train"])
random.shuffle(gpt2_step3_q_in_step1)
random.shuffle(gpt2_step3_q_in_step2)

with open('gpt2_step1.pkl',"wb") as tf:
    pickle.dump(gpt2_step1,tf)
with open('gpt2_step2.pkl',"wb") as tf:
    pickle.dump(gpt2_step2,tf)
np.save('gpt2_eval.npy',gpt2_eval)
np.save('gpt2_step3_q_in_step1.npy',gpt2_step3_q_in_step1)
np.save('gpt2_step3_q_in_step2.npy',gpt2_step3_q_in_step2)
