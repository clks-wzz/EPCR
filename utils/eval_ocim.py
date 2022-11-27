import numpy as np
import sklearn
from sklearn.metrics import roc_curve, auc

differ_thresh=0.01

def hex2dec(string_num):
    return str(int(string_num.upper(), 16))

def get_err_threhold(fpr, tpr, threshold):
    RightIndex=(tpr+(1-fpr)-1); 
    right_index = np.argmax(RightIndex)
    best_th = threshold[right_index]
    err = fpr[right_index]

    differ_tpr_fpr_1=tpr+fpr-1.0
    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]    

    #print(err, best_th)
    return err, best_th, right_index

def performances(test_scores, test_labels):
    fpr,tpr,threshold = roc_curve(test_labels, test_scores, pos_label=1)
    err, best_th, right_index = get_err_threhold(fpr, tpr, threshold)
    final_auc = auc(fpr, tpr)

    type1 = len([i for i in range(len(test_scores)) if test_scores[i] <= best_th and test_labels[i] == 1])
    type2 = len([i for i in range(len(test_scores)) if test_scores[i] >  best_th and test_labels[i] == 0])

    final_acc = 1. - (type1 + type2) / len(test_scores)
    frr = 1 - tpr
    hter = (fpr + frr) / 2.0

    final_hter = hter[right_index]
    
    results = [final_acc * 100., final_auc * 100., final_hter * 100]
    return results

def get_scores_ocim(scores_test):
    test_scores = np.zeros([len(scores_test)])
    Test_labels = np.zeros([len(scores_test)])
    for i in range(len(scores_test)):
        file_name, file_score, label = scores_test[i]
        test_scores[i] = float(file_score)
        Test_labels[i] = 1. - label

    Performances_this = performances(test_scores,Test_labels)
    return Performances_this

def eval_ocim(lines):
    scores_test = []
    
    lines_video = {}
    statistic_video = {}
    for l, scores in enumerate(lines):
        name, logits, label, true_label = scores
        file_name = name.split('/')[-2]
        lines_video.setdefault(file_name, {'probs': [], 'label': float(label), 'true_label': true_label})
        lines_video[file_name]['probs'].append(logits[0])
        #lines_video[file_name]['label'] = float(label)

        statistic_video.setdefault(label, set())
        statistic_video[label].add(file_name)
    
    for key in statistic_video.keys():
        print('label:', key, len(statistic_video[key]))    
    
    for key in lines_video.keys():
        file_name = key
        prob = np.mean(np.array(lines_video[file_name]['probs']))
        true_label = lines_video[file_name]['true_label']
        label = lines_video[file_name]['label']
        
        scores_test.append([file_name, prob, label])

    pf_list = get_scores_ocim(scores_test)
    pf_list = ['%.5f'%(x) for x in pf_list]
    pf_str = ' '.join(pf_list)
    return pf_str

