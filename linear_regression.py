#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os

import xgboost
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score



def main(args):
    model_file = args.model_file
    model_dir = os.path.dirname(model_file)
    os.makedirs(model_dir, exist_ok=True)

    train_record = args.train_record
    x_train, label_train = get_data(train_record, negative_ratio=1)
    eval_record = args.eval_record
    x_eval, label_eval = get_data(eval_record, negative_ratio=0.035)

    data_train = xgboost.DMatrix(x_train, label=label_train)
    data_test = xgboost.DMatrix(x_eval, label=label_eval)

    sum_pos = sum(label_train)
    sum_neg = len(label_train) - sum_pos
    params = {'max_depth': 6, 'eta': 0.3, 'silent': 0,
              'objective': 'binary:logistic',
              'scale_pos_weight': sum_neg / sum_pos,
              'max_delta_step': 3,
              'eval_metric': ['error', 'auc', 'map', 'aucpr']}

    watchlist = [(data_test, 'eval'), (data_train, 'train')]
    num_round = 20
    gbm = xgboost.train(params, data_train, num_round, watchlist,
                        early_stopping_rounds=5)
    model_dir = os.path.dirname(model_file)
    os.makedirs(model_dir, exist_ok=True)
    gbm.save_model(model_file)
    print("Validating...")
    check = gbm.predict(data_test,
                        ntree_limit=gbm.best_iteration + 1)
    print('check.shape', check.shape, check[:100])
    score = average_precision_score(label_eval, check)
    print('area under the precision-recall curve: {:.6f}'.format(score))

    check2 = check.round()
    score = precision_score(label_eval, check2)
    print('precision score: {:.6f}'.format(score))

    score = recall_score(label_eval, check2)
    print('recall score: {:.6f}'.format(score))
    print('f1_score: {:.6f}'.format(f1_score(label_eval, check2)))
    lr_auc = roc_auc_score(label_eval, check)
    print('ROC AUC=: {:.6f}'.format(lr_auc))
    lr_precision, lr_recall, thresh = precision_recall_curve(label_eval, check)
    print(lr_precision, lr_recall, thresh)
    lr_auc = auc(lr_recall, lr_precision)

    # # Compute micro-average ROC curve and ROC area
    # fpr, tpr, _ = roc_curve(label_eval, check)
    # roc_auc = auc(fpr, tpr)
    import matplotlib.pyplot as plt

    plt.figure()
    lw = 2
    plt.plot(lr_recall, lr_precision, color='darkorange',
             lw=lw, label='AUC (area = %0.2f)' % lr_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    no_skill = len(label_eval[label_eval == 1]) / len(label_eval)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    plt.show()
    # classifier_model = xgboost.Booster()
    # classifier_model.load_model(model_file)
    xgboost.plot_importance(gbm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-mf', '--model_file', type=str, default=None)
    parser.add_argument('-tr', '--train_record', type=str, default=None)
    parser.add_argument('-er', '--eval_record', type=str, default=None)
    main(parser.parse_args())
