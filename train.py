#coding=utf-8
import argparse
import config
import models
import tensorflow as tf
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
con = config.Config()

parser = argparse.ArgumentParser()
# 数据集所在路径
parser.add_argument('--path', '-p', default='./data/HUAWEI/')
# 所使用的模型
parser.add_argument('--model', '-m', default='AnalogyPlus')
# 训练速度
parser.add_argument('--alpha', '-a', default='0.2')
# 训练次数
parser.add_argument('--times', '-t', default='1000')
# 表示向量的维度
parser.add_argument('--dimension', '-d', default='100')

args = parser.parse_args()

con.set_in_path(args.path if args.path[-1] == '/' else args.path + '/')
con.set_train_times(int(args.times))
con.set_alpha(float(args.alpha))
con.set_dimension(int(args.dimension))

# 设置测试实体预测准确率
con.set_test_link_prediction(True)
# 设置测试三元组预测准确率
con.set_test_triple_classification(True)

# 一些关于训练的相对固定的参数
con.set_work_threads(8)
con.set_nbatches(100)
con.set_margin(1.0)
con.set_bern(0)
con.set_lmbda(0.0001)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method('SGD')

# 训练好的模型会通过tf.Saver()自动保存（便于直接使用）
con.set_export_files('./res/model.vec.tf', 0)
# 模型的参数也会自动输出到json中（便于阅读）
con.set_out_files('./res/embedding.vec.json')

con.init()

model_dict = {
        'transe': models.TransE,
        'transh': models.TransH,
        'transr': models.TransR,
        'transd': models.TransD,
        'analogy': models.Analogy,
        'complex': models.ComplEx,
        'distmult': models.DistMult,
        'hole': models.HolE,
        'analogyplus': models.AnalogyPlus
        }

con.set_model(model_dict[args.model.lower()])

# 训练模型
con.run()
# 对训练好的模型进行测试，测试的内容需要在之前set_xxxx(True)
con.test()
