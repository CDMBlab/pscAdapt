import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import random
import math

# seed_everything()
import torch
import numpy as np
seed=666
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import random
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

import lr_schedule
import loss_utility
from utils import *
from networks import *
from vat import VATLoss
from similarity_loss import *

torch.set_num_threads(2)

def pscAdapt(args, data_set):
    ## prepare data
    batch_size = args.batch_size
    kwargs = {'num_workers': 0, 'pin_memory': True}

    source_name = args.source_name #"TM_baron_mouse_for_baron"
    target_name = args.target_name #"baron_human"
    domain_to_indices = np.where(data_set['accessions'] == source_name)[0]
    train_set = {'features': data_set['features'][domain_to_indices], 'labels': data_set['labels'][domain_to_indices],
                 'accessions': data_set['accessions'][domain_to_indices]}
    domain_to_indices = np.where(data_set['accessions'] == target_name)[0]
    test_set = {'features': data_set['features'][domain_to_indices], 'labels': data_set['labels'][domain_to_indices],
                'accessions': data_set['accessions'][domain_to_indices]}
    print('source labels:', np.unique(train_set['labels']), ' target labels:', np.unique(test_set['labels']))
    test_set_eval = {'features': data_set['features'][domain_to_indices], 'labels': data_set['labels'][domain_to_indices],
                     'accessions': data_set['accessions'][domain_to_indices]}
    print(train_set['features'].shape, test_set['features'].shape)

    data = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_set['features']), torch.LongTensor(matrix_one_hot(train_set['labels'], int(max(train_set['labels'])+1)).long()))
    source_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

    data = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_set['features']), torch.LongTensor(matrix_one_hot(test_set['labels'], int(max(train_set['labels'])+1)).long()))
    target_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    target_test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False,**kwargs)
    class_num = max(train_set['labels'])+1
    class_num_test = max(test_set['labels']) + 1

    ### re-weighting the classifier
    cls_num_list = [np.sum(train_set['labels'] == i) for i in range(class_num)]
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()


    ## set base network
    embedding_size = args.embedding_size
    base_network = FeatureExtractor(num_inputs=train_set['features'].shape[1], embed_size = embedding_size).cuda()
    label_predictor = LabelPredictor(base_network.output_num(), class_num).cuda()
    total_model = nn.Sequential(base_network, label_predictor)

    print("output size of FeatureExtractor and LabelPredictor: ", base_network.output_num(), class_num)
    ad_net = scAdversarialNetwork(base_network.output_num(), 1024).cuda()
    src=1
    if (src==1):
        base_network.train()
        label_predictor.train()
        # set optimizer
        config_optimizer = {"lr_type": "inv", "lr_param": {"lr": 0.0005, "gamma": 0.0005, "power": 0.75}}
        parameter_list = base_network.get_parameters() + label_predictor.get_parameters()
        optimizer = optim.SGD(parameter_list, lr=5e-4, weight_decay=5e-4, momentum=0.9, nesterov=True)
        schedule_param = config_optimizer["lr_param"]
        lr_scheduler = lr_schedule.schedule_dict[config_optimizer["lr_type"]]
        len_train_source = len(source_loader)
        for epoc in range(args.num_iterations_pre):
            with torch.no_grad():
                if epoc % (500) == 0 and epoc != 0:
                    feature_source = base_network(torch.FloatTensor(train_set['features']).cuda())
                    output_source = label_predictor.forward(feature_source)
                    softmax_out = nn.Softmax(dim=1)(output_source)
                    predict_prob_arr, predict_label_arr = torch.max(softmax_out, 1)
                    acc_by_label = np.zeros(class_num)
                    all_label = train_set['labels']
                    for i in range(class_num):
                        acc_by_label[i] = np.sum(predict_label_arr.cpu().data.numpy()[all_label == i] == i) / np.sum(all_label == i)
                    # np.set_printoptions(suppress=True)
                    print('iter:', epoc, "average acc over all source cell types: ", round(np.nanmean(acc_by_label), 4))
                    print("acc of each source cell type: ", np.round(acc_by_label, 4))
            # zero gradients for optimizer
            optimizer = lr_scheduler(optimizer, epoc, **schedule_param)
            optimizer.zero_grad()
            if epoc % len_train_source == 0:
                iter_source = iter(source_loader)
            inputs_source, labels_source = iter_source.__next__()
            inputs_source, labels_source, = inputs_source.cuda(), labels_source.cuda()

            feature_source = base_network(inputs_source)
            output_source = label_predictor(feature_source)

            # classifier_loss = nn.CrossEntropyLoss(output_source,torch.max(labels_source, dim=1)[1])
            classifier_loss = nn.CrossEntropyLoss(weight=per_cls_weights)(output_source, torch.max(labels_source, dim=1)[1])
            source_loss = classifier_loss
            source_loss.backward()
            optimizer.step()
        src=+1
    ## set optimizer
    config_optimizer = {"lr_type": "inv", "lr_param": {"lr": 0.00005, "gamma": 0.001, "power": 0.75}}
    parameter_list = base_network.get_parameters() + ad_net.get_parameters() + label_predictor.get_parameters()
    optimizer = optim.SGD(parameter_list, lr=5e-5, weight_decay=5e-4, momentum=0.9, nesterov=True)
    schedule_param = config_optimizer["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[config_optimizer["lr_type"]]

    ## train
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    epoch_global = 0.0

    hit = False
    s_global_centroid = torch.zeros(class_num, embedding_size).cuda()
    t_global_centroid = torch.zeros(class_num, embedding_size).cuda()
    for epoch in range(args.num_iterations):
        if epoch % (1000) == 0 and epoch != 0:
            feature_target = base_network(torch.FloatTensor(test_set['features']).cuda())
            output_target = label_predictor.forward(feature_target)
            softmax_out = nn.Softmax(dim=1)(output_target)
            predict_prob_arr, predict_label_arr = torch.max(softmax_out, 1)
            if epoch == args.epoch_th:
                data = torch.utils.data.TensorDataset(torch.FloatTensor(test_set['features']), predict_label_arr.cpu())
                target_loader_align = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True,**kwargs)
            result_path = args.result_path  # "../results/"
            model_file = result_path + 'final_model_' + str(epoch) + source_name + target_name + '.ckpt'
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            with torch.no_grad():
                code_arr_s = base_network(Variable(torch.FloatTensor(train_set['features']).cuda()))
                code_arr_t = base_network(Variable(torch.FloatTensor(test_set_eval['features']).cuda()))
                code_arr = np.concatenate((code_arr_s.cpu().data.numpy(), code_arr_t.cpu().data.numpy()), 0)

            digit_label_dict = pd.read_csv(args.dataset_path + 'digit_label_dict.csv')
            digit_label_dict = pd.DataFrame(zip(digit_label_dict.iloc[:, 0], digit_label_dict.index),
                                            columns=['digit', 'label'])
            digit_label_dict = digit_label_dict.to_dict()['label']
            # transform digit label to cell type name
            y_pred_label = [digit_label_dict[x] if x in digit_label_dict else x for x in
                            predict_label_arr.cpu().data.numpy()]
            if epoch % (5000) == 0 and epoch != 0:
                pred_labels_file = result_path + 'pred_labels_' + source_name + "_" + target_name + "_" + str(epoch) + ".csv"
                pd.DataFrame([predict_prob_arr.cpu().data.numpy(), y_pred_label],index=["pred_probability", "pred_label"]).to_csv(pred_labels_file, sep=',')
                embedding_file = result_path + 'embeddings_' + source_name + "_" + target_name + "_" + str(epoch) + ".csv"
                pd.DataFrame(code_arr).to_csv(embedding_file, sep=',')
            ### only for evaluation
            acc_by_label = np.zeros(class_num_test )
            all_label = test_set['labels']
            for i in range(class_num_test):
                acc_by_label[i] = np.sum(predict_label_arr.cpu().data.numpy()[all_label == i] == i) / np.sum(all_label == i)
            print('iter:', epoch, "average acc over all source cell types: ", round(np.nanmean(acc_by_label), 3))
            print("acc of each source cell type: ", np.round(acc_by_label, 3))


        ## train one iter
        base_network.train(True)
        ad_net.train(True)
        label_predictor.train(True)

        optimizer = lr_scheduler(optimizer, epoch, **schedule_param)
        optimizer.zero_grad()
        # optimizer_centloss.zero_grad()

        if epoch % len_train_source == 0:
            iter_source = iter(source_loader)
            epoch_global = epoch_global + 1
        if epoch % len_train_target == 0:
            if epoch < args.epoch_th:
                iter_target = iter(target_loader)
            else:
                hit = True
                iter_target = iter(target_loader_align)
        inputs_source, labels_source = iter_source.__next__()
        inputs_target, labels_target = iter_target.__next__()
        inputs_source, inputs_target, labels_source, labels_target = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda(), labels_target.cuda()

        feature_source = base_network(inputs_source)
        feature_target = base_network(inputs_target)
        features = torch.cat((feature_source, feature_target), dim=0)

        output_source = label_predictor.forward(feature_source)
        output_target = label_predictor.forward(feature_target)

        ######## VAT and BNM loss
        # LDS should be calculated before the forward for cross entropy
        vat_loss = VATLoss(xi=args.xi, eps=args.eps, ip=args.ip)
        lds_loss = vat_loss(total_model, inputs_target)

        softmax_tgt = nn.Softmax(dim=1)(output_target[:, 0:class_num])
        _, s_tgt, _ = torch.svd(softmax_tgt)
        BNM_loss = -torch.mean(s_tgt)

        ########domain alignment loss
        if args.method == 'DANN':
            domain_prob_discriminator_1_source = ad_net.forward(feature_source)
            domain_prob_discriminator_1_target = ad_net.forward(feature_target)

            adv_loss = loss_utility.BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_source), \
                                                     predict_prob=domain_prob_discriminator_1_source)  # domain matching
            adv_loss += loss_utility.BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_target), \
                                                      predict_prob=1 - domain_prob_discriminator_1_target)

            transfer_loss = adv_loss


        ######CrossEntropyLoss
        # classifier_loss = nn.CrossEntropyLoss(weight=per_cls_weights)(output_source, torch.max(labels_source, dim=1)[1])
        classifier_loss = loss_utility.CrossEntropyLoss(labels_source.float(), nn.Softmax(dim=1)(output_source))

        ######semantic_loss and center loss
        cell_th = args.cell_th
        epoch_th = args.epoch_th
        if epoch < args.epoch_th or hit == False:
            similarity_loss = torch.FloatTensor([0.0]).cuda()
            pass
        elif hit == True:
            labels_s = torch.argmax(labels_source, dim=1)

            matrix_s = feature_source.view(batch_size, 16, -1)
            matrix_t = feature_target.view(batch_size, 16, -1)
            # 相似度矩阵
            matrix_s_s = cosine_similarity_matrix(matrix_s)
            matrix_t_s = cosine_similarity_matrix(matrix_t)
            # selected_matrix, selected_labels = Threshold_Selected(matrix_t_s, predict_prob_arr, predict_label_w, 0.9)
            similarity_loss = subtract_tensors_by_label(labels_s, matrix_s_s, labels_target, matrix_t_s)

        if epoch > epoch_th:
            lds_loss = torch.FloatTensor([0.0]).cuda()
        if epoch <= args.num_iterations:
            progress = epoch / args.epoch_th #args.num_iterations
        else:
            progress = 1
        lambd = 2 / (1 + math.exp(-10 * progress)) - 1
        if epoch % (1000) == 0 and epoch != 0:
            print("similarity loss:",similarity_loss)
        total_loss = classifier_loss + lambd * args.DA_coeff * transfer_loss + lambd*args.BNM_coeff*BNM_loss + lambd * args.alpha * lds_loss + 20 * similarity_loss
        total_loss.backward()
        optimizer.step()
