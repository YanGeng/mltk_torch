import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from sklearn import metrics

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import BertModel, AlbertModel, BertConfig, BertTokenizer, AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, LoraModel, get_peft_model, PeftConfig, PeftModel

from dataloader import TextDataset, BatchTextCall, TextDatasetMultLabel, BatchTextCallLLM
from model import MultiClass, MultiClassLLM
from utils import load_config, get_files_in_dir

from accelerate import Accelerator


def choose_bert_type(path, bert_type="tiny_albert"):
    """
    choose bert type for chinese, tiny_albert or macbert（bert）
    return: tokenizer, model
    """

    if bert_type == "albert":
        model_config = BertConfig.from_pretrained(path)
        model = AlbertModel.from_pretrained(path, config=model_config)
    elif bert_type == "bert" or bert_type == "roberta":
        model_config = BertConfig.from_pretrained(path)
        model = BertModel.from_pretrained(path, config=model_config)
    elif bert_type == "Qwen2-1.5B":
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
    else:
        model_config, model = None, None
        print("ERROR, not choose model!")

    return tokenizer, model


def predict(model, text1, text2, label2ind_dict, save_path, vocab_path):
    model.load_state_dict(torch.load(save_path))

    model.eval()
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    token_res = tokenizer(text1)
    token = token_res.get('input_ids').squeeze(1)
    mask = token_res.get('attention_mask').squeeze(1)
    segment = token_res.get('token_type_ids').squeeze(1)

    # token = token.cuda()
    # segment = segment.cuda()
    # mask = mask.cuda()

    out = model(token, segment, mask)
    predic = torch.max(out.data, 1)[1].cpu().numpy()

    result = label2ind_dict.get(predic[0])
    return result

def evaluation(accelerator, model, test_dataloader, loss_func, label2ind_dict, classify_type, valid_or_test="test"):
    # model.load_state_dict(torch.load(save_path))

    model.eval()
    total_loss = 0
    predict_all = None
    labels_all = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ind, (token, mask, label) in enumerate(test_dataloader):
#         token = token.to(device)
#         segment = segment.to(device)
#         mask = mask.to(device)
        if config.classify_type == 'multi_label':
            label2 = label.float()
#         label2 = label.to(device)
        # label2 = F.one_hot(label).float()
        # label2 = label2.to(device)
#         print(token.shape)
#         print(token)
#         print('############ current device before infe ##############' + str(accelerator.device))
#         accelerator.wait_for_everyone()
        out = model(token, mask)
#         print('############ current device ##############' + str(accelerator.device))
#         accelerator.wait_for_everyone()

        out = accelerator.gather_for_metrics(out)
        label2 = accelerator.gather_for_metrics(label2)
        loss = loss_func(out[0], label2)
        total_loss += loss.detach().item()

        if config.classify_type != 'multi_label':
            predic = torch.max(out[0].data, 1)[1]
        else:
            predic = out[0].data
            predic = torch.where(predic > 0.5, 1, 0)
        # predic = out[0].data.cpu().numpy()
        
        label = accelerator.gather_for_metrics(label)
        label = label.data.cpu().numpy()
        
#         predic = accelerator.gather_for_metrics(predic)
        predic = predic.data.cpu().numpy()
        
        labels_all = label if labels_all is None else np.append(labels_all, label, axis=0)
        predict_all = predic if predict_all is None else np.append(predict_all, predic, axis=0)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if valid_or_test == "test":
        report = metrics.classification_report(labels_all, predict_all, target_names=label2ind_dict.keys(), digits=4)
        if config.classify_type == 'multi_label':
            confusion = metrics.multilabel_confusion_matrix(labels_all, predict_all)
        else:
            confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, total_loss / len(test_dataloader), report, confusion
    return acc, total_loss / len(test_dataloader)


def train(config):
    # label2ind_dict = {'finance': 0, 'realty': 1, 'stocks': 2, 'education': 3, 'science': 4, 'society': 5, 'politics': 6,
    #                   'sports': 7, 'game': 8, 'entertainment': 9}
    label2ind_dict = {'others': 0,'is_merchant_duty': 1,'is_merchant_compensation': 2,'is_merchant_break': 3,'is_merchant_goods_less': 4,'is_merchant_wrong_delivery': 5
        ,'is_merchant_refund': 6,'is_merchant_goods_exchange': 7, 'is_merchant_goods_reissue': 8, 'merchant_deny_others': 9,'merchant_deny_wrong_delivery': 10,'merchant_express_user_buy_wrong': 11
        ,'merchant_express_user_buy_wrong_size': 12,'merchant_deny_goods_less': 13,'merchant_express_not_my_goods': 14,'user_affirm_mistake': 15
        ,'user_acc_or_no_need_afs': 16, 'user_affirm_buy_wrong_size': 17,'user_affirm_buy_wrong': 18,'user_affirm_found': 19}

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    torch.backends.cudnn.benchmark = True

    # load_data(os.path.join(data_dir, "cnews.train.txt"), label_dict)
    accelerator = Accelerator()
    device = accelerator.device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, bert_encode_model = choose_bert_type(config.pretrained_path, bert_type=config.bert_type)
    multi_classification_model = MultiClassLLM(bert_encode_model, 1024,
                                            num_classes=config.num_classes, classify_type=config.classify_type)
    
    

    print('use_lora is: {}'.format(config.use_lora))

    if config.use_lora:
        if config.restore_flag == True:
            lora_config = PeftConfig.from_pretrained(config.lora_save_path)
            print(lora_config)
#             model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
#             multi_classification_model.load_state_dict(torch.load(lora_config.base_model_name_or_path))
            multi_classification_model = PeftModel.from_pretrained(multi_classification_model, config.lora_save_path, is_trainable=True)
            print("----------------------------------------------------------------------")
            print("Success restore model from: {}".format(config.restore_path))
        else:
#             lora_config = LoraConfig(
#                 r=16,
#                 lora_alpha=16,
#                 target_modules=["query", "value"],
# #                 target_modules=["q_proj", "v_proj"],
#                 lora_dropout=0.1,
#                 bias="none",
#                 # modules_to_save=["classifier"],
#             )
            lora_config = LoraConfig(multi_classification_model)
            lora_config.target_modules = ["q_proj", "v_proj"]
            lora_config.modules_to_save = ["fc"]
            print(lora_config)
            multi_classification_model = get_peft_model(multi_classification_model, lora_config)
        print('############ lora model ##############')
        print(multi_classification_model)
    else:
        if config.restore_flag == True:
            multi_classification_model.load_state_dict(torch.load(config.restore_path))
            print("----------------------------------------------------------------------")
            print("Success restore model from: {}".format(config.restore_path))

    # tokenizer = BertTokenizer.from_pretrained(config.pretrained_path)
    train_dataset_call = BatchTextCallLLM(tokenizer, max_len=config.sent_max_len)

    train_data_file = get_files_in_dir(config.train_data_dir)
    print('train_data_file is {}'.format(train_data_file))
    train_dataset = TextDatasetMultLabel(train_data_file, num_classes=config.num_classes)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0,
                                  collate_fn=train_dataset_call)

#     valid_data_file = get_files_in_dir(config.valid_data_dir)
#     print('valid_data_file is {}'.format(valid_data_file))
#     valid_dataset = TextDatasetMultLabel(valid_data_file, num_classes=config.num_classes)
#     valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0,
#                                   collate_fn=train_dataset_call)

    test_data_file = get_files_in_dir(config.test_data_dir)
    print('test_data_file is {}'.format(test_data_file))
    test_dataset = TextDatasetMultLabel(test_data_file, num_classes=config.num_classes)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0,
                                 collate_fn=train_dataset_call)
    
    multi_classification_model.to(device)
    # multi_classification_model.cuda()

    num_train_optimization_steps = len(train_dataloader) * config.epoch
    param_optimizer = list(multi_classification_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=config.lr, correct_bias=not config.bertadam)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                             int(num_train_optimization_steps * config.warmup_proportion),
                                                             num_train_optimization_steps)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

    multi_classification_model, optimizer, scheduler, train_dataloader, test_dataloader = accelerator.prepare(multi_classification_model, optimizer, scheduler, train_dataloader, test_dataloader)

    if config.classify_type == 'multi_class':
        loss_func = F.cross_entropy
    elif config.classify_type == 'multi_label':
        # loss_func = F.binary_cross_entropy_with_logits
        loss_func = F.binary_cross_entropy

    loss_total, top_acc = [], 0
    if config.restore_flag == True:
        acc, loss, report, confusion = evaluation(accelerator, multi_classification_model,
                                                  test_dataloader, loss_func, label2ind_dict,
                                                  config.classify_type)
        print("Accuracy: %.4f Loss in test %.4f" % (acc, loss))
        if top_acc <= acc:
            top_acc = acc
            print(report, '\n', confusion)
        
    for epoch in range(config.epoch):
        multi_classification_model.train()
        start_time = time.time()
        tqdm_bar = tqdm(train_dataloader, desc="Training epoch{epoch}".format(epoch=epoch), disable=not accelerator.is_main_process)
        for i, (token, mask, label) in enumerate(tqdm_bar):
        # for aaa in enumerate(train_dataloader):
            # token = token.to(device)
            # segment = segment.to(device)
            # mask = mask.to(device)
            if config.classify_type == 'multi_label':
                label = label.float()
            # label = label.to(device)
            # label2 = F.one_hot(label).float()
            # label2 = label2.to(device)

            multi_classification_model.zero_grad()
            out = multi_classification_model(token, mask)
            loss = loss_func(out[0], label)
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_total.append(loss.detach().item())
        print("Epoch: %03d; loss = %.4f cost time  %.4f" % (epoch, np.mean(loss_total), time.time() - start_time))

        acc, loss, report, confusion = evaluation(accelerator, multi_classification_model,
                                                  test_dataloader, loss_func, label2ind_dict,
                                                  config.classify_type)
        print("Accuracy: %.4f Loss in test %.4f" % (acc, loss))
        
        accelerator.wait_for_everyone()
        if top_acc < acc:
            top_acc = acc
            if accelerator.is_main_process:
                # torch.save(multi_classification_model.state_dict(), config.save_path)
#                 accelerator.save(net_dict, config.save_path)
#                 accelerator.save_state(output_dir='./ckpt/duty/acc_bert')
                unwrapped_model = accelerator.unwrap_model(multi_classification_model)
#                 print('############ unwrapped_model model ##############')
#                 print(unwrapped_model)
                # save
#                 accelerator.save(unwrapped_model.state_dict(), config.save_path)
                if config.use_lora:
                    unwrapped_model.save_pretrained(config.lora_save_path)
                else:
                    accelerator.save(unwrapped_model.state_dict(), config.save_path)
                print(report, '\n', confusion)
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bert classification')
    parser.add_argument("-c", "--config", type=str, default="./config_multilabel.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    print(type(config.lr), type(config.batch_size))
    config.lr = float(config.lr)
    train(config)
