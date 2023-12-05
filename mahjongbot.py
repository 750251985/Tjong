#!/usr/bin/env python
# encoding: utf-8
'''
@author: lb
@time: 2023/02/15 01:27
深度学习
'''
import json
import time

from MahjongGB import MahjongFanCalculator
from MahjongGB import MahjongShanten
from MahjongGB import ThirteenOrphansShanten
from MahjongGB import SevenPairsShanten
from MahjongGB import HonorsAndKnittedTilesShanten
from MahjongGB import KnittedStraightShanten
from MahjongGB import RegularShanten
import torch
import torch.optim as optim
from enum import Enum
import numpy as np
import orch.nn as nn
import torch.nn.functional as F
import sys
import os
import random
from copy import deepcopy
import argparse

from TIT_1head import Config_1head, TIT_1head

parser = argparse.ArgumentParser(description='Model')
parser.add_argument('-t', '--train', action='store_true', default=True, help='whether to train model')
parser.add_argument('-s', '--save', action='store_true', default=True, help='whether to store model')
parser.add_argument('-l', '--load', action='store_true', default=True, help='whether to load model')
parser.add_argument('-lp', '--load_path', type=str, default='./models/dl_TIT_1head_afternorm_1680087559.7798574.pt', help='from where to load model')
parser.add_argument('-sp', '--save_path', type=str, default='./models/', help='save model path')
parser.add_argument('-b', '--batch_size', type=int, default=1024, help='training batch size')
parser.add_argument('--training_data', type=str, default='./data', help='path to training data folder')
parser.add_argument('--botzone', action='store_true', default=False, help='whether to run the model on botzone')
parser.add_argument('-pi', '--print_interval', type=int, default=102400, help='how often to print')
parser.add_argument('-si', '--save_interval', type=int, default=1024000, help='how often to save')
args = parser.parse_args()
args.model_name = 'dl_TIT_1head'
args.cuda = 'cuda:0'

class requests(Enum):
    initialHand = 1
    drawCard = 2
    DRAW = 4
    PLAY = 5
    PENG = 6
    CHI = 7
    GANG = 8
    BUGANG = 9
    MINGGANG = 10
    ANGANG = 11

class responses(Enum):
    Pass = 0
    Discard = 1
    Win = 2
    # 需要区分明杠和暗杠
    MINGGANG = 3
    ANGANG = 4
    BUGANG = 5
    Pon = 6
    Chow = 7
    Listen = 8
    need_cards = [0, 1, 1, 4, 4, 4, 3, 3, 1]

class cards(Enum):
    # 饼万条
    B = 0
    W = 9
    T = 18
    # 风
    F = 27
    # 箭牌
    J = 31

class myModel(nn.Module):
    def __init__(self, obs_array_dim, num_extra_feats, num_cards, num_actions):
        super(myModel, self).__init__()
        activation_fn = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'gelu': nn.GELU}
        self.config = Config_1head(
            patch_dim=num_cards,
            num_blocks=3,
            features_dim=1+(num_actions-1)*num_cards,  # 输出1+7*34
            embed_dim_inner=1024,
            num_heads_inner=4,
            attention_dropout_inner=0.,
            ffn_dropout_inner=0.1,
            context_len_inner=int(np.ceil((obs_array_dim+num_extra_feats)/num_cards)),
            embed_dim_outer=1024,
            num_heads_outer=8,
            attention_dropout_outer=0.,
            ffn_dropout_outer=0.1,
            context_len_outer=4,    # K outer的切片个数，跨越的时间步幅
            observation_type='array',
            C=1, H=0, W=0, D=obs_array_dim+num_extra_feats,
            activation_fn_inner=activation_fn['gelu'],
            activation_fn_outer=activation_fn['gelu'],
            activation_fn_other=activation_fn['tanh'],
            dim_expand_inner=2,
            dim_expand_outer=2,
            have_position_encoding=1,
            share_tit_blocks=0  # EnhancedTIT使用1，vanilla_tit 0
        )
        self.tit = TIT_1head(self.config)
        print(self.config.__dict__)

    # discard,chow Kong,
    def forward(self, card_feats, extra_feats, device):
        obs = np.concatenate((card_feats,extra_feats), axis=1)
        obs_tensor = torch.from_numpy(obs).to(device).to(torch.float32)
        probs = self.tit(obs_tensor)
        return probs
    def train_backward(self, loss, optim):
        optim.zero_grad()
        loss.backward()
        optim.step()


# 添加 牌墙无牌不能杠
class MahjongHandler():
    def __init__(self, train, model_path, load_model=False, save_model=True, batch_size=1000):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device(args.cuda if use_cuda else "cpu")
        print('using ' + str(self.device))
        self.train = train
        self.training_data = {"doc":[],"training":[]}
        self.model_path = model_path
        self.load_model = load_model
        self.save_model = save_model
        self.total_cards = 34
        self.learning_rate = 1e-4
        self.total_actions = len(responses) - 2
        self.model = myModel(
            obs_array_dim=10*34,
            num_extra_feats=20,
            num_cards=self.total_cards,
            num_actions=self.total_actions
        ).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.best_precision = 0
        self.batch_size = batch_size
        self.print_interval = args.print_interval
        self.save_interval = args.save_interval
        self.round_count = 0
        self.match = np.zeros(self.total_actions)
        self.count = np.zeros(self.total_actions)
        if self.load_model:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            try:
                self.round_count = checkpoint['progress']
            except KeyError:
                self.round_count = 0
        if not train:
            self.model.eval()
        self.reset(True)
        self.t = None

    def reset(self, initial=False):
        self.hand_free = np.zeros(self.total_cards, dtype=int)  #自由手牌
        self.history = np.zeros(self.total_cards, dtype=int)  #全局历史记录
        self.player_history = np.zeros((4, self.total_cards), dtype=int)  # 玩家出牌历史记录
        self.player_on_table = np.zeros((4, self.total_cards), dtype=int)  # 玩家弃牌
        self.hand_fixed = self.player_on_table[0]  #玩家鸣牌
        self.player_last_play = np.zeros(4, dtype=int)  #四个人的上一张牌
        self.player_angang = np.zeros(4, dtype=int)  #暗杠，特殊标记
        self.fan_count = 0  #番数
        self.hand_fixed_data = []
        self.turnID = 0  #回合数
        self.tile_count = [21, 21, 21, 21]  # 大众麻将中无效，每个人的剩余牌数，大众麻将中使用总剩余牌数
        self.myPlayerID = 0
        self.quan = 0
        self.prev_request = ''
        self.an_gang_card = ''
        # test training acc
        if self.train and not initial and self.round_count % self.print_interval == 0:
            # 获取训练数据
            training_data = self.training_data['doc']
            with torch.no_grad():
                print('-' * 50)
                # 训练数据分为两类：'弃牌', '鸣牌‘分别训练
                for kind in ['弃牌', '鸣牌']:
                    data = training_data[kind]
                    probs = torch.tensor([]).to(device=self.device)
                    start = 0
                    flag = True
                    it = iter(range(self.batch_size, len(data["target"]), self.batch_size))
                    while flag:
                        x = next(it, len(data["target"]))
                        if len(data["card_feats"][start:x]) > 0:
                            probs_temp = self.model(np.array(data['card_feats'][start:x]),
                                               np.array(data['extra_feats'][start:x]), self.device, kind)  # 概率输出
                            start = x
                            probs = torch.cat((probs, probs_temp), dim=0)
                            if x == len(data["target"]):
                                flag = False
                    target_tensor = torch.from_numpy(np.array(data['target'])).to(device=self.device, dtype=torch.int64)
                    losses = F.cross_entropy(probs, target_tensor).to(torch.float32)
                    mask_tensor = torch.from_numpy(np.array(data['mask'])).to(torch.float32).to(self.device)
                    pred = torch.argmax(probs*mask_tensor, dim=1)
                    acc = (pred == target_tensor).sum() / probs.shape[0]
                    print('{}: acc {} loss {}'.format(kind, float(acc.cpu()), float(losses.mean().cpu())))
                self.training_data.save_data(self.round_count)
        if self.save_model and not initial and self.round_count % self.save_interval == 0:
            state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                     'progress': self.round_count}
            timestemp=time.time()
            print("save_model: "+args.model_name +"_"+str(timestemp)+".pt")
            torch.save(state, args.save_path + args.model_name +"_"+str(timestemp)+".pt", _use_new_zipfile_serialization=False)
        self.round_count += 1
        self.loss = []

    def step_for_train(self, pon_hand, kon_hand, chow_hand, hand, history, dealer, seat, response_target=None):
        # 构建历史记录，为后续输入特征做准备
        request = self.build_hand_history(pon_hand.split(","), kon_hand.split(","), chow_hand.split(","), hand, history,
                                          dealer, seat)
        # 动作掩码
        available_action_mask, available_card_mask = self.build_available_action_mask(request)
        # 构建牌特征
        card_feats = self.build_input(self.hand_free, self.history, self.player_history,
                                      self.player_on_table, self.player_last_play, available_card_mask)
        available_card_mask = available_card_mask.flatten(order='C')
        available_card_mask = available_card_mask[33:]
        # 构建局势特征
        extra_feats = np.concatenate((self.player_angang[1:], [self.hand_free.sum()],
                                      available_action_mask, *np.eye(4)[[self.quan, self.myPlayerID]],
                                      self.tile_count))
        # 无效数据过滤
        def judge_response(available_action_mask):
            if available_action_mask.sum() == available_action_mask[responses.PASS.value]:
                return False
            return True

        if self.train and response_target is not None and judge_response(available_action_mask): # 已经排除了only pass
            rand = random.uniform(0, 100)
            if rand > 90:
                training_data = self.training_data.doc
            else:
                training_data = self.training_data.training
                for kind in ['弃牌', '鸣牌']:
                    # 数据满足一批大小，训练一次
                    if len(training_data[kind]['card_feats']) >= self.batch_size:
                        data = training_data[kind]
                        probs = self.model(np.array(data['card_feats']),
                                           np.array(data['extra_feats']), self.device, kind)
                        target_tensor = torch.from_numpy(np.array(data['target'])).to(device=self.device,
                                                                                      dtype=torch.int64)

                        losses = F.cross_entropy(probs, target_tensor).to(torch.float32)
                        #反向传播
                        self.model.train_backward(losses, self.optimizer)
                        # 清空数据缓存
                        self.training_data.reset(kind)

            response_target = response_target.split(' ')
            response_name = response_target[0]
            if response_name == 'GANG':
                if len(response_target) > 1:
                    response_name = 'ANGANG'
                    self.an_gang_card = response_target[-1]
                else:
                    response_name = 'MINGGANG'
            if available_action_mask.sum() > 1:  # 最少是一个play动作，>1 代表有动作发生,
                if responses[response_name] not in [responses.PLAY, responses.CHI, responses.PENG, responses.ANGANG, responses.BUGANG]:
                    data = training_data["鸣牌"]
                    data['card_feats'].append(card_feats.flatten(order='C'))
                    data['mask'].append(available_card_mask)
                    data['extra_feats'].append(extra_feats)
                    target_index = 0
                    if responses[response_name].value > 0:  # 非PASS，为：明杠、HU
                        target_index = 1 + (responses[response_name].value - 1) * 34 + self.getCardInd(
                            request[-1])
                    data['target'].append(target_index)
                if responses[response_name] in [responses.CHI, responses.ANGANG, responses.BUGANG]:
                    data = training_data["鸣牌"]
                    data['mask'].append(available_card_mask)
                    data['card_feats'].append(card_feats.flatten(order='C'))
                    data['extra_feats'].append(extra_feats)
                    target_index = 0
                    if self.getCardInd(response_target[1]) > 0:
                        target_index = 1+(responses[response_name].value-1)*34 + self.getCardInd(response_target[1])
                    data['target'].append(target_index)

            if responses[response_name] in [responses.PLAY, responses.CHI, responses.PENG]: # 需要弃牌
                if responses[response_name] == responses.PLAY:
                    play_target = self.getCardInd(response_target[1])
                    card_mask = available_card_mask
                else:
                    if responses[response_name] == responses.CHI:
                        chi_peng_ind = self.getCardInd(response_target[1])
                    else:
                        chi_peng_ind = self.getCardInd(request[-1])
                    play_target = self.getCardInd(response_target[-1])
                    card_feats, extra_feats, card_mask = self.simulate_chi_peng(request, responses[response_name],
                                                                                chi_peng_ind, True)
                    card_mask = card_mask.flatten(order='C')
                    card_mask = card_mask[33:]
                data = training_data['弃牌']
                data['card_feats'].append(card_feats.flatten(order='C'))
                data['extra_feats'].append(extra_feats)
                data['mask'].append(card_mask)
                target_index = 0
                if play_target > 0:
                    target_index = 1 + play_target  # responses['play'].value - 1 = 0
                data['target'].append(target_index)

        self.prev_request = request
        self.turnID += 1

    # 实战中做决策的步骤：
    def step(self, pon_hand, kon_hand, chow_hand, hand, history, dealer, seat):

        request = self.build_hand_history(pon_hand.split(","), kon_hand.split(","), chow_hand.split(","), hand, history,
                                          dealer, seat)
        if self.turnID <= 1:
            response = 'PASS'
        else:
            available_action_mask, available_card_mask = self.build_available_action_mask(request)
            card_feats = self.build_input(self.hand_free, self.history, self.player_history,
                                          self.player_on_table, self.player_last_play, available_card_mask)
            extra_feats = np.concatenate((self.player_angang[1:], [self.hand_free.sum()],
                                          available_action_mask, *np.eye(4)[[self.quan, self.myPlayerID]],
                                          self.tile_count))
            card_feats = card_feats.flatten(order='C')
            available_card_mask = available_card_mask.flatten("C")[33:]
            cards = []
            probs = self.model(np.array([card_feats]), np.array([extra_feats]), self.device)
            vals = probs.data.cpu().numpy()[0]
            vals = vals * available_card_mask
            ind = np.argmax(vals)
            if ind == 0:
                action = ind
            else:
                action = int((ind - 1 + 34) / 34)
                cards.append(int((ind - 1 + 34)) % 34)

            if responses(action) in [responses.CHI, responses.PENG]:  # 根据交互方式，后跟打出的牌
                if responses(action) == responses.CHI:
                    chi_peng_ind = cards[0]
                else:
                    chi_peng_ind = self.getCardInd(request[-1])
                card_feats, extra_feats, card_mask = self.simulate_chi_peng(request, responses(action), chi_peng_ind,
                                                                            True)
                card_feats = card_feats.flatten("C")
                card_mask = card_mask.flatten("C")[33:]
                card_probs = self.model(np.array([card_feats]), np.array([extra_feats]), self.device)
                card_vals = card_probs.data.cpu().numpy()[0]
                card_vals = card_vals * card_mask
                ind = np.argmax(card_vals)
                cards.append(int((ind - 1) % 34))
            response = self.build_output(responses(action), cards)
            if responses(action) == responses.ANGANG:
                self.an_gang_card = self.getCardName(cards[0])

        self.prev_request = request
        self.turnID += 1
        return response

    def build_input(self, my_free, history, play_history, on_table, last_play, available_card_mask):
        temp = np.array([my_free, 4 - history])
        one_hot_last_play = np.eye(self.total_cards)[last_play]
        card_feats = np.concatenate((temp, on_table, play_history, one_hot_last_play, available_card_mask))
        return card_feats

    def build_result_summary(self, response, response_target):
        if response_target.split(' ')[0] == 'CHI':
            print(response, response_target)
        resp_name = response.split(' ')[0]
        resp_target_name = response_target.split(' ')[0]
        if resp_target_name == 'GANG':
            if len(response_target.split(' ')) > 1:
                resp_target_name = 'ANGANG'
            else:
                resp_target_name = 'MINGGANG'
        if resp_name == 'GANG':
            if len(response.split(' ')) > 1:
                resp_name = 'ANGANG'
            else:
                resp_name = 'MINGGANG'
        self.count[responses[resp_target_name].value] += 1
        if response == response_target:
            self.match[responses[resp_name].value] += 1

    def simulate_chi_peng(self, request, response, chi_peng_ind, only_feature=False):
        last_card_played = self.getCardInd(request[-1])
        available_card_mask = np.zeros((self.total_actions, self.total_cards), dtype=int)
        available_card_play_mask = available_card_mask[responses['PLAY'].value]
        my_free, on_table = self.hand_free.copy(), self.player_on_table.copy()
        if response == responses.CHI:
            my_free[chi_peng_ind - 1:chi_peng_ind + 2] -= 1
            my_free[last_card_played] += 1
            on_table[0][chi_peng_ind - 1:chi_peng_ind + 2] += 1
            is_chi = True
        else:
            chi_peng_ind = last_card_played
            my_free[last_card_played] -= 2
            on_table[0][last_card_played] += 3
            is_chi = False
        self.build_available_card_mask(available_card_play_mask, responses.PLAY, last_card_played,
                                       chi_peng_ind=chi_peng_ind, is_chi=is_chi)
        card_feats = self.build_input(my_free, self.history, self.player_history, on_table, self.player_last_play, available_card_mask)
        if only_feature:
            action_mask = np.zeros(self.total_actions, dtype=int)
            action_mask[responses.PLAY.value] = 1
            extra_feats = np.concatenate((self.player_angang[1:], [my_free.sum()], action_mask,
                                          *np.eye(4)[[self.quan, self.myPlayerID]], self.tile_count))
            return card_feats, extra_feats, available_card_mask
        card_play_probs = self.model(card_feats, self.device, decide_cards=True, card_mask=available_card_play_mask)
        return card_play_probs

    def build_available_action_mask(self, request):
        available_action_mask = np.zeros(self.total_actions, dtype=int)
        available_card_mask = np.zeros((self.total_actions, self.total_cards), dtype=int)
        requestID = int(request[0])
        playerID = int(request[1])
        myPlayerID = self.myPlayerID
        try:
            last_card = request[-1]
            last_card_ind = self.getCardInd(last_card)
        except:
            last_card = ''
            last_card_ind = 0
        # 摸牌回合
        if requests(requestID) == requests.drawCard:
            for response in [responses.PLAY, responses.ANGANG, responses.BUGANG]:
                if self.tile_count[self.myPlayerID] == 0 and response in [responses.ANGANG, responses.BUGANG]:
                    continue
                self.build_available_card_mask(available_card_mask[response.value], response, last_card_ind)
                if available_card_mask[response.value].sum() > 0:
                    available_action_mask[response.value] = 1
            # 杠上开花
            if requests(int(self.prev_request[0])) in [requests.ANGANG, requests.BUGANG]:
                isHu = self.judgeHu(last_card, playerID, True)
            # 这里胡的最后一张牌其实不一定是last_card，因为可能是吃了上家胡，需要知道上家到底打的是哪张
            else:
                isHu = self.judgeHu(last_card, playerID, False)
            if isHu >= 8:
                available_action_mask[responses.HU.value] = 1
                self.fan_count = isHu
        else:
            available_action_mask[responses.PASS.value] = 1
            available_card_mask[responses.PASS.value, :] = 1
            # 别人出牌
            if requests(requestID) in [requests.PENG, requests.CHI, requests.PLAY]:
                if playerID != myPlayerID:
                    for response in [responses.PENG, responses.MINGGANG, responses.CHI]:
                        # 不是上家
                        if response == responses.CHI and (self.myPlayerID - playerID) % 4 != 1:
                            continue
                        # 最后一张，不能吃碰杠
                        if self.tile_count[(playerID + 1) % 4] == 0:
                            continue
                        self.build_available_card_mask(available_card_mask[response.value], response, last_card_ind)
                        if available_card_mask[response.value].sum() > 0:
                            available_action_mask[response.value] = 1
                    # 是你必须现在决定要不要抢胡
                    isHu = self.judgeHu(last_card, playerID, False, dianPao=True)
                    if isHu >= 8:
                        available_action_mask[responses.HU.value] = 1
                        self.fan_count = isHu
            # 抢杠胡
            if requests(requestID) == requests.BUGANG and playerID != myPlayerID:
                isHu = self.judgeHu(last_card, playerID, True, dianPao=True)
                if isHu >= 8:
                    available_action_mask[responses.HU.value] = 1
                    self.fan_count = isHu
        return available_action_mask, available_card_mask

    def build_available_card_mask(self, available_card_mask, response, last_card_ind, chi_peng_ind=None, is_chi=False):
        if response == responses.PLAY:
            # 正常出牌
            if chi_peng_ind is None:
                for i, card_num in enumerate(self.hand_free):
                    if card_num > 0:
                        available_card_mask[i] = 1
            else:
                # 吃了再出
                if is_chi:
                    for i, card_num in enumerate(self.hand_free):
                        if i in [chi_peng_ind - 1, chi_peng_ind, chi_peng_ind + 1] and i != last_card_ind:
                            if card_num > 1:
                                available_card_mask[i] = 1
                        elif card_num > 0:
                            available_card_mask[i] = 1
                else:
                    for i, card_num in enumerate(self.hand_free):
                        if i == chi_peng_ind:
                            if card_num > 2:
                                available_card_mask[i] = 1
                        elif card_num > 0:
                            available_card_mask[i] = 1
        elif response == responses.PENG:
            if self.hand_free[last_card_ind] >= 2:
                available_card_mask[last_card_ind] = 1
        elif response == responses.CHI:
            # 数字牌才可以吃
            if last_card_ind < cards.F.value:
                card_name = self.getCardName(last_card_ind)
                card_number = int(card_name[1])
                for i in [-1, 0, 1]:
                    middle_card = card_number + i
                    if middle_card >= 2 and middle_card <= 8:
                        can_chi = True
                        for card in range(last_card_ind + i - 1, last_card_ind + i + 2):
                            if card != last_card_ind and self.hand_free[card] == 0:
                                can_chi = False
                        if can_chi:
                            available_card_mask[last_card_ind + i] = 1
        elif response == responses.ANGANG:
            for card in range(len(self.hand_free)):
                if self.hand_free[card] == 4:
                    available_card_mask[card] = 1
        elif response == responses.MINGGANG:
            if self.hand_free[last_card_ind] == 3:
                available_card_mask[last_card_ind] = 1
        elif response == responses.BUGANG:
            for card in range(len(self.hand_free)):
                if self.hand_fixed[card] == 3 and self.hand_free[card] == 1:
                    for card_combo in self.hand_fixed_data:
                        if card_combo[1] == self.getCardName(card) and card_combo[0] == 'PENG':
                            available_card_mask[card] = 1
        else:
            available_card_mask[last_card_ind] = 1
        return available_card_mask

    def judgeHu(self, last_card, playerID, isGANG, dianPao=False):
        hand = []
        for ind, cardcnt in enumerate(self.hand_free):
            for _ in range(cardcnt):
                hand.append(self.getCardName(ind))
        if self.history[self.getCardInd(last_card)] == 4:
            isJUEZHANG = True
        else:
            isJUEZHANG = False
        if self.tile_count[(playerID + 1) % 4] == 0:
            isLAST = True
        else:
            isLAST = False
        if not dianPao:
            hand.remove(last_card)
        try:
            ans = MahjongFanCalculator(tuple(self.hand_fixed_data), tuple(hand), last_card, 0,
                                       playerID == self.myPlayerID,
                                       isJUEZHANG, isGANG, isLAST, self.myPlayerID, self.quan)
        except Exception as err:
            if str(err) == 'ERROR_NOT_WIN':
                return 0
        else:
            fan_count = 0
            for fan in ans:
                fan_count += fan[0]
            return fan_count

    def build_hand_history(self, pon_hand, kon_hand, chow_hand, hand, history, dealer, seat):
        request = []
        myPlayerID = seat
        quan = dealer
        self.myPlayerID = int(myPlayerID)
        self.other_players_id = [(self.myPlayerID - i) % 4 for i in range(4)]
        self.player_positions = {}
        for position, id in enumerate(self.other_players_id):
            self.player_positions[id] = position
        self.quan = int(quan)
        for i in range(int(len(hand) / 2)):
            cardInd = self.getCardInd(hand[2 * i:2 * i + 2])
            self.hand_free[cardInd] += 1
        for i in range(len(history)):
            request = list(history[i].split(","))
            if len(request) == 3:
                self.maintain_status(request, self.history, self.player_history,
                                     self.player_on_table, self.player_last_play, self.player_angang,
                                     self.player_listen)
        return request

    def maintain_status(self, request, my_free, history, play_history, on_table, last_play, angang):
        playerID = int(request[0])
        actionId = int(requests[request[1]].value)
        player_position = self.player_positions[playerID]
        if requests(actionId) in [requests.drawCard, requests.DRAW]:
            self.tile_count[playerID] -= 1
        elif requests(actionId) == requests.Discard:
            play_card = self.getCardInd(request[-1])
            play_history[player_position][play_card] += 1
            last_play[player_position] = play_card
            history[play_card] += 1
            if self.tile_count[playerID] > 1:
                self.tile_count[playerID] -= 1
        elif requests(actionId) == requests.Pon:
            last_card_ind = self.getCardInd(self.prev_request[-1][0:2])
            on_table[player_position][last_card_ind] = 3
            if player_position != 0:
                history[last_card_ind] += 2
            else:
                # 自己碰，记录Pon来源于哪个玩家
                last_player = int(self.prev_request[0])
                last_player_pos = self.player_positions[last_player]
                self.hand_fixed_data.append(('PENG', self.getMyCardName(self.prev_request[-1]), last_player_pos))
        elif requests(actionId) == requests.Chow:
            last_card_ind = self.getCardInd(self.prev_request[-1])
            middle_card = request[-1][2:4]
            middle_card_ind = self.getCardInd(middle_card)
            on_table[player_position][middle_card_ind - 1:middle_card_ind + 2] += 1
            if player_position != 0:
                history[middle_card_ind - 1:middle_card_ind + 2] += 1
                history[last_card_ind] -= 1
            else:
                # Chow,中间牌名，123代表上家的牌是第几张
                self.hand_fixed_data.append(
                    ('CHI', self.getMyCardName(middle_card), last_card_ind - middle_card_ind + 2))
        elif requests(actionId) == requests.Kon:
            if self.prev_request == [] and on_table[player_position][self.getCardInd(request[-1][0:2])] == 0:
                # 暗杠
                request[1] = requests.ANGANG.name
                if player_position == 0:
                    gangCard = request[-1][0:2]
                    # print(gangCard)
                    if gangCard == '':
                        print(self.prev_request)
                        print(request)
                    gangCardInd = self.getCardInd(gangCard)
                    # 记录gang来源于哪个玩家（可能来自自己，暗杠）
                    self.hand_fixed_data.append(('GANG', self.getMyCardName(gangCard), 0))
                    on_table[0][gangCardInd] = 4
                    self.kon_plus += 1
                else:
                    angang[player_position] += 1
            else:
                if self.prev_request[-1] != request[-1][0:2]:
                    if on_table[player_position][self.getCardInd(request[-1][0:2])] == 0:
                        # 暗杠
                        request[1] = requests.ANGANG.name
                        if player_position == 0:
                            gangCard = request[-1][0:2]
                            # print(gangCard)
                            if gangCard == '':
                                print(self.prev_request)
                                print(request)
                            gangCardInd = self.getCardInd(gangCard)
                            # 记录gang来源于哪个玩家（可能来自自己，暗杠）
                            self.hand_fixed_data.append(('GANG', self.getMyCardName(gangCard), 0))
                            on_table[0][gangCardInd] = 4
                            self.kon_plus += 1
                        else:
                            angang[player_position] += 1
                    else:
                        # 补杠
                        bugang_card_ind = self.getCardInd(request[-1])
                        request[1] = requests.BUGANG.name
                        history[bugang_card_ind] = 4
                        on_table[player_position][bugang_card_ind] = 4
                        if player_position == 0:
                            self.kon_plus += 1
                            for id, comb in enumerate(self.hand_fixed_data):
                                if comb[1] == self.getMyCardName(request[-1][0:2]):
                                    self.hand_fixed_data[id] = ('GANG', comb[1], comb[2])
                                    break
                else:
                    # 明杠
                    gangCardInd = self.getCardInd(self.prev_request[-1][0:2])
                    request[1] = requests.MINGGANG.name
                    history[gangCardInd] = 4
                    on_table[player_position][gangCardInd] = 4
                    if player_position == 0:
                        # 记录gang来源于哪个玩家
                        last_player = int(self.prev_request[0])
                        self.hand_fixed_data.append(
                            ('GANG', self.getMyCardName(self.prev_request[-1]), self.player_positions[last_player]))
                        self.kon_plus += 1
        elif requests(actionId) == requests.Listen:
            play_card = self.getCardInd(request[-1])
            play_history[player_position][play_card] += 1
            last_play[player_position] = play_card
            history[play_card] += 1
            listen[player_position] = 1
        self.prev_request = request
        return request

    def build_output(self, response, cards_ind):
        if (responses.need_cards.value[
                response.value] == 1 and response != responses.CHI) or response == responses.PENG:
            response_name = response.name
            if response == responses.ANGANG:
                response_name = 'GANG'
            return '{} {}'.format(response_name, self.getCardName(cards_ind[0]))
        if response == responses.CHI:
            return 'CHI {} {}'.format(self.getCardName(cards_ind[0]), self.getCardName(cards_ind[1]))
        response_name = response.name
        if response == responses.MINGGANG:
            response_name = 'GANG'
        return response_name

    def getCardInd(self, cardName):
        if cardName[0] == 'H':
            print('hua ' + self.fname)
        return cards[cardName[0]].value + int(cardName[1]) - 1

    def getCardName(self, cardInd):
        num = 1
        while True:
            if cardInd in cards._value2member_map_:
                break
            num += 1
            cardInd -= 1
        return cards(cardInd).name + str(num)


def train_main():
    train = args.train
    load = args.load
    save = args.save
    model_path = args.load_path
    batch_size = args.batch_size
    # 麻将处理器，记录处理麻将对局数据
    my_bot = MahjongHandler(train=train, model_path=model_path, load_model=load, save_model=save, batch_size=batch_size)
    count = 0
    restore_count = my_bot.round_count
    trainning_data_files = os.listdir(args.training_data)
    while True:
        for fname in trainning_data_files:
            with open('{}/{}'.format(args.training_data, fname), 'r') as f:
                rounds_data = json.load(f)
                random.shuffle(rounds_data)
                for round_data in rounds_data:
                    for j in range(4):
                        count += 1
                        if count < restore_count:
                            continue
                        if count % 2000 == 0:
                            print(count)
                        train_requests = round_data["requests"][j]
                        first_request = '0 {} {}'.format(j, 0)
                        train_requests.insert(0, first_request)
                        train_responses = ['PASS'] + round_data["responses"][j]
                        for pon_hand, kon_hand, chow_hand, hand, history, dealer, seat, _response in zip(train_requests, train_responses):
                            my_bot.step_for_train(pon_hand, kon_hand, chow_hand, hand, history, dealer, seat, _response, round_data['fname'])
                        my_bot.reset()
        count = 0
        my_bot.round_count = 1
        restore_count = 0

if __name__ == '__main__':
    print(args)
    train_main()  #训练网络