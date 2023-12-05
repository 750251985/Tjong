#!/usr/bin/env python
# encoding: utf-8
'''
@author: lb
@time: 2023/02/15 01:27
深度学习，直接使用numpy解析好的数据
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
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import random
from copy import deepcopy
import argparse
import torch.utils.data as du

from TIT_3heads_3fc import Config_3heads, TIT_3heads

parser = argparse.ArgumentParser(description='Model')
parser.add_argument('-t', '--train', action='store_true', default=True, help='whether to train model')
parser.add_argument('-s', '--save', action='store_true', default=True, help='whether to store model')
parser.add_argument('-l', '--load', action='store_true', default=True, help='whether to load model')
parser.add_argument('-lp', '--load_path', type=str, default='/home/lb/pyproject/Mahjong-RL-botzone-master/models/dl_TIT_3heads_blocks3_fc3_play_best_play71.57.pt', help='from where to load model')
parser.add_argument('-sp', '--save_path', type=str, default='./models/', help='save model path')
parser.add_argument('-b', '--batch_size', type=int, default=1024, help='training batch size')
parser.add_argument('-eps', '--epochs', type=int, default=50, help='training epochs')
parser.add_argument('--training_data', type=str, default='/media/lb/F88876BF88767C46/lb/processed_data1', help='path to training data folder')
parser.add_argument('--botzone', action='store_true', default=False, help='whether to run the model on botzone')
parser.add_argument('-pi', '--print_interval', type=int, default=1, help='how often to print')
parser.add_argument('-si', '--save_interval', type=int, default=1, help='how often to save')
args = parser.parse_args()
args.model_name = 'dl_TIT_3heads_blocks3_fc3_play'
args.cuda = 'cuda:1'

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
    PASS = 0
    PLAY = 1
    HU = 2
    # 需要区分明杠和暗杠
    MINGGANG = 3
    ANGANG = 4
    BUGANG = 5
    PENG = 6
    CHI = 7
    need_cards = [0, 1, 0, 0, 1, 1, 0, 1]
    loss_weight = [6, 1, 9, 6, 6, 6, 6, 6]


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
        self.config = Config_3heads(
            algo='vanilla_tit_ppo',
            patch_dim=num_cards,
            num_blocks=3,
            features_dim_action=num_actions,  # action输出
            features_dim=num_cards,  # chigang判断/play 34 输出
            embed_dim_inner=1024,
            num_heads_inner=4,
            attention_dropout_inner=0.,
            ffn_dropout_inner=0.,
            context_len_inner=int(np.ceil((obs_array_dim+num_extra_feats)/num_cards)),
            embed_dim_outer=1024,
            num_heads_outer=8,
            attention_dropout_outer=0.,
            ffn_dropout_outer=0.,
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
        self.tit = TIT_3heads(self.config)
        print(self.config.__dict__)

    # play, chi_gang,
    def forward(self, card_feats, extra_feats, device, decide_which):
        assert decide_which in ['play', 'action', 'chi_gang']
        # obs = np.concatenate((card_feats,extra_feats), axis=1)
        # obs_tensor = torch.from_numpy(obs).to(device).to(torch.float32)
        obs_tensor = torch.cat((card_feats,extra_feats), dim=1).to(device).to(torch.float32)
        if decide_which == 'play':
            probs = self.tit(obs_tensor,decide_which)
        elif decide_which == 'action':
            probs = self.tit(obs_tensor,decide_which)
        else:
            probs = self.tit(obs_tensor, decide_which)
        return probs
    def train_backward(self, loss, optim):
        optim.zero_grad()
        loss.backward()
        optim.step()


# 添加 牌墙无牌不能杠
class MahjongHandler():
    def __init__(self, train, model_path, load_model=False, save_model=True, botzone=False, batch_size=1000):
        use_cuda = torch.cuda.is_available()
        self.botzone = botzone
        self.device = torch.device(args.cuda if use_cuda else "cpu")
        if not botzone:
            print('using ' + str(self.device))
        self.train = train
        self.model_path = model_path
        self.load_model = load_model
        self.save_model = save_model
        self.total_cards = 34
        self.learning_rate = 1e-4
        self.action_loss_weight = responses.loss_weight.value
        self.action_weight = torch.from_numpy((np.array(responses.loss_weight.value))).to(device=self.device,
                                                                                          dtype=torch.float32)
        self.card_loss_weight = 2
        self.total_actions = len(responses) - 2
        self.model = myModel(
            obs_array_dim=22*34,
            num_extra_feats=24,
            num_cards=self.total_cards,
            num_actions=self.total_actions
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.best_precision = 0
        self.batch_size = batch_size
        self.print_interval = args.print_interval
        self.save_interval = args.save_interval
        # self.round_count = 0
        self.match = np.zeros(self.total_actions)
        self.count = np.zeros(self.total_actions)
        if self.load_model:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            if not botzone:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                # try:
                #     self.round_count = checkpoint['progress']
                # except KeyError:
                #     self.round_count = 0
        if not train:
            self.model.eval()
        self.reset()
        self.t = None

    def reset(self):
        self.hand_free = np.zeros(self.total_cards, dtype=int)
        self.history = np.zeros(self.total_cards, dtype=int)
        self.player_history = np.zeros((4, self.total_cards), dtype=int)
        self.player_on_table = np.zeros((4, self.total_cards), dtype=int)
        self.hand_fixed = self.player_on_table[0]
        self.player_last_play = np.zeros(4, dtype=int)
        self.player_angang = np.zeros(4, dtype=int)
        self.fan_count = 0
        self.hand_fixed_data = []
        self.turnID = 0
        self.tile_count = [21, 21, 21, 21]
        self.myPlayerID = 0
        self.quan = 0
        self.prev_request = ''
        self.an_gang_card = ''
        self.recorder = {'play': [], 'action': [], 'chi_gang': []}
        # self.round_count += 1

    def save_bestandcurrent_model(self, acc):
        state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'acc': acc}
        timestemp=time.time()
        print("save_model: "+args.model_name + "_" + str(timestemp)+".pt")
        torch.save(state, args.save_path + args.model_name + "_" + str(timestemp)+".pt", _use_new_zipfile_serialization=False)
        torch.save(self.recorder, args.save_path+args.model_name + "_recorder"+str(timestemp), _use_new_zipfile_serialization=False)
        self.recorder = {'play': [], 'action': [], 'chi_gang': []}
        if acc > self.best_precision:
            path = args.save_path + args.model_name + "_" + 'best_play{:.2}'.format(acc) + ".pt"
            torch.save(state, path, _use_new_zipfile_serialization=False)
            self.best_precision = acc.cpu()
    def step_for_test(self, kinds, epoch):
        self.model.eval()
        acc_play = 0.0
        for kind in kinds:
            test_data_files = os.listdir(args.training_data + '/{}/test'.format(kind))
            right = torch.tensor(0).to(device=self.device, dtype=torch.float32)
            total = torch.tensor(1).to(device=self.device, dtype=torch.float32)
            losses = torch.tensor(0).to(device=self.device, dtype=torch.float32)
            print('-' * 25, epoch, '-' * 25)
            for fname in test_data_files:
                dataload = np.load(args.training_data + '/{}/test/{}'.format(kind, fname), allow_pickle=True)
                datadict = dataload.tolist()
                target = torch.from_numpy(np.array(datadict['target'])).to(torch.int64)
                mask = torch.from_numpy(np.array(datadict['mask'])).to(torch.float32)
                card_feats = torch.from_numpy(np.array(datadict['card_feats'])).to(torch.float32)
                extra_feats = torch.from_numpy(np.array(datadict['extra_feats'])).to(torch.float32)
                train_loader = du.DataLoader(du.TensorDataset(card_feats, extra_feats, mask, target), batch_size=self.batch_size,
                                             shuffle=False)
                for batch_idx, (card_feats, extra_feats, mask, target) in enumerate(train_loader):
                    with torch.no_grad():
                        probs = self.model(card_feats, extra_feats, self.device, kind)
                        target_tensor = target.to(device=self.device, dtype=torch.int64)
                        losses = F.cross_entropy(probs, target_tensor).to(torch.float32)
                        mask_tensor = mask.to(torch.float32).to(self.device)
                        pred = torch.argmax(probs * mask_tensor, dim=1)
                        right += (pred == target_tensor).sum()
                        total += int(probs.shape[0])
                        # 披露细节
                        # if kind == 'action':
                        #     counts = np.zeros(self.total_actions, dtype=float)
                        #     matches = np.zeros(self.total_actions, dtype=float)
                        #     for p, t in zip(list(pred.cpu()), list(target_tensor.cpu())):
                        #         p = int(p)
                        #         t = int(t)
                        #         if p == t:
                        #             matches[p] += 1
                        #         counts[t] += 1
                        #     accs = matches / counts
                        #     for i in range(self.total_actions):
                        #         print('{}: {},{} {:.2%}'.format(responses(i).name, matches[i], counts[i], accs[i]))
            acc = right/total
            self.recorder.get(kind).append((losses, acc, time.time()))
            print('{}: acc {:.2%} loss {}'.format(kind, float(acc.cpu()), float(losses.mean().cpu())))
            print('-' * 25, epoch, '-' * 25)
            if kind =='play':
                acc_play = acc
        return acc_play

    def step_for_train(self, train_loader, kind, epoch, length, findex):
        self.model.train()
        for batch_idx, (card_feats, extra_feats, mask, target) in enumerate(train_loader):
            probs = self.model(card_feats,
                               extra_feats, self.device, kind)
            target_tensor = target.to(device=self.device, dtype=torch.int64)
            if kind == 'action':
                losses = F.cross_entropy(probs, target_tensor, weight=self.action_weight).to(torch.float32)
            else:
                losses = F.cross_entropy(probs, target_tensor).to(torch.float32)
            self.model.train_backward(losses, self.optimizer)

            if (batch_idx) / 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, findex, length, findex / length, losses.item()))

    def step(self, request=None, response_target=None, fname=None):
        if fname:
            self.fname = fname
        if request is None:
            if self.turnID == 0:
                inputJSON = json.loads(input())
                request = inputJSON['requests'][0].split(' ')
            else:
                request = input().split(' ')
        else:
            request = request.split(' ')

        request = self.build_hand_history(request)
        if self.turnID <= 1:
            response = 'PASS'
        else:
            def make_decision(probs):
                vals = probs.data.cpu().numpy()[0]
                max = -1000000
                decision = 0
                for i, val in enumerate(vals):
                    if val != 0 and val > max:
                        max = val
                        decision = i
                return decision

            available_action_mask, available_card_mask = self.build_available_action_mask(request)
            card_feats = self.build_input(self.hand_free, self.history, self.player_history,
                                          self.player_on_table, self.player_last_play, available_card_mask)
            extra_feats = np.concatenate((self.player_angang[1:], [self.hand_free.sum()],
                                          available_action_mask, *np.eye(4)[[self.quan, self.myPlayerID]],
                                          self.tile_count))
            action_probs = self.model(np.array([card_feats]),
                                      np.array([extra_feats]), self.device,
                                      'action', np.array([available_action_mask]))
            action = make_decision(action_probs)
            cards = []
            if responses(action) in [responses.CHI, responses.ANGANG, responses.BUGANG]:
                card_probs = self.model(np.array([card_feats]),
                                        np.array([extra_feats]), self.device,
                                        'chi_gang', np.array([available_card_mask[action]]))
                card_ind = make_decision(card_probs)
                cards.append(card_ind)
            if responses(action) in [responses.PLAY, responses.CHI, responses.PENG]:
                if responses(action) == responses.PLAY:
                    card_mask = available_card_mask[action]
                else:
                    if responses(action) == responses.CHI:
                        chi_peng_ind = cards[0]
                    else:
                        chi_peng_ind = self.getCardInd(request[-1])
                    card_feats, extra_feats, card_mask = self.simulate_chi_peng(request, responses(action),
                                                                                chi_peng_ind, True)
                card_probs = self.model(np.array([card_feats]),
                                        np.array([extra_feats]), self.device,
                                        'play', np.array([card_mask]))
                card_ind = make_decision(card_probs)
                cards.append(card_ind)
            response = self.build_output(responses(action), cards)
            if responses(action) == responses.ANGANG:
                self.an_gang_card = self.getCardName(cards[0])

        self.prev_request = request
        self.turnID += 1
        if self.botzone:
            print(json.dumps({"response": response}))
        else:
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
            return card_feats, extra_feats, available_card_play_mask
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
                if not self.botzone:
                    print(hand, last_card, self.hand_fixed_data)
                    print(self.fname)
                    print(err)
                    return 0
        else:
            fan_count = 0
            for fan in ans:
                fan_count += fan[0]
            return fan_count

    def build_hand_history(self, request):
        # 第0轮，确定位置
        if self.turnID == 0:
            _, myPlayerID, quan = request
            self.myPlayerID = int(myPlayerID)
            self.other_players_id = [(self.myPlayerID - i) % 4 for i in range(4)]
            self.player_positions = {}
            for position, id in enumerate(self.other_players_id):
                self.player_positions[id] = position
            self.quan = int(quan)
            return request
        # 第一轮，发牌
        if self.turnID == 1:
            for i in range(5, 18):
                cardInd = self.getCardInd(request[i])
                self.hand_free[cardInd] += 1
                self.history[cardInd] += 1
            return request
        if int(request[0]) == 3:
            request[0] = str(requests[request[2]].value)
        elif int(request[0]) == 2:
            request.insert(1, self.myPlayerID)
        request = self.maintain_status(request, self.hand_free, self.history, self.player_history,
                                       self.player_on_table, self.player_last_play, self.player_angang)
        return request

    def maintain_status(self, request, my_free, history, play_history, on_table, last_play, angang):
        requestID = int(request[0])
        playerID = int(request[1])
        player_position = self.player_positions[playerID]
        if requests(requestID) in [requests.drawCard, requests.DRAW]:
            self.tile_count[playerID] -= 1
        if requests(requestID) == requests.drawCard:
            my_free[self.getCardInd(request[-1])] += 1
            history[self.getCardInd(request[-1])] += 1
        elif requests(requestID) == requests.PLAY:
            play_card = self.getCardInd(request[-1])
            play_history[player_position][play_card] += 1
            last_play[player_position] = play_card
            # 自己
            if player_position == 0:
                my_free[play_card] -= 1
            else:
                history[play_card] += 1
        elif requests(requestID) == requests.PENG:
            # 上一步一定有play
            last_card_ind = self.getCardInd(self.prev_request[-1])
            play_card_ind = self.getCardInd(request[-1])
            on_table[player_position][last_card_ind] = 3
            play_history[player_position][play_card_ind] += 1
            last_play[player_position] = play_card_ind
            if player_position != 0:
                history[last_card_ind] += 2
                history[play_card_ind] += 1
            else:
                # 记录peng来源于哪个玩家
                last_player = int(self.prev_request[1])
                last_player_pos = self.player_positions[last_player]
                self.hand_fixed_data.append(('PENG', self.prev_request[-1], last_player_pos))
                my_free[last_card_ind] -= 2
                my_free[play_card_ind] -= 1
        elif requests(requestID) == requests.CHI:
            # 上一步一定有play
            last_card_ind = self.getCardInd(self.prev_request[-1])
            middle_card, play_card = request[3:5]
            middle_card_ind = self.getCardInd(middle_card)
            play_card_ind = self.getCardInd(play_card)
            on_table[player_position][middle_card_ind - 1:middle_card_ind + 2] += 1
            if player_position != 0:
                history[middle_card_ind - 1:middle_card_ind + 2] += 1
                history[last_card_ind] -= 1
                history[play_card_ind] += 1
            else:
                # CHI,中间牌名，123代表上家的牌是第几张
                self.hand_fixed_data.append(('CHI', middle_card, last_card_ind - middle_card_ind + 2))
                my_free[middle_card_ind - 1:middle_card_ind + 2] -= 1
                my_free[last_card_ind] += 1
                my_free[play_card_ind] -= 1
        elif requests(requestID) == requests.GANG:
            # 暗杠
            if requests(int(self.prev_request[0])) in [requests.drawCard, requests.DRAW]:
                request[2] = requests.ANGANG.name
                if player_position == 0:
                    gangCard = self.an_gang_card
                    # print(gangCard)
                    if gangCard == '' and not self.botzone:
                        print(self.prev_request)
                        print(request)
                        print(self.fname)
                    gangCardInd = self.getCardInd(gangCard)
                    # 记录gang来源于哪个玩家（可能来自自己，暗杠）
                    self.hand_fixed_data.append(('GANG', gangCard, 0))
                    on_table[0][gangCardInd] = 4
                    my_free[gangCardInd] = 0
                else:
                    angang[player_position] += 1
            else:
                # 明杠
                gangCardInd = self.getCardInd(self.prev_request[-1])
                request[2] = requests.MINGGANG.name
                history[gangCardInd] = 4
                on_table[player_position][gangCardInd] = 4
                if player_position == 0:
                    # 记录gang来源于哪个玩家
                    last_player = int(self.prev_request[1])
                    self.hand_fixed_data.append(
                        ('GANG', self.prev_request[-1], self.player_positions[last_player]))
                    my_free[gangCardInd] = 0
        elif requests(requestID) == requests.BUGANG:
            bugang_card_ind = self.getCardInd(request[-1])
            history[bugang_card_ind] = 4
            on_table[player_position][bugang_card_ind] = 4
            if player_position == 0:
                for id, comb in enumerate(self.hand_fixed_data):
                    if comb[1] == request[-1]:
                        self.hand_fixed_data[id] = ('GANG', comb[1], comb[2])
                        break
                my_free[bugang_card_ind] = 0
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

    my_bot = MahjongHandler(train=train, model_path=model_path, load_model=load, save_model=save, batch_size=batch_size)
    epochs = args.epochs
    # kinds = ['play', 'action', 'chi_gang']
    kinds = ['play']

    for epoch in range(epochs):
        epoch_start_time = time.time()
        for kind in kinds:
            train_data_files = os.listdir(args.training_data + '/{}/train'.format(kind))
            train_data_files = sorted(train_data_files)
            length = len(train_data_files)
            for fname in sorted(train_data_files):
                dataload = np.load(args.training_data+'/{}/train/{}'.format(kind, fname), allow_pickle=True)
                datadict = dataload.tolist()
                target = torch.from_numpy(np.array(datadict['target'])).to(torch.int64)
                mask = torch.from_numpy(np.array(datadict['mask'])).to(torch.float32)
                card_feats = torch.from_numpy(np.array(datadict['card_feats'])).to(torch.float32)
                extra_feats = torch.from_numpy(np.array(datadict['extra_feats'])).to(torch.float32)
                train_loader = du.DataLoader(du.TensorDataset(card_feats, extra_feats, mask, target), batch_size=batch_size,
                                             shuffle=False)
                my_bot.step_for_train(train_loader, kind, epoch, length, train_data_files.index(fname))
            epoch_end_time = time.time()
            print('epoch-kind:', epoch, '-', kind, '耗时：', epoch_end_time-epoch_start_time)
        acc_play = my_bot.step_for_test(kinds, epoch)
        my_bot.save_bestandcurrent_model(acc_play)
        my_bot.reset()

if __name__ == '__main__':
    print(args)
    train_main()

# nohup python -u DL_TIT_3heads_agent.py > log_TIT_3heads_03130315.log 2>&1 &