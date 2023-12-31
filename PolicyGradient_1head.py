#!/usr/bin/env python
# encoding: utf-8
'''
@file: PolicyGradient.py
@time: ？？？？
强化学习模型
'''

import json
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
from itertools import permutations

import sys
import os
import math
from copy import deepcopy
from torch.distributions import Categorical
import torch.multiprocessing as mp
from multiprocessing.managers import BaseManager
import argparse
import time
from normalization import Normalization, RewardScaling

from TIT_1head import Config_1head, TIT_1head

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

parser = argparse.ArgumentParser(description='Policy-Gradient Model')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('-s', '--save', action='store_true', default=True, help='whether to store model')
parser.add_argument('-t', '--test', action='store_true', default=True, help='whether to test model')
parser.add_argument('-o', '--old_path', type=str, default='./models/dl_TIT_3heads_blocks3_fc3_1682935896.489467.pt',
# parser.add_argument('-o', '--old_path', type=str, default='./models/rl_pg_vanilla_r0_0530-081512.pt',
                    help='path to old model as components')
parser.add_argument('-n', '--new_path', type=str, default='./models/dl_TIT_3heads_blocks3_fc3_1682935896.489467.pt',
# parser.add_argument('-n', '--new_path', type=str, default='./models/rl_pg_vanilla_r0_0530-081512.pt',
                    help='path to load training model')
parser.add_argument('-S', '--save_path', type=str, default='./models/', help='path to save model')
parser.add_argument('-p', '--num_process_per_gpu', type=int, default=1, help='number of processes to run per gpu')
# pi, si的含义为经过多少个episode，若rn=10, rt=10, 则一个episode为10*10=100games
parser.add_argument('-pi', '--print_interval', type=int, default=50, help='how often to print')
parser.add_argument('-si', '--save_interval', type=int, default=100, help='how often to save')
parser.add_argument('-ti', '--train_interval', type=int, default=5, help='how often to train backward')
parser.add_argument('-testi', '--test_interval', type=int, default=50, help='how often to test backward')
parser.add_argument('-ji', '--join_interval', type=int, default=50, help='how often to update shared model')
parser.add_argument('--eps_clip', type=float, default=0.2, help='epsilon for p/q clipped')
parser.add_argument('--entropy_weight', type=float, default=1e-3, help='initial entropy loss weight')
parser.add_argument('--entropy_target', type=float, default=1e-4, help='targeted entropy value')
parser.add_argument('--entropy_step', type=float, default=0.01, help='entropy change per step')
parser.add_argument('-tn', '--total_number', type=int, default=5000, help='epochs to run in total，全部epochs，控制停止')
parser.add_argument('-te', '--test_epochs', type=int, default=2, help='test epochs to run，一次测试循环次数')
parser.add_argument('-rn', '--round_number', type=int, default=200,
                    help='round number*repeated_times to run in parallel，一个round跑几圈')
parser.add_argument('-rt', '--repeated_times', type=int, default=4,
                    help='the repeated times for one round，一副牌的对战次数，暂时用作一圈跑几盘，')  # 一副牌东南西北，玩家各打一次,>2,否则减mean后分数全为零
parser.add_argument('-e', '--epochs', type=int, default=1, help='training epochs for stored data')
parser.add_argument('-lp', '--log_path', type=str, default='./', help='log path, set to "none" for no logging')
parser.add_argument('-bs', '--batch_size', type=int, default=900, help='max batch size to update model')
parser.add_argument('-ga', '--gamma', type=float, default=0.96, help='cumulate reward discount factor')
args = parser.parse_args()
args.model_name = 'rl_pg_vanilla_r3'
args.cuda = 'cuda:1'
args.ps = '"ps：+discount reward'


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
    loss_weight = [1, 1, 5, 2, 2, 2, 2, 2]


class cards(Enum):
    # 饼万条
    B = 0
    W = 9
    T = 18
    # 风
    F = 27
    # 箭牌
    J = 31


# Memory for a certain kind of action
class memory_for_kind:
    def __init__(self):
        self.states = {'card_feats': [],
                       'extra_feats': [],
                       'masks': []}
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.turnIDs = []

    def add_memory(self, card_feats=None, extra_feats=None, mask=None, action=None, logprob=None, turnID=None,
                   reward=None):
        if card_feats is not None:
            self.states['card_feats'].append(card_feats)
            self.states['extra_feats'].append(extra_feats)
            self.states['masks'].append(mask)
        if action is not None:
            self.actions.append(action)
        if logprob is not None:
            self.logprobs.append(logprob)
        if turnID is not None:
            self.turnIDs.append(turnID)
        if reward is not None:
            # turn_length = len(self.actions)-len(self.rewards)
            # if turn_length > 0:
            #     # 计算未来折扣奖励
            #     rewards = []
            #     rewards.extend([reward/turn_length]*turn_length)
            #     discounted_rewards = []
            #     cum_reward = 0
            #     for reward in reversed(rewards):
            #         cum_reward = reward + args.gamma * cum_reward
            #         discounted_rewards.append(cum_reward)
            #     discounted_rewards.reverse()
            #     self.rewards.extend(discounted_rewards)
            turn_length = len(self.actions) - len(self.rewards)
            if turn_length > 0:
                self.rewards.extend([reward / turn_length] * turn_length)

    def add_rewards(self, rewards=None):
        turn_length = len(self.actions) - len(self.rewards)
        if len(rewards) > 0 and len(rewards) == turn_length:
            self.rewards.extend(rewards)

    def merge_memory(self, memory):
        for key, val in self.states.items():
            val.extend(memory.states[key])
        self.rewards.extend(memory.rewards)
        self.actions.extend(memory.actions)
        self.logprobs.extend(memory.logprobs)
        self.turnIDs.extend(memory.turnIDs)

    def clear_memory(self):
        for value in self.states.values():
            del value[:]
        del self.actions[:]
        del self.rewards[:]
        del self.logprobs[:]
        del self.turnIDs[:]


class Memory:  # collected from old policy
    def __init__(self):
        self.memory = {
            'action': memory_for_kind(),
            'chi_gang': memory_for_kind(),
            'play': memory_for_kind()
        }

    def get_memory(self):
        return self.memory

    def add_memory(self, kind=None, card_feats=None, extra_feats=None, mask=None, action=None, logprob=None,
                   turnID=None, reward=None):
        if kind is None:
            for memory in self.memory.values():
                memory.add_memory(card_feats, extra_feats, mask, action, logprob, turnID, reward)
        else:
            self.memory[kind].add_memory(card_feats, extra_feats, mask, action, logprob, turnID, reward)

    def add_reward(self, kind, rewards=None):
        kind_memory = self.memory[kind]
        kind_memory.add_rewards(rewards)

    def merge_memory(self, round_memory):
        for kind, memory in round_memory.memory.items():
            self.memory[kind].merge_memory(memory)

    def clear_memory(self):
        for memory in self.memory.values():
            memory.clear_memory()


# Model
class Policy(nn.Module):
    def __init__(self, obs_array_dim=22 * 34, num_extra_feats=24, num_cards=34, num_actions=8):
        super().__init__()
        self.entropy_weight = args.entropy_weight
        activation_fn = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'gelu': nn.GELU}
        self.config = Config_1head(
            algo='vanilla_tit_ppo',
            patch_dim=num_cards,
            num_blocks=3,
            features_dim=1 + (num_actions - 1) * num_cards,  # 输出1+7*34
            embed_dim_inner=1024,
            num_heads_inner=4,
            attention_dropout_inner=0.,
            ffn_dropout_inner=0.1,
            context_len_inner=int(np.ceil((obs_array_dim + num_extra_feats) / num_cards)),
            embed_dim_outer=1024,
            num_heads_outer=8,
            attention_dropout_outer=0.,
            ffn_dropout_outer=0.1,
            context_len_outer=4,  # K outer的切片个数，跨越的时间步幅
            observation_type='array',
            C=1, H=0, W=0, D=obs_array_dim + num_extra_feats,
            activation_fn_inner=activation_fn['gelu'],
            activation_fn_outer=activation_fn['gelu'],
            activation_fn_other=activation_fn['tanh'],
            dim_expand_inner=2,
            dim_expand_outer=2,
            have_position_encoding=1,
            share_tit_blocks=0  # EnhancedTIT使用1，vanilla_tit 0
        )
        self.tit = TIT_1head(self.config)
        self.softmax = nn.Softmax(dim=1)
        print(self.config.__dict__)

    def forward(self, card_feats, extra_feats, device, decide_which):
        assert decide_which in ['play', 'chi_gang']
        obs = np.concatenate((card_feats, extra_feats), axis=1)
        obs_tensor = torch.from_numpy(obs).to(device).to(torch.float32)
        probs = self.tit(obs_tensor, decide_which)
        probs = self.softmax(probs)
        return probs

    def mask_unavailable_actions(self, result, valid_actions_tensor):
        replace_nan = torch.isnan(result)
        result_no_nan = result.masked_fill(mask=replace_nan, value=1e-9)
        masked_actions = result_no_nan * valid_actions_tensor
        return masked_actions

    # THIS IS IMPORTANT!
    def get_feature_size(self):
        return self.features_dim


class MahjongEnv:
    def __init__(self, old_model_path=None, shared_model=None, cuda='cpu', shared_memory=None, lock=None):
        use_cuda = torch.cuda.is_available()
        self.repeated_times = args.repeated_times  # 一副牌的对战次数
        self.round_number = args.round_number * self.repeated_times  # 并行游戏同时对战，所以相乘  ，伪并行，串行收集state，批送入模型。
        self.shared_memory = shared_memory
        self.local_memory = Memory()
        self.memory_each_rounds = [Memory() for _ in range(4 * self.round_number)]
        self.device = torch.device(cuda if use_cuda else "cpu")
        print('using ' + str(self.device))
        self.total_cards = 34
        self.total_actions = len(responses) - 2
        self.model = Policy().to(self.device)
        self.old_model = Policy().to(self.device)
        self.shared_model = shared_model
        self.old_model_code = 0
        self.new_model_code = 1
        self.models = [self.old_model, self.model]
        # 加载预训练好的actor网络
        old_check_point = torch.load(old_model_path, map_location=self.device)
        old_dict = self.model.state_dict()
        for k in old_check_point['model'].keys():
            old_dict[k] = old_check_point['model'][k]
        self.old_model.load_state_dict(old_dict)
        self.old_model.eval()
        self.model.load_state_dict(shared_model.state_dict())
        self.model.eval()
        # state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
        # torch.save(state, self.model_path, _use_new_zipfile_serialization=False)
        self.lock = lock
        self.cuda = cuda
        self.round_count = 0
        self.train_count = 0
        self.bots = []
        self.winners = np.zeros(4, dtype=int)
        self.win_steps = []
        self.losses = []
        self.scores = np.zeros((self.round_number, 4), dtype=float)
        self.scores4count = np.zeros((self.round_number, 4), dtype=float)
        # 新model位置
        self.new_model_pos = [0]
        for _ in range(self.round_number):
            bots = []
            # for i in range(4):
            bots.append(MahjongHandler(self.model, self.device))
            bots.append(MahjongHandler(self.old_model, self.device))
            bots.append(MahjongHandler(self.old_model, self.device))
            bots.append(MahjongHandler(self.old_model, self.device))
            # if i % 2 == self.old_model_code:
            #     bots.append(MahjongHandler(self.old_model, self.device))
            # else:
            #     bots.append(MahjongHandler(self.model, self.device))
            self.bots.append(bots)
        self.train = True
        self.reset(True)

    def reset(self, initial=False):
        self.tile_walls = []  # 牌墙
        self.quans = []  # 圈风
        self.mens = []  # 门风
        self.bots_orders = []  # 座次
        self.hand_fixed_data_bots_orders = {}  # 专门记录其他玩家鸣牌轮次（玩家index,trunID）
        for i in range(self.round_number):
            self.hand_fixed_data_bots_orders[i] = []
        # 初始摸牌bot为0
        self.drawers = []
        all_tiles = np.arange(self.total_cards)  # 全部牌
        # 构建牌墙
        all_tiles = all_tiles.repeat(4)
        quan, men = 0, 0
        for round_id in range(self.round_number):  # 初始化并行牌局
            # perms = permutations(range(4), 4)  # 4人座次全排列
            if round_id % self.repeated_times == 0:  # 洗牌
                np.random.shuffle(all_tiles)
                quan = np.random.choice(4)  # 改变圈风
                men = np.random.choice(4)  # 改变门风
            # 改变门风
            # 用pop，从后面摸牌
            self.tile_walls.append(np.reshape(all_tiles, (4, -1)).tolist())  # 每个人分牌
            self.quans.append(quan)
            self.mens.append(men)
            # 这一局bots的order，牌墙永远下标和bot一致
            self.bots_orders.append([self.bots[round_id][(i + self.mens[-1]) % 4] for i in range(4)])
            self.drawers.append(0)
        # print("all_tiles:",all_tiles)
        if not initial:  # 真重置
            if self.train:
                with self.lock:
                    # 每局结束将memory汇总
                    self.shared_memory.merge_memory(self.local_memory)
                self.round_count += self.round_number
            self.local_memory.clear_memory()
            for bots in self.bots_orders:
                for bot in bots:
                    bot.reset()
            # self.scores = np.zeros((self.round_number, 4), dtype=float)  # 在外部初始化，记录score

    def run_rounds(self):
        turnID = 0
        player_responses = [['PASS'] * 4 for _ in range(self.round_number)]
        finished = np.zeros(self.round_number, dtype=int)
        state_dimension = 5
        while finished.sum() < self.round_number:
            data = {
                # card_feats, extra_feats, action_mask, card_mask, bot
                self.models.index(self.old_model): [[], [], [], [], []],
                self.models.index(self.model): [[], [], [], [], []],
                # which bot the data came from
                "order": [[], []]
            }
            lengths = {
                self.models.index(self.old_model): 0,
                self.models.index(self.model): 0,
            }
            for round_id in range(self.round_number):
                if finished[round_id]:
                    continue
                if turnID == 0:
                    for id, player in enumerate(self.bots_orders[round_id]):
                        player.step('0 %d %d' % (id, self.quans[round_id]))
                elif turnID == 1:
                    for id, player in enumerate(self.bots_orders[round_id]):
                        request = ['1']
                        for i in range(4):
                            request.append('0')
                        for i in range(13):
                            request.append(self.getCardName(self.tile_walls[round_id][id].pop()))
                        request = ' '.join(request)
                        player.step(request)
                else:
                    requests = self.parse_response(player_responses[round_id], round_id)
                    # print("player_responses in env", player_responses[round_id])
                    # print("requests in env", requests)
                    if requests[0] in ['hu', 'huangzhuang']:
                        outcome = requests[0]
                        if outcome == 'hu':
                            winner_id = int(requests[1])
                            self.winners[self.getBotInd(round_id, winner_id)] += 1
                            self.win_steps.append(turnID)
                            fan_count = int(requests[2])
                            dianpaoer = requests[3]
                            if dianpaoer == 'None':
                                dianpaoer = None
                            if dianpaoer is not None:
                                dianpaoer = int(dianpaoer)
                            self.calculate_scores(round_id, winner_id, dianpaoer, fan_count, mode='naive')
                        finished[round_id] = 1
                    else:
                        for i in range(4):
                            this_bot = self.bots_orders[round_id][i]
                            this_model = this_bot.model
                            model_code = self.models.index(this_model)
                            state = self.bots_orders[round_id][i].step(requests[i])
                            for j in range(state_dimension):
                                data[model_code][j].append(state[j])
                            data["order"][model_code].append((round_id, i))
                            lengths[model_code] += 1

            # all bots stepped, if turnID > 1, should produce responses through model
            if turnID > 1:
                for index, model in enumerate(self.models):
                    actions, cards = self.step(data[index], data["order"][index], model, turnID)
                    action_num = len(actions)
                    for i in range(action_num):
                        round_id, bot_id = data["order"][index][i]
                        response = self.bots_orders[round_id][bot_id].build_response(actions[i], cards[i])
                        player_responses[round_id][bot_id] = response
                        # 记录其他玩家鸣牌轮次
                        if responses(actions[i]) in [responses.CHI, responses.PENG, responses.MINGGANG,
                                                     responses.ANGANG, responses.BUGANG]:
                                self.hand_fixed_data_bots_orders[round_id].append((bot_id, turnID, response,
                                                                                   self.bots_orders[round_id][bot_id].prev_request))
            turnID += 1
        # if self.train:
        #     # end while, all games finished, merge all bots' memory to local memory
        #     self.regularize_scores()

        for i in range(self.round_number):
            for j in range(4):
                if self.train:
                    if self.scores[i][j] != 0 and self.getBotInd(i, j) in self.new_model_pos:
                        round_memory = self.memory_each_rounds[i * 4 + j].get_memory()
                        for kind in round_memory.keys():
                            rewards = self.reshape_reward(kind, round_memory.get(kind), self.bots_orders[i], j,
                                                          self.scores[i], self.hand_fixed_data_bots_orders[i])
                            self.memory_each_rounds[i * 4 + j].add_reward(kind, rewards)
                        self.local_memory.merge_memory(self.memory_each_rounds[i * 4 + j])
                    self.memory_each_rounds[i * 4 + j].clear_memory()
                # 将score按bot顺序调整
                self.scores4count[i][j] = self.scores[i][(j - self.mens[i]) % 4]

    def step(self, state, order, model, turnID):
        number_of_data = len(state[0])
        actions = np.zeros(number_of_data, dtype=int)
        cards = [[] for _ in range(number_of_data)]
        training_data = {
            'action': {
                'card_feats': [],
                'extra_feats': [],
                'masks': [],
                'mapping': []
            },
            'play': {
                'card_feats': [],
                'extra_feats': [],
                'masks': [],
                'mapping': []
            },
            'chi_gang': {
                'card_feats': [],
                'extra_feats': [],
                'masks': [],
                'mapping': []
            }
        }

        # collect all bots' states
        def collect_data(kind, card_feats, extra_feats, mask, idx):
            if mask.sum() > 1:
                training_data[kind]['card_feats'].append(card_feats)
                training_data[kind]['masks'].append(mask)
                training_data[kind]['extra_feats'].append(extra_feats)
                training_data[kind]['mapping'].append(idx)
            else:
                action = np.argmax(mask)
                if kind == 'action':
                    actions[idx] = action
                else:
                    cards[idx].append(action)

        # run forward with data together
        def run_forward(kind, turnID):
            feature_size = len(training_data[kind]['card_feats'])
            if feature_size == 0:
                return
            extra_feats = np.array(training_data[kind]['extra_feats'])
            card_feats = np.array(training_data[kind]['card_feats'])
            batch_number = feature_size // args.batch_size + int(bool(feature_size % args.batch_size))
            probs = torch.tensor([]).to(device=self.device)
            for i in range(batch_number):
                start = i * args.batch_size
                end = min(feature_size, start + args.batch_size)
                probs_temp = model(
                    card_feats[start:end],
                    extra_feats[start:end],
                    self.device,
                    kind, )
                probs = torch.cat((probs, probs_temp), dim=0)
            # print(kind,training_data[kind]['masks'])
            mask = torch.from_numpy(np.array(training_data[kind]['masks'])).to(self.device).to(torch.float32)
            probs_dist = Categorical(probs * mask)
            if self.train and model != self.old_model:
                action_tensor = probs_dist.sample()
            else:
                action_tensor = torch.argmax(probs * mask, dim=1)
            # print(kind,action_tensor)
            log_probs = probs_dist.log_prob(action_tensor).detach().cpu().numpy()
            action_numpy = action_tensor.cpu().numpy()
            action_idx = training_data[kind]['mapping']
            for idx, action in zip(action_idx, action_numpy):
                if kind == 'action':
                    actions[idx] = int(action)
                else:
                    cards[idx].append(int(action))
            if self.train and model != self.old_model:
                for idx, action, log_prob, card_feat, extra_feat, mask in zip(action_idx, action_numpy, log_probs,
                                                                              training_data[kind]['card_feats'],
                                                                              training_data[kind]['extra_feats'],
                                                                              training_data[kind]['masks']):
                    # if log_prob >= -0.1:
                    #     continue
                    original_order = order[idx][0] * 4 + order[idx][1]
                    self.memory_each_rounds[original_order].add_memory(kind, card_feat, extra_feat, mask, action,
                                                                       log_prob, turnID)

        # kind = 'action'
        # for i, card_feats, extra_feats, available_action_mask, _, _ in zip(np.arange(number_of_data), *state):
        #     collect_data(kind, card_feats, extra_feats, available_action_mask, i)
        #
        # run_forward(kind, turnID)

        kind = 'chi_gang'
        for i, card_feats, extra_feats, _, available_card_mask, _ in zip(np.arange(number_of_data), *state):
            action = actions[i]
            if responses(action) in [responses.CHI, responses.ANGANG, responses.BUGANG]:
                card_mask = available_card_mask[action]
                collect_data(kind, card_feats, extra_feats, card_mask, i)

        run_forward(kind, turnID)

        kind = 'play'

        for i, card_feats, extra_feats, _, available_card_mask, bot in zip(np.arange(number_of_data), *state):
            action = actions[i]
            if responses(action) in [responses.PLAY, responses.CHI, responses.PENG]:
                if responses(action) == responses.PLAY:
                    card_mask = available_card_mask[action]
                else:
                    request = bot.prev_request
                    if responses(action) == responses.CHI:
                        chi_peng_ind = cards[i][0]
                    else:
                        chi_peng_ind = self.getCardInd(request[-1])
                    card_feats, extra_feats, card_mask = bot.simulate_chi_peng(request, responses(action), chi_peng_ind)
                collect_data(kind, card_feats.flatten(order="C"), extra_feats, card_mask, i)

        run_forward(kind, turnID)

        return actions, cards

    def reshape_reward(self, kind, kind_m, bots, botInd, payoffs, hand_fixed_data_bots_order_round):
        rewards = []
        winner = payoffs.tolist().index(payoffs.max())
        diaopao = payoffs.tolist().index(payoffs.min())
        winnercard = bots[winner].hand_free + bots[winner].hand_fixed
        if winner == botInd and payoffs[winner] > 0:
            if kind == 'play':
                for i, mem in enumerate(kind_m.states["card_feats"]):
                    mem = mem.reshape(22, 34)
                    card = (mem[0, :] + mem[2, :]).astype(int)
                    hand = []
                    for ind, cardcnt in enumerate(card):
                        for _ in range(cardcnt):
                            hand.append(self.getCardName(ind))
                    card[kind_m.actions[i]] = card[kind_m.actions[i]] - 1
                    r = card & winnercard
                    r = r.sum() * payoffs[botInd] / 13
                    if len(rewards) == 0:
                        rewards.append(r)
                    else:
                        rewards.append(r - rewards[-1])
            elif kind == 'action':
                rewards.extend([(payoffs[botInd] * 3 / 13) ** 0.5] * len(kind_m.actions))
            else:
                rewards.extend([(payoffs[botInd] * 2 / 13) ** 0.5] * len(kind_m.actions))
        elif diaopao == botInd and payoffs[winner] > 0:
            if kind == 'play':
                ids = []
                for data in hand_fixed_data_bots_order_round:
                    try:
                        temp = data[-1]
                        if data[0] == winner and bots[winner].player_positions[int(temp[0]) if len(temp)==2 else int(temp[1])] == botInd:
                            ids.append(data[1])
                    except Exception:
                        print(data)
                for j in range(len(kind_m.actions)):
                    if kind_m.turnIDs[j] + 1 in ids:
                        rewards.append(-((-payoffs[botInd] * 3 / 13) ** 0.5))
                    else:
                        rewards.append(0)
                rewards[-1] = -((-payoffs[botInd] * 4 / 13) ** 0.5)
            elif kind == 'action':
                rewards.extend([0] * len(kind_m.actions))
            else:
                rewards.extend([0] * len(kind_m.actions))
        else:
            rewards.extend([0] * len(kind_m.actions))
        def discount_reward(rewards):
            discounted_rewards = []
            if len(rewards) > 0:
                # 计算未来折扣奖励
                cum_reward = 0
                for reward in reversed(rewards):
                    cum_reward = reward + args.gamma * cum_reward
                    discounted_rewards.append(cum_reward)
                    cum_reward=discounted_rewards[-1]
                discounted_rewards.reverse()
            return discounted_rewards

        # if self.history[self.getCardInd(last_card)] == 4:
        #     isJUEZHANG = True
        # else:
        #     isJUEZHANG = False
        # if self.tile_count[(playerID + 1) % 4] == 0:
        #     isLAST = True
        # else:
        #     isLAST = False
        # if not dianPao:
        #     hand.remove(last_card)
        # rewards = []
        # if payoff > 0:
        #     rewards.extend([(payoff/2)**0.5]*len(kind_m.actions))
        # elif payoff < 0:
        #     if len(kind_m.actions) > 0:
        #         if kind == 'play':
        #             rewards.extend([-1] * len(kind_m.actions))
        #             rewards[-1] = -((-payoff/2)**0.5)
        #         else:
        #             rewards.extend([0] * len(kind_m.actions))
        # else:
        #     rewards.extend([0]*len(kind_m.actions))

        return discount_reward(rewards)

    # 不和牌，分数都是0，不会调用这个函数
    def calculate_scores(self, round_id, winner_id=0, dianpaoer=None, fan_count=0, difen=8, mode='naive'):
        assert mode in ['naive', 'botzone']
        if mode == 'botzone':
            for i in range(4):
                if i == winner_id:
                    self.scores[round_id][i] = 10
                    if dianpaoer is None:
                        # 自摸
                        self.scores[round_id][i] = 3 * (difen + fan_count)
                    else:
                        self.scores[round_id][i] = 3 * difen + fan_count
                else:
                    if dianpaoer is None:
                        self.scores[round_id][i] = -0.5 * (difen + fan_count)
                    else:
                        if i == dianpaoer:
                            self.scores[round_id][i] = -2 * (difen + fan_count)
                        else:
                            self.scores[round_id][i] = -0.5 * difen
        else:
            for i in range(4):
                if i == winner_id:
                    self.scores[round_id][i] = fan_count
                elif i == dianpaoer:
                    self.scores[round_id][i] = -fan_count
                else:
                    self.scores[round_id][i] = 0

    # 将得分减去平均得分
    def regularize_scores(self):
        # print(self.scores)
        self.scores[::] -= np.mean(self.scores, dtype=np.float32)
        # for i in range(self.round_number // self.repeated_times):  # 不太理解同一手牌得分正则的含义，改为所有得分正则化
        #     start_round = i * self.repeated_times
        #     end_round = (i + 1) * self.repeated_times
        #     m = np.mean(self.scores[start_round:end_round], axis=0)
        #     self.scores[start_round:end_round] -= np.mean(self.scores[start_round:end_round], axis=0)
        # print(self.scores)

    def parse_response(self, player_responses, round_id):
        requests = []
        for id, response in enumerate(player_responses):
            response = response.split(' ')
            response_name = response[0]
            if response_name == 'HU':
                return ['hu', id, response[1], response[2]]
            if response_name == 'PENG':
                requests = []
                for i in range(4):
                    requests.append('3 %d PENG %s' % (id, response[1]))
                self.drawers[round_id] = (id + 1) % 4
                break
            if response_name == "GANG":
                requests = []
                for i in range(4):
                    requests.append('3 %d GANG' % (id))
                self.drawers[round_id] = id
                break
            if response_name == 'CHI':
                for i in range(4):
                    requests.append('3 %d CHI %s %s' % (id, response[1], response[2]))
                self.drawers[round_id] = (id + 1) % 4
            if response_name == 'PLAY':
                for i in range(4):
                    requests.append('3 %d PLAY %s' % (id, response[1]))
                self.drawers[round_id] = (id + 1) % 4
            if response_name == 'BUGANG':
                for i in range(4):
                    requests.append('3 %d BUGANG %s' % (id, response[1]))
                self.drawers[round_id] = id
        # 所有人pass，摸牌
        if len(requests) == 0:
            if len(self.tile_walls[round_id][self.drawers[round_id]]) == 0:
                return ['huangzhuang', 0]
            draw_card = self.tile_walls[round_id][self.drawers[round_id]].pop()
            for i in range(4):
                if i == self.drawers[round_id]:
                    requests.append('2 %s' % self.getCardName(draw_card))
                else:
                    requests.append('3 %d DRAW' % self.drawers[round_id])
        return requests

    def print_log(self, type, current_round, print_interval, winners, player_scores, time_cost):
        win_sum = sum(winners)
        print('total winning: ', winners[:])
        print('total scores: ', player_scores[:])
        total_rounds = current_round * self.round_number
        rounds_this_stage = print_interval * self.round_number
        print(
            '{}: total rounds: {}, during the last {} rounds, new bot winning rate: {:.2%}, old bot winning rate: {:.2%}\n'
            'Hu {} rounds，Huang-zhuang {} rounds，hu ratio {:.2%}, average rounds to hu: {}, took {:.2f} minutes per 10000 rounds'.format(
                type, total_rounds, rounds_this_stage + 1e-9,
                                    sum(winners[1::2]) / rounds_this_stage + 1e-9,
                                    sum(winners[::2]) / rounds_this_stage + 1e-9, win_sum,
                                    rounds_this_stage - win_sum,
                                    win_sum / rounds_this_stage + 1e-9,
                                    sum(self.win_steps) / (len(self.win_steps) + 1e-9),
                time_cost
            ))
        if args.log_path != 'none':
            with open(args.log_path + '_' + args.model_name + '_' + type + '.log', 'a+') as f:
                print(
                    '{} {} {:.2%} {:.2%} {} {} {:.2%} {} {:.2f}'.format(
                        total_rounds, rounds_this_stage,
                        sum(winners[1::2]) / rounds_this_stage + 1e-9,
                        sum(winners[::2]) / rounds_this_stage + 1e-9, win_sum,
                        rounds_this_stage - win_sum,
                        win_sum / rounds_this_stage + 1e-9,
                        sum(self.win_steps) / len(self.win_steps) + 1e-9,
                        sum(player_scores[1::2]),
                        sum(player_scores[::2]),
                        time_cost
                    ), file=f)
        self.win_steps = []

    def getBotInd(self, round_id, bot_id):
        return self.bots[round_id].index(self.bots_orders[round_id][bot_id])

    def getCardInd(self, cardName):
        return cards[cardName[0]].value + int(cardName[1]) - 1

    def getCardName(self, cardInd):
        num = 1
        while True:
            if cardInd in cards._value2member_map_:
                break
            num += 1
            cardInd -= 1
        return cards(cardInd).name + str(num)


# 维护对局环境
class MahjongHandler:
    def __init__(self, model, device):
        self.total_cards = 34
        self.total_actions = len(responses) - 2
        self.model = model
        self.optimizer = optim
        self.device = device
        self.reset()

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
        self.dianpaoer = None
        self.doc_data = {
            "old_probs": [],
            "new_probs": [],
            "entropy": []
        }

    def step(self, request):  # 获取当前state
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
            self.prev_request = request
            self.turnID += 1
            return
        else:
            available_action_mask, available_card_mask = self.build_available_action_mask(request)
            card_feats = self.build_input(self.hand_free, self.history, self.player_history,
                                          self.player_on_table, self.player_last_play, available_card_mask)
            extra_feats = np.concatenate((self.player_angang[1:], available_action_mask,
                                          [self.hand_free.sum()], *np.eye(4)[[self.quan, self.myPlayerID]],
                                          self.tile_count))
            self.prev_request = request
            self.turnID += 1
            card_feats = card_feats.flatten(order="C")
            return card_feats, extra_feats, available_action_mask, available_card_mask, self

    def build_input(self, my_free, history, play_history, on_table, last_play, available_card_mask):
        temp = np.array([my_free, 4 - history])
        one_hot_last_play = np.eye(self.total_cards)[last_play]
        card_feats = np.concatenate((temp, on_table, play_history, one_hot_last_play, available_card_mask))
        return card_feats

    def build_response(self, action, cards):
        # print(action, cards)
        response = self.build_output(responses(action), cards)
        if responses(action) == responses.ANGANG:
            self.an_gang_card = self.getCardName(cards[0])
        self.turnID += 1
        self.response = response
        return response

    def simulate_chi_peng(self, request, response, chi_peng_ind):
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
        card_feats = self.build_input(my_free, self.history, self.player_history, on_table, self.player_last_play,
                                      available_card_mask)

        action_mask = np.zeros(self.total_actions, dtype=int)
        action_mask[responses.PLAY.value] = 1
        extra_feats = np.concatenate((self.player_angang[1:], [my_free.sum()], action_mask,
                                      *np.eye(4)[[self.quan, self.myPlayerID]], self.tile_count))
        return card_feats, extra_feats, available_card_play_mask

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
            # print(self.myPlayerID, hand, last_card, self.hand_fixed_data)
            ans = MahjongFanCalculator(tuple(self.hand_fixed_data), tuple(hand), last_card, 0,
                                       playerID == self.myPlayerID,
                                       isJUEZHANG, isGANG, isLAST, self.myPlayerID, self.quan)
        except Exception as err:
            # print(hand, last_card, self.hand_fixed_data)
            # print(err)
            if str(err) == 'ERROR_NOT_WIN':
                return 0
            else:
                with open('error.txt', 'a+') as f:
                    print(self.prev_request, file=f)
                    print(self.response, file=f)
                    print(hand, last_card, self.hand_fixed_data, file=f)
                    print(err, file=f)
                print(self.prev_request)
                print(self.response)
                print(hand, last_card, self.hand_fixed_data)
                return 0
        else:
            fan_count = 0
            # with open('hu.txt', 'a+') as f:
            #     print(ans, file=f)
            for fan in ans:
                fan_count += fan[0]
            if dianPao:
                self.dianpaoer = playerID
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
            request.insert(1, str(self.myPlayerID))
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
                    if gangCard == '':
                        print(self.prev_request)
                        print(request)
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
        if response == responses.HU:
            return '{} {} {}'.format(response_name, self.fan_count, self.dianpaoer)
        return response_name

    def getCardInd(self, cardName):
        return cards[cardName[0]].value + int(cardName[1]) - 1

    def getCardName(self, cardInd):
        num = 1
        while True:
            if cardInd in cards._value2member_map_:
                break
            num += 1
            cardInd -= 1
        return cards(cardInd).name + str(num)


def ensure_shared_grads(model, shared_model, device):
    """ ensure proper initialization of global grad"""
    # NOTE: due to no backward passes has ever been ran on the global model
    # NOTE: ref: https://discuss.pytorch.org/t/problem-on-variable-grad-data/957
    for shared_param, local_param in zip(shared_model.parameters(),
                                         model.parameters()):
        if 'cuda' in str(device):
            # GPU
            if local_param.grad is None:
                shared_param._grad = None
            else:
                shared_param._grad = local_param.grad.clone().cpu()  # pylint: disable=W0212
        else:
            # CPU
            if shared_param.grad is not None:
                return
            else:
                shared_param._grad = local_param.grad  # pylint: disable=W0212


def update(all_memories: dict, model, shared_model, optimizer, device):
    model.load_state_dict(shared_model.state_dict())
    model.zero_grad()
    optimizer.zero_grad()
    for kind, memory in all_memories.items():
        # Monte Carlo estimation of rewards
        rewards = memory.rewards
        if len(rewards) == 0:
            continue
        # Normalize rewards
        rewards = torch.tensor(rewards).to(device, dtype=torch.float32)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        card_feats = np.array(memory.states['card_feats'])
        feature_size = card_feats.shape[0]
        batch_number = feature_size // args.batch_size + int(bool(feature_size % args.batch_size))
        print('Training for {} model, training data length {}'.format(kind, feature_size))
        extra_feats = np.array(memory.states['extra_feats'])
        # masks = np.array(memory.states['masks'])
        old_actions = torch.tensor(memory.actions).to(device)
        old_logprobs = torch.tensor(memory.logprobs).to(device)
        # print(old_actions.shape, old_logprobs.shape)

        # Train policy for K epochs: sampling and updating
        for _ in range(args.epochs):
            for i in range(batch_number):
                start = i * args.batch_size
                end = min(feature_size, start + args.batch_size)
                # Evaluate old actions and values using current policy
                new_probs = model(
                    card_feats[start:end],
                    extra_feats[start:end],
                    device,
                    kind,
                    # masks[start:end]
                )
                # mask_epochs = torch.from_numpy(masks[start:end]).to(device=device)
                # new_probs_dist = Categorical(new_probs * mask_epochs)
                new_probs_dist = Categorical(new_probs)
                new_logprobs = new_probs_dist.log_prob(old_actions[start:end])
                entropy = new_probs_dist.entropy()
                # '''参考代码'''
                # policy_gradient = []
                # for log_prob, reward in zip(new_logprobs, discounted_rewards):
                #     policy_gradient.append(-log_prob * reward)
                # policy_gradient = torch.stack(policy_gradient).sum()
                # '''参考代码'''
                # Importance ratio: p/q
                ratios = torch.exp(new_logprobs - old_logprobs[start:end].detach())

                # Actor loss using Surrogate loss
                surr1 = ratios * rewards[start:end]
                surr2 = torch.clamp(ratios, 1 - args.eps_clip, 1 + args.eps_clip) * rewards[start:end]
                shared_model.entropy_weight = shared_model.entropy_weight + args.entropy_step * (
                        args.entropy_target - float(entropy.mean().data.cpu()))
                loss = - torch.min(surr1, surr2) - shared_model.entropy_weight * entropy
                # loss = - surr1 - shared_model.entropy_weight * entropy
                # Backward gradients
                loss.mean().backward()
    torch.nn.utils.clip_grad_norm_(shared_model.parameters(), 0.5)
    ensure_shared_grads(model, shared_model, device)
    optimizer.step()


"""
PolicyGradient.py -p 4 -rn 50 -rt 40 -o models/super_model_2 -n models/rl_pg -S models/rl_pg_new -s -lp logs/new_log -bs 8000 -lr 2e-6 -ti 8 -ji 10 -pi 160 -si 320 -e 1
"""


class SharedAdam(optim.Adam):
    # pylint: disable=C0103
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()
                state['max_exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                state['max_exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """
        Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
        Returns:
            loss
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead'
                    )
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size.item())  # first arg has to be scalar

        return loss


# thread for training
def train_thread(cuda, shared_model, global_episode_counter, global_training_counter, optimizer, lock, shared_memory,
                 winners):
    # shared_model must be on cpu, so create a duplicate to update on gpu
    use_cuda = torch.cuda.is_available()
    model_for_update = Policy().to(cuda if use_cuda else "cpu")
    # for n, p in model_for_update.named_parameters():
    #     if 'card_net' in n:
    #         p.requires_grad = False
    start_time = time.perf_counter()
    start_counter = global_episode_counter.value
    save = args.save
    test = args.test
    old_model_path = args.old_path
    save_path = args.save_path
    print_interval = args.print_interval
    save_interval = args.save_interval
    train_interval = args.train_interval
    test_interval = args.test_interval

    env = MahjongEnv(old_model_path=old_model_path,
                     shared_model=shared_model,
                     cuda=cuda,
                     shared_memory=shared_memory,
                     lock=lock)

    while global_episode_counter.value < args.total_number:
        env.run_rounds()  # 一个episode= rn * rt
        env.reset(False)
        with lock:
            global_episode_counter.value += 1
            current_round = global_episode_counter.value
            player_scores = [0., 0., 0., 0.]
            for i in range(4):
                winners[i] += env.winners[i]
                player_scores[i] += np.sum(env.scores4count, axis=0)[i]
            env.winners = np.zeros(4, dtype=int)
            env.scores4count = np.zeros((env.round_number, 4), dtype=float)

            # update policy
            if current_round % train_interval == 0:
                print('round:', current_round)
                update(shared_memory.get_memory(), model_for_update, shared_model, optimizer, env.device)
                shared_memory.clear_memory()
                global_training_counter.value += 1

            # update local policy
            if global_training_counter.value - env.train_count >= args.join_interval:
                env.train_count = global_training_counter.value
                env.model.load_state_dict(shared_model.state_dict())

            if current_round % print_interval == 0:
                total_rounds = (current_round - start_counter) * env.round_number
                this_time = time.perf_counter()
                time_cost = (this_time - start_time) / (60 * (total_rounds / 10000))
                env.print_log('train', current_round, print_interval, winners, player_scores, time_cost)
                for i in range(4):
                    winners[i] = 0

            if test and current_round % test_interval == 0:
                print('-' * 30)
                test_start_time = time.perf_counter()
                player_scores_test = [0., 0., 0., 0.]
                for i in range(args.test_epochs):
                    env.train = False
                    repeated_times = env.repeated_times
                    env.repeated_times = 2
                    env.run_rounds()  # 一个episode= rn * rt
                    env.reset(False)
                    for i in range(4):
                        winners[i] += env.winners[i]
                        player_scores_test[i] += np.sum(env.scores4count, axis=0)[i]
                    env.winners = np.zeros(4, dtype=int)
                    env.scores4count = np.zeros((env.round_number, 4), dtype=float)
                total_rounds = args.test_epochs * env.round_number
                test_this_time = time.perf_counter()
                time_cost = (test_this_time - test_start_time) / (60 * (total_rounds / 10000))
                env.print_log('test', args.test_epochs, args.test_epochs, winners, player_scores_test, time_cost)
                for i in range(4):
                    winners[i] = 0
                env.train = True
                env.repeated_times = repeated_times
            if save and current_round % save_interval == 0:
                env.losses = []
                env.model.load_state_dict(shared_model.state_dict())
                stemp = time.strftime('%m%d-%H%M%S', time.localtime(time.time()))
                print('total rounds: %d, saving model...' % current_round,
                      save_path + args.model_name + '_' + str(stemp) + ".pt")
                state = {'model': shared_model.state_dict(), 'optimizer': optimizer.state_dict(),
                         'counter': current_round}
                torch.save(state, save_path + args.model_name + '_' + str(stemp) + ".pt",
                           _use_new_zipfile_serialization=False)


# # 这两个类也许是多余的，抄的网上
# class MyManager(BaseManager):
#     pass
#
# def ManagerStarter():
#     m = MyManager()
#     m.start()
#     return m

def main():
    mp.set_start_method('spawn')  # required to avoid Conv2d froze issue
    num_processes_per_gpu = args.num_process_per_gpu
    new_model_path = args.new_path
    lr = args.learning_rate
    shared_model = Policy(
        obs_array_dim=22 * 34,
        num_extra_feats=24,
        num_cards=34,
        num_actions=8,
    )
    # for n, p in shared_model.named_parameters():
    #     if 'card_net' in n:
    #         p.requires_grad = False
    checkpoint = torch.load(new_model_path, map_location=torch.device(args.cuda))
    new_dict = shared_model.state_dict()
    for k in checkpoint['model'].keys():
        new_dict[k] = checkpoint['model'][k]
    shared_model.load_state_dict(new_dict)
    shared_model.share_memory()
    # gpu_count = torch.cuda.device_count()
    # gpu_count = 1
    # num_processes = gpu_count * num_processes_per_gpu if gpu_count > 0 else num_processes_per_gpu

    lock = mp.Lock()
    optimizer = SharedAdam(shared_model.parameters(), lr=lr)
    optimizer.share_memory()

    shared_memory = Memory()
    # multiprocesses, Hogwild! style update
    processes = []
    try:
        # init_episode_counter_val = checkpoint['counter']
        init_episode_counter_val = 0
        optimizer.load_state_dict(checkpoint['optimizer'])
    except KeyError:
        init_episode_counter_val = 0
        max_winning_rate = 0.0
    global_episode_counter = mp.Value('i', init_episode_counter_val)
    global_training_counter = mp.Value('i', 0)
    winners_count = mp.Array('i', 4, lock=True)
    # player_scores = mp.Array('f', 4, lock=True)
    # each worker_thread creates its own environment and trains agents
    train_thread(args.cuda, shared_model, global_episode_counter,
                 global_training_counter, optimizer, lock, shared_memory, winners_count)


if __name__ == '__main__':
    print(args)
    main()

# nohup python -u PolicyGradient_1head.py > log_PG_vanilla_04110101.log 2>&1 &
# 23.4.11 修改了reward 正则化
