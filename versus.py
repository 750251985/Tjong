#!/usr/bin/env python
# encoding: utf-8
'''
@file: versus.py
@time: ？？？？
AI对战
'''

import torch
from enum import Enum
import numpy as np

import sys
import os
import torch.multiprocessing as mp
import argparse
import time

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

parser = argparse.ArgumentParser(description='Versus')
parser.add_argument('-m1p', '--model_cnn_path', type=str, default='./models/dl_CNN_3heads_1682951596.106185.pt', help='path to cnn model as components')
parser.add_argument('-m2p', '--model_rnn_path', type=str, default='./models/dl_GRU_3heads_1682244759.4946852.pt', help='path to rnn model as components')
parser.add_argument('-m3p', '--model_mlp_path', type=str, default='./models/dl_MLP_3heads_1681998305.70236.pt', help='path to mlp model as components')
parser.add_argument('-m4p', '--model_vit_path', type=str, default='./models/dl_vit_block6_21677488505.613873', help='path to vit model as components')
parser.add_argument('-m5p', '--model_tit1_path', type=str, default='./models/dl_TIT_1head_afternorm_1680087559.7798574.pt', help='path to vit model as components')
parser.add_argument('-m6p', '--model_tit_pg_path', type=str, default='./models/rl_pg_vanilla_r_0519-220410.pt', help='path to tit model as components')
parser.add_argument('-m7p', '--model_tit_wqds_path', type=str, default='./bots/wqds.pt', help='path to tit model as components')
parser.add_argument('-m8p', '--model_tit_jks_path', type=str, default='./bots/jks.pt', help='path to tit model as components')
parser.add_argument('-m9p', '--model_tit_ndwzm_path', type=str, default='./bots/ndwzm.pt', help='path to tit model as components')
parser.add_argument('-m10p', '--model_tit_ylzz_path', type=str, default='./bots/dl_TIT_3heads_blocks3_fc3.pt', help='path to tit model as components')
parser.add_argument('-m11p', '--model_resnet_path', type=str, default='./models/dl_ResNet_3heads_1682252322.684695.pt', help='path to resnet model as components')
parser.add_argument('-m12p', '--model_tit_path', type=str, default='./models/dl_TIT_3heads_blocks3_fc3_1682935896.489467.pt', help='path to resnet model as components')
# pi, si的含义为经过多少个episode，若rn=10, rt=10, 则一个episode为10*10=100games
parser.add_argument('-pi', '--print_interval', type=int, default=24, help='how often to print')
parser.add_argument('-tn', '--total_number', type=int, default=24*100, help='epochs to run in total，全部epochs，控制停止')
parser.add_argument('-rn', '--round_number', type=int, default=1, help='round number*repeated_times to run in parallel，一个round跑几圈')
parser.add_argument('-rt', '--repeated_times', type=int, default=1,
                    help='the repeated times for one round，一副牌的对战次数，暂时用作一圈跑几盘，')  # 一副牌东南西北，玩家各打一次,>2,否则减mean后分数全为零
args = parser.parse_args()
args.model_name = 'pg Vs ylzz wqds jks'
args.cuda = 'cpu'

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


class LogMahjong:
    def __init__(self):
        self.log = {"players": "", "request": [], "response": []}

    def get_log(self):
        return self.log

    def add_request(self, request=None):
        self.log["request"].append(request)

    def add_response(self, response=None):
        self.log["response"].append(response)
    def clear_memory(self):
        self.log = {"players": "", "request": [], "response": []}

from DL_CNN_3heads_agent import agent as cnn_agent
from DL_MLP_3heads_agent import agent as mpl_agent
from DL_ResNet_3heads_agent import agent as resnet_agent
from DL_VIT1_agent import agent as vit_agent
from DL_RNN_3heads_agent import agent as gru_agent
from DL_TIT_1head_agent import agent as tit1_agent
from DL_TIT_3heads_agent import agent as tit_agent
from bots.ndwzm import agent as ndwzm
from bots.jks import agent as jks
from bots.wqds import agent as wqds
from bots.ylzz import agent as ylzz


class MahjongEnv:
    def __init__(self, cuda='cpu', lock=None):
        use_cuda = torch.cuda.is_available()
        self.repeated_times = args.repeated_times  #  一副牌的对战次数
        self.round_number = args.round_number * self.repeated_times  #并行游戏同时对战，所以相乘  ，伪并行，串行收集state，批送入模型。
        self.log_each_rounds = [LogMahjong() for _ in range(4 * self.round_number)]
        self.device = torch.device(cuda if use_cuda else "cpu")
        print('using ' + str(self.device))
        self.total_cards = 34
        self.total_actions = len(responses) - 2
        self.model_code_0 = 0
        self.model_code_1 = 1
        # self.model3_code = 2
        # self.model4_code = 3
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
        bot4 = tit1_agent(model_path=args.model_tit1_path, cuda=cuda)
        bot3 = ylzz(model_path=args.model_tit_pg_path, cuda=cuda)
        bot2 = wqds(model_path=args.model_tit_wqds_path, cuda=cuda)
        bot1 = jks(model_path=args.model_tit_jks_path, cuda=cuda)
        for _ in range(self.round_number):  # 不存在并发，没必要新建如此多的bots，均为handler
            bots = []
            bots.append(bot1)
            bots.append(bot2)
            bots.append(bot3)
            bots.append(bot4)
            self.bots.append(bots)
        self.train = False
        self.reset(True)

    def reset(self, initial=False):
        self.tile_walls = []   # 牌墙
        self.quans = []  # 圈风
        self.mens = []  # 门风
        self.bots_orders = []  #座次
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
                quan = np.random.choice(4)  #改变圈风
            men = (round_id % 4)  #改变门风
              #改变门风
            # 用pop，从后面摸牌
            self.tile_walls.append(np.reshape(all_tiles, (4, -1)).tolist())  # 每个人分牌
            self.quans.append(quan)
            self.mens.append(men)
            # 这一局bots的order，牌墙永远下标和bot一致
            self.bots_orders.append([self.bots[round_id][(i + self.mens[-1]) % 4] for i in range(4)])
            self.drawers.append(0)
        # print("all_tiles:",all_tiles)
        if not initial:  # 真重置
            self.round_count += self.round_number
            for bots in self.bots_orders:
                for bot in bots:
                    bot.reset()
            for id, log in enumerate(self.log_each_rounds):
                log.clear_memory()
            # self.scores = np.zeros((self.round_number, 4), dtype=float)  # 在外部初始化，记录score

    def run_rounds(self):
        turnID = 0
        player_responses = [['PASS'] * 4 for _ in range(self.round_number)]
        finished = np.zeros(self.round_number, dtype=int)
        while finished.sum() < self.round_number:
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
                            self.calculate_scores(round_id, winner_id, dianpaoer, fan_count, mode='botzone')
                        finished[round_id] = 1
                    else:
                        for i in range(4):
                            player_responses[round_id][i] = self.bots_orders[round_id][i].step(requests[i])
            turnID += 1
        for i in range(self.round_number):
            for j in range(4):
                # 将score按bot顺序调整
                self.scores4count[i][j] = self.scores[i][(j - self.mens[i]) % 4]

    # 不和牌，分数都是0，不会调用这个函数
    def calculate_scores(self, round_id, winner_id=0, dianpaoer=None, fan_count=0, difen=8, mode='botzone'):
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
                    self.scores[round_id][i] = -0.5 * fan_count
                else:
                    self.scores[round_id][i] = 0

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
        print('total winning: ',winners[:])
        print('total scores: ', player_scores[:])
        total_rounds = current_round * self.round_number
        rounds_this_stage = print_interval * self.round_number
        print(
            '{}: total rounds: {}, during the last {} rounds, bot 13 winning rate: {:.2%}, bot 24 winning rate: {:.2%}\n'
            'Hu {} rounds，Huang-zhuang {} rounds，hu ratio {:.2%}, average rounds to hu: {}, took {:.2f} minutes per 10000 rounds'.format(
                type, total_rounds, rounds_this_stage,
                sum(winners[::2]) / rounds_this_stage,
                sum(winners[1::2]) / rounds_this_stage,
                win_sum,
                rounds_this_stage - win_sum,
                win_sum / rounds_this_stage,
                sum(self.win_steps) / (len(self.win_steps)+1e-9),
                time_cost
            ))
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

# thread for training
def versus_thread(cuda, global_episode_counter, lock, winners, player_scores):

    start_time = time.perf_counter()
    start_counter = global_episode_counter.value
    print_interval = args.print_interval

    env = MahjongEnv(cuda=cuda, lock=lock)

    while global_episode_counter.value < args.total_number:
        env.run_rounds()  # 一个episode= rn * rt
        env.reset(False)
        with lock:
            global_episode_counter.value += 1
            current_round = global_episode_counter.value
            for i in range(4):
                winners[i] += env.winners[i]
                player_scores[i] += np.sum(env.scores4count, axis=0)[i]
            env.winners = np.zeros(4, dtype=int)
            env.scores4count = np.zeros((env.round_number, 4), dtype=float)

            if current_round % print_interval == 0:
                total_rounds = (current_round - start_counter) * env.round_number
                this_time = time.perf_counter()
                time_cost = (this_time - start_time) / (60 * (total_rounds / 10000))
                env.print_log(args.model_name, current_round, print_interval, winners, player_scores, time_cost)
                for i in range(4):
                    winners[i] = 0


def main():
    mp.set_start_method('spawn')  # required to avoid Conv2d froze issue
    lock = mp.Lock()
    global_episode_counter = mp.Value('i', 0)
    winners_count = mp.Array('i', 4, lock=True)
    player_scores = mp.Array('f', 4, lock=True)
    versus_thread(args.cuda, global_episode_counter, lock, winners_count, player_scores)

if __name__ == '__main__':
    print(args)
    main()


# 23.4.11 修改了reward 正则化
# nohup python -u versus.py > log_tit_vs_ndwzm_05191234.log 2>&1 &