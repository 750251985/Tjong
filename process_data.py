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
from enum import Enum
import numpy as np
import os
import random
from copy import deepcopy
import argparse

parser = argparse.ArgumentParser(description='Model')
parser.add_argument('--training_data', type=str, default='./data', help='path to training data folder')
parser.add_argument('-si', '--save_interval', type=int, default=102400, help='how often to save')
args = parser.parse_args()

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


# store all data
class dataManager:
    def __init__(self):
        self.reset('all')

    def reset(self, kind, type='test'):
        assert kind in ['all', 'play', 'action', 'chi_gang']
        doc_dict = {
            "card_feats": [],
            "extra_feats": [],
            "mask": [],
            "target": []
        }
        if kind == 'all':
            self.doc = {
                "play": deepcopy(doc_dict),
                "action": deepcopy(doc_dict),
                "chi_gang": deepcopy(doc_dict)
            }
            self.training = {
                "play": deepcopy(doc_dict),
                "action": deepcopy(doc_dict),
                "chi_gang": deepcopy(doc_dict)
            }
        else:
            # for key, item in self.training[which_part].items():
            #     self.doc[which_part][key].extend(item)
            if type == 'test':
                self.doc[kind] = deepcopy(doc_dict)
            elif type == 'train':
                self.training[kind] = deepcopy(doc_dict)
    # 可以将所有训练数据保存成numpy，但是占据空间过大，不推荐
    def save_data(self, save_type, kind, round):
        if save_type=='test':
            # data = self.doc[kind]
            # data = np.concatenate((np.array(data["card_feats"]),
            #                 np.array(data["extra_feats"]),
            #                 np.array(data["mask"]),
            #                 np.array(data["target"])), axis=1)
            np.save('/media/lb/F88876BF88767C46/lb/processed_data1/{}/test/round_{}.npy'.format(kind, 'test'+str(round)), self.doc[kind])
            self.reset(kind, 'test')
        else:
            np.save('/media/lb/F88876BF88767C46/lb/processed_data1/{}/train/round_{}.npy'.format(kind,'train'+str(round)), self.training[kind])
            # dataload = np.load('/media/lb/F88876BF88767C46/lb/processed_data/{}/round_{}.npy'.format(which_part,str(round)), allow_pickle=True)
            # datadict = dataload.tolist()
            # print(len(datadict))
            self.reset(kind, 'train')

# 添加 牌墙无牌不能杠
class MahjongHandler():
    def __init__(self):
        self.dataManager = dataManager()
        self.total_cards = 34
        self.total_actions = len(responses) - 2
        self.save_interval = args.save_interval
        self.round_count = 0
        self.match = np.zeros(self.total_actions)
        self.count = np.zeros(self.total_actions)
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
        self.round_count += 1

    def step_for_train(self, request=None, response_target=None, fname=None):
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
        if self.turnID > 1:
            available_action_mask, available_card_mask = self.build_available_action_mask(request)
            card_feats = self.build_input(self.hand_free, self.history, self.player_history,
                                          self.player_on_table, self.player_last_play, available_card_mask)
            # available_card_mask = available_card_mask.flatten(order='C')
            extra_feats = np.concatenate((self.player_angang[1:], [self.hand_free.sum()],
                                          available_action_mask, *np.eye(4)[[self.quan, self.myPlayerID]],
                                          self.tile_count))

            def judge_response(available_action_mask):
                if available_action_mask.sum() == available_action_mask[responses.PASS.value]:
                    return False
                return True

            if response_target is not None and judge_response(available_action_mask): # 已经排除了pass
                rand = random.uniform(0, 100)
                if rand > 90:
                    training_data = self.dataManager.doc
                    for kind in ['play', 'action', 'chi_gang']:
                        if len(training_data[kind]['card_feats']) >= self.save_interval:
                            self.dataManager.save_data('test', kind, self.round_count)
                else:
                    training_data = self.dataManager.training
                    for kind in ['play', 'action', 'chi_gang']:
                        if len(training_data[kind]['card_feats']) >= self.save_interval:
                            self.dataManager.save_data('train', kind, self.round_count)

                response_target = response_target.split(' ')
                response_name = response_target[0]
                if response_name == 'GANG':
                    if len(response_target) > 1:
                        response_name = 'ANGANG'
                        self.an_gang_card = response_target[-1]
                    else:
                        response_name = 'MINGGANG'
                if available_action_mask.sum() > 1:  # 最少是一个play动作，>1 代表有动作发生
                    action_target = responses[response_name].value
                    data = training_data["action"]
                    data['card_feats'].append(card_feats.flatten(order='C'))
                    data['extra_feats'].append(extra_feats)
                    data['mask'].append(available_action_mask)
                    data['target'].append(action_target)

                available_card_mask = available_card_mask[responses[response_name].value]
                if responses[response_name] in [responses.CHI, responses.ANGANG, responses.BUGANG]:
                    data = training_data["chi_gang"]
                    data['card_feats'].append(card_feats.flatten(order='C'))
                    data['extra_feats'].append(extra_feats)
                    data['mask'].append(available_card_mask)
                    data['target'].append(self.getCardInd(response_target[1]))

                if responses[response_name] in [responses.PLAY, responses.CHI, responses.PENG]:
                    if responses[response_name] == responses.PLAY:
                        play_target = self.getCardInd(response_target[1])
                        card_mask = available_card_mask
                    else:
                        if responses[response_name] == responses.CHI:
                            chi_peng_ind = self.getCardInd(response_target[1])
                        else:
                            chi_peng_ind = self.getCardInd(request[-1])
                        play_target = self.getCardInd(response_target[-1])
                        card_feats, extra_feats, card_mask = self.simulate_chi_peng(request, responses[response_name], chi_peng_ind)
                    data = training_data['play']
                    data['card_feats'].append(card_feats.flatten(order='C'))
                    data['extra_feats'].append(extra_feats)
                    data['mask'].append(card_mask)
                    data['target'].append(play_target)

        self.prev_request = request
        self.turnID += 1

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
        card_feats = self.build_input(my_free, self.history, self.player_history, on_table, self.player_last_play, available_card_mask)
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
                else:  # 其他鸣牌后出牌
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
            is4thTile = True
        else:
            is4thTile = False
        if self.tile_count[(playerID + 1) % 4] == 0:
            isLAST = True
        else:
            isLAST = False
        if not dianPao:
            hand.remove(last_card)
        try:
            ans = MahjongFanCalculator(tuple(self.hand_fixed_data), tuple(hand), last_card, 0,
                                       playerID == self.myPlayerID,
                                       is4thTile, isGANG, isLAST, self.myPlayerID, self.quan)
        except Exception as err:
            if str(err) == 'ERROR_NOT_WIN':
                return 0
            else:
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
        if int(request[0]) == 3:  # 他人动作
            request[0] = str(requests[request[2]].value)
        elif int(request[0]) == 2:  # 自己动作
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
                    if gangCard == '':
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


def data_main():
    my_bot = MahjongHandler()
    count = 0
    start = time.localtime()
    print(time.strftime("%Y-%m-%d %H:%M:%S", start))
    trainning_data_files = os.listdir(args.training_data)
    for fname in trainning_data_files:
        if fname[-1] == 'y':
            continue
        with open('{}/{}'.format(args.training_data, fname), 'r') as f:
            rounds_data = json.load(f)
            random.shuffle(rounds_data)
            print(fname)
            for round_data in rounds_data:
                for j in range(4):
                    count += 1
                    if count % 2000 == 0:
                        print(count)
                    train_requests = round_data["requests"][j]
                    first_request = '0 {} {}'.format(j, 0)
                    train_requests.insert(0, first_request)
                    train_responses = ['PASS'] + round_data["responses"][j]
                    for _request, _response in zip(train_requests, train_responses):
                        my_bot.step_for_train(_request, _response, fname)
                    my_bot.reset()
    for kind in ['play', 'action', 'chi_gang']:
        my_bot.dataManager.save_data('train', kind, my_bot.round_count)
        my_bot.dataManager.save_data('test', kind, my_bot.round_count)
    print(count)
    print(my_bot.round_count)
    end = time.localtime()
    print(time.strftime("%Y-%m-%d %H:%M:%S", end))
    print('cost：', end - start)

if __name__ == '__main__':
    data_main()

# nohup python -u process_data.py > log_process_data_03130315.log 2>&1 &