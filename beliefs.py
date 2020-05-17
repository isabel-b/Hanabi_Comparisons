import pandas as pd
import os
import numpy as np
import sys



n=2
hand_size = 5



def letter_to_num(color):
    if color == 'r':
        return 0
    if color == 'o':
        return 1
    if color == 'y':
        return 2
    if color == 'g':
        return 3
    if color == 'b':
        return 4
    return -1

def decode_hands(hands_list):
    actual_hands = np.zeros((len(hands_list), hand_size, 5, 5))
    cardlens = np.zeros(len(hands_list))
    for i, hand in enumerate(hands_list):
        cards = hand.split(',')
        cardlens[i] = len(cards)
        for j, card in enumerate(cards):
            value = int(card[0]) - 1
            color = letter_to_num(card[1])
            actual_hands[i,j,color,value] = 1
    return actual_hands, cardlens


# hand_knowledge: hand numpy array
# 0 -> cannot be
# 1 -> could be
def cannots_to_probs(hand_knowledge, candidates):
    # Get sum of possible cards each card could be
    knol_with_candidates = hand_knowledge * candidates
    hkSum = np.sum(knol_with_candidates, axis=(2, 3), keepdims=True)
    return np.divide(knol_with_candidates, hkSum, out=np.zeros_like(knol_with_candidates), where=hkSum!=0)

def probs_to_cannots(beliefs):
    return np.divide(beliefs, beliefs, out=np.zeros_like(beliefs), where=beliefs!=0)

def cannots_correctness(actual_hand, hand_knowledge):
    # if actual_hand && hand_knowledge == False: incorrect
    # add up how many cards in hand are incorrect.
    is_correct = actual_hand * hand_knowledge
    return np.sum(is_correct)

# Convert separate color and value cannots/musts to total:
# First, set up np array. Start with all 1s.
# Musts: put 0 for all others (e.g. must be green -> 0 in red, blue, etc rows)
# Cannots: put 0 for cannots.


# probs
# Input:
#       Actual hand: numpy array: 4/5 cards per hand * 5 colors * 5 numbers
#       Beliefs about hand: numpy array, same format
def calculate_loss(actual_hand, hand_beliefs):
    return np.mean(np.sum((-1)*actual_hand*np.clip(np.log(hand_beliefs,where=hand_beliefs>0,out=-5*np.ones_like(hand_beliefs)),-5,0), axis=(2,3)))





def get_info_score(hands, fp, current_line_list, candidates):
    p = 0
    c = 0
    actual_hands, handlens = decode_hands(hands)
    hand_beliefs = np.ones((len(hands), hand_size, 5, 5))
    while p < n:
        fp.readline() # p
        while c < hand_size:
            fp.readline() # roygb
            for i in list(range(5)):
                val = fp.readline()
                color = 0
                for cl in val[3:8]:
                    if cl == "K" and c < handlens[p]:
                        hand_beliefs[p,c,color,i] = 1
                    elif c < handlens[p]:
                        hand_beliefs[p, c, color, i] = 0
                    color += 1
            c += 1
        p += 1
        c = 0
        #if c >= hand_size-1 and p >= n-1:
    hand_probs = cannots_to_probs(hand_beliefs, candidates)
    correct = cannots_correctness(actual_hands, hand_beliefs)
    loss = calculate_loss(actual_hands, hand_probs)
    return [correct, loss], fp



def get_torch_score(hands, fp, current_line_list, candidates):
    p = 0
    c = 0
    actual_hands, handlens = decode_hands(hands)
    hand_beliefs = np.ones((len(hands), hand_size, 5, 5))
    beliefs=True
    while p < n:
        if beliefs:
            fp.readline() # p
        while c < hand_size:
            if beliefs:
                line = fp.readline()
                # print(line)
                # print('here')
                words = line.split()
            if len(words) and words[0] == "currently":
                hand_beliefs[p,c]=candidates/np.sum(candidates)
                beliefs=False
            else:
                for i in list(range(5)):
                    col = fp.readline()
                    value = 0
                    for cl in col.split()[1:6]:
                        hand_beliefs[p,c,i,value] = float(cl)
                        value += 1
            c += 1
        p += 1
        c = 0
        if p < n and beliefs:
            fp.readline()
    hand_knol = probs_to_cannots(hand_beliefs)
    correct = cannots_correctness(actual_hands, hand_knol)
    loss = calculate_loss(actual_hands, hand_beliefs)
    return [correct, loss], fp


def get_smart_score(hands, fp, current_line_list, candidates):
    p = 0
    c = 0
    actual_hands, handlens = decode_hands(hands)
    hand_beliefs = np.ones((len(hands), hand_size, 5, 5))
    while p < n:
        fp.readline() # p
        while c < handlens[p]:
            fp.readline() # roygb
            for i in list(range(5)):
                val = fp.readline()
                color = 0
                for cl in val[1:6]:
                    if cl == "K":
                        hand_beliefs[p,c,color,i] = 1
                    else:
                        hand_beliefs[p, c, color, i] = 0
                    color += 1
            fp.readline()
            fp.readline()
            fp.readline()
            fp.readline()
            c += 1
        p += 1
        c = 0
        fp.readline()
    hand_probs = cannots_to_probs(hand_beliefs, candidates)
    correct = cannots_correctness(actual_hands, hand_beliefs)
    loss = calculate_loss(actual_hands, hand_probs)
    return [correct, loss], fp


def get_holmes_score(hands, fp, current_line_list, candidates):
    score = 0
    player = current_line_list[-2][0]
    p = 0
    c = 0
    actual_hands, handlens = decode_hands(hands)
    hand_beliefs = np.ones((len(hands), hand_size, 5, 5))
    while p < n:
        while c < hand_size:
            pos = fp.tell()
            line = fp.readline()
            # print(line)
            if line[0].isdigit() and int(line[0])>p:
                p+=1
                if p == n:
                    break
                c = 0
            if c > 0 and line[0:4]!="Card":
                fp.seek(pos)
                break
            if c > 0 or p > 0:
                fp.readline()
            # values:
            for i in list(range(5)):
                val = fp.readline()
                if val[1] == '.':
                    hand_beliefs[p, c, :, i] = 0
                if val[2] == 'M':
                    for v in list(range(5)):
                        if v != i:
                            hand_beliefs[p, c, :, v] = 0
            # colors:
            for i in list(range(5)):
                col = fp.readline().split()[1]
                if col[1] == '.':
                    hand_beliefs[p, c, i, :] = 0
                if col[2] == 'M':
                    for r in list(range(5)):
                        if r != i:
                            hand_beliefs[p, c, r, :] = 0
            c += 1
        p += 1
        c = 0
    hand_probs = cannots_to_probs(hand_beliefs, candidates)
    correct = cannots_correctness(actual_hands, hand_beliefs)
    loss = calculate_loss(actual_hands, hand_probs)
    return [correct, loss], fp



def get_bot_score(botname, hands, fp, words, candidates):
    if botname[0:6] == "Holmes":
        return get_holmes_score(hands, fp, words, candidates)
    if botname[0:5] == "Smart":
        return get_smart_score(hands, fp, words, candidates)
    if botname[0:6] == "Simple":
        return get_holmes_score(hands, fp, words, candidates)
    if botname[0:5] == "Torch":
        return get_torch_score(hands, fp, words, candidates)
    if botname[0:4] == "Info":
        return get_info_score(hands, fp, words, candidates)
    if botname[0:2] == "SB":
        return [0,0], fp
    print('ERROR - invalid bot name')
    print(botname)
    return 10000


# path = '/Volumes/Untitled'

# belief_scores = []



def main():
    global n, hand_size
    file_path = sys.argv[1]
    n = int(sys.argv[2])
    score = []
    if n == 2:
        hand_size=5
        b_score_0_acc = []
        b_score_0_loss = []
        b_score_1_acc = []
        b_score_1_loss = []
        b_score_0_ingame_acc = []
        b_score_1_ingame_acc = []
        b_score_0_ingame_loss = []
        b_score_1_ingame_loss = []
    elif n == 4:
        hand_size=4
        b_score_0_acc = []
        b_score_0_loss = []
        b_score_1_acc = []
        b_score_1_loss = []
        b_score_2_acc = []
        b_score_2_loss = []
        b_score_3_acc = []
        b_score_3_loss = []
        b_score_0_ingame_acc = []
        b_score_1_ingame_acc = []
        b_score_2_ingame_acc = []
        b_score_3_ingame_acc = []
        b_score_0_ingame_loss = []
        b_score_1_ingame_loss = []
        b_score_2_ingame_loss = []
        b_score_3_ingame_loss = []
        b_score_0_ingame_acc = []
        b_score_1_ingame_acc = []
        b_score_2_ingame_acc = []
        b_score_3_ingame_acc = []
        b_score_0_ingame_loss = []
        b_score_1_ingame_loss = []
        b_score_2_ingame_loss = []
        b_score_3_ingame_loss = []
    else:
        print('error')
        exit()

    candidates = np.array([[3,2,2,2,1],[3,2,2,2,1],[3,2,2,2,1],[3,2,2,2,1],[3,2,2,2,1]])
    with open(file_path, 'r') as r_f:
        line = r_f.readline()
        while line:
            words = line.split()
            if words and words[0] == "post_process,":
                words.pop(0)
            if words and len(words)>3 and (words[2] == "played" or words[2] == "discarded"):
                card = words[6][1:3]
                value=int(card[0])-1
                color = letter_to_num(card[1])
                if candidates[color, value] > 0:
                    candidates[color, value] -= 1
                else:
                    print('error - too many cards of same type played')
            if words and words[0] == "Current" and words[1] == "hands:":
                hands = words[2:2+n]
                # print(hands)
                # print('just printed hands')
                line = r_f.readline()
                words = line.split()
                # print(words)
                bot_0 = words[1].split(':')[1]
                s,  r_f = get_bot_score(bot_0, hands, r_f, words, candidates)
                b_score_0_ingame_acc.append(s[0])
                b_score_0_ingame_loss.append(s[1])
                line = r_f.readline()
                words = line.split()
                # print(words)
                bot_1 = words[0]
                # print('bot 1', bot_1)
                s, r_f = get_bot_score(bot_1, hands, r_f, words, candidates)
                b_score_1_ingame_acc.append(s[0])
                b_score_1_ingame_loss.append(s[1])
                if n > 2:
                    line = r_f.readline()
                    words = line.split()
                    # print(words)
                    bot_2 = words[0]
                    # print('bot 2', bot_2)
                    s, r_f = get_bot_score(bot_2, hands, r_f, words, candidates)
                    b_score_2_ingame_acc.append(s[0])
                    b_score_2_ingame_loss.append(s[1])
                    line = r_f.readline()
                    words = line.split()
                    # print(words)
                    bot_3 = words[0]
                    # print('bot 3', bot_3)
                    s, r_f = get_bot_score(bot_3, hands, r_f, words, candidates)
                    b_score_3_ingame_acc.append(s[0])
                    b_score_3_ingame_loss.append(s[1])
            line = r_f.readline()
            if words and words[0] == "Final":
                candidates = np.array([[3,2,2,2,1],[3,2,2,2,1],[3,2,2,2,1],[3,2,2,2,1],[3,2,2,2,1]])
                score.append(float(words[4]))
                b_score_0_acc.append(np.asscalar(np.nanmean(np.array(b_score_0_ingame_acc))))
                b_score_1_acc.append(np.asscalar(np.nanmean(np.array(b_score_1_ingame_acc))))
                b_score_0_ingame_acc = []
                b_score_1_ingame_acc = []
                if n > 2:
                    b_score_2_acc.append(np.asscalar(np.nanmean(np.array(b_score_2_ingame_acc))))
                    b_score_3_acc.append(np.asscalar(np.nanmean(np.array(b_score_3_ingame_acc))))
                    b_score_2_ingame_acc = []
                    b_score_3_ingame_acc = []
                b_score_0_loss.append(np.asscalar(np.nanmean(np.array(b_score_0_ingame_loss))))
                b_score_1_loss.append(np.asscalar(np.nanmean(np.array(b_score_1_ingame_loss))))
                b_score_0_ingame_loss = []
                b_score_1_ingame_loss = []
                if n > 2:
                    b_score_2_loss.append(np.asscalar(np.nanmean(np.array(b_score_2_ingame_loss))))
                    b_score_3_loss.append(np.asscalar(np.nanmean(np.array(b_score_3_ingame_loss))))
                    b_score_2_ingame_loss = []
                    b_score_3_ingame_loss = []

        if n == 2:
            df = pd.DataFrame(list(zip(b_score_0_acc, b_score_1_acc, b_score_0_loss, b_score_1_loss, score)), columns=['p0Acc', 'p1Acc', 'p0loss', 'p1loss', 'score'])
            df.to_csv(file_path[0:-4]+bot_0.strip(':')+bot_1.strip(':')+".csv")
        else:
            df = pd.DataFrame(list(zip(b_score_0_acc, b_score_1_acc, b_score_2_acc, b_score_3_acc, b_score_0_loss, b_score_1_loss, b_score_2_loss, b_score_3_loss, score)), columns=['p0Acc', 'p1Acc', 'p2Acc', 'p3Acc', 'p0loss', 'p1loss', 'p2loss', 'p3loss', 'score'])
            df.to_csv(file_path[0:-4]+bot_0.strip(':')+bot_1.strip(':')+bot_2.strip(':')+bot_3.strip(':')+".csv")



# path='logs4pt'
#
# files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
# print(files)
# not_files = ['Info-Info-Info-Info.txt', 'Info-Info-Info-Smart.txt', 'Info-Info-Smart-Info.txt', 'Info-Info-Smart-Smart.txt', 'Info-Smart-Info-Info.txt', 'Info-Smart-Info-Smart.txt', 'Info-Smart-Smart-Info.txt', 'Smart-Info-Info-Info.txt'] + ['Smart-Info-Smart-Smart.txt', 'Simple-Simple-Simple-Simple.txt', 'Smart-Smart-Info-Smart.txt', 'Smart-Info-Info-Smart.txt', 'Smart-Smart-Smart-Info.txt']
# for f in files:
#     print(f)
#     if f[-4:]==".txt" and f not in not_files:
#         with open(os.path.join(path, f), 'r') as r_f:
#             line = r_f.readline()
#             score_0 = []
#             score_1 = []
#             score_2 = []
#             score_3 = []
#             while line:
#                 words = line.split()
#                 if words and words[0] == "Current" and words[1] == "hands:":
#                     hands = words[2:2+n]
#                     print(hands)
#                     print('just printed hands')
#                     line = r_f.readline()
#                     words = line.split()
#                     bot_0 = words[1].split(':')[1]
#                     s,  r_f = get_bot_score(bot_0, hands, r_f, words)
#                     score_0.append(s)
#                     line = r_f.readline()
#                     words = line.split()
#                     print(words)
#                     bot_1 = words[0]
#                     print('bot 1', bot_1)
#                     s, r_f = get_bot_score(bot_1, hands, r_f, words)
#                     score_1.append(s)
#                     line = r_f.readline()
#                     words = line.split()
#                     print(words)
#                     bot_2 = words[0]
#                     print('bot 2', bot_2)
#                     s, r_f = get_bot_score(bot_2, hands, r_f, words)
#                     score_2.append(s)
#                     line = r_f.readline()
#                     words = line.split()
#                     print(words)
#                     bot_3 = words[0]
#                     print('bot 3', bot_3)
#                     s, r_f = get_bot_score(bot_3, hands, r_f, words)
#                     score_3.append(s)
#                 line = r_f.readline()
#             x = np.array(score_0)
#             s0 = np.nanmean(x, axis=0)
#             x = np.array(score_1)
#             s1 = np.nanmean(x, axis=0)
#             x = np.array(score_2)
#             s2 = np.nanmean(x, axis=0)
#             x = np.array(score_3)
#             s3 = np.nanmean(x, axis=0)
#             belief_scores.append([bot_0+bot_1+bot_2+bot_3]+[s0[0], s1[0], s2[0], s3[0], s0[1], s1[1], s2[1], s3[1]])
#
# df_score = pd.DataFrame(belief_scores)
# df_score.to_csv('4pbeliefst.csv')
#
# score_0 = []
# score_1 = []
#
# with open(os.path.join(path, 'Smart-InfoSearch.txt'), 'r') as r_f:
#     line = r_f.readline()
#
#     while line:
#         print(line)
#         words = line.split()
#         if words and words[0] == "Current" and words[1] == "hands:":
#             hands = words[2:2+n]
#             print(hands)
#             print('just printed hands')
#             line = r_f.readline()
#             words = line.split()
#             bot_0 = words[1].split(':')[1]
#             s,  r_f = get_bot_score(bot_0, hands, r_f, words)
#             score_0.append(s)
#             line = r_f.readline()
#             words = line.split()
#             bot_1 = words[0]
#             s, r_f = get_bot_score(bot_1, hands, r_f, words)
#             score_1.append(s)
#         line = r_f.readline()
#     belief_scores = [score_0, score_1]
#     x = np.array(belief_scores)
#     print(x.shape)
#     np.mean(x, axis=1)
#
#
# path1='logssearch'
# path2='logssearch2'
# path3='logssearch3'
# path=path1
# score_0 = []
# score_1 = []
# with open(os.path.join(path, 'Torch-InfoSearch.txt'), 'r') as r_f:
#     line = r_f.readline()
#     while line:
#         words = line.split()
#         if words and words[0] == "Current" and words[1] == "hands:":
#             hands = words[2:2+n]
#             print(hands)
#             print('just printed hands')
#             line = r_f.readline()
#             words = line.split()
#             bot_0 = words[1].split(':')[1]
#             s,  r_f = get_bot_score(bot_0, hands, r_f, words)
#             score_0.append(s)
#             line = r_f.readline()
#             words = line.split()
#             bot_1 = words[0]
#             s, r_f = get_bot_score(bot_1, hands, r_f, words)
#             score_1.append(s)
#         line = r_f.readline()
#
# path='logssearch4'
#
# with open(os.path.join(path, 'Torch-TorchSearch.txt'), 'r') as r_f:
#     line = r_f.readline()
#     while line:
#         words = line.split()
#         if words and words[0] == "Current" and words[1] == "hands:":
#             hands = words[2:2+n]
#             print(hands)
#             print('just printed hands')
#             line = r_f.readline()
#             words = line.split()
#             bot_0 = words[1].split(':')[1]
#             s,  r_f = get_bot_score(bot_0, hands, r_f, words)
#             score_0.append(s)
#             line = r_f.readline()
#             words = line.split()
#             bot_1 = words[0]
#             s, r_f = get_bot_score(bot_1, hands, r_f, words)
#             score_1.append(s)
#         line = r_f.readline()
#
# path=path3
#
# with open(os.path.join(path, 'Torch-TorchSearch.txt'), 'r') as r_f:
#     line = r_f.readline()
#     while line:
#         words = line.split()
#         if words and words[0] == "Current" and words[1] == "hands:":
#             hands = words[2:2+n]
#             print(hands)
#             print('just printed hands')
#             line = r_f.readline()
#             words = line.split()
#             bot_0 = words[1].split(':')[1]
#             s,  r_f = get_bot_score(bot_0, hands, r_f, words)
#             score_0.append(s)
#             line = r_f.readline()
#             words = line.split()
#             bot_1 = words[0]
#             s, r_f = get_bot_score(bot_1, hands, r_f, words)
#             score_1.append(s)
#         line = r_f.readline()
#
# with open(os.path.join(path, 'Torch-TorchSearch2.txt'), 'r') as r_f:
#     line = r_f.readline()
#     while line:
#         words = line.split()
#         if words and words[0] == "Current" and words[1] == "hands:":
#             hands = words[2:2+n]
#             print(hands)
#             print('just printed hands')
#             line = r_f.readline()
#             words = line.split()
#             bot_0 = words[1].split(':')[1]
#             s,  r_f = get_bot_score(bot_0, hands, r_f, words)
#             score_0.append(s)
#             line = r_f.readline()
#             words = line.split()
#             bot_1 = words[0]
#             s, r_f = get_bot_score(bot_1, hands, r_f, words)
#             score_1.append(s)
#         line = r_f.readline()
#
# with open(os.path.join(path, 'Torch-TorchSearch3.txt'), 'r') as r_f:
#     line = r_f.readline()
#     while line:
#         words = line.split()
#         if words and words[0] == "Current" and words[1] == "hands:":
#             hands = words[2:2+n]
#             print(hands)
#             print('just printed hands')
#             line = r_f.readline()
#             words = line.split()
#             bot_0 = words[1].split(':')[1]
#             s,  r_f = get_bot_score(bot_0, hands, r_f, words)
#             score_0.append(s)
#             line = r_f.readline()
#             words = line.split()
#             bot_1 = words[0]
#             s, r_f = get_bot_score(bot_1, hands, r_f, words)
#             score_1.append(s)
#         line = r_f.readline()
#
# belief_scores = [score_0, score_1]
# x = np.array(belief_scores)
# print(x.shape)
# np.mean(x, axis=1)
#
#
#
# score=0
# count=0
# perfect_games = 0
# path=path3
#
# with open(os.path.join(path, 'Torch-TorchSearch.txt'), 'r') as r_f:
#     line = r_f.readline()
#     while line:
#         words = line.split()
#         if len(words) and words[0] == "Final":
#             score+=int(words[4])
#             count += 1
#             if int(words[4]) == 25:
#                 perfect_games += 1
#         line = r_f.readline()
#
# with open(os.path.join(path, 'Torch-TorchSearch2.txt'), 'r') as r_f:
#     line = r_f.readline()
#     while line:
#         words = line.split()
#         if len(words) and words[0] == "Final":
#             score+=int(words[4])
#             count += 1
#             if int(words[4]) == 25:
#                 perfect_games += 1
#         line = r_f.readline()
#
#
# with open(os.path.join(path, 'Torch-TorchSearch3.txt'), 'r') as r_f:
#     line = r_f.readline()
#     while line:
#         words = line.split()
#         if len(words) and words[0] == "Final":
#             score+=int(words[4])
#             count += 1
#             if int(words[4]) == 25:
#                 perfect_games += 1
#         line = r_f.readline()
#
# path='logssearch4'
#
# with open(os.path.join(path, 'Torch-TorchSearch.txt'), 'r') as r_f:
#     line = r_f.readline()
#     while line:
#         words = line.split()
#         if len(words) and words[0] == "Final":
#             score+=int(words[4])
#             count += 1
#             if int(words[4]) == 25:
#                 perfect_games += 1
#         line = r_f.readline()
#
#
# print('mean:', float(score)/count)
# mean=float(score)/count
# print('perfect:', perfect_games)
# print('count:', count)
# var=0
# path=path3
#
# with open(os.path.join(path, 'Torch-TorchSearch.txt'), 'r') as r_f:
#     line = r_f.readline()
#     while line:
#         words = line.split()
#         if len(words) and words[0] == "Final":
#             var+=(mean - int(words[4]))**2
#         line = r_f.readline()
#
#
# with open(os.path.join(path, 'Torch-TorchSearch2.txt'), 'r') as r_f:
#     line = r_f.readline()
#     while line:
#         words = line.split()
#         if len(words) and words[0] == "Final":
#             var+=(mean - int(words[4]))**2
#         line = r_f.readline()
#
# with open(os.path.join(path, 'Torch-TorchSearch3.txt'), 'r') as r_f:
#     line = r_f.readline()
#     while line:
#         words = line.split()
#         if len(words) and words[0] == "Final":
#             var+=(mean - int(words[4]))**2
#         line = r_f.readline()
#
# path='logssearch4'
# with open(os.path.join(path, 'Torch-TorchSearch.txt'), 'r') as r_f:
#     line = r_f.readline()
#     while line:
#         words = line.split()
#         if len(words) and words[0] == "Final":
#             var+=(mean - int(words[4]))**2
#         line = r_f.readline()
#
#
# with open(os.path.join(path, 'Smart-InfoSearch.txt'), 'r') as r_f:
#     line = r_f.readline()
#     while line:
#         words = line.split()
#         if len(words) and words[0] == "Final":
#             var+=(mean - int(words[4]))**2
#         line = r_f.readline()
# with open(os.path.join(path, 'Smart-InfoSearch2.txt'), 'r') as r_f:
#     line = r_f.readline()
#     while line:
#         words = line.split()
#         if len(words) and words[0] == "Final":
#             var+=(mean - int(words[4]))**2
#         line = r_f.readline()
#
# with open(os.path.join(path, 'Smart-InfoSearch3.txt'), 'r') as r_f:
#     line = r_f.readline()
#     while line:
#         words = line.split()
#         if len(words) and words[0] == "Final":
#             var+=(mean - int(words[4]))**2
#         line = r_f.readline()
#
# print('var', var)
# print('var:', var/(count-1))




if __name__ == "__main__":
    main()
