import random

def poker_game():
    """
    :rtype: List[List[str]]
    """

    # 红桃, 黑桃, 梅花, 方块
    suit = ['H', 'S', 'C', 'D']
    nums = [str(i) for i in range(1, 11)] + ['J', 'Q', 'K']

    # 生成所有的扑克牌，形如[H1, H2, ..., HK, S1, S2, ..., DK]
    all_cards = []
    for s in suit:
        for n in nums:
            all_cards.append(s + n)

    # 加入大小王
    all_cards += ['RJ', "BJ"]

    # 洗牌
    random.shuffle(all_cards)

    # 发牌规则：3个玩家轮流摸牌
    results = [[] for _ in range(3)]
    for i, card in enumerate(all_cards):
        results[i % 3].append(card)

    # 对玩家手牌排序
    for player_hand in results:
        # 将手牌按花色和点数进行排序
        player_hand.sort(key=lambda card: (suit.index(card[0]) if card[0] in suit else 4, # 花色排序，王放最后
                                           nums.index(card[1:]) if card[1:] in nums else 13)) # 点数排序，王放最后

    return results


print(poker_game())