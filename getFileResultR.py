import os

dataset_seq = [2]
cm_seq = [1, 2]
# model is optional
model_seq = ['mng', 'mngr', 'mngpw', 'mngrpw', 'mngsr', 'mngsrpw',
             'mngap', 'mngapr', 'mngappw', 'mngaprpw', 'mngapsr', 'mngapsrpw',
             'mhd', 'mhed', 'mhdpw', 'mhedpw',
             'mpmis', 'mr']
ppp_seq = [1, 2, 3]
wpiwp_seq = [bool(1), bool(0)]
prod_seq, prod2_seq = [1, 2], [1, 2, 3]
wallet_distribution_seq = [1, 2]

for data_setting in dataset_seq:
    dataset_name = 'email_undirected' * (data_setting == 1) + 'dnc_email_directed' * (data_setting == 2) + 'email_Eu_core_directed' * (data_setting == 3) + \
                    'WikiVote_directed' * (data_setting == 4) + 'NetPHY_undirected' * (data_setting == 5)
    for cm in cm_seq:
        cascade_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
        for model_name in model_seq:
            profit, cost, time_avg, time_total = [[[] for _ in range(len(ppp_seq))] for _ in range(len(wallet_distribution_seq))], [[[] for _ in range(len(ppp_seq))] for _ in range(len(wallet_distribution_seq))], [[[] for _ in range(len(ppp_seq))] for _ in range(len(wallet_distribution_seq))], [[[] for _ in range(len(ppp_seq))] for _ in range(len(wallet_distribution_seq))]
            ratio_profit, ratio_cost, number_an, number_seed = [[[] for _ in range(len(ppp_seq))] for _ in range(len(wallet_distribution_seq))], [[[] for _ in range(len(ppp_seq))] for _ in range(len(wallet_distribution_seq))], [[[] for _ in range(len(ppp_seq))] for _ in range(len(wallet_distribution_seq))], [[[] for _ in range(len(ppp_seq))] for _ in range(len(wallet_distribution_seq))]
            for wallet_distribution in wallet_distribution_seq:
                wallet_distribution_type = 'm50e25' * (wallet_distribution == 1) + 'm99e96' * (wallet_distribution == 2)
                for ppp in ppp_seq:
                    for prod_setting in prod_seq:
                        for wpiwp in wpiwp_seq:
                            for prod_setting2 in prod2_seq:
                                product_name = 'item_lphc' * (prod_setting == 1) + 'item_hplc' * (prod_setting == 2) + '_ce' * (prod_setting2 == 2) + '_ee' * (prod_setting2 == 3)
                                num_product = len(open('item/' + product_name + '.txt').readlines())

                                path1 = 'result/r_' + dataset_name + '_' + cascade_model
                                path2 = path1 + '/' + model_name + '_' + wallet_distribution_type + '_wpiwp' * wpiwp
                                result_name = path2 + '/' + product_name + '_ppp' + str(ppp)
                                try:
                                    with open(result_name + '/1profit.txt') as f:
                                        for line in f:
                                            profit[wallet_distribution - 1][ppp - 1].append(line)
                                    f.close()
                                    with open(result_name + '/2cost.txt') as f:
                                        for line in f:
                                            cost[wallet_distribution - 1][ppp - 1].append(line)
                                    f.close()
                                    with open(result_name + '/3time_avg.txt') as f:
                                        for line in f:
                                            time_avg[wallet_distribution - 1][ppp - 1].append(line)
                                    f.close()
                                    with open(result_name + '/4time_total.txt') as f:
                                        for line in f:
                                            time_total[wallet_distribution - 1][ppp - 1].append(line)
                                    f.close()
                                except FileNotFoundError:
                                    profit[wallet_distribution - 1][ppp - 1].append('')
                                    cost[wallet_distribution - 1][ppp - 1].append('')
                                    time_total[wallet_distribution - 1][ppp - 1].append('')
                                    time_avg[wallet_distribution - 1][ppp - 1].append('')
                                    continue

                        for prod_setting2 in [1, 2, 3]:
                            product_name = 'item_lphc' * (prod_setting == 1) + 'item_hplc' * (prod_setting == 2) + '_ce' * (prod_setting2 == 2) + '_ee' * (prod_setting2 == 3)
                            num_product = len(open('item/' + product_name + '.txt').readlines())
                            for wpiwp in wpiwp_seq:
                                path1 = 'result/r_' + dataset_name + '_' + cascade_model
                                path2 = path1 + '/' + model_name + '_' + wallet_distribution_type + '_wpiwp' * wpiwp
                                result_name = path2 + '/' + product_name + '_ppp' + str(ppp)
                                try:
                                    with open(result_name + '/5ratio_profit.txt') as f:
                                        for line in f:
                                            ratio_profit[wallet_distribution - 1][ppp - 1].append(line)
                                    f.close()
                                    with open(result_name + '/6ratio_cost.txt') as f:
                                        for line in f:
                                            ratio_cost[wallet_distribution - 1][ppp - 1].append(line)
                                    f.close()
                                    with open(result_name + '/7number_pn.txt') as f:
                                        for line in f:
                                            number_an[wallet_distribution - 1][ppp - 1].append(line)
                                    f.close()
                                    with open(result_name + '/8number_seed.txt') as f:
                                        for line in f:
                                            number_seed[wallet_distribution - 1][ppp - 1].append(line)
                                    f.close()
                                except FileNotFoundError:
                                    for num in range(num_product):
                                        ratio_profit[wallet_distribution - 1][ppp - 1].append('\n')
                                        ratio_cost[wallet_distribution - 1][ppp - 1].append('\n')
                                        number_seed[wallet_distribution - 1][ppp - 1].append('\n')
                                        number_an[wallet_distribution - 1][ppp - 1].append('\n')
                                    continue

            for wallet_distribution in wallet_distribution_seq:
                for ppp in ppp_seq:
                    write_file = 'w' * (wallet_distribution == 1 and ppp == 1) + 'a' * (wallet_distribution != 1 or ppp != 1)
                    path1 = 'result/r_' + dataset_name + '_' + cascade_model + '/r'
                    if not os.path.isdir(path1):
                        os.mkdir(path1)
                    path = 'result/r_' + dataset_name + '_' + cascade_model + '/r/' + model_name
                    fw = open(path + '_1profit.txt', write_file)
                    for lnum, line in enumerate(profit[wallet_distribution - 1][ppp - 1]):
                        fw.write(str(line) + '\n')
                        if lnum % 6 == 5:
                            fw.write('\n' * 10)
                    fw.close()
                    fw = open(path + '_2cost.txt', write_file)
                    for lnum, line in enumerate(cost[wallet_distribution - 1][ppp - 1]):
                        fw.write(str(line) + '\n')
                        if lnum % 6 == 5:
                            fw.write('\n' * 10)
                    fw.close()
                    fw = open(path + '_3time_avg.txt', write_file)
                    for lnum, line in enumerate(time_avg[wallet_distribution - 1][ppp - 1]):
                        fw.write(str(line) + '\n')
                        if lnum % 6 == 5:
                            fw.write('\n' * 10)
                    fw.close()
                    fw = open(path + '_4time_total.txt', write_file)
                    for lnum, line in enumerate(time_total[wallet_distribution - 1][ppp - 1]):
                        fw.write(str(line) + '\n')
                        if lnum % 6 == 5:
                            fw.write('\n' * 10)
                    fw.close()

                    fw = open(path + '_5ratio_profit.txt', write_file)
                    for lnum, line in enumerate(ratio_profit[wallet_distribution - 1][ppp - 1]):
                        if (lnum % 6 == 0 and lnum != 0 and (wallet_distribution == 1 and ppp == 1)) or (lnum % 6 == 0 and (wallet_distribution != 1 or ppp != 1)):
                            fw.write('\n' * 9)
                        fw.write(str(line))
                    fw.close()
                    fw = open(path + '_6ratio_cost.txt', write_file)
                    for lnum, line in enumerate(ratio_cost[wallet_distribution - 1][ppp - 1]):
                        if (lnum % 6 == 0 and lnum != 0 and (wallet_distribution == 1 and ppp == 1)) or (lnum % 6 == 0 and (wallet_distribution != 1 or ppp != 1)):
                            fw.write('\n' * 9)
                        fw.write(str(line))
                    fw.close()
                    fw = open(path + '_7number_pn.txt', write_file)
                    for lnum, line in enumerate(number_an[wallet_distribution - 1][ppp - 1]):
                        if (lnum % 6 == 0 and lnum != 0 and (wallet_distribution == 1 and ppp == 1)) or (lnum % 6 == 0 and (wallet_distribution != 1 or ppp != 1)):
                            fw.write('\n' * 9)
                        fw.write(str(line))
                    fw.close()
                    fw = open(path + '_8number_seed.txt', write_file)
                    for lnum, line in enumerate(number_seed[wallet_distribution - 1][ppp - 1]):
                        if (lnum % 6 == 0 and lnum != 0 and (wallet_distribution == 1 and ppp == 1)) or (lnum % 6 == 0 and (wallet_distribution != 1 or ppp != 1)):
                            fw.write('\n' * 9)
                        fw.write(str(line))
                    fw.close()