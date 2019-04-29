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
total_budget = 10

for data_setting in dataset_seq:
    dataset_name = 'email_undirected' * (data_setting == 1) + 'dnc_email_directed' * (data_setting == 2) + 'email_Eu_core_directed' * (data_setting == 3) + \
                   'WikiVote_directed' * (data_setting == 4) + 'NetPHY_undirected' * (data_setting == 5)
    for cm in cm_seq:
        cascade_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
        for model_name in model_seq:
            for ppp in ppp_seq:
                for wpiwp in wpiwp_seq:
                    for prod_setting in prod_seq:
                        for prod_setting2 in prod2_seq:
                            product_name = 'item_lphc' * (prod_setting == 1) + 'item_hplc' * (prod_setting == 2) + '_ce' * (prod_setting2 == 2) + '_ee' * (prod_setting2 == 3)
                            num_product = len(open('item/' + product_name + '.txt').readlines())
                            for wallet_distribution in wallet_distribution_seq:
                                wallet_distribution_type = 'm50e25' * (wallet_distribution == 1) + 'm99e96' * (wallet_distribution == 2)

                                profit, cost = [], []
                                time_avg, time_total = [], []

                                ratio_profit, ratio_cost = [[] for _ in range(num_product)], [[] for _ in range(num_product)]
                                number_seed, number_an = [[] for _ in range(num_product)], [[] for _ in range(num_product)]

                                for bud in range(1, total_budget + 1):
                                    path_result_name = 'result/' + model_name + '_' + wallet_distribution_type + '_ppp' + str(ppp) + '_wpiwp' * wpiwp
                                    if 'ap' in model_name:
                                        result_name = path_result_name + '/' + dataset_name + '_' + cascade_model + '_' + product_name + '/b' + str(bud) + '_i1.txt'
                                    else:
                                        result_name = path_result_name + '/' + dataset_name + '_' + cascade_model + '_' + product_name + '/b' + str(bud) + '_i10.txt'
                                    try:
                                        print(result_name)

                                        with open(result_name) as f:
                                            for lnum, line in enumerate(f):
                                                if lnum <= 2 or lnum == 5:
                                                    continue
                                                elif lnum == 3:
                                                    (l) = line.split()
                                                    profit.append(l[2].rstrip(','))
                                                    cost.append(l[-1])
                                                elif lnum == 4:
                                                    (l) = line.split()
                                                    time_total.append(l[2].rstrip(','))
                                                    time_avg.append(l[-1])
                                                elif lnum == 6:
                                                    (l) = line.split()
                                                    for nl in range(2, len(l)):
                                                        ratio_profit[nl-2].append(l[nl])
                                                elif lnum == 7:
                                                    (l) = line.split()
                                                    for nl in range(2, len(l)):
                                                        ratio_cost[nl - 2].append(l[nl])
                                                elif lnum == 8:
                                                    (l) = line.split()
                                                    for nl in range(2, len(l)):
                                                        number_seed[nl-2].append(l[nl])
                                                elif lnum == 9:
                                                    (l) = line.split()
                                                    for nl in range(2, len(l)):
                                                        number_an[nl - 2].append(l[nl])
                                                else:
                                                    break
                                        f.close()
                                    except FileNotFoundError:
                                        profit.append('')
                                        cost.append('')
                                        time_total.append('')
                                        time_avg.append('')
                                        for num in range(num_product):
                                            ratio_profit[num].append('')
                                            ratio_cost[num].append('')
                                            number_seed[num].append('')
                                            number_an[num].append('')
                                        continue

                                    path1 = 'result/r_' + dataset_name + '_' + cascade_model
                                    if not os.path.isdir(path1):
                                        os.mkdir(path1)
                                    path2 = path1 + '/' + model_name + '_' + wallet_distribution_type + '_wpiwp' * wpiwp
                                    if not os.path.isdir(path2):
                                        os.mkdir(path2)
                                    path = path2 + '/' + product_name + '_ppp' + str(ppp)
                                    if not os.path.isdir(path):
                                        os.mkdir(path)

                                    fw = open(path + '/1profit.txt', 'w')
                                    for line in profit:
                                        fw.write(str(line) + '\t')
                                    fw.close()
                                    fw = open(path + '/2cost.txt', 'w')
                                    for line in cost:
                                        fw.write(str(line) + '\t')
                                    fw.close()
                                    fw = open(path + '/3time_avg.txt', 'w')
                                    for line in time_avg:
                                        fw.write(str(line) + '\t')
                                    fw.close()
                                    fw = open(path + '/4time_total.txt', 'w')
                                    for line in time_total:
                                        fw.write(str(line) + '\t')
                                    fw.close()
                                    fw = open(path + '/5ratio_profit.txt', 'w')
                                    for line in ratio_profit:
                                        for l in line:
                                            fw.write(str(l) + '\t')
                                        fw.write('\n')
                                    fw.close()
                                    fw = open(path + '/6ratio_cost.txt', 'w')
                                    for line in ratio_cost:
                                        for l in line:
                                            fw.write(str(l) + '\t')
                                        fw.write('\n')
                                    fw.close()
                                    fw = open(path + '/7number_pn.txt', 'w')
                                    for line in number_an:
                                        for l in line:
                                            fw.write(str(l) + '\t')
                                        fw.write('\n')
                                    fw.close()
                                    fw = open(path + '/8number_seed.txt', 'w')
                                    for line in number_seed:
                                        for l in line:
                                            fw.write(str(l) + '\t')
                                        fw.write('\n')
                                    fw.close()
