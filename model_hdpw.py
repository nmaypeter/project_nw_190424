from SeedSelection_HighDegree import *
import os

if __name__ == '__main__':
    dataset_seq = [2]
    prod_seq, prod2_seq = [1, 2], [1, 2, 3]
    cm_seq = [1, 2]
    wallet_distribution_seq = [1, 2]
    total_budget = 10
    wpiwp_seq = [bool(1), bool(0)]
    sample_number = 10
    ppp_seq = [1, 2, 3]
    monte_carlo, eva_monte_carlo = 10, 100
    for data_setting in dataset_seq:
        dataset_name = 'email_undirected' * (data_setting == 1) + 'dnc_email_directed' * (data_setting == 2) + 'email_Eu_core_directed' * (data_setting == 3) + \
                       'WikiVote_directed' * (data_setting == 4) + 'NetPHY_undirected' * (data_setting == 5)
        for cm in cm_seq:
            cascade_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
            for prod_setting in prod_seq:
                for prod_setting2 in prod2_seq:
                    product_name = 'item_lphc' * (prod_setting == 1) + 'item_hplc' * (prod_setting == 2) + '_ce' * (prod_setting2 == 2) + '_ee' * (prod_setting2 == 3)
                    for wallet_distribution in wallet_distribution_seq:
                        wallet_distribution_type = 'm50e25' * (wallet_distribution == 1) + 'm99e96' * (wallet_distribution == 2)
                        for wpiwp in wpiwp_seq:

                            iniG = IniGraph(dataset_name)
                            iniP = IniProduct(product_name)

                            seed_cost_dict = iniG.constructSeedCostDict()
                            graph_dict = iniG.constructGraphDict(cascade_model)
                            product_list = iniP.getProductList()
                            num_node = len(seed_cost_dict)
                            num_product = len(product_list)
                            product_weight_list = getProductWeight(product_list, wallet_distribution_type)

                            seed_set_sequence, ss_time_sequence = [[] for _ in range(total_budget)], [[] for _ in range(total_budget)]
                            sshdpw_main = SeedSelectionHDPW(graph_dict, seed_cost_dict, product_list, product_weight_list)
                            for sample_count in range(sample_number):
                                ss_strat_time = time.time()
                                begin_budget = 1
                                now_budget = 0.0
                                seed_set = [set() for _ in range(num_product)]
                                degree_dict = sshdpw_main.constructDegreeDict(dataset_name)
                                ss_acc_time = round(time.time() - ss_strat_time, 2)
                                temp_sequence = [[begin_budget, now_budget, seed_set, degree_dict, ss_acc_time]]
                                while len(temp_sequence) != 0:
                                    ss_strat_time = time.time()
                                    [begin_budget, now_budget, seed_set, degree_dict, ss_acc_time] = temp_sequence.pop(0)
                                    print('@ mhdpw seed selection @ dataset_name = ' + dataset_name + '_' + cascade_model + ', dist = ' + str(wallet_distribution_type) + ', wpiwp = ' + str(wpiwp) +
                                          ', product_name = ' + product_name + ', budget = ' + str(begin_budget) + ', sample_count = ' + str(sample_count))
                                    mep_g = selectDegreeSeed(degree_dict)
                                    mep_k_prod, mep_i_node = mep_g[0], mep_g[1]

                                    while now_budget < begin_budget and mep_i_node != '-1':
                                        sc = seed_cost_dict[mep_i_node]
                                        if now_budget + sc >= begin_budget and begin_budget < total_budget and len(temp_sequence) == 0:
                                            ss_time = round(time.time() - ss_strat_time + ss_acc_time, 2)
                                            temp_deg = iniG.getNodeOutDegree(mep_i_node)
                                            temp_degree_dict = copy.deepcopy(degree_dict)
                                            if temp_deg in temp_degree_dict:
                                                temp_degree_dict[str(temp_deg)].add((mep_k_prod, mep_i_node))
                                            else:
                                                temp_degree_dict[str(temp_deg)] = {(mep_k_prod, mep_i_node)}
                                            temp_sequence.append([begin_budget + 1, now_budget, copy.deepcopy(seed_set), temp_degree_dict, ss_time])

                                        if now_budget + sc > begin_budget:
                                            mep_g = selectDegreeSeed(degree_dict)
                                            mep_k_prod, mep_i_node = mep_g[0], mep_g[1]
                                            if mep_i_node == '-1':
                                                break
                                            continue

                                        seed_set[mep_k_prod].add(mep_i_node)
                                        now_budget += sc

                                        mep_g = selectDegreeSeed(degree_dict)
                                        mep_k_prod, mep_i_node = mep_g[0], mep_g[1]

                                    ss_time = round(time.time() - ss_strat_time + ss_acc_time, 2)
                                    print('ss_time = ' + str(ss_time) + 'sec')
                                    seed_set_sequence[begin_budget - 1].append(seed_set)
                                    ss_time_sequence[begin_budget - 1].append(ss_time)

                                for bud in range(total_budget):
                                    if len(seed_set_sequence[bud]) != sample_count + 1:
                                        seed_set_sequence[bud].append(0)
                                        ss_time_sequence[bud].append(ss_time_sequence[bud - 1][-1])

                            eva_start_time = time.time()
                            result = [[[] for _ in range(len(ppp_seq))] for _ in range(total_budget)]
                            for bud in range(1, total_budget + 1):
                                for ppp in ppp_seq:
                                    ppp_strategy = 'random' * (ppp == 1) + 'expensive' * (ppp == 2) + 'cheap' * (ppp == 3)
                                    pps_start_time = time.time()
                                    avg_pro, avg_bud = 0.0, 0.0
                                    avg_sn_k, avg_pnn_k = [0 for _ in range(num_product)], [0 for _ in range(num_product)]
                                    avg_pro_k, avg_bud_k = [0.0 for _ in range(num_product)], [0.0 for _ in range(num_product)]

                                    eva_main = Evaluation(graph_dict, seed_cost_dict, product_list, ppp, wpiwp)
                                    iniW = IniWallet(dataset_name, product_name, wallet_distribution_type)
                                    wallet_list = iniW.getWalletList()
                                    personal_prob_list = eva_main.setPersonalPurchasingProbList(wallet_list)
                                    for sample_count, sample_seed_set in enumerate(seed_set_sequence[bud - 1]):
                                        if sample_seed_set != 0:
                                            print('@ mhdpw evaluation @ dataset_name = ' + dataset_name + '_' + cascade_model + ', dist = ' + wallet_distribution_type + ', wpiwp = ' + str(wpiwp) +
                                                  ', product_name = ' + product_name + ', budget = ' + str(bud) + ', ppp = ' + ppp_strategy + ', sample_count = ' + str(sample_count))
                                            sample_pro_acc, sample_bud_acc = 0.0, 0.0
                                            sample_sn_k_acc, sample_pnn_k_acc = [0.0 for _ in range(num_product)], [0 for _ in range(num_product)]
                                            sample_pro_k_acc, sample_bud_k_acc = [0.0 for _ in range(num_product)], [0.0 for _ in range(num_product)]

                                            for _ in range(eva_monte_carlo):
                                                pro, pro_k_list, pnn_k_list = eva_main.getSeedSetProfit(sample_seed_set, copy.deepcopy(wallet_list), copy.deepcopy(personal_prob_list))
                                                sample_pro_acc += pro
                                                for kk in range(num_product):
                                                    sample_pro_k_acc[kk] += pro_k_list[kk]
                                                    sample_pnn_k_acc[kk] += pnn_k_list[kk]
                                            sample_pro_acc = round(sample_pro_acc / eva_monte_carlo, 4)
                                            for kk in range(num_product):
                                                sample_pro_k_acc[kk] = round(sample_pro_k_acc[kk] / eva_monte_carlo, 4)
                                                sample_pnn_k_acc[kk] = round(sample_pnn_k_acc[kk] / eva_monte_carlo, 2)
                                                sample_sn_k_acc[kk] = len(sample_seed_set[kk])
                                                for sample_seed in sample_seed_set[kk]:
                                                    sample_bud_acc += seed_cost_dict[sample_seed]
                                                    sample_bud_k_acc[kk] += seed_cost_dict[sample_seed]
                                                    sample_bud_acc = round(sample_bud_acc, 2)
                                                    sample_bud_k_acc[kk] = round(sample_bud_k_acc[kk], 2)

                                            result[bud - 1][ppp - 1].append([sample_pro_acc, sample_bud_acc, sample_sn_k_acc, sample_pnn_k_acc, sample_seed_set])
                                            avg_pro += sample_pro_acc
                                            avg_bud += sample_bud_acc
                                            for kk in range(num_product):
                                                avg_sn_k[kk] += sample_sn_k_acc[kk]
                                                avg_pnn_k[kk] += sample_pnn_k_acc[kk]
                                                avg_pro_k[kk] += sample_pro_k_acc[kk]
                                                avg_bud_k[kk] += sample_bud_k_acc[kk]

                                            print('eva_time = ' + str(round(time.time() - eva_start_time, 2)) + 'sec')
                                            print(result[bud - 1][ppp - 1][sample_count])
                                            print('avg_profit = ' + str(round(avg_pro / (sample_count + 1), 4)) + ', avg_budget = ' + str(round(avg_bud / (sample_count + 1), 4)))
                                            print('------------------------------------------')
                                        else:
                                            result[bud - 1][ppp - 1].append(result[bud - 2][ppp - 1][sample_count])

                                    avg_pro = round(avg_pro / sample_number, 4)
                                    avg_bud = round(avg_bud / sample_number, 2)
                                    for kk in range(num_product):
                                        avg_sn_k[kk] = round(avg_sn_k[kk] / sample_number, 2)
                                        avg_pnn_k[kk] = round(avg_pnn_k[kk] / sample_number, 2)
                                        avg_pro_k[kk] = round(avg_pro_k[kk] / sample_number, 4)
                                        avg_bud_k[kk] = round(avg_bud_k[kk] / sample_number, 2)

                                    total_time = round(sum(ss_time_sequence[bud - 1]), 2)
                                    path1 = 'result/mhdpw_' + wallet_distribution_type + '_ppp' + str(ppp) + '_wpiwp' * wpiwp
                                    if not os.path.isdir(path1):
                                        os.mkdir(path1)
                                    path = 'result/mhdpw_' + wallet_distribution_type + '_ppp' + str(ppp) + '_wpiwp' * wpiwp + '/' + dataset_name + '_' + cascade_model + '_' + product_name
                                    if not os.path.isdir(path):
                                        os.mkdir(path)
                                    fw = open(path + '/b' + str(bud) + '_i' + str(sample_number) + '.txt', 'w')
                                    fw.write('hdpw, ppp = ' + str(ppp) + ', total_budget = ' + str(bud) + ', dist = ' + wallet_distribution_type + ', wpiwp = ' + str(wpiwp) + '\n' +
                                             'dataset_name = ' + dataset_name + '_' + cascade_model + ', product_name = ' + product_name + '\n' +
                                             'total_budget = ' + str(bud) + ', sample_count = ' + str(sample_number) + '\n' +
                                             'avg_profit = ' + str(avg_pro) + ', avg_budget = ' + str(avg_bud) + '\n' +
                                             'total_time = ' + str(total_time) + ', avg_time = ' + str(round(total_time / sample_number, 4)) + '\n')
                                    fw.write('\nprofit_ratio =')
                                    for kk in range(num_product):
                                        fw.write(' ' + str(avg_pro_k[kk]))
                                    fw.write('\nbudget_ratio =')
                                    for kk in range(num_product):
                                        fw.write(' ' + str(avg_bud_k[kk]))
                                    fw.write('\nseed_number =')
                                    for kk in range(num_product):
                                        fw.write(' ' + str(avg_sn_k[kk]))
                                    fw.write('\ncustomer_number =')
                                    for kk in range(num_product):
                                        fw.write(' ' + str(avg_pnn_k[kk]))
                                    fw.write('\n')

                                    for t, r in enumerate(result[bud - 1][ppp - 1]):
                                        fw.write('\n' + str(t) + '\t' + str(round(r[0], 4)) + '\t' + str(round(r[1], 4)) + '\t' + str(r[2]) + '\t' + str(r[3]) + '\t' + str(r[4]))
                                    fw.close()