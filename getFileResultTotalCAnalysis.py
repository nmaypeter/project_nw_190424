dataset_seq = [2]
cm_seq = [1, 2]
model_seq = ['mng', 'mngr', 'mngpw', 'mngrpw', 'mngsr', 'mngsrpw',
             'mngap', 'mngapr', 'mngappw', 'mngaprpw', 'mngapsr', 'mngapsrpw',
             'mhd', 'mhed', 'mhdpw', 'mhedpw',
             'mpmis', 'mr']
ppp_seq = [1, 2, 3]
wpiwp_seq = [bool(1), bool(0)]
prod_seq, prod2_seq = [1, 2], [1, 2, 3]
wallet_distribution_seq = [1, 2]
total_budget = 10

for bud in range(1, total_budget + 1):
    for data_setting in dataset_seq:
        dataset_name = 'email_undirected' * (data_setting == 1) + 'dnc_email_directed' * (data_setting == 2) + 'email_Eu_core_directed' * (data_setting == 3) + \
                        'WikiVote_directed' * (data_setting == 4) + 'NetPHY_undirected' * (data_setting == 5)
        for cm in cm_seq:
            cascade_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
            profit = []
            for wpiwp in wpiwp_seq:
                for wallet_distribution in wallet_distribution_seq:
                    wallet_distribution_type = 'm50e25' * (wallet_distribution == 1) + 'm99e96' * (wallet_distribution == 2)
                    for ppp in ppp_seq:
                        for prod_setting in prod_seq:
                            for prod_setting2 in prod2_seq:
                                product_name = 'item_lphc' * (prod_setting == 1) + 'item_hplc' * (prod_setting == 2) + '_ce' * (prod_setting2 == 2) + '_ee' * (prod_setting2 == 3)
                                model_str = ''
                                for model_name in model_seq:
                                    try:
                                        path_result_name = 'result/' + model_name + '_' + wallet_distribution_type + '_ppp' + str(ppp) + '_wpiwp' * wpiwp
                                        if 'ap' in model_name:
                                            result_name = path_result_name + '/' + dataset_name + '_' + cascade_model + '_' + product_name + '/b' + str(bud) + '_i1.txt'
                                        else:
                                            result_name = path_result_name + '/' + dataset_name + '_' + cascade_model + '_' + product_name + '/b' + str(bud) + '_i10.txt'
                                        print(result_name)

                                        with open(result_name) as f:
                                            for lnum, line in enumerate(f):
                                                if lnum == 3:
                                                    (l) = line.split()
                                                    model_str += (l[2].rstrip(',')) + '\t'
                                                    break
                                        f.close()
                                    except FileNotFoundError:
                                        model_str += '\t'
                                profit.append(model_str)

            fw = open('result/r_' + dataset_name + '_' + cascade_model + '/comparison_total_profit_analysis_b' + str(bud) + '.txt', 'w')
            for lnum, line in enumerate(profit):
                if lnum % (len(wallet_distribution_seq) * len(ppp_seq) * len(prod_seq) * len(prod2_seq)) == 0 and lnum != 0:
                    fw.write('\n')
                fw.write(str(line) + '\n')
            fw.close()