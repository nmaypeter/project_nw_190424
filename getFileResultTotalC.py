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

for data_setting in dataset_seq:
    dataset_name = 'email_undirected' * (data_setting == 1) + 'dnc_email_directed' * (data_setting == 2) + 'email_Eu_core_directed' * (data_setting == 3) + \
                    'WikiVote_directed' * (data_setting == 4) + 'NetPHY_undirected' * (data_setting == 5)
    for cm in cm_seq:
        cascade_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
        for ppp in ppp_seq:
            for wallet_distribution in wallet_distribution_seq:
                wallet_distribution_type = 'm50e25' * (wallet_distribution == 1) + 'm99e96' * (wallet_distribution == 2)
                profit = []
                for prod_setting in prod_seq:
                    for prod_setting2 in prod2_seq:
                        product_name = 'item_lphc' * (prod_setting == 1) + 'item_hplc' * (prod_setting == 2) + '_ce' * (prod_setting2 == 2) + '_ee' * (prod_setting2 == 3)
                        for wpiwp in wpiwp_seq:
                            for model_name in model_seq:
                                path1 = 'result/r_' + dataset_name + '_' + cascade_model
                                path2 = path1 + '/' + model_name + '_' + wallet_distribution_type + '_wpiwp' * wpiwp
                                path = path2 + '/' + product_name + '_ppp' + str(ppp)
                                result_name = path + '/1profit.txt'
                                try:
                                    print(result_name)

                                    with open(result_name) as f:
                                        for lnum, line in enumerate(f):
                                            if lnum == 0:
                                                profit.append(line)
                                            else:
                                                break
                                    f.close()
                                except FileNotFoundError:
                                    profit.append('')
                                    continue

                fw = open('result/r_' + dataset_name + '_' + cascade_model + '/comparison_total_profit_' + wallet_distribution_type + '_ppp' + str(ppp) + '.txt', 'w')
                for lnum, line in enumerate(profit):
                    if lnum % (len(model_seq) * 2) == 0 and lnum != 0:
                        fw.write('\n' * 3)
                    fw.write(str(line) + '\n')
                fw.close()
