from Diffusion import *


def selectDegreeSeed(d_dict):
    # -- get the node with highest degree --
    mep = (-1, '-1')
    max_degree = -1
    while mep[1] == '-1':
        while max_degree == -1:
            for deg in list(d_dict.keys()):
                if int(deg) > max_degree:
                    max_degree = int(deg)

            if max_degree == -1:
                return mep, d_dict

            if d_dict[str(max_degree)] == set():
                del d_dict[str(max_degree)]
                max_degree = -1

        if d_dict[str(max_degree)] == set():
            del d_dict[str(max_degree)]
            max_degree = -1
            continue

        mep = choice(list(d_dict[str(max_degree)]))
        d_dict[str(max_degree)].remove(mep)

    return mep


class SeedSelectionHD:
    def __init__(self, g_dict, s_c_dict, prod_list):
        ### g_dict: (dict) the graph
        ### s_c_dict: (dict) the set of cost for seeds
        ### prod_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### num_node: (int) the number of nodes
        ### num_product: (int) the kinds of products
        self.graph_dict = g_dict
        self.seed_cost_dict = s_c_dict
        self.product_list = prod_list
        self.num_node = len(s_c_dict)
        self.num_product = len(prod_list)

    def constructDegreeDict(self, data_name):
        # -- display the degree and the nodes with the degree --
        ### d_dict: (dict) the degree and the nodes with the degree
        ### d_dict[deg]: (set) the set for deg-degree nodes
        d_dict = {}
        with open(IniGraph(data_name).data_degree_path) as f:
            for line in f:
                (i, deg) = line.split()
                if deg == '0':
                    continue
                for k in range(self.num_product):
                    if deg in d_dict:
                        d_dict[deg].add((k, i))
                    else:
                        d_dict[deg] = {(k, i)}
        f.close()

        return d_dict

    def constructExpendDegreeDict(self):
        # -- display the degree and the nodes with the degree --
        ### d_dict: (dict) the degree and the nodes with the degree
        ### d_dict[deg]: (set) the set for deg-degree nodes
        d_dict = {}
        for i in self.graph_dict:
            i_set = {i}
            for ii in self.graph_dict[i]:
                if ii not in i_set:
                    i_set.add(ii)
            for ii in self.graph_dict[i]:
                if ii in self.graph_dict:
                    for iii in self.graph_dict[ii]:
                        if iii not in i_set:
                            i_set.add(iii)

            deg = str(len(i_set))
            for k in range(self.num_product):
                if deg in d_dict:
                    d_dict[deg].add((k, i))
                else:
                    d_dict[deg] = {(k, i)}

        return d_dict


class SeedSelectionHDPW:
    def __init__(self, g_dict, s_c_dict, prod_list, pw_list):
        ### g_dict: (dict) the graph
        ### s_c_dict: (dict) the set of cost for seeds
        ### prod_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### num_node: (int) the number of nodes
        ### num_product: (int) the kinds of products
        ### pw_list: (list) the product weight list
        self.graph_dict = g_dict
        self.seed_cost_dict = s_c_dict
        self.product_list = prod_list
        self.num_node = len(s_c_dict)
        self.num_product = len(prod_list)
        self.pw_list = pw_list

    def constructDegreeDict(self, data_name):
        # -- display the degree and the nodes with the degree --
        ### d_dict: (dict) the degree and the nodes with the degree
        ### d_dict[deg]: (set) the set for deg-degree nodes
        d_dict = {}
        with open(IniGraph(data_name).data_degree_path) as f:
            for line in f:
                (i, deg) = line.split()
                if deg == '0':
                    continue
                for k in range(self.num_product):
                    deg = str(round(float(deg) * self.pw_list[k]))
                    if deg in d_dict:
                        d_dict[deg].add((k, i))
                    else:
                        d_dict[deg] = {(k, i)}
        f.close()

        return d_dict

    def constructExpendDegreeDict(self):
        # -- display the degree and the nodes with the degree --
        ### d_dict: (dict) the degree and the nodes with the degree
        ### d_dict[deg]: (set) the set for deg-degree nodes
        d_dict = {}
        for i in self.graph_dict:
            i_set = {i}
            for ii in self.graph_dict[i]:
                if ii not in i_set:
                    i_set.add(ii)
            for ii in self.graph_dict[i]:
                if ii in self.graph_dict:
                    for iii in self.graph_dict[ii]:
                        if iii not in i_set:
                            i_set.add(iii)

            for k in range(self.num_product):
                deg = str(round(len(i_set) * self.pw_list[k]))
                if deg in d_dict:
                    d_dict[deg].add((k, i))
                else:
                    d_dict[deg] = {(k, i)}

        return d_dict


if __name__ == '__main__':
    dataset_name = 'email_undirected'
    cascade_model = 'ic'
    product_name = 'item_lphc'
    wallet_distribution_type = 'm50e25'
    total_budget = 10
    whether_passing_information_with_purchasing = bool(1)
    personal_purchasing_prob = 'random'
    eva_monte_carlo = 100

    iniG = IniGraph(dataset_name)
    iniP = IniProduct(product_name)

    seed_cost_dict = iniG.constructSeedCostDict()
    graph_dict = iniG.constructGraphDict(cascade_model)
    product_list = iniP.getProductList()
    num_node = len(seed_cost_dict)
    num_product = len(product_list)

    # -- initialization for each budget --
    start_time = time.time()
    sshd = SeedSelectionHD(graph_dict, seed_cost_dict, product_list)

    # -- initialization for each sample_number --
    now_budget = 0.0
    seed_set = [set() for _ in range(num_product)]

    degree_dict = sshd.constructDegreeDict(dataset_name)
    mep_g = selectDegreeSeed(degree_dict)
    mep_k_prod, mep_i_node = mep_g[0], mep_g[1]

    # -- main --
    while now_budget < total_budget and mep_i_node != '-1':
        if now_budget + seed_cost_dict[mep_i_node] > total_budget:
            mep_g = selectDegreeSeed(degree_dict)
            mep_k_prod, mep_i_node = mep_g[0], mep_g[1]
            if mep_i_node == '-1':
                break
            continue

        seed_set[mep_k_prod].add(mep_i_node)
        now_budget += seed_cost_dict[mep_i_node]

        mep_g = selectDegreeSeed(degree_dict)
        mep_k_prod, mep_i_node = mep_g[0], mep_g[1]

    print('seed selection time: ' + str(round(time.time() - start_time, 2)) + 'sec')
    eva = Evaluation(graph_dict, seed_cost_dict, product_list, personal_purchasing_prob, whether_passing_information_with_purchasing)
    iniW = IniWallet(dataset_name, product_name, wallet_distribution_type)
    wallet_list = iniW.getWalletList()
    personal_prob_list = eva.setPersonalPurchasingProbList(wallet_list)

    sample_pro_acc, sample_bud_acc = 0.0, 0.0
    sample_sn_k_acc, sample_pnn_k_acc = [0.0 for _ in range(num_product)], [0 for _ in range(num_product)]
    sample_pro_k_acc, sample_bud_k_acc = [0.0 for _ in range(num_product)], [0.0 for _ in range(num_product)]

    for _ in range(eva_monte_carlo):
        pro, pro_k_list, pnn_k_list = eva.getSeedSetProfit(seed_set, copy.deepcopy(wallet_list), copy.deepcopy(personal_prob_list))
        sample_pro_acc += pro
        for kk in range(num_product):
            sample_pro_k_acc[kk] += pro_k_list[kk]
            sample_pnn_k_acc[kk] += pnn_k_list[kk]
    sample_pro_acc = round(sample_pro_acc / eva_monte_carlo, 4)
    for kk in range(num_product):
        sample_pro_k_acc[kk] = round(sample_pro_k_acc[kk] / eva_monte_carlo, 4)
        sample_pnn_k_acc[kk] = round(sample_pnn_k_acc[kk] / eva_monte_carlo, 2)
        sample_sn_k_acc[kk] = len(seed_set[kk])
        for sample_seed in seed_set[kk]:
            sample_bud_acc += seed_cost_dict[sample_seed]
            sample_bud_k_acc[kk] += seed_cost_dict[sample_seed]
            sample_bud_acc = round(sample_bud_acc, 2)
            sample_bud_k_acc[kk] = round(sample_bud_k_acc[kk], 2)

    print('seed set: ' + str(seed_set))
    print('profit: ' + str(sample_pro_acc))
    print('budget: ' + str(sample_bud_acc))
    print('seed number: ' + str(sample_sn_k_acc))
    print('purchasing node number: ' + str(sample_pnn_k_acc))
    print('ratio profit: ' + str(sample_pro_k_acc))
    print('ratio budget: ' + str(sample_bud_k_acc))
    print('total time: ' + str(round(time.time() - start_time, 2)) + 'sec')
