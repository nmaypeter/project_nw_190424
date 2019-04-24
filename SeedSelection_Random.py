from Diffusion import *


def selectRandomSeed(rn_set):
    # -- select a seed for a random product randomly --
    mep = (-1, '-1')
    if len(rn_set) != 0:
        mep = choice(list(rn_set))
        rn_set.remove(mep)

    return mep


class SeedSelectionRandom:
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

    def constructRandomNodeSet(self):
        rn_set = set()
        for k in range(self.num_product):
            for i in self.graph_dict:
                rn_set.add((k, i))

        return rn_set


if __name__ == '__main__':
    dataset_name = 'email_undirected'
    product_name = 'item_lphc'
    cascade_model = 'ic'
    wallet_distribution_name = 'm50e25'
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
    ssr = SeedSelectionRandom(graph_dict, seed_cost_dict, product_list)

    # -- initialization for each sample_number --
    now_budget = 0.0
    seed_set = [set() for _ in range(num_product)]

    random_node_set = ssr.constructRandomNodeSet()
    mep_g = selectRandomSeed(random_node_set)
    mep_k_prod, mep_i_node = mep_g[0], mep_g[1]

    # -- main --
    while now_budget < total_budget and mep_i_node != '-1':
        if now_budget + seed_cost_dict[mep_i_node] > total_budget:
            mep_g = selectRandomSeed(random_node_set)
            mep_k_prod, mep_i_node = mep_g[0], mep_g[1]
            if mep_i_node == '-1':
                break
            continue

        seed_set[mep_k_prod].add(mep_i_node)
        now_budget += seed_cost_dict[mep_i_node]

        mep_g = selectRandomSeed(random_node_set)
        mep_k_prod, mep_i_node = mep_g[0], mep_g[1]

    print('seed selection time: ' + str(round(time.time() - start_time, 2)) + 'sec')
    eva = Evaluation(graph_dict, seed_cost_dict, product_list, personal_purchasing_prob, whether_passing_information_with_purchasing)
    iniW = IniWallet(dataset_name, product_name, wallet_distribution_name)
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