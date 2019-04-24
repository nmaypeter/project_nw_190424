from generateDistribution import *
import time


class IniGraph:
    def __init__(self, data_name):
        ### dataset_name, data_data_path, data_weight_path, data_degree_path: (str)
        self.data_name = data_name
        self.data_data_path = 'data/' + data_name + '/data.txt'
        self.data_ic_weight_path = 'data/' + data_name + '/weight_ic.txt'
        self.data_wc_weight_path = 'data/' + data_name + '/weight_wc.txt'
        self.data_degree_path = 'data/' + data_name + '/degree.txt'

    def setEdgeWeight(self):
        #  -- set weight on edge --
        with open(self.data_data_path) as f:
            num_node = 0
            out_degree_list, in_degree_list = [], []
            for line in f:
                (node1, node2) = line.split()
                num_node = max(num_node, int(node1), int(node2))
                out_degree_list.append(node1)
                in_degree_list.append(node2)
        f.close()

        fw = open(self.data_degree_path, 'w')
        for i in range(0, num_node + 1):
            fw.write(str(i) + '\t' + str(out_degree_list.count(str(i))) + '\n')
        fw.close()

        fw_ic = open(self.data_ic_weight_path, 'w')
        fw_wc = open(self.data_wc_weight_path, 'w')
        with open(self.data_data_path) as f:
            for line in f:
                (node1, node2) = line.split()
                fw_ic.write(node1 + '\t' + node2 + '\t0.1\n')
                fw_wc.write(node1 + '\t' + node2 + '\t' + str(round(1 / in_degree_list.count(node2), 2)) + '\n')
        fw_ic.close()
        fw_wc.close()

    def getNodeOutDegree(self, i_node):
        #  -- get the out-degree --
        deg = 0
        with open(self.data_degree_path) as f:
            for line in f:
                (node, degree) = line.split()
                if node == i_node:
                    deg = int(degree)
                    break
        f.close()

        return deg

    def constructSeedCostDict(self):
        # -- calculate the cost for each seed --
        ### s_cost_dict: (dict) the set of cost for each seed
        ### s_cost_dict[ii]: (float2) the degree of ii's seed
        ### num_node: (int) the number of nodes in data
        ### max_deg: (int) the maximum degree in data
        s_cost_dict = {}
        with open(self.data_degree_path) as f:
            num_node, max_deg = 0, 0
            seed_cost_list = []
            for line in f:
                (node, degree) = line.split()
                num_node = max(num_node, int(node))
                max_deg = max(max_deg, int(degree))
                seed_cost_list.append([node, degree])

            for i in range(num_node + 1):
                s_cost_dict[str(i)] = round(int(seed_cost_list[i][1]) / max_deg, 2)
        f.close()

        return s_cost_dict

    def constructGraphDict(self, cas):
        # -- build graph --
        ### graph: (dict) the graph
        ### graph[node1]: (dict) the set of node1's receivers
        ### graph[node1][node2]: (str) the weight one the edge of node1 to node2
        path = self.data_ic_weight_path * (cas == 'ic') + self.data_wc_weight_path * (cas == 'wc')
        graph = {}
        with open(path) as f:
            for line in f:
                (node1, node2, wei) = line.split()
                if node1 in graph:
                    graph[node1][node2] = str(wei)
                else:
                    graph[node1] = {node2: str(wei)}
        f.close()
        return graph

    def getTotalNumNode(self):
        #  -- get the num_node --
        ### num_node: (int) the number of nodes in data
        num_node = 0
        with open(self.data_data_path) as f:
            for line in f:
                (node1, node2) = line.split()
                num_node = max(int(node1), int(node2), num_node)
        f.close()
        print('num_node = ' + str(round(num_node + 1, 2)))

        return num_node

    def getTotalNumEdge(self):
        # -- get the num_edge --
        num_edge = 0
        with open(self.data_data_path) as f:
            for _ in f:
                num_edge += 1
        f.close()
        print('num_edge = ' + str(round(num_edge, 2)))

        return num_edge

    def getMaxDegree(self):
        # -- get the max_deg --
        ### max_deg: (int) the maximum degree in data
        with open(self.data_degree_path) as f:
            max_deg = 0
            for line in f:
                (node, degree) = line.split()
                max_deg = max(max_deg, int(degree))
        f.close()
        print('max_deg = ' + str(round(max_deg, 2)))

        return max_deg


class IniProduct:
    def __init__(self, prod_name):
        ### prod_name: (str)
        self.prod_name = prod_name

    def getProductList(self):
        # -- get product list from file
        ### prod_list: (list) [profit, cost, price]
        prod_list = []
        with open('item/' + self.prod_name + '.txt') as f:
            for line in f:
                (p, c, r, pr) = line.split()
                prod_list.append([float(p), float(c), round(float(p) + float(c), 2)])

        return prod_list

    def getTotalPrice(self):
        # -- get total_price from file
        ### total_price: (float2) the sum of prices
        total_price = 0.0
        with open('item/' + self.prod_name + '.txt') as f:
            for line in f:
                (p, c, r, pr) = line.split()
                total_price += float(pr)
        print('total_price = ' + str(round(total_price, 2)))

        return round(total_price, 2)


class IniWallet:
    def __init__(self, data_name, prod_name, wallet_dist_name):
        ### dataset_name: (str)
        self.data_name = data_name
        self.prod_name = prod_name
        self.wallet_dist_name = wallet_dist_name

    def setNodeWallet(self, num_node):
        # -- set node's personal budget (wallet) --
        price_list = []
        with open('item/' + self.prod_name + '.txt') as f:
            for line in f:
                (p, c, r, pr) = line.split()
                price_list.append(float(pr))
        f.close()

        mu, sigma = 0, 1
        if self.wallet_dist_name == 'm50e25':
            mu = np.mean(price_list)
            sigma = (max(price_list) - mu) / 0.6745
        elif self.wallet_dist_name == 'm99e96':
            mu = sum(price_list)
            sigma = abs(min(price_list) - mu) / 3

        fw = open('data/' + self.data_name + '/' + self.prod_name.replace('item', 'wallet') + '_' + str(self.wallet_dist_name) + '.txt', 'w')
        for i in range(0, num_node + 1):
            wal = 0
            while wal <= 0:
                q = stats.norm.rvs(mu, sigma)
                pd = stats.norm.pdf(q, mu, sigma)
                wal = get_quantiles(pd, mu, sigma)
            fw.write(str(i) + '\t' + str(round(wal, 2)) + '\n')
        fw.close()

    def getWalletList(self):
        # -- get wallet_list from file --
        w_list = []
        with open('data/' + self.data_name + '/' + self.prod_name.replace('item', 'wallet') + '_' + str(self.wallet_dist_name) + '.txt') as f:
            for line in f:
                (node, wal) = line.split()
                w_list.append(float(wal))
        f.close()

        return w_list

    def getTotalWallet(self):
        # -- get total_wallet from file --
        total_w = 0.0
        with open('data/' + self.data_name + '/' + self.prod_name.replace('item', 'wallet') + '_' + str(self.wallet_dist_name) + '.txt') as f:
            for line in f:
                (node, wallet) = line.split()
                total_w += float(wallet)
        f.close()
        print('total wallet = ' + self.prod_name + '_dis' + str(self.wallet_dist_name) + ' = ' + str(round(total_w, 2)))

        return total_w


if __name__ == '__main__':
    start_time = time.time()
    dataset_seq = [1, 2]
    prod_seq, prod2_seq = [1, 2], [1]
    cm_seq = [1]
    wallet_distribution_seq = [1, 2]
    for data_setting in dataset_seq:
        dataset_name = 'email_undirected' * (data_setting == 1) + 'dnc_email_directed' * (data_setting == 2) + 'email_Eu_core_directed' * (data_setting == 3) + \
                       'WikiVote_directed' * (data_setting == 4) + 'NetPHY_undirected' * (data_setting == 5)
        for prod_setting in prod_seq:
            for prod_setting2 in prod2_seq:
                product_name = 'item_lphc' * (prod_setting == 1) + 'item_hplc' * (prod_setting == 2) + '_ce' * (prod_setting2 == 2) + '_ee' * (prod_setting2 == 3)
                for cm in cm_seq:
                    cascade_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
                    for wallet_distribution in wallet_distribution_seq:
                        wallet_distribution_name = 'm50e25' * (wallet_distribution == 1) + 'm99e96' * (wallet_distribution == 2)

                        iniG = IniGraph(dataset_name)
                        iniP = IniProduct(product_name)
                        iniW = IniWallet(dataset_name, product_name, wallet_distribution_name)

                        # iniG.setEdgeWeight()
                        number_node = iniG.getTotalNumNode()
                        # number_edge = iniG.getTotalNumEdge()
                        # max_degree = iniG.getMaxDegree()

                        # sum_price = iniP.getTotalPrice()
                        iniW.setNodeWallet(number_node)

                        # seed_cost_dict = iniG.constructSeedCostDict()
                        # graph_dict = iniG.constructGraphDict(cascade_model)
                        # product_list = iniP.getProductList()
                        # wallet_list = iniW.getWalletList()
                        total_wallet = iniW.getTotalWallet()

    print('total time: ' + str(round(time.time() - start_time, 4)) + 'sec')

    ### -- sum_price --
    ### -- item_lphc, item_hplc = 1.44 --
    ### -- item_lphc_ce, item_hplc_ce = 1.32 --
    ### -- item_lphc_ee, item_hplc_ee = 1.68 --

    ### -- num_node --
    ### -- email_undirected = 1134 --
    ### -- dnc_email_directed = 2030 --
    ### -- email_Eu_core_directed = 1005 --
    ### -- WikiVote_directed = 8298 --
    ### -- NetPHY_undirected = 37154 --

    ### -- num_edge --
    ### -- email_undirected = 10902 --
    ### -- dnc_email_directed = 5598 --
    ### -- email_Eu_core_directed = 25571 --
    ### -- WikiVote_directed = 201524 --
    ### -- NetPHY_undirected = 348322 --

    ### -- max_degree --
    ### -- email_undirected = 71 --
    ### -- dnc_email_directed = 331 --
    ### -- email_Eu_core_directed = 334 --
    ### -- WikiVote_directed = 1065 --
    ### -- NetPHY_undirected = 178 --

    ### -- total wallet --
    ### -- email_undirected --
    ### -- item_lphc_dism50e25 = 612.39 --
    ### -- item_lphc_dism99e96 = 1612.21 --
    ### -- item_hplc_dism50e25 = 610.77 --
    ### -- item_hplc_dism99e96 = 1618.81 --
    ### -- dnc_email_directed --
    ### -- item_lphc_dism50e25 = 1101.13 --
    ### -- item_lphc_dism99e96 = 2909.05 --
    ### -- item_hplc_dism50e25 = 1088.12 --
    ### -- item_hplc_dism99e96 = 2944.66 --
