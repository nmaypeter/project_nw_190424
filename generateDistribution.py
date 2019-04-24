from random import choice
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def get_quantiles(pd, mu, sigma):

    discrim = -2 * sigma**2 * np.log(pd * sigma * np.sqrt(2 * np.pi))

    # no real roots
    if discrim < 0:
        return None

    # one root, where x == mu
    elif discrim == 0:
        return mu

    # two roots
    else:
        return choice([mu - np.sqrt(discrim), mu + np.sqrt(discrim)])


if __name__ == '__main__':
    price_list_g = [[0.24, 0.48, 0.72], [0.24, 0.48, 0.6], [0.24, 0.48, 0.96]]

    # mu_g = np.mean(price_list_g[0])
    mu_g = sum(price_list_g[0])
    # sigma_g = (max(price_list_g[0]) - mu_g) / 0.6745
    sigma_g = abs(min(price_list_g[0]) - mu_g) / 3
    X_g = np.arange(0, 2, 0.001)

    Y_g = [stats.norm.pdf(X_g, mu_g, sigma_g), stats.norm.cdf(X_g, mu_g, sigma_g), stats.norm.sf(X_g, mu_g, sigma_g)]
    X_label = ['wallet guess', 'wallet guess', 'number of nodes with purchasing ability guess']
    Y_label = ['probability density', 'probability', 'probability']
    title = ['pdf', 'cdf', 'ccdf']
    for _ in range(len(price_list_g) - 1):
        Y_g.append(Y_g[-1])
        X_label.append(X_label[-1])
        Y_label.append(Y_label[-1])
        title.append(title[-1])

    for index in range(len(Y_g)):
        plt.plot(X_g, Y_g[index])
        if index >= len(price_list_g) - 1:
            for pk in range(len(price_list_g[index - 2])):
                plt.plot(price_list_g[index - 2][pk], float(stats.norm.sf(price_list_g[index - 2][pk], mu_g, sigma_g)), '*')
                plt.text(price_list_g[index - 2][pk], float(stats.norm.sf(price_list_g[index - 2][pk], mu_g, sigma_g)),
                         round(float(stats.norm.sf(price_list_g[index - 2][pk], mu_g, sigma_g)), 4), ha='center', va='bottom', fontsize=9)
        plt.xlabel(X_label[index])
        plt.ylabel(Y_label[index])
        plt.title(title[index] + ' of normal distribution: μ = ' + str(mu_g) + ', σ = ' + str(round(sigma_g, 4)))
        plt.grid()
        plt.show()

    product_weight_list = [round(float(Y_g[2][np.argwhere(X_g == p)]), 4) for p in price_list_g[0]]
    print(product_weight_list)
    print(round(float(stats.norm.sf(price_list_g[0][0] + price_list_g[0][1], mu_g, sigma_g)), 4))
    print(round(float(stats.norm.sf(price_list_g[0][0] + price_list_g[0][2], mu_g, sigma_g)), 4))
    print(round(float(stats.norm.sf(price_list_g[0][1] + price_list_g[0][2], mu_g, sigma_g)), 4))
    print(round(float(stats.norm.sf(sum(price_list_g[0]), mu_g, sigma_g)), 4))