import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



data = pd.read_csv('E:\\study\\Thesis\\epymarl\\results\\rware\\FINAL_COMA_tiny_500.csv')


if __name__ == '__main__':
    x = np.arange(1,501)
    # print(data.columns)
    y1 = data["coma_ige"]
    y2 = data["coma"]
    y3 = data["coma_icm"]
    # y4 = data["ge"]
    index = data["x_axis"].tolist()

    coma_ige = plt.plot(np.arange(0,500), y1, color='olivedrab', marker='o', ms=0.5)
    coma = plt.plot(np.arange(0,500), y2, color='gold', marker='^', ms=0.5)
    coma_icm = plt.plot(np.arange(0,500), y3, color='lightskyblue', marker='s', ms=0.5)
    # ge = plt.plot(np.arange(1,21), y4, color='mediumpurple', marker='p', ms=4)

    plt.fill_between(np.arange(0,500), data['coma_ige'] - 0.04, data['coma_ige'] + 0.04, color="honeydew", alpha=0.95)
    plt.fill_between(np.arange(0,500), data['coma'] - 0.05, data['coma'] + 0.05, color="lightyellow", alpha=0.95)
    plt.fill_between(np.arange(0,500), data['coma_icm'] - 0.03, data['coma_icm'] + 0.03, color="aliceblue", alpha=0.95)
    # plt.fill_between(np.arange(1,21), data['ge'] - 0.013, data['ge'] + 0.013, color="lavender", alpha=0.9)


    # plt.xticks(rotation=20)
    
    # plt.xticks(x,index, horizontalalignment='right')
    plt.xticks(np.arange(0,500, step=50))
    plt.yticks(np.arange(0,1, step=0.2))

    plt.ylabel('Reward')
    plt.xlabel('Timesteps (0.2 * 1e6)')


    plt.legend(["coma_ige-ge","coma","coma_icm"], loc='upper left')#,"ge"

    plt.title("Tiny-4-Agent")
    plt.savefig("E:/study/Thesis/epymarl/results/rware/coma-TINY-4g-500.png",dpi=500,bbox_inches = 'tight')
    plt.show()