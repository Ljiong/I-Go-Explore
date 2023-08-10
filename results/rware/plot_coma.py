import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



data = pd.read_csv('E:\\study\\Thesis\\epymarl\\results\\rware\\FINAL_COMA_SMALL(2).csv')


if __name__ == '__main__':
    x = np.arange(1,21)
    # print(data.columns)
    y1 = data["coma_ige"]
    y2 = data["coma"]
    y3 = data["coma_icm"]
    # y4 = data["ge"]
    index = data["x_axis"].tolist()

    coma_ige = plt.plot(np.arange(0,20), y1, color='olivedrab', marker='o', ms=3)
    coma = plt.plot(np.arange(0,20), y2, color='gold', marker='^', ms=3)
    coma_icm = plt.plot(np.arange(0,20), y3, color='lightskyblue', marker='s', ms=3)
    # ge = plt.plot(np.arange(1,21), y4, color='mediumpurple', marker='p', ms=4)

    plt.fill_between(np.arange(0,20), data['coma_ige'] - 0.018, data['coma_ige'] + 0.018, color="honeydew", alpha=0.9)
    plt.fill_between(np.arange(0,20), data['coma'] - 0.004, data['coma'] + 0.004, color="lightyellow", alpha=0.9)
    plt.fill_between(np.arange(0,20), data['coma_icm'] - 0.01, data['coma_icm'] + 0.01, color="aliceblue", alpha=0.9)
    # plt.fill_between(np.arange(1,21), data['ge'] - 0.013, data['ge'] + 0.013, color="lavender", alpha=0.9)


    # plt.xticks(rotation=20)
    
    # plt.xticks(x,index, horizontalalignment='right')
    plt.xticks(np.arange(0,20, step=5))

    plt.ylabel('Reward')
    plt.xlabel('Timesteps (0.2 * 1e6)')


    plt.legend(["coma_ige-ge","coma","coma_icm"], loc='upper left')#,"ge"

    plt.title("SMALL-4-Agent")
    plt.savefig("E:/study/Thesis/epymarl/results/rware/coma-SMALL-4g.png",dpi=500,bbox_inches = 'tight')
    plt.show()