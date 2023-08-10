import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



data = pd.read_csv('E:\\study\\Thesis\\epymarl\\results\\rware\\FINAL_maddpg_tiny.csv')


if __name__ == '__main__':
    x = np.arange(1,26)
    # print(data.columns)
    y1 = data["maddpg_ige"]
    y2 = data["maddpg"]
    y3 = data["maddpg_icm"]
    # y4 = data["ge"]
    index = data["x_axis"].tolist()

    maddpg_ige = plt.plot(np.arange(0,25), y1, color='olivedrab', marker='o', ms=3)
    maddpg = plt.plot(np.arange(0,25), y2, color='gold', marker='^', ms=3)
    maddpg_icm = plt.plot(np.arange(0,25), y3, color='lightskyblue', marker='s', ms=3)
    # ge = plt.plot(np.arange(1,21), y4, color='mediumpurple', marker='p', ms=4)

    plt.fill_between(np.arange(0,25), data['maddpg_ige'] - 0.67, data['maddpg_ige'] + 0.67, color="honeydew", alpha=0.9)
    plt.fill_between(np.arange(0,25), data['maddpg'] - 0.77, data['maddpg'] + 0.77, color="lightyellow", alpha=0.9)
    plt.fill_between(np.arange(0,25), data['maddpg_icm'] - 0.76, data['maddpg_icm'] + 0.76, color="aliceblue", alpha=0.9)
    # plt.fill_between(np.arange(1,21), data['ge'] - 0.013, data['ge'] + 0.013, color="lavender", alpha=0.9)


    # plt.xticks(rotation=20)
    
    # plt.xticks(x,index, horizontalalignment='right')
    plt.xticks(np.arange(0,25, step=5))

    plt.ylabel('Reward')
    plt.xlabel('Timesteps (0.2 * 1e6)')


    plt.legend(["maddpg_ige-ge","maddpg","maddpg_icm"], loc='upper left')#,"ge"

    plt.title("Tiny-4-Agent")
    plt.savefig("E:/study/Thesis/epymarl/results/rware/MADDPG-TINY-4g.png",dpi=500,bbox_inches = 'tight')
    plt.show()