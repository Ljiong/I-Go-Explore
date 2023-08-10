import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



data = pd.read_csv('E:\\study\\Thesis\\epymarl\\results\\rware\\FINAL_mappo_SMALL.csv')


if __name__ == '__main__':
    x = np.arange(1,1001)
    # print(data.columns)
    y1 = data["mappo_ige"]
    y2 = data["mappo"]
    y3 = data["mappo_icm"]
    # y4 = data["ge"]
    index = data["x_axis"].tolist()

    mappo_ige = plt.plot(np.arange(0,1000), y1, color='olivedrab', marker='o', ms=0.5)
    mappo = plt.plot(np.arange(0,1000), y2, color='gold', marker='^', ms=0.5)
    mappo_icm = plt.plot(np.arange(0,1000), y3, color='lightskyblue', marker='s', ms=0.5)
    # ge = plt.plot(np.arange(1,21), y4, color='mediumpurple', marker='p', ms=4)

    plt.fill_between(np.arange(0,1000), data['mappo_ige'] - 12.53, data['mappo_ige'] + 12.53, color="honeydew", alpha=0.9)
    plt.fill_between(np.arange(0,1000), data['mappo'] - 13.54, data['mappo'] + 13.54, color="lightyellow", alpha=0.9)
    plt.fill_between(np.arange(0,1000), data['mappo_icm'] - 12.98, data['mappo_icm'] + 12.98, color="aliceblue", alpha=0.9)
    # plt.fill_between(np.arange(1,21), data['ge'] - 0.013, data['ge'] + 0.013, color="lavender", alpha=0.9)


    # plt.xticks(rotation=20)
    
    # plt.xticks(x,index, horizontalalignment='right')
    plt.xticks(np.arange(0,1000, step=100))

    plt.ylabel('Reward')
    plt.xlabel('Timesteps (0.2 * 1e6)')


    plt.legend(["mappo_ige-ge","mappo","mappo_icm"], loc='upper left')#,"ge"

    plt.title("Small-4-Agent")
    plt.savefig("E:/study/Thesis/epymarl/results/rware/MAPPO-SMALL-4g.png",dpi=500,bbox_inches = 'tight')
    plt.show()