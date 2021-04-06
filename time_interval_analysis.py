import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def bar_graph():
    # data
    time = [10, 20, 30, 60, 180]  # in terms of days
    performance = [(8.2188, 0.6047, 9.0604, 2.6340), (10.8215, 0.6105, 8.0006, 2.6579),
                   (15.3175, 0.6166, 10.3766, 3.5249),
                   (17.1332, 0.5846, 12.1121, 4.2286), (27.4012, 0.5928, 18.0708, 7.6385)]  # in terms of RMSE

    ten = [x for x in performance[0]]
    twenty = [x for x in performance[1]]
    thirty = [x for x in performance[2]]
    sixty = [x for x in performance[3]]
    one_eighty = [x for x in performance[4]]

    lake = [ten[0], twenty[0], thirty[0], sixty[0], one_eighty[0]]
    river = [ten[1], twenty[1], thirty[1], sixty[1], one_eighty[1]]
    waterspring = [ten[2], twenty[2], thirty[2], sixty[2], one_eighty[2]]
    aquifer = [ten[3], twenty[3], thirty[3], sixty[3], one_eighty[3]]
    names = ['lake', 'river', 'waterspring', 'aquifer']  # names of models

    # plotting
    N = len(time)
    barwidth = 0.5
    xloc = np.arange(N)

    p1 = [plt.bar(xloc, lake, width=barwidth, label='Lake'),
          plt.bar(xloc, waterspring, width=barwidth, label='Waterspring'),
          plt.bar(xloc, aquifer, width=barwidth, label='Aquifer'),
          plt.bar(xloc, river, width=barwidth, label='River')]

    # adding labels
    plt.ylabel('Validation RMSE - (lower is better)')
    plt.xlabel('Look Ahead Time - (In Days)')
    plt.title('Time Interval Analysis')
    plt.xticks(xloc, time)
    plt.legend()

    plt.show()


bar_graph()
