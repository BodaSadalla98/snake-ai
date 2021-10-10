import matplotlib.pyplot as plt
from IPython  import display

def plot(scores, mean_scores):

    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()

    plt.title('Training...')
    plt.xlabel('Number of games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)

    plt.show(block=False)
    plt.pause(.1)