def drawresult():

    fig = plt.figure()
    fig.set_size_inches(10.5, 7.5)

    ax11 = fig.add_subplot(2, 2, 1)
    ax11.set_xlim([0, 10])
    ax11.set_ylim([0.94, 1.001])
    ax11.set_title('Training Accuracy')
    ax11.margins(x=0.1, y=0.05)

    ax12 = fig.add_subplot(2, 2, 2)
    ax12.set_xlim([0, 10])
    ax12.set_ylim([0.0, 0.5])
    ax12.set_title('Training Loss')
    ax12.margins(x=0.1, y=0.5)

    ax21 = fig.add_subplot(2, 2, 3)
    ax21.set_xlim([0, 10])
    ax21.set_ylim([0.97, 1.001])
    ax21.set_title('Validation Accuracy')

    ax22 = fig.add_subplot(2, 2, 4)
    ax22.set_xlim([1, 10])
    ax22.set_ylim([0.0, 0.1])
    ax22.set_title('Validation Loss')

    print(running_loss)

    ax11.plot(train_accuracy)
    ax12.plot(train_losses)
    ax21.plot(test_accuracy)
    ax22.plot(test_losses)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    outputs = net(images)

def drawResult():

    fig = plt.figure()
    fig.set_size_inches(10.5, 7.5)
    plot_title = ["1", "2", "3", "4"]

    subplots = []

    for i in range(1, 4):
        subplot = fig.add_subplot(2, 2, i)
        subplot.set_xlim([0, 10])
        subplot.set_ylim([0.94, 1.001])
        subplot.set_title(plot_title[i])
        subplot.margins(x=0.1, y=0.05)
        subplots[0] = subplot

    subplots[0].plot(train_accuracy)
    subplots[1].plot(train_losses)
    subplots[2].plot(test_accuracy)
    subplots[3].plot(test_losses)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()
