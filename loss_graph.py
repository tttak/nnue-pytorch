import datetime
import matplotlib.pyplot as plt
import tensorflow

DATA_SET = [
    # {
    #     'file_path': r"C:\home\nodchip\nnue-pytorch\logs\default\version_31\events.out.tfevents.1654993958.endwalker.41572.0",
    #     # 'label': 'HalfKP',
    #     # 'label': 'halfkp_256x2-32-32',
    #     'label': 'StepLR step=75 gamma=0.3',
    # },
    # {
    #     'file_path': r"C:\home\nodchip\nnue-pytorch\logs\default\version_32\events.out.tfevents.1655176423.endwalker.20100.0",
    #     'label': 't=0.8',
    # },
    # {
    #     'file_path': r"C:\home\nodchip\nnue-pytorch\logs\default\version_35\events.out.tfevents.1655366553.endwalker.24056.0",
    #     'label': 'HalfKP^',
    # },
    # {
    #     'file_path': r"C:\home\nodchip\nnue-pytorch\logs\default\version_40\events.out.tfevents.1655686310.endwalker.45968.0",
    #     'label': 'halfkp_1024x2-8-32',
    # },
    {
        'file_path': r"C:\home\nodchip\nnue-pytorch\logs\default\version_41\events.out.tfevents.1656322267.endwalker.45728.0",
        # 'label': 'StepLR step=1 gamma=0.992',
        # 'label': 'Ranger',
        # 'label': 'lr=0.0008.75',
        'label': 'no-mirror',
    },
    # {
    #     'file_path': r"C:\home\nodchip\nnue-pytorch\logs\default\version_44\events.out.tfevents.1656554368.endwalker.6968.0",
    #     'label': 'RAdam',
    # },
    # {
    #     'file_path': r"C:\home\nodchip\nnue-pytorch\logs\default\version_46\events.out.tfevents.1656726562.endwalker.35780.0",
    #     'label': 'NVLAMB without StepLR',
    # },
    # {
    #     'file_path': r"C:\home\nodchip\nnue-pytorch\logs\default\version_47\events.out.tfevents.1656912294.endwalker.9876.0",
    #     'label': 'NVLAMB',
    # },
    # {
    #     'file_path': r"C:\home\nodchip\nnue-pytorch\logs\default\version_58\events.out.tfevents.1657336478.endwalker.8776.0",
    #     'label': 'lr=0.00003125',
    # },
    # {
    #     'file_path': r"C:\home\nodchip\nnue-pytorch\logs\default\version_57\events.out.tfevents.1657313104.endwalker.28360.0",
    #     'label': 'lr=0.0000625',
    # },
    # {
    #     'file_path': r"C:\home\nodchip\nnue-pytorch\logs\default\version_56\events.out.tfevents.1657285755.endwalker.31280.0",
    #     'label': 'lr=0.000125',
    # },
    # {
    #     'file_path': r"C:\home\nodchip\nnue-pytorch\logs\default\version_51\events.out.tfevents.1657153753.endwalker.6540.0",
    #     'label': 'lr=0.00025',
    # },
    # {
    #     'file_path': r"C:\home\nodchip\nnue-pytorch\logs\default\version_52\events.out.tfevents.1657177744.endwalker.21252.0",
    #     'label': 'lr=0.0005',
    # },
    # {
    #     'file_path': r"C:\home\nodchip\nnue-pytorch\logs\default\version_53\events.out.tfevents.1657211100.endwalker.7184.0",
    #     'label': 'lr=0.0010',
    # },
    # {
    #     'file_path': r"C:\home\nodchip\nnue-pytorch\logs\default\version_54\events.out.tfevents.1657234489.endwalker.20596.0",
    #     'label': 'lr=0.0020',
    # },
    # {
    #     'file_path': r"C:\home\nodchip\nnue-pytorch\logs\default\version_55\events.out.tfevents.1657258888.endwalker.29520.0",
    #     'label': 'lr=0.0040',
    # },
    # {
    #     'file_path': r"C:\home\nodchip\nnue-pytorch\logs\default\version_59\events.out.tfevents.1657422609.endwalker.1792.0",
    #     'label': 'lr=0.00025',
    # },
    {
        'file_path': r"C:\home\nodchip\nnue-pytorch\logs\default\version_62\events.out.tfevents.1660124365.endwalker.10072.0",
        'label': 'mirror',
    },
]


GRAPH_INDEX = 0


def draw_each_tag():
    global GRAPH_INDEX
    for tag in ['train_loss', 'val_loss']:
        plt.figure(figsize=(1600.0/100.0, 1200.0/100.0))

        for data_index, data in enumerate(DATA_SET):
            file_path = data['file_path']
            label = data['label']

            xs = list()
            ys = list()
            for e in tensorflow.compat.v1.train.summary_iterator(file_path):
                for v in e.summary.value:
                    if v.tag == tag:
                        xs.append(e.step)
                        ys.append(v.simple_value)
            plt.plot(xs, ys, label=label)

        plt.legend()
        plt.grid()
        plt.title(tag)

        now = datetime.datetime.now()
        filename = f'image.{now.strftime("%Y-%m-%d-%H-%M-%S")}.{GRAPH_INDEX}.{tag}.png'
        GRAPH_INDEX += 1
        plt.savefig(filename)

        # plt.show()

        plt.close()


def draw_all():
    global GRAPH_INDEX
    plt.figure(figsize=(1600.0/100.0, 1200.0/100.0))
    for tag in ['train_loss', 'val_loss']:
        for data_index, data in enumerate([DATA_SET[-1]]):
            file_path = data['file_path']
            label = data['label']

            xs = list()
            ys = list()
            for e in tensorflow.compat.v1.train.summary_iterator(file_path):
                for v in e.summary.value:
                    if v.tag == tag:
                        xs.append(e.step)
                        ys.append(v.simple_value)
            plt.plot(xs, ys, label=label + " " + tag)

    plt.legend()
    plt.grid()

    now = datetime.datetime.now()
    filename = f'image.{now.strftime("%Y-%m-%d-%H-%M-%S")}.{GRAPH_INDEX}.all.png'
    GRAPH_INDEX += 1
    plt.savefig(filename)

    # plt.show()

    plt.close()


def main():
    draw_each_tag()
    draw_all()


if __name__ == '__main__':
  main()
