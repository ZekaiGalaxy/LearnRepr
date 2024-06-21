from utils import *
import matplotlib.pyplot as plt
path = 'huggingface_proxy_ss'
acc = load_file(f'{path}/train_acc_fn_0.json')
keys = list(acc.keys())

def plot(data='train', all=0):
    if all == 0:
        results = {k:[] for k in keys}

        # train
        x_axis = []
        for i in range(10):
            for phase in ['fn','fp']:
                x_axis.append(f"{i}_{phase}")
                p = f'{path}/{data}_acc_{phase}_{i}.json'
                acc = load_file(p)
                for k in keys:
                    results[k].append(max(0,acc[k]))

        # Plotting the data
        plt.figure(figsize=(10, 6))

        # Plotting the data
        for k in keys:
            plt.plot(x_axis, results[k], label=k)

        # Showing the plot
        plt.tight_layout()
        plt.savefig(path+f'/plot_{data}_acc.png')
    else:
        results = []
        x_axis = []
        for i in range(10):
            for phase in ['fn','fp']:
                x_axis.append(f"{i}_{phase}")
                p = f'{path}/{data}_acc_{phase}_{i}.json'
                acc = load_file(p)
                results.append(max(0,acc['all']))
        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, results, label='all')
        plt.savefig(path+f'/plot_{data}_acc_all.png')

plot('train', all=0)
plot('test', all=0)
plot('train', all=1)
plot('test', all=1)
            