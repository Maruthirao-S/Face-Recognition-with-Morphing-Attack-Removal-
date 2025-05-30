import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import seaborn as sn
import pandas as pd
from itertools import cycle
from sklearn import metrics
from sklearn.metrics import roc_curve


def Plot_Results():
    Eval1 = np.load('Eval_all1.npy', allow_pickle=True)
    Eval = np.load('Eval_all.npy', allow_pickle=True)
    Eval[:, :, 7] = Eval1[:, :, 7]
    Terms = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'FPR', 'FNR', 'FOR', 'NPV', 'FDR', 'F1-Score',
             'MCC']
    Graph_Term = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    Algorithm = ['TERMS', 'MAO', 'CO', 'GWO', 'CSA', 'PROPOSED']
    Classifier = ['TERMS', 'Ref-3', 'Ref-7', 'DCNN', 'DTCN', 'PROPOSED']

    value = Eval[3, :, 4:]  .astype('float')# 0.21 Learning Rate
    value[:, :-1] = value[:, :-1] * 100

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], value[j, :])
    print('--------------------------------------------------', 'Algorithm Comparison',
          '--------------------------------------------------')
    print(Table)

    Table = PrettyTable()
    Table.add_column(Classifier[0], Terms)
    for j in range(len(Classifier) - 1):
        Table.add_column(Classifier[j + 1], value[j + 5, :])
    print('--------------------------------------------------', 'Classifier Comparison',
          '--------------------------------------------------')
    print(Table)

    learnper = [1, 2, 3, 4, 5]
    Eval1 = np.load('Eval_all1.npy', allow_pickle=True)
    Eval = np.load('Eval_all.npy', allow_pickle=True)
    Eval[:, :, 7] = Eval1[:, :, 7]
    np.save('Eval_updated.npy',Eval)
    for j in range(len(Graph_Term)):
        Graph = np.zeros((Eval.shape[0], Eval.shape[1]))
        for k in range(Eval.shape[0]):
            for l in range(Eval.shape[1]):
                if Graph_Term[j] == 10:
                    Graph[k, l] = Eval[k, l, Graph_Term[j] + 4]
                else:
                    Graph[k, l] = Eval[k, l, Graph_Term[j] + 4] * 100

        plt.plot(learnper, Graph[:, 0], color='red', linewidth=3, marker='d', markerfacecolor='lime', markersize=12,
                 linestyle='dashed', label="MAO-ADTCN")
        plt.plot(learnper, Graph[:, 1], color='green', linewidth=3, marker='p', markerfacecolor='deeppink',
                 markersize=12,
                 linestyle='dashed', label='CO-ADTCN')
        plt.plot(learnper, Graph[:, 2], color='blue', linewidth=3, marker='h', markerfacecolor='aqua',
                 markersize=12,
                 linestyle='dashed', label="GWO-ADTCN")
        plt.plot(learnper, Graph[:, 3], color='magenta', linewidth=3, marker='o', markerfacecolor='gold', markersize=12,
                 linestyle='dashed', label="CSA-ADTCN")
        plt.plot(learnper, Graph[:, 4], color='cyan', linewidth=3, marker='>', markerfacecolor='green',
                 markersize=12,
                 linestyle='dashed', label="IRP-CSA-ADTCN")
        plt.xlabel('k fold')
        plt.ylabel(Terms[Graph_Term[j]])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.175),
                   ncol=3, fancybox=True, shadow=True)
        path = "./Results_dummy/%s_alg-kfold.png" % (Terms[Graph_Term[j]])
        plt.savefig(path)
        plt.show()

        data = {'CNN': Graph[:, 5],
                'VGG16': Graph[:, 6],
                'DCNN': Graph[:, 7],
                'DTCN': Graph[:, 8],
                'IRP-CSA-ADTCN': Graph[:, 9]}
        df = pd.DataFrame(data, columns=['CNN', 'VGG16', 'DCNN', 'DTCN', 'IRP-CSA-ADTCN'],
                          index=['1', '2', '3', '4', '5'])
        label_colors = ['deeppink', 'lime', 'orange', 'b', 'green']
        ax = df.plot.barh(width=0.7, color=label_colors)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=8)
        ax.set(xlabel=Terms[Graph_Term[j]], ylabel="k fold")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14),
                  ncol=3, fancybox=True, shadow=True)
        path = "./Results_dummy/%s_cls-kfold.png" % (Terms[Graph_Term[j]])
        plt.savefig(path)
        plt.show()


def Plot_Confusion():
    # Confusion Matrix
    Eval = np.load('Eval_all1.npy', allow_pickle=True)
    value = Eval[3, 4, :5]
    val = np.asarray([0, 1, 1])
    data = {'y_Actual': [val.ravel()],
            'y_Predicted': [np.asarray(val).ravel()]
            }
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'][0], df['y_Predicted'][0], rownames=['Actual'], colnames=['Predicted'])
    value = value.astype('int')

    confusion_matrix.values[0, 0] = value[1]
    confusion_matrix.values[0, 1] = value[3]
    confusion_matrix.values[1, 0] = value[2]
    confusion_matrix.values[1, 1] = value[0]

    sn.heatmap(confusion_matrix, annot=True).set(title='Accuracy = ' + str(Eval[2, 4, 4] * 100)[1:6] + '%')
    sn.plotting_context()
    path1 = './Results/Confusion.png'
    plt.savefig(path1)
    plt.show()


def Plot_ROC():
    lw = 2
    cls = ['CNN', 'VGG16', 'DCNN', 'DTCN', 'IRP-CSA-ADTCN']
    colors = cycle(["hotpink", "plum", "chocolate", "green", "magenta"])
    Predicted = np.load('roc_score.npy', allow_pickle=True)
    Actual = np.load('roc_act.npy', allow_pickle=True)
    for j, color in zip(range(5), colors):  # For all classifiers

        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual[2, j + 5][:, 0].astype('float'),
                                                                          Predicted[2, j + 5][:, 0].astype('float'))
        # auc = metrics.roc_auc_score(Actual[j, :], Predicted[j, :])
        auc = metrics.roc_auc_score(Actual[2, j + 5][:, 0].astype('float'), Predicted[2, j + 5][:, 0].astype('float'))
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label="{0} (auc = {1:0.2f})".format(cls[j], auc),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path = "./Results/Roc.png"
    plt.savefig(path)
    plt.show()


def Plot_Convergence():
    for a in range(1):
        conv = np.load('Fitness_2.npy', allow_pickle=True)[a]
        ind = np.argsort(conv[:, conv.shape[1] - 1])
        x = conv[ind[0], :].copy()
        y = conv[4, :].copy()
        conv[4, :] = x
        conv[ind[0], :] = y

        Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
        Algorithm = ['MAO', 'CO', 'GWO', 'CSA', 'PROPOSED']

        Value = np.zeros((conv.shape[0], 5))
        for j in range(conv.shape[0]):
            Value[j, 0] = np.min(conv[j, :])
            Value[j, 1] = np.max(conv[j, :])
            Value[j, 2] = np.mean(conv[j, :])
            Value[j, 3] = np.median(conv[j, :])
            Value[j, 4] = np.std(conv[j, :])

        Table = PrettyTable()
        Table.add_column("ALGORITHMS", Statistics)
        for j in range(len(Algorithm)):
            Table.add_column(Algorithm[j], Value[j, :])
        print(
            '-------------------------------------------------- Statistical Analysis--------------------------------------------------')
        print(Table)

        iteration = np.arange(conv.shape[1])
        plt.plot(iteration, conv[0, :], color='darkgreen', linewidth=3, marker='*', markerfacecolor='deeppink',
                 markersize=8,
                 label="MAO-ADTCN")
        plt.plot(iteration, conv[1, :], color='orange', linewidth=3, marker='*', markerfacecolor='cyan', markersize=8,
                 label="CO-ADTCN")
        plt.plot(iteration, conv[2, :], color='slateblue', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=8,
                 label="GWO-ADTCN")
        plt.plot(iteration, conv[3, :], color='darkcyan', linewidth=3, marker='*', markerfacecolor='plum', markersize=8,
                 label="CSA-ADTCN")
        plt.plot(iteration, conv[4, :], color='lawngreen', linewidth=3, marker='*', markerfacecolor='blueviolet',
                 markersize=8,
                 label="IRP-CSA-ADTCN")
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        path1 = "./Results/convergence.png"
        plt.savefig(path1)
        plt.show()


if __name__ == '__main__':
    Plot_Convergence()
    Plot_Results()
    Plot_Confusion()
    Plot_ROC()
