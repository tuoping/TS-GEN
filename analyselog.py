import os
import numpy as np
import matplotlib.pyplot as plt
import sys

font={'family':'serif',
      # 'style':'italic',  # 斜体
      'weight':'normal',
      # 'color':'red',
      'size': 18
}  
def setfigform_simple(xlabel, ylabel=None, xlimit = (None,None), ylimit = (None, None)):
    # plt.legend(fontsize = 16, frameon=False),
    plt.xlabel(xlabel, fontdict = font)
    plt.ylabel(ylabel, fontdict = font)
    plt.xlim(xlimit)
    plt.ylim(ylimit)
    # plt.xticks(fontsize = font['size'], fontname = "serif")
    # plt.yticks(fontsize = font['size'], fontname = "serif")
    plt.tick_params(direction="in")

if os.path.exists("EXCLUDE_EPOCHS"):
    exclude_epochs = np.loadtxt("EXCLUDE_EPOCHS")
else:
    exclude_epochs = []

def readlog(dir, trainlosskeyword="\'train_loss\'", exclude_epochs=exclude_epochs):
    alltrainsteps_baseline = []
    alltrainlosses_baseline = []
    allconditional_bool = []
    with open(os.path.join(dir, "log.out")) as fp:
        lines = fp.readlines()
        for line in lines:
            l = line.split()
            if trainlosskeyword in line:
                for idx_t,t in enumerate(l):
                    if "\'epoch\'" in t:
                        if float(l[idx_t+1].replace(",","")) in exclude_epochs:
                            print(float(l[idx_t+1].replace(",","")), exclude_epochs)
                            if len(alltrainlosses_baseline)>len(alltrainsteps_baseline):
                                alltrainlosses_baseline.pop(-1)
                            break
                        alltrainsteps_baseline.append(float(l[idx_t+1].replace(",","").replace("np.float64(","").replace(")","")))
                        # break
                    if trainlosskeyword in t:
                        alltrainlosses_baseline.append(float(l[idx_t+1].replace(",","").replace("np.float64(","").replace(")","")))
                    if "conditional_batch" in t:
                        allconditional_bool.append(float(l[idx_t+1].replace(",","").replace("np.float64(","").replace(")","")))
    return alltrainlosses_baseline, alltrainsteps_baseline, allconditional_bool


def plot_1losses(dir_dir_b1024, key="\'train_loss\'", after_epoch=None, before_epoch=None, ymin=None, ymax=None):
    plt.rcParams["figure.figsize"] = (6,5)
    fig = plt.figure()
    alltrainlosses_dir_b1024, alltrainsteps_dir_b1024, allconditional_bool = readlog(dir_dir_b1024, trainlosskeyword=key)
    # np.save(key, np.vstack([alltrainsteps_dir_b1024, alltrainlosses_dir_b1024]))
    before_idx = None
    after_idx = None
    if len(alltrainlosses_dir_b1024) != 0:
        if after_epoch is not None and after_epoch != "None":
            after_idx = np.where(np.array(alltrainsteps_dir_b1024)==float(after_epoch))[0][0]
            print("after_idx = ", after_idx)
        if before_epoch is not None and before_epoch != "None":
            before_idx = np.where(np.array(alltrainsteps_dir_b1024)==float(before_epoch))[0][0]
            print("before_idx = ", before_idx)
        
        
    ### remove the loss value when restart training 
    # if "loss_gen" in key:
    #     stable_idx = np.where(np.array(alltrainlosses_dir_b1024)<-1.1)[0]
    # else:
    stable_idx = np.arange(len(alltrainsteps_dir_b1024), dtype=int)
    alltrainlosses_dir_b1024 = np.array(alltrainlosses_dir_b1024)[stable_idx]
    alltrainsteps_dir_b1024 = np.array(alltrainsteps_dir_b1024)[stable_idx]
    allconditional_bool = np.array(allconditional_bool)
    # plotting
    positive_idx = np.where(np.array(alltrainlosses_dir_b1024[after_idx:before_idx])>0)[0][:]
    negative_idx = np.where(np.array(alltrainlosses_dir_b1024[after_idx:before_idx])<=0)[0][:]
    print("positive_idx = ", positive_idx)
    print("negative_idx = ", negative_idx)
    print("allconditional_bool = ", allconditional_bool)
    print("alltrainsteps_dir_b1024 = ", alltrainsteps_dir_b1024[after_idx:before_idx])
    print("alltrainlosses_dir_b1024 = ", alltrainlosses_dir_b1024[after_idx:before_idx])
    # plt.scatter(np.array(alltrainsteps_dir_b1024[after_idx:before_idx])[positive_idx], np.array(alltrainlosses_dir_b1024[after_idx:before_idx])[positive_idx], c=allconditional_bool[after_idx:before_idx][positive_idx], label="$L>0$", cmap='bwr', marker="x")
    # plt.scatter(np.array(alltrainsteps_dir_b1024[after_idx:before_idx])[negative_idx], -np.array(alltrainlosses_dir_b1024[after_idx:before_idx])[negative_idx], c=allconditional_bool[after_idx:before_idx][positive_idx], cmap="bwr", label="$L<0$", marker="o")
    plt.scatter(np.array(alltrainsteps_dir_b1024[after_idx:before_idx])[positive_idx], np.array(alltrainlosses_dir_b1024[after_idx:before_idx])[positive_idx], c=np.array(allconditional_bool)[positive_idx], label="$L>0$", marker="x")
    print("negative alltrainsteps_dir_b1024 = ", alltrainsteps_dir_b1024[after_idx:before_idx][negative_idx])
    print("negative alltrainlosses_dir_b1024 = ", alltrainlosses_dir_b1024[after_idx:before_idx][negative_idx])
    print("negative allconditional_bool = ", allconditional_bool[negative_idx])
    plt.scatter(np.array(alltrainsteps_dir_b1024[after_idx:before_idx])[negative_idx], -np.array(alltrainlosses_dir_b1024[after_idx:before_idx])[negative_idx], c=np.array(allconditional_bool)[negative_idx], label="$L<0$", marker="o", vmin=0, vmax=1, s=10)
    cbar = plt.colorbar()
    cbar.set_label("Ratio of conditional training per batch", fontsize=font['size']-4)
    if len(positive_idx) > 0:
        plt.axhline(np.array(alltrainlosses_dir_b1024[after_idx:before_idx])[positive_idx][-1], ls="--")
    if len(negative_idx) > 0:
        plt.axhline(-np.array(alltrainlosses_dir_b1024[after_idx:before_idx])[negative_idx][-1], ls="--", c="r")
    plt.semilogy()
    setfigform_simple("epoch","loss", ylimit=(ymin, ymax))
    plt.legend()
    plt.title(dir_dir_b1024, fontdict=font)
    
    fig.tight_layout()
    plt.savefig(os.path.join(dir_dir_b1024, key), bbox_inches="tight")
    # plt.show()


import argparse

parser = argparse.ArgumentParser(description="DCD → Extended XYZ with triclinic lattice")
parser.add_argument("--after_epoch", type=int, default=0, )
parser.add_argument("--before_epoch",  type=int, default=None)
parser.add_argument("--ymin_val",  type=float, default=None)
parser.add_argument("--ymax_val",  type=float, default=None)
parser.add_argument("--ymin_train",  type=float, default=None)
parser.add_argument("--ymax_train",  type=float, default=None)
parser.add_argument("--key",  type=str, default="\'val_loss_gen\'")
args = parser.parse_args()

dir = f"./"
plot_1losses(dir, key="\'train_loss\'", after_epoch=args.after_epoch, before_epoch=args.before_epoch, ymin=args.ymin_train, ymax=args.ymax_train)
plot_1losses(dir, key=args.key, after_epoch=args.after_epoch, before_epoch=args.before_epoch, ymin=args.ymin_val, ymax=args.ymax_val)
