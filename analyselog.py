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
    return alltrainlosses_baseline, alltrainsteps_baseline


def plot_1losses(dir_dir_b1024, key="\'train_loss\'", after_epoch=None, before_epoch=None):
    plt.rcParams["figure.figsize"] = (6,5)
    fig = plt.figure()
    alltrainlosses_dir_b1024, alltrainsteps_dir_b1024 = readlog(dir_dir_b1024, trainlosskeyword=key)
    np.save(key, np.vstack([alltrainsteps_dir_b1024, alltrainlosses_dir_b1024]))
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
    # plotting
    positive_idx = np.where(np.array(alltrainlosses_dir_b1024[after_idx:before_idx])>0)[0][::2]
    negative_idx = np.where(np.array(alltrainlosses_dir_b1024[after_idx:before_idx])<=0)[0][::2]
    plt.scatter(np.array(alltrainsteps_dir_b1024[after_idx:before_idx])[positive_idx], np.array(alltrainlosses_dir_b1024[after_idx:before_idx])[positive_idx], c=np.arange(len(positive_idx)), label="$L>0$", marker="x")
    plt.scatter(np.array(alltrainsteps_dir_b1024[after_idx:before_idx])[negative_idx], -np.array(alltrainlosses_dir_b1024[after_idx:before_idx])[negative_idx], c=np.arange(len(negative_idx)), cmap="plasma", label="$L<0$", marker="x")
    if len(positive_idx) > 0:
        plt.axhline(np.array(alltrainlosses_dir_b1024[after_idx:before_idx])[positive_idx][-1], ls="--")
    if len(negative_idx) > 0:
        plt.axhline(-np.array(alltrainlosses_dir_b1024[after_idx:before_idx])[negative_idx][-1], ls="--", c="r")
    plt.semilogy()
    setfigform_simple("epoch","loss")
    plt.legend()
    plt.title(dir_dir_b1024, fontdict=font)
    
    fig.tight_layout()
    plt.savefig(os.path.join(dir_dir_b1024, key), bbox_inches="tight")
    # plt.show()



dir = f"./"
if sys.argv[2] == "None": 
    before_epoch = None
else:
    before_epoch = int(sys.argv[2])
plot_1losses(dir, after_epoch=int(sys.argv[1]), before_epoch=before_epoch)
plot_1losses(dir, key="\'val_loss\'", after_epoch=int(sys.argv[1]), before_epoch=before_epoch)
plot_1losses(dir, key="\'train_loss_energy\'", after_epoch=int(sys.argv[1]), before_epoch=before_epoch)
plot_1losses(dir, key="\'val_loss_energy\'", after_epoch=int(sys.argv[1]), before_epoch=before_epoch)
# plot_1losses(dir, key="\'train_loss_cell\'", after_epoch=int(sys.argv[1]), before_epoch=before_epoch)
# plot_1losses(dir, key="\'val_loss_cell\'", after_epoch=int(sys.argv[1]), before_epoch=before_epoch)
plot_1losses(dir, key="\'train_loss_gen\'", after_epoch=int(sys.argv[1]), before_epoch=before_epoch)
plot_1losses(dir, key="\'val_loss_gen\'", after_epoch=int(sys.argv[1]), before_epoch=before_epoch)
# plot_1losses(dir, key="\'train_loss_score\'", after_epoch=int(sys.argv[1]), before_epoch=before_epoch)
# plot_1losses(dir, key="\'val_loss_score\'", after_epoch=int(sys.argv[1]), before_epoch=before_epoch)
