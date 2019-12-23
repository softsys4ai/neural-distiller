import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

RESULTS_FILE = "experiment3_AR_FGM.csv"
EXP = "experiment2"
EPS = "0.050"
ATTACK = "FGM"
CMAP = "gnuplot2"
df_plot = pd.read_csv(RESULTS_FILE)
df = df_plot[df_plot.interval != 0]
df_min = df['eps_'+EPS].min()
df_max = df['eps_'+EPS].max()
hm_data1 = pd.pivot_table(df_plot, values='eps_' + EPS,
                          index=['temp'],
                          columns='interval')
plot = sns.heatmap(hm_data1, cmap=CMAP, vmin=df_min, vmax=df_max)
plot.invert_yaxis()
plt.title("Adversarial Accuracy w.r.t Temperature and Epoch Interval ("+ATTACK+", Epsilon "+EPS+")")
fig = plot.get_figure()
fig.savefig(EXP+"_AR_"+ATTACK+"_Eps_"+EPS+"_"+CMAP+".png")
plt.show()
