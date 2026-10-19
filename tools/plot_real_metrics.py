"""Grafik precision-recall dan confusion matrix dari CSV test."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,default=Path("reports/gop"))
    args=parser.parse_args()
    metrics=json.loads((args.results/"metrics.json").read_text())
    with (args.results/"predictions_test_gop_calibrated.csv").open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    y=np.array([float(r["target_raw"])<.5 for r in rows])
    risk=np.array([1-float(r["prediction_raw"])/2 for r in rows])
    precision,recall,_=precision_recall_curve(y,risk)
    detection=metrics["results"]["test"]["gop_calibrated"]["phoneme"]["mispronunciation"]
    confusion=detection["confusion_matrix"]
    matrix=np.array([[confusion["tn"],confusion["fp"]],[confusion["fn"],confusion["tp"]]])
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    axes[0].step(recall,precision,where="post",color="#17648c",label=f"GOP + MAE boosting (AP={detection['pr_auc_average_precision']:.3f})")
    axes[0].axhline(y.mean(),ls="--",color="#ad614c",label=f"Constant baseline (AP={y.mean():.3f})")
    axes[0].scatter([detection["recall"]],[detection["precision"]],color="#172d3f",s=30,zorder=5,label="Locked validation threshold")
    axes[0].set(xlabel="Recall of expert-labelled phone errors",ylabel="Precision",title="Reused test benchmark: precision-recall",xlim=(0,1),ylim=(0,1.02))
    axes[0].legend(fontsize=8,loc="upper right")
    axes[0].grid(alpha=.15)
    axes[1].imshow(matrix,cmap="Blues")
    for (i,j),value in np.ndenumerate(matrix):
        axes[1].text(j,i,f"{value:,}",ha="center",va="center",color="white" if value>matrix.max()/2 else "#172d3f",fontsize=14)
    axes[1].set(xticks=[0,1],xticklabels=["No practice flag","Practice flag"],yticks=[0,1],yticklabels=["Expert: not an error","Expert: error"],title=f"Confusion matrix (F1={detection['f1']:.3f})",xlabel="Model decision at locked threshold")
    fig.savefig(args.results/"test_detection.png",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    main()
