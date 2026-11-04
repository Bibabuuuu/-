"""Visualize sampled choices and matched physical probes without rerunning physics."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import Normalize
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/"outputs/rby1-research/s3-joint-candidates"
plan=json.loads((DEST/"plan.json").read_text());summary=json.loads((DEST/"summary.json").read_text())
plt.rcParams.update({"font.family":FontProperties(fname="C:/Windows/Fonts/msyh.ttc").get_name(),"axes.unicode_minus":False,
    "axes.spines.top":False,"axes.spines.right":False,"font.size":10})
fig,axs=plt.subplots(2,3,figsize=(14.5,9.1))
fig.subplots_adjust(left=.065,right=.91,top=.84,bottom=.155,hspace=.48,wspace=.3)
fig.suptitle("缓冲量随环境改变：保留负荷、空间与动作幅度的取舍",fontsize=18,y=.975)
fig.text(.065,.92,"上排：通过参考几何筛选的候选；颜色表示最小间隙，黑边为所查候选中的非支配点。",fontsize=10.5)
fig.text(.065,.88,"下排：相同底盘参考下，保持上身与部分缓冲的实际配对；20 N·m仍为诊断条件，四次主对照均未触限。",fontsize=10.5)
norm=Normalize(0,45);cmap=plt.get_cmap("viridis")
for ax,name,title in zip(axs[0],["low","upper_bar","right_column"],["低障碍：上方空间允许较长缓冲","横挡：更柔和与更大间隙需要取舍","右立柱：可用缓冲缩小到毫米级"]):
    d=next(x for x in plan["decisions"] if x["scene"]["name"]==name)
    for x in d["candidates"]:
        if not x["geometry_admitted"]:continue
        pareto=x["id"] in d["sampled_nondominated_ids"]
        ax.scatter(x["buffer_stroke_m"]*1000,x["predicted_acceleration_mps2"],c=[1000*x["minimum_distance_m"]],
            cmap=cmap,norm=norm,s=34,edgecolors=".15" if pareto else "none",linewidths=.7,alpha=.85)
    for role,marker in [("held","s"),("buffer","*")]:
        x=d[role+"_probe"]
        if x:ax.scatter(1000*x["buffer_stroke_m"],x["predicted_acceleration_mps2"],marker=marker,s=130,
            facecolors="none",edgecolors="#c05045",linewidths=1.4)
    ax.set(title=title,xlabel="上身参考前伸量（mm）",ylabel="预测物体加速度峰值（m/s²）")
    ax.grid(alpha=.15)
cbax=fig.add_axes([.929,.585,.012,.23])
fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cbax,label="参考最小间隙（mm）")
axs[0,2].plot([],[],marker="s",mfc="none",mec="#c05045",ls="none",label="保持探针")
axs[0,2].plot([],[],marker="*",mfc="none",mec="#c05045",ls="none",label="部分缓冲探针")
axs[0,2].legend(frameon=False,fontsize=9,loc="upper right")
for scene,ax in [("upper_bar",axs[1,0]),("right_column",axs[1,1])]:
    for role,label,color in [("held","上身保持","#b65b4b"),("buffer","部分缓冲","#238675")]:
        with np.load(DEST/scene/role/"trace.npz") as z:
            a=z["data"];cols=z["columns"].tolist();t=a[:,0]
            v=a[:,[cols.index("object_v"+k+"_mps") for k in "xyz"]]
            ax.plot(t,np.linalg.norm(np.gradient(v,t,axis=0,edge_order=2),axis=1),color=color,lw=1.7,label=label)
    if scene=="right_column" and (DEST/scene/"buffer_halfdt/trace.npz").exists():
        with np.load(DEST/scene/"buffer_halfdt/trace.npz") as z:
            a=z["data"];cols=z["columns"].tolist();t=a[:,0]
            v=a[:,[cols.index("object_v"+k+"_mps") for k in "xyz"]]
            ax.plot(t,np.linalg.norm(np.gradient(v,t,axis=0,edge_order=2),axis=1),color=".4",ls="--",lw=1.,label="缓冲：半步长")
    ax.set(xlim=(0,.9),xlabel="触发后时间（s）",ylabel="实际加速度模长（m/s²）",
        title="横挡：约13.1 mm缓冲的实际效果" if scene=="upper_bar" else "右立柱：约4.47 mm缓冲的实际效果")
    ax.grid(alpha=.15);ax.legend(frameon=False)
for role,label,color in [("held","上身保持","#b65b4b"),("buffer","部分缓冲","#238675")]:
    with np.load(DEST/"right_column"/role/"grasp_trace.npz") as g:
        t=g["time"];moment=np.max(np.linalg.norm(g["contact_load_tcp_hand"][:,:,3:],axis=-1),axis=1)
        axs[1,2].plot(t,moment,color=color,lw=1.7,label=label)
if (DEST/"right_column/buffer_halfdt/grasp_trace.npz").exists():
    with np.load(DEST/"right_column/buffer_halfdt/grasp_trace.npz") as g:
        axs[1,2].plot(g["time"],np.max(np.linalg.norm(g["contact_load_tcp_hand"][:,:,3:],axis=-1),axis=1),color=".4",ls="--",lw=1.,label="缓冲：半步长")
axs[1,2].set(title="右立柱：手部接触力矩",xlim=(0,.9),xlabel="触发后时间（s）",ylabel="单手TCP力矩模长最大值（N·m）")
axs[1,2].grid(alpha=.15);axs[1,2].legend(frameon=False)
fig.text(.065,.09,"居中立柱：当前有限动作族内0个可行候选，未强行执行。部分缓冲探针按统一规则选取，尚非最终偏好或最优策略。",fontsize=10)
fig.text(.065,.045,"423次参考评估约48 s（含共享参考生成和加密核验），仍暂停仿真时钟；接触慢漂、实时接管与硬件能力尚未解决。",fontsize=10,color=".35")
fig.savefig(DEST/"joint_candidates_comparison.png",dpi=180);plt.close(fig)
print(DEST/"joint_candidates_comparison.png")
