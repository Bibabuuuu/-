"""Plot saved matched-space comparison; never run physics here."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/"outputs/rby1-research/s3-upper-buffer"
plt.rcParams.update({"font.family":FontProperties(fname="C:/Windows/Fonts/msyh.ttc").get_name(),
    "axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False,"font.size":10})
c=json.loads((DEST/"contract.json").read_text())
fig,axs=plt.subplots(2,3,figsize=(14,8.7))
fig.subplots_adjust(left=.07,right=.98,bottom=.15,top=.82,hspace=.4,wspace=.32)
fig.suptitle("同一低障碍场景：腰臂缓冲减少负荷，但没有消除停车慢漂",fontsize=18,y=.98)
fig.text(.07,.915,"同初态约0.4 m/s、0.5 kg；同20 N·m诊断轮上限（均未触限）；障碍、接触和控制增益一致。",fontsize=10.5)
items=[("fast_held","底盘快停，上身保持", "#b65b4b"),
       ("soft_held","底盘柔和停，接近空间边界", "#3c70a3"),
       ("fast_buffer","底盘快停＋腰臂缓冲", "#238675")]
for name,label,color in items:
    with np.load(DEST/name/"trace.npz") as z:
        a=z["data"]; cols=z["columns"].tolist(); t=a[:,0]
        v=a[:,[cols.index("object_v"+k+"_mps") for k in "xyz"]]
        acceleration=np.linalg.norm(np.gradient(v,t,axis=0,edge_order=2),axis=1)
        axle=a[:,cols.index("axle_x_m")]; obj=a[:,cols.index("object_com_x_m")]
        axs[0,0].plot(t,acceleration,color=color,lw=1.6,label=label)
        axs[1,0].plot(t,1000*(axle-axle[0]),color=color,lw=1.6)
        axs[1,1].plot(t,1000*(obj-obj[0]),color=color,lw=1.6)
    with np.load(DEST/name/"grasp_trace.npz") as z:
        moment=np.max(np.linalg.norm(z["contact_load_tcp_hand"][:,:,3:],axis=-1),axis=1)
        rp=z["relative_position"]
        relative=1000*np.max(np.linalg.norm(rp-rp[0],axis=-1),axis=1)
        axs[0,1].plot(t,moment,color=color,lw=1.6)
        axs[0,2].plot(t,relative,color=color,lw=1.6)
        start=c["common_observation_start_s"]; mask=t>=start
        anchor=np.array([[np.interp(start,t,rp[:,s,k]) for k in range(3)] for s in range(2)])
        drift=1000*np.max(np.linalg.norm(rp[mask]-anchor,axis=-1),axis=1)
        axs[1,2].plot(t[mask]-start,drift,color=color,lw=1.6,ls="--" if name=="fast_buffer" else "-")
titles=["物体加速度峰值：1.003 / 0.589 / 0.502", "手部接触力矩随之降低", "相对位移：未出现同等幅度改善",
        "底盘行进约120 / 160 / 120 mm", "物体行进约120 / 160 / 180 mm", "共同停车窗：三条慢漂曲线近乎重合"]
ys=["加速度模长（m/s²）","单手TCP力矩模长最大值（N·m）","相对初始的手—物位置变化（mm）",
    "底盘前进量（mm）","物体质心前进量（mm）","相对窗口起点的位置变化（mm）"]
for i,ax in enumerate(axs.flat):
    ax.set(title=titles[i],ylabel=ys[i],xlabel="共同停车窗内时间（s）" if i==5 else "触发后时间（s）")
    ax.grid(alpha=.18)
    if i<5: ax.set_xlim(0,1.9); ax.axvline(c["common_observation_start_s"],color=".55",ls=":",lw=.7)
    else: ax.set_xlim(0,1)
axs[1,0].axhline(160,color=".45",ls=":",lw=.8)
handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,frameon=False,loc="upper center",bbox_to_anchor=(.52,.895))
fig.text(.07,.075,"缓冲代价：上身相对底盘多前伸约60 mm、物体参考晚停约0.30 s；相较柔和底盘对照，物体多前进约20 mm。",fontsize=10)
fig.text(.07,.035,"当前是已知障碍下的离线动作对照。相对位姿与接触切向路程受接触数值模型影响，不能直接当真实滑移或硬件性能。",fontsize=10,color=".35")
fig.savefig(DEST/"upper_buffer_comparison.png",dpi=180)
plt.close(fig)
print(DEST/"upper_buffer_comparison.png")
