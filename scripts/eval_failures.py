"""Sort SmolVLA eval episodes into outcome categories from the stored qpos (eval_traj.npz). Usage: python scripts/eval_failures.py <dir with eval.json + eval_traj.npz> ..."""
import json,sys,numpy as np,mujoco
sys.path.insert(0,'src')
from rohub.scene import load, CUBE_HALF
from rohub.pick_expert import in_bowl, TARGET_XY
from rohub.scene import BOWL_R
scene=load(task="pick"); m,d=scene.model,scene.data
sid=m.site("gripperframe").id
def analyse(dirp):
    ev=json.load(open(dirp+'/eval.json')); tr=np.load(dirp+'/eval_traj.npz')
    cats={}; rows=[]
    for i,e in enumerate(ev['episodes']):
        q=tr[str(e['seed'])]
        cz=q[:,-5]; cxy=q[:,-7:-5]
        start=np.array(e['cube_xy'])
        mind=9; tmin=None; gz=[]
        for t,qq in enumerate(q):
            d.qpos[:]=qq; mujoco.mj_kinematics(m,d)
            dd=np.linalg.norm(d.site_xpos[sid]-qq[-7:-4])
            if dd<mind: mind, tmin = dd, t
        lifted=cz.max()>CUBE_HALF+0.02
        disp=np.linalg.norm(cxy[-1]-start)
        dbowl=np.linalg.norm(cxy[-1]-TARGET_XY)
        if e['success']: c='success'
        elif not lifted and disp<0.01: c='never moved cube'
        elif not lifted: c='pushed, no lift'
        elif dbowl<BOWL_R+0.03: c='lifted, ended on/at bowl rim'
        elif cz[-1]>CUBE_HALF+0.01: c='lifted, still held at timeout'
        else: c='lifted, dropped away from bowl'
        cats[c]=cats.get(c,0)+1
        rows.append((e['seed'],i%4,c,round(mind*100,1),round(float(cz.max()-CUBE_HALF)*100,1),round(dbowl*100,1),e['frames']))
    return cats,rows
for p in sys.argv[1:]:
    cats,rows=analyse(p); print(p,cats)
    per={}
    for r in rows: per.setdefault(r[1],[0,0]); per[r[1]][0]+=r[2]=='success'; per[r[1]][1]+=1
    print(' success by anchor clip (s%4):',per)
    import statistics
    for c in set(r[2] for r in rows):
        rr=[r for r in rows if r[2]==c]
        print(f'  {c}: n={len(rr)} median min gripper-cube dist {statistics.median(r[3] for r in rr)} cm, median max lift {statistics.median(r[4] for r in rr)} cm, median frames {statistics.median(r[6] for r in rr)}')
