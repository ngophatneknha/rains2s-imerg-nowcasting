import numpy as np, p0_lib as L, json
sv, st = L.load_seg('val'), L.load_seg('test'); nv, nt = len(sv)-8, len(st)-8
Pv, Pt = L.den(sv[5:5+nv]), L.den(st[5:5+nt]); out={}
for T in (0.5,1,2,5):
  for h in (1,2,3):
    Ov, Ot = L.obs_mm(sv,0,nv,h), L.obs_mm(st,0,nt,h)
    K=int((Ov>T).sum()); tau=float(np.partition(Pv.ravel(),-K)[-K])
    def st_(P,O,t):
        p,o=P>t,O>T; H=(p&o).sum(); M=(~p&o).sum(); F=(p&~o).sum(); return dict(CSI=H/(H+M+F),POD=H/(H+M),FAR=F/(H+F),Bias=(H+F)/(H+M))
    out[f'{T}|{h}']=dict(tau=tau,cal=st_(Pt,Ot,tau),raw=st_(Pt,Ot,T))
    print(T,h*30,'tau %.3f'%tau,{k:round(float(v),4) for k,v in out[f'{T}|{h}']['cal'].items()},'raw CSI %.4f bias %.2f'%(out[f'{T}|{h}']['raw']['CSI'],out[f'{T}|{h}']['raw']['Bias']))
json.dump(out,open('pers_cal.json','w'),default=float,indent=1)
