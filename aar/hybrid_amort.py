"""Hybrid amortized 4D: feed-forward init + few-step teacher optimization.

Exp-83/84 showed a single forward can't predict good 4D gaussians (grounded caps ~16.5;
free gaussians have no anchor and won't optimize when *predicted*). But per-scene *optimization*
of free gaussians reaches 21.94 (Exp-82). Hybrid = get the best of both:
  1) feed-forward GroundedDecoderK gives an INITIAL gaussian cloud (front-side structure, ~16.5),
  2) convert it to free nn.Parameters,
  3) let a few-step RetimeGS-teacher optimization place/extend them (incl. growing to the back).
Optimization (with per-gaussian gradients) does what prediction couldn't. We log the convergence
curve to see if the feed-forward init reaches per-scene quality FASTER than from scratch.
Usage: hybrid_amort.py --scene walk --teacher repro_4d_clean --ff_ckpt experiments/outputs/amort_mono_allview/student.pt
"""
import os, json, argparse, math, time, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from gsplat import rasterization
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/aar")
from teacher_retime import TeacherRetimeGS, load_cams
from exp20_distill import GroundedDecoderK, DIMS, PARAMS, cam_rays

ap = argparse.ArgumentParser(); ap.add_argument("--scene", default="walk")
ap.add_argument("--teacher", default="repro_4d_clean")
ap.add_argument("--ff_ckpt", default="experiments/outputs/amort_mono_allview/student.pt")
ap.add_argument("--iters", type=int, default=3000); ap.add_argument("--res", type=int, default=512)
ap.add_argument("--n_frames", type=int, default=9)       # full action = use all 33 frames (t axis becomes [0,1] over 33 timestamps)
ap.add_argument("--cpu_data", type=int, default=0)       # keep the RGB/mask tensors on CPU, stream one (cam,t) to GPU per step — needed for 33 frames on a 4GB GPU
ap.add_argument("--antifloat", type=int, default=0)      # aggressive floater removal: prune spatial outliers + oversized gaussians during densify
ap.add_argument("--tau_mul", type=float, default=0)      # scale the feed-forward init time-window (0 = auto 9/n_frames): narrower windows for more frames -> less temporal blur
ap.add_argument("--gh", type=int, default=64); ap.add_argument("--gw", type=int, default=88)
ap.add_argument("--K", type=int, default=2); ap.add_argument("--input_res", type=int, default=256)
ap.add_argument("--from_scratch", type=int, default=0)   # 1 = random init instead of feed-forward (baseline for speedup)
ap.add_argument("--densify", type=int, default=0)        # 1 = densify (clone high-grad) + prune (low-opacity) during optimization
ap.add_argument("--add_back", type=int, default=0)       # add N back-side gaussians to init (optimizer pulls them into place)
ap.add_argument("--teacher_init", type=int, default=0)   # 1 = back-side positions from teacher gaussians (plausible structure = Lyra proxy); 0 = random
ap.add_argument("--mirror_init", type=int, default=0)    # 1 = back-side positions by mirroring person FF gaussians through the person-centroid plane ⊥ cam0 view (DEPLOYABLE symmetry prior, no teacher)
ap.add_argument("--gt_supervise", type=int, default=0)   # 1 = optimize against REAL sparse observations (train cams × train frames, person matted on black) instead of teacher renders (DEPLOYABLE, no teacher)
ap.add_argument("--all_cam_gt", type=int, default=0)      # with gt_supervise: use ALL 32 cams × train frames (=160 imgs, the teacher's own training data / official protocol); eval 'both' then matches the teacher's condition
ap.add_argument("--pseudo_dir", default="")               # with gt_supervise: dir of RIFE pseudo-frames (matted, from TRAIN frames only — no held-data leakage) mixed into the target pool = deployable continuous-time supervision
ap.add_argument("--save_model", type=int, default=0)      # save optimized free-gaussian params to out/model_<tag>_<iters>.pt (for visualization)
ap.add_argument("--dump_init", default="")                # save the PRE-optimization init (front + mirror-back, activated attrs) for visualizing stage A/B
ap.add_argument("--load_init", default="")                # load a saved init dump (skip VGGT feed-forward + mirror) — for fast pareto sweeps
ap.add_argument("--n_sup", type=int, default=0)           # SPARSE-camera: supervise with only N uniformly-spaced cameras; the rest are held (novel) for eval. 0 = use the even/odd protocol
ap.add_argument("--ff_cams", default="0")                 # feed-forward init input view(s), comma-sep. Sparse-view sets these = the supervision cams
ap.add_argument("--ff_dtype", default="fp32")             # VGGT feed-forward precision: fp32 (best) | bf16 | fp16 — loaded low-precision on CPU FIRST so GPU never holds an fp32 copy (4GB-GPU deploy)
ap.add_argument("--ff_offload", type=int, default=0)      # CPU-offload VGGT layer-by-layer (fp32) → <4GB peak, ZERO precision loss, feed-forward pass slower
ap.add_argument("--prune_op", type=float, default=0.005)  # densify prune: opacity threshold (default = Exp-85c behavior)
ap.add_argument("--min_n", type=int, default=0)           # densify prune: capacity FLOOR — keep at least top-k gaussians by opacity (visual detail vs prune-regularization trade-off; Exp-92)
ap.add_argument("--max_n", type=int, default=0)           # densify CAP (0=off) — keep at most top-k by opacity; prevents gaussian explosion / OOM on unstable scenes (e.g. bear)
ap.add_argument("--densify_until", type=float, default=0.7)  # stop densify at this fraction of iters (lower=earlier); unstable scenes explode late, so stopping densify early keeps them from crashing
ap.add_argument("--seed", type=int, default=0)            # fix ALL randomness (mirror sampling, color init, densify) so a given config reproduces bit-for-bit — required for a credible paper
ap.add_argument("--flow", type=float, default=0.0)        # >0 = GaussianFlow RAFT optical-flow supervision on adjacent train frames -> constrains v1/v2/v3 motion -> better retiming (teacher uses --flow, we didn't; needs gt_supervise, no pseudo)
ap.add_argument("--ssim_w", type=float, default=0.0)      # D-SSIM weight (0=pure MSE=blurry; 0.2=standard 3DGS L1+SSIM -> structure/edges -> sharper)
ap.add_argument("--lpips", type=float, default=0.0)       # >0 = LPIPS(vgg) perceptual loss -> sharper fine detail/texture (teacher uses it; downsampled for speed)
ap.add_argument("--sil_w", type=float, default=0.0)       # penalise rendered coverage outside the foreground silhouette;
                                                          # the photometric target is black there, so dark gaussians are
                                                          # otherwise free and show up as haze on a non-black client
ap.add_argument("--aniso_w", type=float, default=0.0)     # penalty on near-degenerate (needle) gaussians: they face the
ap.add_argument("--aniso_r", type=float, default=10.0)    # training rig and render as dark spikes from free viewpoints
ap.add_argument("--lpips_every", type=int, default=1)     # compute LPIPS every K iters (K>1 = ~K x cheaper perceptual term -> faster HD recipe; periodic anti-floater nudge)
ap.add_argument("--eval_perceptual", type=int, default=0) # 1 = also report SSIM + LPIPS on held sets in final (for perceptual-metric tables)
ap.add_argument("--lr_decay", type=int, default=0)        # 1 = exp-decay all LRs over final 10% iters -> stable convergence -> sharper (Adam w/o decay keeps jittering -> blur)
ap.add_argument("--all_frames", type=int, default=0)      # 1 = supervise ALL frames (no held-time) = deployment/reconstruction demo -> clean video (matches other 4DGS demos = train-frame reconstruction, NOT held-time interpolation)
ap.add_argument("--data_root", default="data/sc_mv_all")  # capture root; a second dataset lives under its own root
ap.add_argument("--out", default=""); a = ap.parse_args(); dev = "cuda"
torch.manual_seed(a.seed); np.random.seed(a.seed); torch.cuda.manual_seed_all(a.seed)
cap_hits=[0]; n_peak=[0]                        # densify instrumentation: is the capacity cap ever reached?
SD = f"{ROOT}/{a.data_root}/{a.scene}"; out = a.out or f"{ROOT}/experiments/outputs/hybrid_{a.scene}"
os.makedirs(out, exist_ok=True)
frames = sorted([d for d in os.listdir(SD) if d.isdigit()])[:a.n_frames]; T = len(frames)

# ---- cameras, GT (eval), teacher ----
im0 = Image.open(f"{SD}/{frames[0]}/00.png"); W0,H0 = im0.size; sc = a.res/max(W0,H0)
Wr, Hr = int(round(W0*sc)), int(round(H0*sc))
vms, Ks = load_cams(f"{SD}/{frames[0]}/cameras.json", Wr, Hr, dev); ncam = len(vms)
ddev = "cpu" if a.cpu_data else dev                                             # RGB/mask on CPU for 33-frame / 4GB runs (stream one (cam,t) to GPU per use)
rgb = torch.zeros(ncam,T,Hr,Wr,3,device=ddev); msk = torch.zeros(ncam,T,Hr,Wr,device=ddev)
for ci in range(ncam):
    for ti,f in enumerate(frames):
        rgb[ci,ti]=torch.tensor(np.asarray(Image.open(f"{SD}/{f}/{ci:02d}.png").convert("RGB").resize((Wr,Hr)))/255.,dtype=torch.float32,device=ddev)
        mp=f"{SD}/{f}/masks/{ci:02d}.png"
        if os.path.exists(mp): msk[ci,ti]=torch.tensor((np.asarray(Image.open(mp).convert("L").resize((Wr,Hr),Image.NEAREST))>127).astype(np.float32),device=ddev)
tpath = f"{ROOT}/experiments/outputs/{a.teacher}/model_{a.scene}.pt"                # teacher only needed for teacher-sup or teacher_init;
tch = TeacherRetimeGS(tpath, dev) if ((not a.gt_supervise) or a.teacher_init or os.path.exists(tpath)) else None  # deploy config runs on teacher-less scenes
train_c=list(range(0,ncam,2)); held_c=list(range(1,ncam,2)); train_t=list(range(0,T,2)); held_t=list(range(1,T,2))
if a.all_frames: train_t=list(range(T))                                          # deployment/demo: supervise every frame -> clean video (held_t still eval'd but now leaked, so 'both' becomes a train-quality upper bound not a generalization number)
if a.n_sup and a.n_sup<ncam:                                                     # SPARSE-camera: N uniform supervision cams; the rest are held (true novel) views
    sup_c=sorted(set(np.linspace(0,ncam-1,a.n_sup).round().astype(int).tolist()))
    train_c=sup_c; held_c=[c for c in range(ncam) if c not in sup_c]
    if a.ff_cams=="0": a.ff_cams=",".join(map(str,sup_c))                        # sparse-view: feed-forward init from the same sparse input views
    print(f"[sparse-cam] supervise {len(sup_c)} cams {sup_c}; eval novel on {len(held_c)} UNsupervised cams; ff-init cams={a.ff_cams}")
pseudo_rgb=None; ptv=[]
if a.pseudo_dir:                                                             # RIFE pseudo-frames (already matted on black), dirs named t<global time>
    pts=sorted(d for d in os.listdir(a.pseudo_dir) if d.startswith("t")); ptv=[float(d[1:]) for d in pts]
    pseudo_rgb=torch.zeros(ncam,len(pts),Hr,Wr,3,device=ddev)
    for pi,d in enumerate(pts):
        for ci in range(ncam):
            pseudo_rgb[ci,pi]=torch.tensor(np.asarray(Image.open(f"{a.pseudo_dir}/{d}/{ci:02d}.png").convert("RGB").resize((Wr,Hr)))/255.,dtype=torch.float32,device=ddev)
    print(f"[hybrid] pseudo supervision: {len(pts)} RIFE times {['%.3f'%t for t in ptv]} x {ncam} cams")

# ---- feed-forward INITIAL gaussians (GroundedDecoderK on VGGT features of each input view) ----
def feed_forward_init():
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from exp20_distill import load_cameras
    bb = VGGTOmega().eval()                                                       # build on CPU
    bb.load_state_dict(torch.load(f"{ROOT}/hf_cache/models--facebook--VGGT-Omega/snapshots/05654241adc2f218dfb089c373a011f8a7040576/vggt_omega_1b_512.pt", map_location="cpu"))
    _dt={"fp16":torch.half,"bf16":torch.bfloat16}.get(a.ff_dtype)
    if a.ff_offload:                                                              # keep VGGT weights on CPU, stream layers to GPU during forward (fp32, <4GB peak)
        from accelerate import cpu_offload
        bb.aggregator = cpu_offload(bb.aggregator, execution_device=dev)
    else:
        if _dt: bb=bb.to(_dt)                                                     # cast on CPU BEFORE moving → GPU never holds an fp32 copy
        bb=bb.to(dev)
    dec=GroundedDecoderK(feat_dim=2048,K=a.K).to(dev); dec.load_state_dict(torch.load(a.ff_ckpt,map_location=dev))
    _,_,fullc = load_cameras(f"{SD}/{frames[0]}/cameras.json", Wr/W0)
    ff_cams=[int(c) for c in a.ff_cams.split(",")]                                # feed-forward init input view(s); sparse-view sets these = supervision cams
    gy,gx=torch.meshgrid(torch.linspace(-1,1,a.gh,device=dev),torch.linspace(-1,1,a.gw,device=dev),indexing="ij")
    G={k:[] for k in ["xyz","color","opacity","scale","quat","v1","v2","v3","mu_t","tau_l","tau_r","person"]}
    tvals=[t/(T-1) for t in range(T)]
    for cam in ff_cams:                                                          # each input view grounds its own gaussians (all in one world frame)
        names=[f"{SD}/{f}/{cam:02d}.png" for f in frames]
        imgs=load_and_preprocess_images(names, image_resolution=a.input_res).to(dev)
        if _dt: imgs=imgs.to(_dt)
        with torch.no_grad(): agg,ps = bb.aggregator(imgs[None])
        fin=agg[-1]; Cd=fin.shape[-1]; Ftt=fin.shape[1]; _,_,_,Hin,Win=imgs[None].shape
        patch=fin[0,:,ps:,:].float().reshape(Ftt,Hin//16,Win//16,Cd).permute(0,3,1,2)
        grid=torch.stack([gx,gy],-1)[None].expand(Ftt,-1,-1,-1)
        feat=F.grid_sample(patch,grid,align_corners=True).permute(0,2,3,1).reshape(Ftt,a.gh*a.gw,Cd).detach()
        ro,rd = cam_rays(fullc[cam], a.gh, a.gw, dev)                            # this view's rays for grounding
        with torch.no_grad():
            for fi in range(Ftt):
                r=dec(feat[fi]); base=F.softplus(r["depth"][...,0]+2.0)+0.3; depth=base.clone()
                for kk in range(1,a.K): depth[:,kk]=depth[:,kk-1]+F.softplus(r["depth"][:,kk,0])*0.2+0.02
                N,K=depth.shape; xyz=ro[None,None]+depth[...,None]*rd[:,None,:]
                G["xyz"].append(xyz.reshape(N*K,3)); G["color"].append(torch.sigmoid(r["color"]).reshape(N*K,3))
                G["opacity"].append(torch.sigmoid(r["opacity"]).reshape(N*K,1)); G["scale"].append((F.softplus(r["scale"])*0.02+1e-3).reshape(N*K,3))
                G["quat"].append(F.normalize(r["quat"],dim=-1).reshape(N*K,4))
                for kk in ("v1","v2","v3"): G[kk].append((torch.tanh(r[kk])*0.5).reshape(N*K,3))
                G["mu_t"].append(torch.full((N*K,1),tvals[fi],device=dev))
                _tm=a.tau_mul if a.tau_mul>0 else min(1.0,9.0/T)                 # narrower time window when more frames (auto 9/n_frames)
                G["tau_l"].append((F.softplus(r["tau_l"])*0.08*_tm+1e-2*_tm).reshape(N*K,1)); G["tau_r"].append((F.softplus(r["tau_r"])*0.08*_tm+1e-2*_tm).reshape(N*K,1))
                mp=f"{SD}/{frames[fi]}/masks/{cam:02d}.png"                       # person flag per grid pixel (this view's FG mask), for --mirror_init
                pm=torch.tensor((np.asarray(Image.open(mp).convert("L").resize((a.gw,a.gh),Image.NEAREST))>127).astype(np.float32),device=dev) if os.path.exists(mp) else torch.zeros(a.gh,a.gw,device=dev)
                G["person"].append(pm.reshape(-1).repeat_interleave(K))
    del bb; torch.cuda.empty_cache()
    return {k:torch.cat(v,0) for k,v in G.items()}

if a.load_init:                                                                 # fast path: reuse a saved front+mirror init (skip VGGT + mirror)
    g0 = {k: v.to(dev) for k, v in torch.load(a.load_init, map_location=dev).items() if k != "n_front"}
    print(f"[hybrid] {a.scene}: loaded init {g0['xyz'].shape[0]} gaussians from {os.path.basename(a.load_init)}")
    a.add_back = 0                                                               # already merged in the dump; skip the block below
g0 = g0 if a.load_init else feed_forward_init()
if a.add_back>0:                                                                 # add back-side gaussians (Lyra-completion analog)
    N2=a.add_back
    if a.mirror_init:                                                            # plausible back-side STRUCTURE by symmetry: mirror person FF gaussians per-frame through the centroid plane ⊥ cam0 view dir (deployable, no teacher)
        pf=g0["person"].bool(); Rw=vms[0][:3,:3]; cam_c=-(Rw.T@vms[0][:3,3])
        fwd=F.normalize(g0["xyz"][pf].mean(0)-cam_c,dim=0)                       # cam0→person direction (convention-safe view axis)
        parts=[]
        for tvv in sorted(set(g0["mu_t"][pf][:,0].tolist())):
            s=pf&(g0["mu_t"][:,0]==tvv); P=g0["xyz"][s]
            if P.shape[0]<10: continue
            cc=P.mean(0); parts.append(P-2*(((P-cc)@fwd)[:,None])*fwd[None])
        tm=torch.cat(parts,0); idx=torch.randint(0,tm.shape[0],(N2,),device=dev); bxyz=tm[idx]+torch.randn(N2,3,device=dev)*0.02
        print(f"[hybrid] mirror back-side: {int(pf.sum())} person gaussians -> {tm.shape[0]} mirrored -> {N2} sampled")
    elif a.teacher_init:                                                         # plausible back-side STRUCTURE from teacher positions (Lyra proxy)
        tm=tch.P["means"]; idx=torch.randint(0,tm.shape[0],(N2,),device=dev); bxyz=tm[idx]+torch.randn(N2,3,device=dev)*0.02
    else:                                                                        # random positions (no structure)
        ctr=g0["xyz"].mean(0); rad=(g0["xyz"].max(0).values-g0["xyz"].min(0).values).max()*0.6; bxyz=ctr+(torch.rand(N2,3,device=dev)-0.5)*2*rad
    g0["xyz"]=torch.cat([g0["xyz"],bxyz])
    g0["color"]=torch.cat([g0["color"],torch.rand(N2,3,device=dev)*0.4+0.3])
    g0["opacity"]=torch.cat([g0["opacity"],torch.full((N2,1),0.1,device=dev)])
    g0["scale"]=torch.cat([g0["scale"],torch.full((N2,3),0.02,device=dev)])
    q=torch.zeros(N2,4,device=dev); q[:,0]=1; g0["quat"]=torch.cat([g0["quat"],q])
    for kk in ("v1","v2","v3"): g0[kk]=torch.cat([g0[kk],torch.zeros(N2,3,device=dev)])
    g0["mu_t"]=torch.cat([g0["mu_t"],torch.full((N2,1),0.5,device=dev)])
    g0["tau_l"]=torch.cat([g0["tau_l"],torch.full((N2,1),0.5,device=dev)]); g0["tau_r"]=torch.cat([g0["tau_r"],torch.full((N2,1),0.5,device=dev)])
Ninit = g0["xyz"].shape[0]
print(f"[hybrid] {a.scene}: feed-forward init {Ninit} gaussians, teacher={a.teacher}, res {Wr}x{Hr}")
if a.dump_init:                                                                 # front = [0,n_front) with person flag; back(mirror) = [n_front,Ninit)
    torch.save({**{k: v.detach().cpu() for k, v in g0.items()}, "n_front": int(g0["person"].shape[0])}, a.dump_init)
    print(f"[dump_init] saved pre-optim init -> {a.dump_init} (n_front={int(g0['person'].shape[0])}, n_back={Ninit-int(g0['person'].shape[0])})")

# ---- free params (init from feed-forward, or random for the from-scratch baseline) ----
if a.from_scratch:
    lo=g0["xyz"].min(0).values-0.2; hi=g0["xyz"].max(0).values+0.2
    means=torch.nn.Parameter(torch.rand(Ninit,3,device=dev)*(hi-lo)+lo)
    scales=torch.nn.Parameter(torch.full((Ninit,3),math.log(0.02),device=dev))
    quats=torch.nn.Parameter(torch.cat([torch.ones(Ninit,1),torch.zeros(Ninit,3)],1).to(dev))
    opac=torch.nn.Parameter(torch.logit(torch.full((Ninit,),0.1,device=dev)))
    sh0=torch.nn.Parameter(torch.rand(Ninit,1,3,device=dev)*0.1)
    mu_t=torch.nn.Parameter(g0["mu_t"][:,0].clone()); log_tl=torch.nn.Parameter(torch.full((Ninit,),math.log(0.1),device=dev)); log_tr=torch.nn.Parameter(torch.full((Ninit,),math.log(0.1),device=dev))
    v1=torch.nn.Parameter(torch.zeros(Ninit,3,device=dev)); v2=torch.nn.Parameter(torch.zeros(Ninit,3,device=dev)); v3=torch.nn.Parameter(torch.zeros(Ninit,3,device=dev))
else:
    means=torch.nn.Parameter(g0["xyz"].clone()); scales=torch.nn.Parameter(torch.log(g0["scale"].clamp(min=1e-4)))
    quats=torch.nn.Parameter(g0["quat"].clone()); opac=torch.nn.Parameter(torch.logit(g0["opacity"][:,0].clamp(1e-3,1-1e-3)))
    sh0=torch.nn.Parameter(((g0["color"].clamp(0,1)-0.5)/0.28209).reshape(Ninit,1,3))               # inverse SH DC: color = 0.28209*sh0 + 0.5
    mu_t=torch.nn.Parameter(g0["mu_t"][:,0].clone()); log_tl=torch.nn.Parameter(torch.log(g0["tau_l"][:,0].clamp(min=1e-3))); log_tr=torch.nn.Parameter(torch.log(g0["tau_r"][:,0].clamp(min=1e-3)))
    v1=torch.nn.Parameter(g0["v1"].clone()); v2=torch.nn.Parameter(g0["v2"].clone()); v3=torch.nn.Parameter(g0["v3"].clone())
shN=torch.nn.Parameter(torch.zeros(Ninit,15,3,device=dev))
def make_opt():
    return torch.optim.Adam([{"params":[means],"lr":1.6e-4},{"params":[scales],"lr":5e-3},{"params":[quats],"lr":1e-3},
                      {"params":[opac],"lr":5e-2},{"params":[sh0],"lr":2.5e-3},{"params":[shN],"lr":1.25e-4},
                      {"params":[mu_t],"lr":1e-3},{"params":[log_tl],"lr":1e-2},{"params":[log_tr],"lr":1e-2},
                      {"params":[v1],"lr":1e-3},{"params":[v2],"lr":1e-3},{"params":[v3],"lr":1e-3}],eps=1e-15)
params=[means,scales,quats,opac,sh0,shN,mu_t,log_tl,log_tr,v1,v2,v3]
opt=make_opt()
GAMMA=0.005
def temporal_alpha(tv):
    tl=torch.exp(log_tl); tr=torch.exp(log_tr); le=mu_t-tl; re=mu_t+tr
    left=torch.where(le<0.02,torch.ones_like(le),torch.sigmoid((tv-le)/GAMMA)); right=torch.where(re>0.98,torch.ones_like(re),torch.sigmoid((re-tv)/GAMMA))
    return left*right
def render(ci,tv,ret_alpha=False):
    s=tv-0.5; pos=means+v1*s+v2*s*s+v3*s*s*s
    al=(torch.sigmoid(opac)*temporal_alpha(tv)).clamp(1e-4,1-1e-4)
    col,acc,_=rasterization(pos,F.normalize(quats,dim=-1),torch.exp(scales).clamp(max=0.06),al,torch.cat([sh0,shN],1),vms[ci][None],Ks[ci][None],Wr,Hr,sh_degree=3,render_mode="RGB")
    return (col[0].clamp(0,1), acc[0].clamp(0,1)) if ret_alpha else col[0].clamp(0,1)
def project(p,c):                                                                # world point -> pixel (gsplat viewmat+K convention), for flow supervision
    pc=p@vms[c][:3,:3].T+vms[c][:3,3]; z=pc[:,2].clamp(min=1e-4)
    return torch.stack([pc[:,0]/z*Ks[c][0,0]+Ks[c][0,2], pc[:,1]/z*Ks[c][1,1]+Ks[c][1,2]],-1)
def render_flow(c,ta,tb):                                                         # GaussianFlow: per-gaussian 2D screen displacement ta->tb, alpha-composited at ta
    sa=ta-0.5; sb=tb-0.5; pa=means+v1*sa+v2*sa*sa+v3*sa*sa*sa; pb=means+v1*sb+v2*sb*sb+v3*sb*sb*sb
    fg=project(pb,c)-project(pa,c); fg3=torch.cat([fg,torch.zeros_like(fg[:,:1])],-1)
    al=temporal_alpha(ta).clamp(1e-4,1-1e-4)
    out,_,_=rasterization(pa,F.normalize(quats,dim=-1),torch.exp(scales).clamp(max=0.06),al,fg3,vms[[c]],Ks[[c]],Wr,Hr,sh_degree=None,render_mode="RGB",packed=False)
    return out[0][...,:2]
def psnr(m): return -10*math.log10(m+1e-12)
def ssim(x,y):                                                                   # box-window SSIM (3DGS structure loss); x,y=[1,3,H,W]
    win=torch.ones(3,1,11,11,device=x.device)/121.0
    mx=F.conv2d(x,win,padding=5,groups=3); my=F.conv2d(y,win,padding=5,groups=3)
    mx2,my2,mxy=mx*mx,my*my,mx*my
    sx=F.conv2d(x*x,win,padding=5,groups=3)-mx2; sy=F.conv2d(y*y,win,padding=5,groups=3)-my2
    sxy=F.conv2d(x*y,win,padding=5,groups=3)-mxy
    c1,c2=0.01**2,0.03**2
    return (((2*mxy+c1)*(2*sxy+c2))/((mx2+my2+c1)*(sx+sy+c2))).mean()
def fg_psnr(cams,ts):
    es=[]
    with torch.no_grad():
        for ci in cams:
            for ti in ts:
                m=msk[ci,ti][...,None].to(dev)
                if m.sum()<10: continue
                es.append(((m*(render(ci,ti/(T-1))-rgb[ci,ti].to(dev))**2).sum()/(m.sum()*3+1e-6)).item())
    return psnr(float(np.mean(es))) if es else float('nan')

def fg_perceptual(cams,ts,lp):                                                    # SSIM + LPIPS on person-on-black held renders (perceptual-metric tables)
    ss,lps=[],[]
    with torch.no_grad():
        for ci in cams:
            for ti in ts:
                m=msk[ci,ti][...,None].to(dev)
                if m.sum()<10: continue
                prc=(render(ci,ti/(T-1))*m).permute(2,0,1)[None].clamp(0,1)
                gtc=(rgb[ci,ti].to(dev)*m).permute(2,0,1)[None].clamp(0,1)
                ss.append(float(ssim(prc,gtc)))
                if lp is not None:
                    rd=F.interpolate(prc,scale_factor=0.5,mode="bilinear",align_corners=False)*2-1
                    gl=F.interpolate(gtc,scale_factor=0.5,mode="bilinear",align_corners=False)*2-1
                    lps.append(float(lp(rd,gl).mean()))
    return (float(np.mean(ss)) if ss else float('nan'), float(np.mean(lps)) if lps else float('nan'))

tag = "from-scratch" if a.from_scratch else "feed-forward-init"
if a.densify: tag += "+densify" + (f"-minn{a.min_n//1000}k" if a.min_n else "") + (f"-pop{a.prune_op:g}" if a.prune_op!=0.005 else "")
if a.add_back: tag += ("+mirror-back" if a.mirror_init else ("+teacher-back" if a.teacher_init else "+rand-back"))
if a.gt_supervise: tag += (f"+gt-sup-{a.n_sup}cam" if a.n_sup else ("+gt-sup-allcam" if a.all_cam_gt else "+gt-sup"))
if a.pseudo_dir: tag += "+rife-" + os.path.basename(a.pseudo_dir.rstrip("/")).split("_")[-1]
init_ret, init_both = fg_psnr(train_c,held_t), fg_psnr(held_c,held_t)
curve=[{"it":0,"retiming":init_ret,"both":init_both}]
print(f"[hybrid] init eval ({tag}): retiming {init_ret:.2f} | both {init_both:.2f}")
print(f"[peakmem] feed-forward init + setup: {torch.cuda.max_memory_allocated()/1e9:.2f} GB"); torch.cuda.reset_peak_memory_stats()
flow_nb={}                                                                       # ti -> [(tn,{cam:flow[H,W,2]})] adjacent train frames, RAFT GT motion -> supervises v1/v2/v3
if a.flow>0 and a.gt_supervise:
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    raft=raft_large(weights=Raft_Large_Weights.DEFAULT,progress=False).to(dev).eval()
    cams_f=list(range(0,ncam,4)) if a.all_cam_gt else list(train_c)             # 8-cam subset for flow -> fast RAFT; motion constraint needs a few views, not all 32
    for ti in train_t:
        flow_nb[ti]=[]
        for tn in (ti-2,ti+2):
            if tn not in train_t: continue
            fc={}
            for c in cams_f:
                ia=rgb[c,ti].permute(2,0,1)[None].to(dev)*2-1; ib=rgb[c,tn].permute(2,0,1)[None].to(dev)*2-1
                Hp=(Hr+7)//8*8; Wp=(Wr+7)//8*8; ia=F.pad(ia,(0,Wp-Wr,0,Hp-Hr)); ib=F.pad(ib,(0,Wp-Wr,0,Hp-Hr))   # RAFT needs H,W divisible by 8
                with torch.no_grad(): fc[c]=raft(ia,ib)[-1][0,:,:Hr,:Wr].permute(1,2,0).cpu()   # [H,W,2] on CPU to save VRAM
            flow_nb[ti].append((tn,fc))
    del raft; torch.cuda.empty_cache()
    print(f"[flow] RAFT bidirectional: {sum(len(v) for v in flow_nb.values())} frame-pairs x {len(cams_f)} cams")
lpips_fn=None
if a.lpips>0:
    import lpips as _lp; lpips_fn=_lp.LPIPS(net="vgg",verbose=False).to(dev)
    for p in lpips_fn.parameters(): p.requires_grad_(False)
    print("[hq] LPIPS(vgg) perceptual loss enabled")
base_lrs=[g["lr"] for g in opt.param_groups]                                     # for lr_decay
grad_accum=torch.zeros(means.shape[0],device=dev); gc=0
t0=time.time()
for it in range(a.iters):
    if a.lr_decay and it>=int(a.iters*0.9):                                      # exp-decay LRs over final 10% -> sharper convergence (Adam w/o decay jitters -> blur)
        fdec=0.01**((it-int(a.iters*0.9))/max(a.iters-int(a.iters*0.9),1))
        for i,g in enumerate(opt.param_groups): g["lr"]=base_lrs[i]*fdec
    if a.gt_supervise:                                                           # DEPLOYABLE: fit the real sparse capture (train cams × train frames, person matted on black — same structure as teacher renders)
        cs = train_c if a.n_sup else (range(ncam) if a.all_cam_gt else train_c)  # sparse-cam supervises train_c(=sup_c) only
        ci = cs[np.random.randint(len(cs))]
        if ptv and np.random.rand() < len(ptv)/(len(ptv)+len(train_t)):          # uniform over the merged real+pseudo image pool
            pi = np.random.randint(len(ptv)); tv = ptv[pi]
            tgt = pseudo_rgb[ci,pi].to(dev); fg = (tgt.sum(-1,keepdim=True)>0.02).float()
        else:
            ti = train_t[np.random.randint(len(train_t))]
            tv = ti/(T-1); fg = msk[ci,ti][...,None].to(dev); tgt = rgb[ci,ti].to(dev)*fg
    else:
        ci = train_c[np.random.randint(len(train_c))] if np.random.rand()<0.6 else held_c[np.random.randint(len(held_c))]
        tv = float(np.random.rand())
        with torch.no_grad(): tgt = tch.render(vms[ci],Ks[ci],tv,Wr,Hr)
        fg=(tgt.sum(-1,keepdim=True)>0.02).float()
    w=(1+10.0*fg)
    if a.sil_w>0: pred,acc=render(ci,tv,ret_alpha=True)
    else: pred=render(ci,tv)
    l1=(w*(pred-tgt).abs()).sum()/(w.sum()*3)
    if a.sil_w>0: loss_sil=(acc*(1-fg)).sum()/((1-fg).sum()+1e-6)                                     # weighted L1 (fg-weighted); was pure MSE = blurry
    loss=l1+0.01*torch.sigmoid(opac).mean()+0.01*torch.exp(scales).clamp(max=0.06).mean()
    if a.sil_w>0: loss=loss+a.sil_w*loss_sil
    if a.aniso_w>0:                                                               # cap the log-ratio of the extreme axes
        _lr=scales.max(-1).values-scales.min(-1).values                           # scales are log-scale already
        loss=loss+a.aniso_w*F.relu(_lr-math.log(a.aniso_r)).mean()
    if a.ssim_w>0: loss=loss+a.ssim_w*(1-ssim(pred.permute(2,0,1)[None],tgt.permute(2,0,1)[None]))   # D-SSIM structure -> sharper edges
    if a.lpips>0 and lpips_fn is not None and it%a.lpips_every==0:                # LPIPS perceptual (downsampled for speed) -> sharper fine texture; every-K = faster HD recipe
        rd=F.interpolate(pred.permute(2,0,1)[None],scale_factor=0.5,mode="bilinear",align_corners=False).clamp(0,1)*2-1
        gl=F.interpolate(tgt.permute(2,0,1)[None],scale_factor=0.5,mode="bilinear",align_corners=False)*2-1
        loss=loss+a.lpips*lpips_fn(rd,gl).mean()
    if a.flow>0 and flow_nb and not a.pseudo_dir and it>=int(a.iters*0.7) and ti in flow_nb:   # GaussianFlow motion supervision, ONLY AFTER densify ends (flow grad pollutes densify's grad_accum -> gaussian explosion 22万->862万)
        for tn,fc in flow_nb[ti]:
            if ci not in fc: continue
            loss=loss+a.flow*(render_flow(ci,tv,tn/(T-1))-fc[ci].to(dev)).abs().mean()/Wr
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()
    if means.grad is not None:
        with torch.no_grad(): grad_accum=grad_accum+means.grad.norm(dim=-1); gc+=1
    if a.densify and 0<it<int(a.iters*a.densify_until) and it%300==0:            # clone high-grad (under-reconstructed) + prune low-opacity; densify_until<0.7 stops early to avoid late explosion
        with torch.no_grad():
            g=grad_accum/max(gc,1); op=torch.sigmoid(opac); scmax=torch.exp(scales).max(-1).values
            keep=op>a.prune_op
            if a.antifloat:                                                      # kill floaters: spatial outliers (far from the subject) + oversized gaussians
                c=means.median(0).values; dist=(means-c).norm(dim=-1)
                keep=keep&(dist<torch.quantile(dist,0.92)*1.6)&(scmax<0.05)
            if a.min_n and int(keep.sum())<a.min_n:                              # capacity floor: never prune below min_n (keep top-k by opacity)
                k=min(a.min_n,op.numel()); keep=torch.zeros_like(keep); keep[op.topk(k).indices]=True
            if a.max_n and int(keep.sum())>a.max_n:                              # capacity cap: prevent gaussian explosion/OOM (keep top-k by opacity)
                keep=torch.zeros_like(keep); keep[op.topk(a.max_n).indices]=True
                cap_hits[0]+=1                                                    # how often the cap actually binds
            n_peak[0]=max(n_peak[0], int(means.shape[0]))                         # largest field the run ever holds
            thr=torch.quantile(g[keep],0.85) if keep.sum()>10 else torch.tensor(1e9,device=dev)
            clone=keep&(g>thr)&(scmax<0.03)
            ik=keep.nonzero(as_tuple=True)[0]; ic=clone.nonzero(as_tuple=True)[0]
            def c2(t): return torch.cat([t[ik],t[ic]],0).clone()
            nm=c2(means); nm[len(ik):]+=torch.randn(len(ic),3,device=dev)*0.01
            ns,nq,no=c2(scales),c2(quats),c2(opac); ns0,nsN=c2(sh0),c2(shN)
            nmt,ntl,ntr=c2(mu_t),c2(log_tl),c2(log_tr); nv1,nv2,nv3=c2(v1),c2(v2),c2(v3)
        means=torch.nn.Parameter(nm); scales=torch.nn.Parameter(ns); quats=torch.nn.Parameter(nq); opac=torch.nn.Parameter(no)
        sh0=torch.nn.Parameter(ns0); shN=torch.nn.Parameter(nsN); mu_t=torch.nn.Parameter(nmt); log_tl=torch.nn.Parameter(ntl); log_tr=torch.nn.Parameter(ntr)
        v1=torch.nn.Parameter(nv1); v2=torch.nn.Parameter(nv2); v3=torch.nn.Parameter(nv3)
        params=[means,scales,quats,opac,sh0,shN,mu_t,log_tl,log_tr,v1,v2,v3]; opt=make_opt()
        grad_accum=torch.zeros(means.shape[0],device=dev); gc=0
        torch.cuda.empty_cache()          # densify rebuilds params+Adam each time; without this the dead
                                          # blocks accumulate as allocator cache (~1GB/min late) -> OOM when
                                          # the GPU is shared (leaked 6.2GB ctx + 6.4GB root job)
    if it in (250,500,1000,2000,3000) or (it>3000 and it%2000==0) or it==a.iters-1:
        torch.cuda.empty_cache()                                                  # keep reserved flat on a shared GPU
        r_,n_,b_ = fg_psnr(train_c,held_t), fg_psnr(held_c,train_t), fg_psnr(held_c,held_t)
        curve.append({"it":it+1 if it==a.iters-1 else it,"retiming":r_,"novel":n_,"both":b_})
        print(f"[it {it:4d}] {time.time()-t0:.0f}s N={means.shape[0]} | retiming {r_:.2f} | novel {n_:.2f} | both {b_:.2f}")
n_peak[0]=max(n_peak[0], int(means.shape[0]))
final={"train":fg_psnr(train_c,train_t),"novel":fg_psnr(held_c,train_t),"retiming":fg_psnr(train_c,held_t),"both":fg_psnr(held_c,held_t)}
if a.eval_perceptual:                                                             # add SSIM+LPIPS on held sets (both = held-view x held-time; novel = held-view x seen-time)
    _lp=lpips_fn
    if _lp is None:
        import lpips as _lpm; _lp=_lpm.LPIPS(net="vgg",verbose=False).to(dev)
        for p in _lp.parameters(): p.requires_grad_(False)
    b_ss,b_lp=fg_perceptual(held_c,held_t,_lp); n_ss,n_lp=fg_perceptual(held_c,train_t,_lp)
    final.update({"both_ssim":b_ss,"both_lpips":b_lp,"novel_ssim":n_ss,"novel_lpips":n_lp})
    print(f"[perceptual] both SSIM {b_ss:.4f} LPIPS {b_lp:.4f} | novel SSIM {n_ss:.4f} LPIPS {n_lp:.4f}")
json.dump({"scene":a.scene,"tag":tag,"args":vars(a),"N_final":int(means.shape[0]),"N_peak":n_peak[0],"cap_hits":cap_hits[0],"opt_sec":round(time.time()-t0,1),"curve":curve,"final":final},
          open(f"{out}/result_{tag.replace('+','_')}_{a.iters}.json","w"),indent=1)
print(f"[RESULT] {a.scene} hybrid({tag}) | train {final['train']:.2f} | novel-view {final['novel']:.2f} | retiming {final['retiming']:.2f} | both {final['both']:.2f}")
print(f"[peakmem] few-step optimization: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
if a.save_model:
    torch.save({k:v.detach().cpu() for k,v in zip(["means","scales","quats","opac","sh0","shN","mu_t","log_tl","log_tr","v1","v2","v3"],
                [means,scales,quats,opac,sh0,shN,mu_t,log_tl,log_tr,v1,v2,v3])},
               f"{out}/model_{tag.replace('+','_')}_{a.iters}.pt")
    print(f"[saved] {out}/model_{tag.replace('+','_')}_{a.iters}.pt (N={means.shape[0]})")
