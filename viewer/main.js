/* Minimal 4D Gaussian-splat renderer (WebGL2) with WebXR.
 * Decodes OUR field format in-shader: cubic per-splat trajectory + temporal-opacity window.
 * Splat data lives in one RGBA32F texture (6 texels/splat); per frame we upload only the
 * depth-sorted index buffer (u32/instance). FPS telemetry POSTs to /telemetry every 2 s. */
"use strict";
const canvas = document.getElementById("c");
const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, xrCompatible: true, preserveDrawingBuffer: true });
const hudFps = document.getElementById("fps"), hudInfo = document.getElementById("info");
const Q = new URLSearchParams(location.search);
const FIELD = Q.get("f") || "field20.bin";
const SRC = Q.get("src") || "";                          // e.g. http://<LAN-IP>:8123 -> true wireless path
const LOOP_S = 3.0;
const TFIX = parseFloat(new URLSearchParams(location.search).get("t") || "NaN");                                        // clip loop length (s)

// ---------------- shaders ----------------

function buildShaders(){
  const vs = `#version 300 es
  precision highp float; precision highp int; precision highp sampler2D;
  layout(location=0) in vec2 corner;
  layout(location=1) in uint sid;
  uniform sampler2D uData;
  uniform mat4 uView, uProj;
  uniform vec2 uFocal, uViewport;
  uniform float uT;
  out vec4 vColor; out vec2 vPos;
  vec4 tx(int lin){ return texelFetch(uData, ivec2(lin % 1024, lin / 1024), 0); }
  void main(){
    int base = int(sid)*6;
    vec4 A = tx(base+0);   // pos.xyz, v1.x
    vec4 B = tx(base+1);   // v1.yz, v2.xy
    vec4 C = tx(base+2);   // v2.z, v3.xyz
    vec4 D = tx(base+3);   // scale.xyz, le
    vec4 E = tx(base+4);   // re, rgb
    vec4 F = tx(base+5);   // alpha, quat.xyz (w reconstructed)  -- see loader
    float s = uT - 0.5;
    vec3 p = A.xyz + vec3(A.w,B.x,B.y)*s + vec3(B.z,B.w,C.x)*(s*s) + vec3(C.y,C.z,C.w)*(s*s*s);
    float gate = 1.0/(1.0+exp(-(uT-D.w)/0.005)) * (1.0/(1.0+exp(-(E.x-uT)/0.005)));
    float alpha = F.x * gate;
    vec4 cam = uView * vec4(p,1.0);
    if (cam.z > -0.05 || alpha < 0.004){ gl_Position = vec4(0,0,2,1); vColor=vec4(0); vPos=corner; return; }
    vec4 clip = uProj * cam;
    vec3 qv = F.yzw; float qw = sqrt(max(0.0, 1.0 - dot(qv,qv)));
    float x=qv.x, y=qv.y, z=qv.z, w=qw;
    mat3 R = mat3(1.-2.*(y*y+z*z), 2.*(x*y+w*z), 2.*(x*z-w*y),
                  2.*(x*y-w*z), 1.-2.*(x*x+z*z), 2.*(y*z+w*x),
                  2.*(x*z+w*y), 2.*(y*z-w*x), 1.-2.*(x*x+y*y));
    mat3 S = mat3(D.x,0,0, 0,D.y,0, 0,0,D.z);
    mat3 MV = mat3(uView) * (R * S);
    mat3 cov = MV * transpose(MV);
    float iz = 1.0/-cam.z;
    vec2 f = uFocal;
    float Jx = f.x*iz, Jy = f.y*iz;
    float Jxz = f.x*cam.x*iz*iz, Jyz = f.y*cam.y*iz*iz;
    // 2x3 J applied to cov (cov is symmetric): c2 = J cov J^T
    float c00 = Jx*Jx*cov[0][0] + 2.0*Jx*Jxz*cov[0][2] + Jxz*Jxz*cov[2][2];
    float c01 = Jx*Jy*cov[0][1] + Jx*Jyz*cov[0][2] + Jxz*Jy*cov[1][2] + Jxz*Jyz*cov[2][2];
    float c11 = Jy*Jy*cov[1][1] + 2.0*Jy*Jyz*cov[1][2] + Jyz*Jyz*cov[2][2];
    float a = c00+0.3, b = c01, d = c11+0.3;
    float tr = a+d, det = a*d-b*b;
    float disc = sqrt(max(0.0, 0.25*tr*tr - det));
    float l1 = 0.5*tr + disc, l2 = 0.5*tr - disc;
    vec2 e1 = (abs(b) < 1e-6) ? vec2(1,0) : normalize(vec2(b, l1-a));
    vec2 e2 = vec2(-e1.y, e1.x);
    vec2 off = corner.x * e1 * sqrt(max(l1,0.0)) + corner.y * e2 * sqrt(max(l2,0.0));
    gl_Position = vec4(clip.xy/clip.w + 2.0*off/uViewport, clip.z/clip.w, 1.0);
    vColor = vec4(E.yzw * alpha, alpha);   // premultiplied
    vPos = corner;
  }`;
  const fs = `#version 300 es
  precision highp float;
  in vec4 vColor; in vec2 vPos; out vec4 o;
  void main(){
    float r2 = dot(vPos, vPos);
    float g = exp(-r2);                    // corner=±2 spans 2σ; standard splat falloff (was *4 -> spiky)
    if (g < 0.006) discard;
    o = vColor * g;
  }`;
  return [vs, fs];
}

function compile(type, src){
  const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}

// ---------------- load field ----------------
let N = 0, splat;                          // splat: Float32Array view (20 floats/splat as packed)
let px, py, pz, v1x,v1y,v1z,v2x,v2y,v2z,v3x,v3y,v3z;  // SoA copies for fast sorting
async function loadField(){
  const url = SRC ? (SRC.replace(/\/$/,"") + "/" + FIELD) : FIELD;
  const tDl0 = performance.now();
  const buf = await (await fetch(url, {cache:"no-store"})).arrayBuffer();
  const dlMs = performance.now() - tDl0;
  const mbps = buf.byteLength * 8 / 1e6 / (dlMs / 1000);
  fetch("/telemetry", { method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({ kind:"transfer", field: FIELD, src: SRC || "same-origin",
      bytes: buf.byteLength, ms: +dlMs.toFixed(1), mbps: +mbps.toFixed(2) }) }).catch(()=>{});
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== 0x34444753) throw new Error("bad magic");
  N = dv.getUint32(4, true);
  const raw = new Uint8Array(buf, 8);
  // repack into the texture layout: 6 RGBA32F texels/splat
  const H = Math.ceil(N * 6 / 1024);
  const tex = new Float32Array(1024 * H * 4);
  px=new Float32Array(N); py=new Float32Array(N); pz=new Float32Array(N);
  v1x=new Float32Array(N); v1y=new Float32Array(N); v1z=new Float32Array(N);
  v2x=new Float32Array(N); v2y=new Float32Array(N); v2z=new Float32Array(N);
  v3x=new Float32Array(N); v3y=new Float32Array(N); v3z=new Float32Array(N);
  const f = new Float32Array(raw.buffer, raw.byteOffset, N*20);
  const u = raw;
  for (let i=0;i<N;i++){
    const o = i*20, ob = i*80, t = i*24;
    const P=[f[o],f[o+1],f[o+2]], V1=[f[o+3],f[o+4],f[o+5]], V2=[f[o+6],f[o+7],f[o+8]], V3=[f[o+9],f[o+10],f[o+11]];
    const S=[f[o+12],f[o+13],f[o+14]], le=f[o+15], re=f[o+16];
    const r=u[ob+72]/255, g_=u[ob+73]/255, b=u[ob+74]/255, al=u[ob+75]/255;
    let qx=u[ob+76]/255*2-1, qy=u[ob+77]/255*2-1, qz=u[ob+78]/255*2-1, qw=u[ob+79]/255*2-1;
    if (qw < 0){ qx=-qx; qy=-qy; qz=-qz; qw=-qw; }        // canonical w>=0 so shader can reconstruct
    const n = Math.hypot(qx,qy,qz,qw)||1; qx/=n; qy/=n; qz/=n;
    tex.set([P[0],P[1],P[2],V1[0],  V1[1],V1[2],V2[0],V2[1],  V2[2],V3[0],V3[1],V3[2],
             S[0],S[1],S[2],le,     re,r,g_,b,               al,qx,qy,qz], t);
    px[i]=P[0]; py[i]=P[1]; pz[i]=P[2];
    v1x[i]=V1[0]; v1y[i]=V1[1]; v1z[i]=V1[2];
    v2x[i]=V2[0]; v2y[i]=V2[1]; v2z[i]=V2[2];
    v3x[i]=V3[0]; v3y[i]=V3[1]; v3z[i]=V3[2];
  }
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 1024, H, 0, gl.RGBA, gl.FLOAT, tex);
  hudInfo.textContent = `${N.toLocaleString()} splats — ${FIELD}`;
  return t;
}

// ---------------- sorting (counting sort on quantized depth) ----------------
const order = { buf: null, idx: null };
function sortSplats(view, tNow){
  if (!order.idx){ order.idx = new Uint32Array(N); }
  const s = tNow - 0.5, s2 = s*s, s3 = s2*s;
  const zk = new Float32Array(N);
  const r20=view[2], r21=view[6], r22=view[10], r23=view[14];   // view-space z row (column-major)
  let zmin=Infinity, zmax=-Infinity;
  for (let i=0;i<N;i++){
    const x = px[i]+v1x[i]*s+v2x[i]*s2+v3x[i]*s3;
    const y = py[i]+v1y[i]*s+v2y[i]*s2+v3y[i]*s3;
    const z = pz[i]+v1z[i]*s+v2z[i]*s2+v3z[i]*s3;
    const zc = r20*x + r21*y + r22*z + r23;
    zk[i] = zc; if (zc<zmin) zmin=zc; if (zc>zmax) zmax=zc;
  }
  const B = 4096, cnt = new Uint32Array(B+1);
  const sc = (B-1)/Math.max(1e-6, zmax-zmin);
  const bins = new Uint16Array(N);
  for (let i=0;i<N;i++){ const b = ((zk[i]-zmin)*sc)|0; bins[i]=b; cnt[b+1]++; }
  for (let b=0;b<B;b++) cnt[b+1]+=cnt[b];
  for (let i=0;i<N;i++){ order.idx[cnt[bins[i]]++] = i; }   // ascending z (most negative = farthest first)
  return order.idx;
}

// ---------------- GL setup ----------------
let prog, vao, idxBuf, dataTex, uni = {};
async function initGL(){
  const [vsS, fsS] = buildShaders();
  prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, vsS));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fsS));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  for (const k of ["uData","uView","uProj","uFocal","uViewport","uT"]) uni[k] = gl.getUniformLocation(prog, k);
  dataTex = await loadField();
  vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  const quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-2,-2, 2,-2, -2,2, 2,2]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  idxBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, idxBuf);
  gl.bufferData(gl.ARRAY_BUFFER, N*4, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(1); gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_INT, 0, 0);
  gl.vertexAttribDivisor(1, 1);
  gl.disable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);            // premultiplied, back-to-front
}

function drawScene(view, proj, vpW, vpH, tNow, doSort){
  if (doSort){
    const idx = sortSplats(view, tNow);
    gl.bindBuffer(gl.ARRAY_BUFFER, idxBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, idx);
  }
  gl.useProgram(prog); gl.bindVertexArray(vao);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, dataTex);
  gl.uniform1i(uni.uData, 0);
  gl.uniformMatrix4fv(uni.uView, false, view);
  gl.uniformMatrix4fv(uni.uProj, false, proj);
  const fy = proj[5] * vpH / 2, fx = proj[0] * vpW / 2;
  gl.uniform2f(uni.uFocal, fx, fy);
  gl.uniform2f(uni.uViewport, vpW, vpH);
  gl.uniform1f(uni.uT, tNow);
  gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, N);
}

// ---------------- telemetry ----------------
let frames = 0, lastPost = performance.now(), mode = "flat";
function tickFps(){
  frames++;
  const now = performance.now();
  if (now - lastPost > 2000){
    const fps = frames * 1000 / (now - lastPost);
    hudFps.textContent = `${fps.toFixed(1)} fps (${mode})`;
    fetch("/telemetry", { method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ fps: +fps.toFixed(2), mode, n: N, field: FIELD, ua: "q" }) }).catch(()=>{});
    frames = 0; lastPost = now;
  }
}

// ---------------- flat (non-XR) loop: orbit camera ----------------
let theta = 0.3, phi = 0.15, dist = 3.2, center = null, dragging = false, lx=0, ly=0;
canvas.addEventListener("pointerdown", e=>{dragging=true;lx=e.clientX;ly=e.clientY});
addEventListener("pointerup", ()=>dragging=false);
addEventListener("pointermove", e=>{ if(dragging){ theta += (e.clientX-lx)*0.005; phi += (e.clientY-ly)*0.005;
  phi = Math.max(-1.3, Math.min(1.3, phi)); lx=e.clientX; ly=e.clientY; }});
addEventListener("wheel", e=>{ dist *= (1 + Math.sign(e.deltaY)*0.08); dist = Math.max(0.8, Math.min(10, dist)); });

function centerOfField(){
  const c = Q.get("c");                                  // ?c=x,y,z pins the orbit centre so that packages pruned
  if (c) { const v = c.split(",").map(parseFloat);        // to different sizes are viewed from the same camera
           if (v.length === 3 && v.every(x => !isNaN(x))) return v; }
  let cx=0, cy=0, cz=0; const step = Math.max(1, N>>12);
  let m=0; for (let i=0;i<N;i+=step){ cx+=px[i]; cy+=py[i]; cz+=pz[i]; m++; }
  return [cx/m, cy/m, cz/m];
}
function lookAtView(){
  const [cx,cy,cz] = center;
  const ex = cx + dist*Math.cos(phi)*Math.sin(theta);
  const ey = cy + dist*Math.sin(phi);
  const ez = cz + dist*Math.cos(phi)*Math.cos(theta);
  const f = norm3([cx-ex, cy-ey, cz-ez]);
  const up = [0,-1,0];                                    // dataset is y-down
  const r = norm3(cross(f, up)); const u = cross(r, f);
  return new Float32Array([ r[0],u[0],-f[0],0, r[1],u[1],-f[1],0, r[2],u[2],-f[2],0,
    -(r[0]*ex+r[1]*ey+r[2]*ez), -(u[0]*ex+u[1]*ey+u[2]*ez), (f[0]*ex+f[1]*ey+f[2]*ez), 1 ]);
}
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const norm3=a=>{const l=Math.hypot(...a)||1; return [a[0]/l,a[1]/l,a[2]/l];};
function perspective(fovy, asp, near, far){
  const t = 1/Math.tan(fovy/2);
  return new Float32Array([t/asp,0,0,0, 0,t,0,0, 0,0,(far+near)/(near-far),-1, 0,0,2*far*near/(near-far),0]);
}

let t0 = performance.now(), frameNo = 0;
function flatLoop(){
  if (xrSession) return;                                   // XR takes over
  const w = canvas.clientWidth*devicePixelRatio, h = canvas.clientHeight*devicePixelRatio;
  if (canvas.width!==w||canvas.height!==h){ canvas.width=w; canvas.height=h; }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0,0,w,h); gl.clearColor(1,1,1,1); gl.clear(gl.COLOR_BUFFER_BIT);
  const tNow = isNaN(TFIX) ? ((performance.now()-t0)/1000 % LOOP_S) / LOOP_S : TFIX;
  drawScene(lookAtView(), perspective(1.05, w/h, 0.05, 100), w, h, tNow, (frameNo++%2)===0);
  tickFps();
  requestAnimationFrame(flatLoop);
}

// ---------------- WebXR ----------------
let xrSession = null;
async function enterVR(){
  xrSession = await navigator.xr.requestSession("immersive-vr", { optionalFeatures: ["local-floor"] });
  mode = "vr";
  await gl.makeXRCompatible();
  const layer = new XRWebGLLayer(xrSession, gl);
  xrSession.updateRenderState({ baseLayer: layer });
  const space = await xrSession.requestReferenceSpace("local-floor").catch(()=>xrSession.requestReferenceSpace("local"));
  const [cx,cy,cz] = center;
  xrSession.addEventListener("end", ()=>{ xrSession=null; mode="flat"; requestAnimationFrame(flatLoop); });
  const onFrame = (t, frame)=>{
    xrSession.requestAnimationFrame(onFrame);
    const pose = frame.getViewerPose(space); if (!pose) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    gl.clearColor(0,0,0,1); gl.clear(gl.COLOR_BUFFER_BIT);
    const tNow = ((performance.now()-t0)/1000 % LOOP_S) / LOOP_S;
    let first = true;
    for (const view of pose.views){
      const vp = layer.getViewport(view);
      gl.viewport(vp.x, vp.y, vp.width, vp.height);
      // world transform: put the subject 1.6 m in front of the viewer, y-down data -> flip
      const V = new Float32Array(view.transform.inverse.matrix);
      const model = new Float32Array([ -1,0,0,0, 0,-1,0,0, 0,0,1,0, cx, cy+1.3, cz-1.6, 1 ]); // mirror+lift
      // view * modelAdjust: we bake the adjust by translating the eye instead (cheap): V' = V * T
      const Vm = mul4(V, modelFix(cx, cy, cz));
      drawScene(Vm, new Float32Array(view.projectionMatrix), vp.width, vp.height, tNow, first);
      first = false;
    }
    tickFps();
  };
  xrSession.requestAnimationFrame(onFrame);
}
function modelFix(cx, cy, cz){
  // dataset is y-down, meters-ish; rotate 180° about X (flip y,z), then place center at (0,1.3,-1.8)
  const R = [1,0,0,0, 0,-1,0,0, 0,0,-1,0, 0,0,0,1];
  const T = [1,0,0,0, 0,1,0,0, 0,0,1,0, -cx,-cy,-cz,1];    // move center to origin (pre-rotate)
  const P = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,1.3,-1.8,1];     // place in front of viewer
  return mul4(new Float32Array(P), mul4(new Float32Array(R), new Float32Array(T)));
}
function mul4(A,B){ const o = new Float32Array(16);
  for (let c=0;c<4;c++) for (let r=0;r<4;r++){ let s=0; for (let k=0;k<4;k++) s += A[k*4+r]*B[c*4+k]; o[c*4+r]=s; }
  return o; }

// ---------------- boot ----------------
(async ()=>{
  try{
    await initGL();
    center = centerOfField();
    requestAnimationFrame(flatLoop);
    if (navigator.xr && await navigator.xr.isSessionSupported("immersive-vr")){
      const b = document.getElementById("vrbtn");
      b.style.display = "block";
      b.onclick = ()=>enterVR().catch(e=>{ hudInfo.textContent = "XR error: "+e.message; });
    }
  }catch(e){ hudInfo.textContent = "ERROR: " + e.message; console.error(e); }
})();


// ---- CAPTURE HOOK: ?caps=<tag>&thetas=a,b,c  -> capture each pose in ONE page load ----
(function(){
  const tag = Q.get("caps"); if (!tag) return;
  const ths = (Q.get("thetas")||"0.3").split(",").map(parseFloat);
  const ph = parseFloat(Q.get("phi")||"0.10"), ds = parseFloat(Q.get("dist")||"3.0");
  const wait0 = parseInt(Q.get("wait")||"8000",10), gap = parseInt(Q.get("gap")||"2500",10);
  let k = 0;
  const shoot = () => {
    if (k >= ths.length) { document.title = "SHOTS_DONE_" + tag; return; }
    theta = ths[k]; phi = ph; dist = ds;
    setTimeout(() => {
      requestAnimationFrame(() => requestAnimationFrame(() => {
        const data = canvas.toDataURL("image/png");
        fetch("/shot?name=" + encodeURIComponent(tag + k), {method:"POST", body:data})
          .then(()=>{ document.title = "SHOT_" + tag + k; k++; shoot(); })
          .catch(()=>{ k++; shoot(); });
      }));
    }, gap);
  };
  setTimeout(shoot, wait0);
})();
