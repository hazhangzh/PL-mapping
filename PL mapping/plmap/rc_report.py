"""Interactive HTML report for reflection-contrast maps (self-contained, works offline).

Holds R (counts/s per CCD row) of every point and R0; RC = R/R0 - 1 and the fitted
resonance are recomputed in the browser. Clicking points can build an `r0:` list
for the config, to choose the background by hand.
"""
import json
import os

import numpy as np

from . import HC, __version__
from .rc import LABELS
from .report import TEMPLATE as PL_TEMPLATE, _b64, _bin, _clean

CSS = PL_TEMPLATE[PL_TEMPLATE.index("<style>"):PL_TEMPLATE.index("</style>") + len("</style>")]


def write(path, cfg, grid, wl, R, R0, present, res, par, Emid, win, r0info, prefix, log=print):
    from plotly.offline import get_plotlyjs
    nb = int(cfg["report"]["spectral_binning"] or 1)
    ks = np.arange(1, grid.npts + 1)
    sx, sy = grid.stage_xy(ks)
    ok = res.get("ok", np.ones(grid.npts, bool))
    maps = {}
    for key, cmap, fit in (("amplitude", "Magma", True), ("center_meV", "RdBu", True), ("center_nm", "RdBu", True),
                           ("fwhm_meV", "Viridis", True), ("phase_deg", "Viridis", True), ("baseline", "RdBu", True),
                           ("snr", "Inferno", True), ("r2", "Viridis", True),
                           ("feature_p2p", "Magma", False), ("R_mean", "Greys", False), ("rc_level", "RdBu", False)):
        if key in res:
            maps[key] = {"label": LABELS[key], "v": _clean(res[key]), "fit": fit, "cmap": cmap,
                         "group": "Resonance fit" if fit else "Model-free"}
    if "ok" in res:
        maps["ok_map"] = {"label": "good fit (1) / rejected (0)", "group": "Resonance fit", "fit": False,
                          "cmap": "Greys", "v": [float(x) if p else None for x, p in zip(res["ok"], present)]}
    perpix = {k: _clean(res[k]) for k in ("center_meV", "center_meV_err", "center_nm", "fwhm_meV", "fwhm_meV_err",
                                          "amplitude", "amplitude_err", "phase_deg", "snr", "r2") if k in res}
    data = {
        "title": cfg["report"]["title"] or os.path.basename(os.path.normpath(cfg["data"]["folder"])),
        "prefix": prefix, "version": __version__, "nx": grid.nx, "ny": grid.ny, "units": grid.units,
        "x_um": _clean(grid.x_rel_um()), "y_um": _clean(grid.y_rel_um()),
        "stage_x": _clean(sx, 8), "stage_y": _clean(sy, 8),
        "x_step_um": abs(grid.x_step) * grid.to_um, "y_step_um": abs(grid.y_step) * grid.to_um,
        "flip_x": bool(cfg["scan"]["flip_x"]), "present": [bool(x) for x in present],
        "wl": _clean(_bin(wl, nb), 8), "R": _b64(_bin(np.nan_to_num(R), nb)), "R0": _clean(_bin(R0, nb)),
        "window": [float(win[0]), float(win[1])], "Emid": Emid,
        "par": [_clean(p, 9) for p in par], "ok": [bool(x) for x in ok], "perpix": perpix, "maps": maps,
        "r0_points": [int(k) for k in np.flatnonzero(r0info["points"]) + 1], "r0_source": r0info["source"],
        "r0_picked": isinstance(cfg["rc"]["r0"], (list, tuple)),
        "min_snr": cfg["rc"]["min_snr"], "HC": HC,
    }
    html = TEMPLATE.replace("/*CSS*/", CSS).replace("/*PLOTLYJS*/", get_plotlyjs()).replace(
        "/*DATA*/", json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/"))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    log(f"interactive report: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RC map report</title>
/*CSS*/
<style>#r0list{width:100%;font:12px ui-monospace,monospace;padding:6px;border:1px solid var(--line);border-radius:6px;resize:vertical}</style>
<script>/*PLOTLYJS*/</script>
</head><body>
<header><h1 id="title"></h1><div class="sub" id="subtitle"></div></header>
<main>
<section class="card">
  <h2>Map</h2>
  <div class="controls">
    <label>Quantity <select id="qty"></select></label>
    <label><input type="checkbox" id="mask" checked> hide rejected fits</label>
    <label><input type="checkbox" id="flip"> flip x</label>
    <label><input type="checkbox" id="showr0" checked> show R0 points</label>
  </div>
  <div class="controls">
    <label>Colour <select id="cmap"></select></label>
    <label>min <input type="number" id="cmin" step="any"></label>
    <label>max <input type="number" id="cmax" step="any"></label>
    <button id="auto">auto (1–99.5 %)</button>
  </div>
  <div id="map" style="height:470px"></div>
  <div class="hint">Click a pixel (or use the arrow keys) to see its spectrum. x, y in µm from the scan start; y increases downward (file order).</div>
</section>
<section class="card">
  <h2>Spectrum</h2>
  <div id="pixinfo"></div>
  <div class="controls">
    <label><input type="radio" name="ax" value="eV" checked> energy</label>
    <label><input type="radio" name="ax" value="nm"> wavelength</label>
    <label><input type="checkbox" id="zoom" checked> zoom to resonance window</label>
  </div>
  <div id="spec" style="height:360px"></div>
  <div id="rspec" style="height:220px"></div>
  <table id="ptab"></table>
</section>
<section class="card full">
  <h2>Choose the background (R0) by hand</h2>
  <p class="hint">R0 used: <span id="r0src"></span> (small green dots on the map). To pick your own background points, click a featureless
  pixel on the map (dark in the "feature size" map), press <b>add this point</b> (green circles), repeat, then copy the line below into the <code>rc:</code>
  section of the config and run <code>plmap rc config.yaml</code> again.</p>
  <div class="controls"><button id="addr0">add this point</button><button id="delr0">remove this point</button>
    <button id="clrr0">clear list</button><span class="hint" id="r0n"></span></div>
  <textarea id="r0list" rows="3" readonly></textarea>
</section>
</main>
<script id="data" type="application/json">/*DATA*/</script>
<script>
"use strict";
const D=JSON.parse(document.getElementById("data").textContent);
const NX=D.nx,NY=D.ny,NW=D.wl.length,HC=D.HC,WL=D.wl,EN=WL.map(l=>HC/l);
function b64f32(s){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Float32Array(u.buffer);}
const R=b64f32(D.R);
const WIN=WL.map((l,i)=>[l,i]).filter(([l])=>l>=D.window[0]&&l<=D.window[1]).map(([,i])=>i);
function Rspec(i){return R.subarray(i*NW,(i+1)*NW);}
function RC(i){const r=Rspec(i);return Array.from(r,(v,j)=>D.R0[j]?v/D.R0[j]-1:null);}
function model(p,E){const d=p[0]-E,h=p[1]/2,den=d*d+h*h;return p[2]*h*d/den+p[3]*h*h/den+p[4]+p[5]*(E-D.Emid);}
const fmt=(x,d)=>x===null||x===undefined||!isFinite(x)?"–":(+x).toFixed(d);

// ---------- map
const qsel=document.getElementById("qty"),csel=document.getElementById("cmap");
const groups={};for(const [k,m] of Object.entries(D.maps))(groups[m.group]=groups[m.group]||[]).push([k,m]);
for(const [g,items] of Object.entries(groups)){const og=document.createElement("optgroup");og.label=g;
  for(const [k,m] of items){const o=document.createElement("option");o.value=k;o.textContent=m.label;og.appendChild(o);}qsel.appendChild(og);}
const CUSTOM={Inferno:[[0,"#000004"],[0.25,"#4a0c6b"],[0.5,"#a52c60"],[0.75,"#ed6925"],[1,"#fcffa4"]],
  Magma:[[0,"#000004"],[0.25,"#4f127b"],[0.5,"#b5367a"],[0.75,"#fb8761"],[1,"#fcfdbf"]]};
for(const c of ["Magma","Inferno","Viridis","RdBu","Cividis","Greys","Jet","Blues"]){const o=document.createElement("option");o.value=c;o.textContent=c;csel.appendChild(o);}
const cscale=n=>CUSTOM[n]||n;
document.getElementById("flip").checked=D.flip_x;
let pick=D.r0_picked?D.r0_points.slice():[];   // the list being built; the run's own R0 points are drawn faintly
let sel=(()=>{let b=-1,bi=0;const v=(D.maps.amplitude||D.maps.feature_p2p).v;v.forEach((x,i)=>{if(x!==null&&D.ok[i]&&x>b){b=x;bi=i;}});return bi;})();
function values(){const m=D.maps[qsel.value];let v=m.v.slice();if(m.fit&&document.getElementById("mask").checked)v=v.map((x,i)=>D.ok[i]?x:null);return v;}
function pct(a,p){a=a.filter(x=>x!==null&&isFinite(x)).sort((x,y)=>x-y);if(!a.length)return 0;const k=(a.length-1)*p/100,f=Math.floor(k),c=Math.ceil(k);return a[f]+(a[c]-a[f])*(k-f);}
function to2d(v){const z=[];for(let r=0;r<NY;r++)z.push(v.slice(r*NX,(r+1)*NX));return z;}
const CUST=to2d([...Array(NX*NY).keys()].map(i=>[i+1,Math.floor(i/NX),i%NX,D.stage_x[i],D.stage_y[i]]));
function autoRange(v){document.getElementById("cmin").value=+pct(v,1).toPrecision(5);document.getElementById("cmax").value=+pct(v,99.5).toPrecision(5);}
function pts(list){const xs=[],ys=[];if(document.getElementById("showr0").checked)for(const k of list){xs.push(D.x_um[(k-1)%NX]);ys.push(D.y_um[Math.floor((k-1)/NX)]);}return [xs,ys];}
function markers(){const [ux,uy]=pts(D.r0_picked?[]:D.r0_points),[xs,ys]=pts(pick);
  return [{x:ux,y:uy,mode:"markers",type:"scatter",marker:{symbol:"circle",size:3,color:"#2ca02c",opacity:0.45},hoverinfo:"skip",showlegend:false},
          {x:xs,y:ys,mode:"markers",type:"scatter",marker:{symbol:"circle-open",size:7,color:"#00a651",line:{width:2}},hoverinfo:"skip",showlegend:false},
          {x:[D.x_um[sel%NX]],y:[D.y_um[Math.floor(sel/NX)]],mode:"markers",type:"scatter",marker:{symbol:"square-open",size:13,color:"#00e5ff",line:{width:2.5}},hoverinfo:"skip",showlegend:false}];}
function drawMap(reset){const v=values();if(reset){autoRange(v);csel.value=D.maps[qsel.value].cmap;}
  const fl=document.getElementById("flip").checked;
  const hm={type:"heatmap",x:D.x_um,y:D.y_um,z:to2d(v),customdata:CUST,colorscale:cscale(csel.value),reversescale:csel.value==="RdBu",
    zmin:+document.getElementById("cmin").value,zmax:+document.getElementById("cmax").value,colorbar:{thickness:12,len:0.9},hoverongaps:false,
    hovertemplate:"loc%{customdata[0]}  (row %{customdata[1]}, col %{customdata[2]})<br>(%{x}, %{y}) µm<br>stage x=%{customdata[3]}, y=%{customdata[4]} "+D.units+"<br>value %{z:.5g}<extra></extra>"};
  Plotly.react("map",[hm,...markers()],{margin:{l:52,r:10,t:28,b:44},title:{text:D.maps[qsel.value].label,font:{size:13}},plot_bgcolor:"#d9d9d9",
    xaxis:{title:{text:"x (µm)"+(fl?" ← flipped":"")},autorange:fl?"reversed":true,constrain:"domain",zeroline:false},
    yaxis:{title:{text:"y (µm)"},autorange:"reversed",scaleanchor:"x",constrain:"domain",zeroline:false}},{responsive:true,displaylogo:false});}
function moveMarker(){const m=markers();Plotly.restyle("map",{x:m.map(t=>t.x),y:m.map(t=>t.y)},[1,2,3]);}

// ---------- spectra
function drawSpec(){const i=sel,r=Math.floor(i/NX),c=i%NX,eV=document.querySelector("input[name=ax]:checked").value==="eV";
  document.getElementById("pixinfo").textContent=`${D.prefix}_loc${i+1} · row ${r}, col ${c} · (${D.x_um[c]}, ${D.y_um[r]}) µm · stage x = ${D.stage_x[i]}, y = ${D.stage_y[i]} ${D.units}`+(pick.includes(i+1)?" · R0 point":"");
  if(!D.present[i]){Plotly.purge("spec");Plotly.purge("rspec");document.getElementById("ptab").innerHTML="";return;}
  const X=eV?EN:WL,y=RC(i),tr=[{x:X,y:y,type:"scatter",mode:"lines",line:{color:"#8a8a8a",width:1},name:"RC"}];
  const p=D.par[i];let resid=null;
  if(p&&p.every(t=>t!==null)){const xs=WIN.map(j=>X[j]),f=WIN.map(j=>model(p,EN[j]));
    tr.push({x:xs,y:f,type:"scatter",mode:"lines",line:{color:D.ok[i]?"#d62728":"#999",width:1.8,dash:D.ok[i]?"solid":"dot"},name:"fit"+(D.ok[i]?"":" (rejected)")});
    resid={x:xs,y:WIN.map((j,k)=>y[j]-f[k]),type:"scatter",mode:"lines",line:{color:"#555",width:0.8},name:"residual",xaxis:"x",yaxis:"y2",showlegend:false};tr.push(resid);}
  const zoom=document.getElementById("zoom").checked,fr=eV?[HC/D.window[1],HC/D.window[0]]:D.window;
  let yr=null;if(zoom){const v=WIN.map(j=>y[j]).filter(t=>t!==null);const lo=Math.min(...v),hi=Math.max(...v);yr=[lo-0.1*(hi-lo),hi+0.1*(hi-lo)];}
  Plotly.react("spec",tr,{margin:{l:58,r:10,t:30,b:44},legend:{orientation:"h",x:0,y:1.02,yanchor:"bottom"},
    xaxis:{title:{text:eV?"Energy (eV)":"Wavelength (nm)"},range:zoom?fr:null,autorange:!zoom,anchor:"y2"},
    yaxis:{title:{text:"RC = (R − R0)/R0"},domain:resid?[0.3,1]:[0,1],range:yr,autorange:!yr},
    yaxis2:{title:{text:"resid."},domain:[0,0.22],visible:!!resid},
    shapes:[{type:"rect",xref:"x",yref:"paper",x0:fr[0],x1:fr[1],y0:0,y1:1,fillcolor:"rgba(31,78,140,0.05)",line:{width:0},layer:"below"}]},{responsive:true,displaylogo:false});
  Plotly.react("rspec",[{x:X,y:Array.from(Rspec(i)),type:"scatter",mode:"lines",line:{color:"#d62728",width:1},name:"R (this point)"},
    {x:X,y:D.R0,type:"scatter",mode:"lines",line:{color:"#111",width:1},name:"R0"}],
    {margin:{l:58,r:10,t:8,b:40},legend:{orientation:"h",x:0,y:1,yanchor:"bottom"},xaxis:{title:{text:eV?"Energy (eV)":"Wavelength (nm)"}},yaxis:{title:{text:"counts/s/row"}}},{responsive:true,displaylogo:false});
  const g=k=>D.perpix[k]?D.perpix[k][i]:null;
  document.getElementById("ptab").innerHTML=D.perpix.center_meV?"<tr><th>E0 (meV)</th><th>λ0 (nm)</th><th>Γ (meV)</th><th>A</th><th>φ (deg)</th><th>SNR</th><th>R²</th><th>fit</th></tr>"+
    `<tr><td>${fmt(g("center_meV"),2)} ± ${fmt(g("center_meV_err"),2)}</td><td>${fmt(g("center_nm"),3)}</td><td>${fmt(g("fwhm_meV"),2)} ± ${fmt(g("fwhm_meV_err"),2)}</td>`+
    `<td>${fmt(g("amplitude"),4)}</td><td>${fmt(g("phase_deg"),0)}</td><td>${fmt(g("snr"),1)}</td><td>${fmt(g("r2"),3)}</td><td class="${D.ok[i]?"good":"bad"}">${D.ok[i]?"good":"rejected"}</td></tr>`:"";}

// ---------- R0 picking
function showList(){pick.sort((a,b)=>a-b);document.getElementById("r0list").value=`  r0: [${pick.join(", ")}]`;document.getElementById("r0n").textContent=`${pick.length} points`;}
document.getElementById("addr0").onclick=()=>{if(D.present[sel]&&!pick.includes(sel+1))pick.push(sel+1);showList();moveMarker();drawSpec();};
document.getElementById("delr0").onclick=()=>{pick=pick.filter(k=>k!==sel+1);showList();moveMarker();drawSpec();};
document.getElementById("clrr0").onclick=()=>{pick=[];showList();moveMarker();};

// ---------- wiring
document.getElementById("title").textContent=D.title+" — reflection contrast";
document.getElementById("subtitle").textContent=`${NX} × ${NY} points, step ${+D.x_step_um.toFixed(4)} × ${+D.y_step_um.toFixed(4)} µm · `+
  `${D.present.filter(x=>x).length} of ${NX*NY} files · resonance window ${D.window[0].toFixed(1)}–${D.window[1].toFixed(1)} nm · rejected if SNR < ${D.min_snr} or at a bound · plmap ${D.version}`;
document.getElementById("r0src").textContent=D.r0_source;
qsel.value=D.maps.amplitude?"amplitude":"feature_p2p";
qsel.onchange=()=>drawMap(true);
for(const id of ["mask","flip","showr0","cmin","cmax","cmap"])document.getElementById(id).onchange=()=>drawMap(false);
document.getElementById("auto").onclick=()=>drawMap(true);
for(const el of document.querySelectorAll("input[name=ax]"))el.onchange=drawSpec;
document.getElementById("zoom").onchange=drawSpec;
showList();drawMap(true);drawSpec();
document.getElementById("map").on("plotly_click",ev=>{const pt=ev.points.find(p=>p.data.type==="heatmap");if(!pt)return;
  const c=D.x_um.indexOf(pt.x),r=D.y_um.indexOf(pt.y);if(c<0||r<0)return;sel=r*NX+c;moveMarker();drawSpec();});
document.addEventListener("keydown",e=>{if(["INPUT","SELECT","TEXTAREA"].includes(document.activeElement.tagName))return;
  let r=Math.floor(sel/NX),c=sel%NX;const fl=document.getElementById("flip").checked?-1:1;
  if(e.key==="ArrowLeft")c-=fl;else if(e.key==="ArrowRight")c+=fl;else if(e.key==="ArrowUp")r--;else if(e.key==="ArrowDown")r++;else return;
  e.preventDefault();r=Math.max(0,Math.min(NY-1,r));c=Math.max(0,Math.min(NX-1,c));sel=r*NX+c;moveMarker();drawSpec();});
</script>
</body></html>
"""
