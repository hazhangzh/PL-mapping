"""Step 4: self-contained interactive HTML report (plotly.js embedded, works offline).

The page holds the averaged spectra, the background pieces and the fit
parameters; fitted curves are re-evaluated in the browser (Voigt via
Weideman's Faddeeva approximation), which keeps the file small.
"""
import base64
import json
import os

import numpy as np

from . import HC, __version__
from .fitting import DERIVED, LABELS
from .lineshapes import weideman_coeffs


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode("ascii")


def _bin(a, n):
    if n <= 1:
        return a
    m = (a.shape[-1] // n) * n
    a = a[..., :m]
    return a.reshape(a.shape[:-1] + (m // n, n)).mean(-1)


def _clean(v, sig=6):
    v = np.asarray(v, float)
    return [None if not np.isfinite(x) else float(f"{x:.{sig}g}") for x in v]


def write(path, cfg, grid, wl, spectra, present, bg, fit, prefix, log=print):
    from plotly.offline import get_plotlyjs
    nb = int(cfg["report"]["spectral_binning"] or 1)
    res, peaks, model = fit["res"], fit["peaks"], fit["model"]
    raw = np.where(present[:, None], spectra, 0.0)
    dark = bg["dark"] if bg["dark"] is not None else np.zeros_like(wl)
    fm = fit["fit_mask"]
    ks = np.arange(1, grid.npts + 1)
    sx, sy = grid.stage_xy(ks)

    maps = {"integrated_counts": {"label": "Integrated PL in fit range (counts)", "group": "General",
                                  "v": _clean(res["integrated_counts"]), "peak": None, "cmap": "Inferno"}}
    for p in peaks:
        n = p["name"]
        for q in ("intensity", "height", "center_nm", "center_meV", "fwhm_meV", "fwhm_nm", "snr"):
            maps[f"{n}_{q}"] = {"label": f"{n}: {LABELS[q]}", "group": f"Peak {n} ({p['shape']})",
                                "v": _clean(res[f"{n}_{q}"]), "peak": n,
                                "cmap": {"center_nm": "RdBu", "center_meV": "RdBu", "fwhm_meV": "Viridis",
                                         "fwhm_nm": "Viridis"}.get(q, "Inferno")}
    maps["r2"] = {"label": "Fit R²", "group": "Fit quality", "v": _clean(res["r2"]), "peak": None, "cmap": "Viridis"}
    maps["noise"] = {"label": "Residual noise (counts)", "group": "Fit quality", "v": _clean(res["noise"]),
                     "peak": None, "cmap": "Viridis"}
    for p in peaks:
        maps[f"{p['name']}_ok_map"] = {"label": f"{p['name']}: good fit (1) / rejected (0)", "group": "Fit quality",
                                       "v": [float(x) if pr else None for x, pr in zip(res[f"{p['name']}_ok"], present)],
                                       "peak": None, "cmap": "Greys"}

    L, a = weideman_coeffs(32)
    pk = [{"name": p["name"], "shape": p["shape"], "start": s.start, "stop": s.stop,
           "center_nm": p["center_nm"], "window_nm": p["window_nm"]}
          for p, s in zip(peaks, model.slices)]
    perpix = {f"{p['name']}_{q}": _clean(res[f"{p['name']}_{q}"]) for p in peaks
              for q in DERIVED + ("center_nm_err", "fwhm_meV_err", "intensity_err")}
    perpix.update({f"{p['name']}_ok": [bool(x) for x in res[f"{p['name']}_ok"]] for p in peaks})
    mean = fit["mean"]
    data = {
        "title": cfg["report"]["title"] or os.path.basename(os.path.normpath(cfg["data"]["folder"])),
        "prefix": prefix, "version": __version__,
        "nx": grid.nx, "ny": grid.ny, "units": grid.units,
        "x_um": _clean(grid.x_rel_um()), "y_um": _clean(grid.y_rel_um()),
        "stage_x": _clean(sx, 8), "stage_y": _clean(sy, 8),
        "x_step_um": abs(grid.x_step) * grid.to_um, "y_step_um": abs(grid.y_step) * grid.to_um,
        "flip_x": bool(cfg["scan"]["flip_x"]),
        "present": [bool(x) for x in present],
        "wl": _clean(_bin(wl, nb), 8), "binning": nb,
        "raw": _b64(_bin(raw, nb)), "offset": _clean(bg["offset"]),
        "substrate": _clean(_bin(bg["substrate"], nb)), "dark": _clean(_bin(dark, nb)),
        "fit_range": [float(wl[fm].min()), float(wl[fm].max())],
        "peaks": pk, "nbase": model.nbase, "Emid": model.Emid, "baseline": model.baseline,
        "theta": [_clean(t, 9) for t in fit["theta"]],
        "perpix": perpix, "r2": _clean(res["r2"]), "maps": maps,
        "mean": {"wl": _clean(wl[fm], 8), "y": _clean(mean["y"]), "theta": _clean(mean["theta"], 9),
                 "n": int(len(mean["points"])),
                 "derived": [{k: (bool(v) if isinstance(v, (bool, np.bool_)) else _clean([v])[0])
                              for k, v in d.items()} for d in mean["derived"]]},
        "comparison": fit["comparison"], "min_snr": cfg["fit"]["min_snr"],
        "faddeeva": {"L": L, "a": a.tolist()},
        "HC": HC,
    }
    html = TEMPLATE.replace("/*PLOTLYJS*/", get_plotlyjs()).replace(
        "/*DATA*/", json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/"))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    log(f"interactive report: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PL map report</title>
<style>
:root{--bg:#fafaf9;--panel:#fff;--ink:#1c1c1a;--muted:#6b6a66;--line:#e4e2dd;--accent:#1f4e8c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:16px 20px 8px}
h1{font-size:19px;margin:0 0 2px}
.sub{color:var(--muted);font-size:12.5px}
main{display:grid;grid-template-columns:minmax(360px,1fr) minmax(420px,1.25fr);gap:14px;padding:8px 20px 20px}
@media (max-width:980px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card h2{font-size:14px;margin:0 0 8px}
.controls{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;font-size:12.5px;margin-bottom:6px}
.controls label{display:flex;gap:5px;align-items:center}
select,input[type=number]{font:inherit;font-size:12.5px;padding:3px 5px;border:1px solid var(--line);border-radius:6px;background:#fff}
input[type=number]{width:78px}
button{font:inherit;font-size:12px;padding:3px 9px;border:1px solid var(--line);border-radius:6px;background:#f3f2ef;cursor:pointer}
button:hover{background:#e9e7e2}
table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:4px 6px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600}
.bad{color:#b42318}.good{color:#1a7f37}
.full{grid-column:1/-1}
.hint{color:var(--muted);font-size:12px}
code{background:#f1f0ec;padding:1px 4px;border-radius:4px}
#pixinfo{font-size:12.5px;color:var(--muted);margin:2px 0 6px}
</style>
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
  </div>
  <div class="controls" id="bandctl" style="display:none">
    <label>Band <input type="number" id="b0" step="0.5"> – <input type="number" id="b1" step="0.5"> nm</label>
  </div>
  <div class="controls">
    <label>Colour <select id="cmap"></select></label>
    <label>min <input type="number" id="cmin" step="any"></label>
    <label>max <input type="number" id="cmax" step="any"></label>
    <button id="auto">auto (1–99.5 %)</button>
  </div>
  <div id="map" style="height:470px"></div>
  <div class="hint">Click a pixel (or use the arrow keys) to see its spectrum. y increases downward (file order).</div>
</section>
<section class="card">
  <h2>Spectrum</h2>
  <div id="pixinfo"></div>
  <div class="controls">
    <label><input type="radio" name="ax" value="nm" checked> wavelength</label>
    <label><input type="radio" name="ax" value="eV"> energy</label>
    <label><input type="checkbox" id="showraw"> raw (before background)</label>
    <label><input type="checkbox" id="fitonly"> zoom to fit range</label>
  </div>
  <div id="spec" style="height:430px"></div>
  <table id="ptab"></table>
</section>
<section class="card full">
  <h2>Mean spectrum of the brightest points and lineshape check</h2>
  <div style="display:grid;grid-template-columns:minmax(0,1.4fr) minmax(0,1fr);gap:14px">
    <div id="meanplot" style="height:330px"></div>
    <div>
      <table id="ctab"></table>
      <p class="hint">Reduced χ² (summed over the 5 brightest points) when each peak in turn is fitted as a
      Lorentzian, Gaussian or Voigt (others unchanged), divided by the best of the three (1.000 = best).
      Voigt always fits at least as well (one more parameter), so it is recommended only if it beats both others by more than 2 %. To change a lineshape, set
      <code>shape:</code> for that peak in the config and run <code>plmap run config.yaml</code> again (a new <code>fit_*</code> folder is written).</p>
      <table id="mtab"></table>
    </div>
  </div>
</section>
</main>
<script id="data" type="application/json">/*DATA*/</script>
<script>
"use strict";
const D = JSON.parse(document.getElementById("data").textContent);
const NX = D.nx, NY = D.ny, NW = D.wl.length, HC = D.HC;
function b64f32(s){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Float32Array(u.buffer);}
const RAW = b64f32(D.raw);
const WL = D.wl, EN = WL.map(l=>HC/l);
const fitIdx = WL.map((l,i)=>[l,i]).filter(([l])=>l>=D.fit_range[0]&&l<=D.fit_range[1]).map(([,i])=>i);

// ---------- lineshapes (match plmap/lineshapes.py)
const LN2 = Math.log(2), SQPI = Math.sqrt(Math.PI);
const FL = D.faddeeva.L, FA = D.faddeeva.a;
function wRe(x,y){ // Re w(x+iy), y>=0 (Weideman 1994, N=32)
  const nr=FL-y, ni=x, dr=FL+y, di=-x, dd=dr*dr+di*di;
  const Zr=(nr*dr+ni*di)/dd, Zi=(ni*dr-nr*di)/dd;
  let pr=0, pi=0;
  for(const a of FA){const t=pr*Zr-pi*Zi+a; pi=pr*Zi+pi*Zr; pr=t;}
  const d2r=dr*dr-di*di, d2i=2*dr*di, d2m=d2r*d2r+d2i*d2i;
  return 2*(pr*d2r+pi*d2i)/d2m + dr/dd/SQPI;
}
const SHAPE = {
  lorentzian:(x,t)=>{const g=t[2]/2;return t[1]*g/Math.PI/((x-t[0])**2+g*g);},
  gaussian:(x,t)=>{const s=t[2]/(2*Math.sqrt(2*LN2));return t[1]*Math.exp(-0.5*((x-t[0])/s)**2)/(s*Math.sqrt(2*Math.PI));},
  voigt:(x,t)=>{const s=t[2]/(2*Math.sqrt(2*LN2)), g=t[3]/2, r=s*Math.SQRT2;return t[1]*wRe((x-t[0])/r,g/r)/(s*Math.sqrt(2*Math.PI));},
};
function evalFit(theta, xs){
  const nb=D.nbase, base=xs.map(x=>{let v=0;for(let j=0;j<nb;j++)v+=theta[theta.length-nb+j]*(x-D.Emid)**j;return v;});
  const comps=D.peaks.map(p=>{const t=theta.slice(p.start,p.stop);return xs.map(x=>SHAPE[p.shape](x,t));});
  const tot=base.map((b,i)=>comps.reduce((s,c)=>s+c[i],b));
  return {base,comps,tot};
}

// ---------- spectra
function rawSpec(i){return RAW.subarray(i*NW,(i+1)*NW);}
function bgSpec(i){const r=rawSpec(i),o=D.offset[i]||0,out=new Array(NW);for(let j=0;j<NW;j++)out[j]=r[j]-D.dark[j]-o-D.substrate[j];return out;}

// ---------- map
const qsel=document.getElementById("qty");
const groups={};
for(const [k,m] of Object.entries(D.maps)){(groups[m.group]=groups[m.group]||[]).push([k,m]);}
const bandOpt=document.createElement("option");bandOpt.value="__band";bandOpt.textContent="Band integral (choose range)";
for(const [g,items] of Object.entries(groups)){const og=document.createElement("optgroup");og.label=g;
  for(const [k,m] of items){const o=document.createElement("option");o.value=k;o.textContent=m.label;og.appendChild(o);}
  if(g==="General")og.appendChild(bandOpt);
  qsel.appendChild(og);}
const CUSTOM={Inferno:[[0,"#000004"],[0.125,"#1b0c41"],[0.25,"#4a0c6b"],[0.375,"#781c6d"],[0.5,"#a52c60"],[0.625,"#cf4446"],[0.75,"#ed6925"],[0.875,"#fb9b06"],[1,"#fcffa4"]],
  Magma:[[0,"#000004"],[0.125,"#1c1044"],[0.25,"#4f127b"],[0.375,"#812581"],[0.5,"#b5367a"],[0.625,"#e55064"],[0.75,"#fb8761"],[0.875,"#fec287"],[1,"#fcfdbf"]]};
const CMAPS=["Inferno","Magma","Viridis","RdBu","Cividis","Greys","Jet","Hot","Blues"];
const cscale=n=>CUSTOM[n]||n;
const csel=document.getElementById("cmap");
for(const c of CMAPS){const o=document.createElement("option");o.value=c;o.textContent=c;csel.appendChild(o);}
document.getElementById("flip").checked=D.flip_x;
const b0=document.getElementById("b0"),b1=document.getElementById("b1");
b0.value=D.fit_range[0].toFixed(1);b1.value=D.fit_range[1].toFixed(1);

let sel = (()=>{let best=-1,bi=0;const v=D.maps.integrated_counts.v;v.forEach((x,i)=>{if(x!==null&&x>best){best=x;bi=i;}});return bi;})();

function values(){
  const q=qsel.value;
  if(q==="__band"){const lo=Math.min(+b0.value,+b1.value),hi=Math.max(+b0.value,+b1.value);
    const idx=[];WL.forEach((l,j)=>{if(l>=lo&&l<=hi)idx.push(j);});
    return D.present.map((p,i)=>{if(!p)return null;const s=bgSpec(i);let t=0;for(const j of idx)t+=s[j];return t*D.binning;});}
  const m=D.maps[q];let v=m.v.slice();
  if(m.peak&&document.getElementById("mask").checked){const ok=D.perpix[m.peak+"_ok"];v=v.map((x,i)=>ok[i]?x:null);}
  return v;
}
function pct(arr,p){const a=arr.filter(x=>x!==null&&isFinite(x)).sort((x,y)=>x-y);if(!a.length)return 0;
  const k=(a.length-1)*p/100,f=Math.floor(k),c=Math.ceil(k);return a[f]+(a[c]-a[f])*(k-f);}
function label(){return qsel.value==="__band"?`Band integral ${b0.value}–${b1.value} nm (counts)`:D.maps[qsel.value].label;}
function to2d(v){const z=[];for(let r=0;r<NY;r++)z.push(v.slice(r*NX,(r+1)*NX));return z;}
const CUST=to2d([...Array(NX*NY).keys()].map(i=>[i+1,Math.floor(i/NX),i%NX,D.stage_x[i],D.stage_y[i]]));
function autoRange(v){document.getElementById("cmin").value=+pct(v,1).toPrecision(5);document.getElementById("cmax").value=+pct(v,99.5).toPrecision(5);}
function selMarker(){return {x:[D.x_um[sel%NX]],y:[D.y_um[Math.floor(sel/NX)]],mode:"markers",type:"scatter",
  marker:{symbol:"square-open",size:13,color:"#00e5ff",line:{width:2.5}},hoverinfo:"skip",showlegend:false};}
function drawMap(resetRange){
  const v=values();
  if(resetRange)autoRange(v);
  if(resetRange&&qsel.value!=="__band")csel.value=D.maps[qsel.value].cmap;
  const hm={type:"heatmap",x:D.x_um,y:D.y_um,z:to2d(v),customdata:CUST,colorscale:cscale(csel.value),
    reversescale:csel.value==="RdBu",zmin:+document.getElementById("cmin").value,zmax:+document.getElementById("cmax").value,
    colorbar:{thickness:12,len:0.9,title:{text:"",side:"right"}},hoverongaps:false,
    hovertemplate:"loc%{customdata[0]}  (row %{customdata[1]}, col %{customdata[2]})<br>stage x=%{customdata[3]}, y=%{customdata[4]} "+D.units+"<br>value %{z:.5g}<extra></extra>"};
  const lay={margin:{l:52,r:10,t:28,b:44},title:{text:label(),font:{size:13}},plot_bgcolor:"#d9d9d9",
    xaxis:{title:{text:"x (µm)"+(document.getElementById("flip").checked?" ← flipped":"")},autorange:document.getElementById("flip").checked?"reversed":true,constrain:"domain",zeroline:false},
    yaxis:{title:{text:"y (µm)"},autorange:"reversed",scaleanchor:"x",constrain:"domain",zeroline:false}};
  Plotly.react("map",[hm,selMarker()],lay,{responsive:true,displaylogo:false});
}
function moveMarker(){Plotly.restyle("map",{x:[[D.x_um[sel%NX]]],y:[[D.y_um[Math.floor(sel/NX)]]]},[1]);}

// ---------- spectrum panel
const fmt=(x,d)=>x===null||x===undefined||!isFinite(x)?"–":(+x).toFixed(d);
function drawSpec(){
  const i=sel, r=Math.floor(i/NX), c=i%NX, eV=document.querySelector("input[name=ax]:checked").value==="eV";
  document.getElementById("pixinfo").textContent=`${D.prefix}_loc${i+1}  ·  row ${r}, col ${c}  ·  stage x = ${D.stage_x[i]}, y = ${D.stage_y[i]} ${D.units}  ·  (${D.x_um[c]}, ${D.y_um[r]}) µm`
    +(D.r2[i]!==null?`  ·  fit R² = ${fmt(D.r2[i],4)}`:"");
  const X=eV?EN:WL, tr=[];
  if(!D.present[i]){Plotly.react("spec",[],{annotations:[{text:"no data for this point",showarrow:false,xref:"paper",yref:"paper",x:0.5,y:0.5}],xaxis:{visible:false},yaxis:{visible:false}});document.getElementById("ptab").innerHTML="";return;}
  const raw=document.getElementById("showraw").checked;
  const y=raw?Array.from(rawSpec(i)):bgSpec(i);
  tr.push({x:X,y:y,type:"scatter",mode:"lines",line:{color:"#8a8a8a",width:1},name:raw?"raw (mean over CCD rows)":"data (bkg-subtracted)"});
  const th=D.theta[i];
  let resid=null;
  if(!raw&&th&&th.every(t=>t!==null)){
    const xs=fitIdx.map(j=>EN[j]), f=evalFit(th,xs), XX=fitIdx.map(j=>X[j]);
    const pal=["#d62728","#1f77b4","#2ca02c","#9467bd","#ff7f0e","#17becf"];
    D.peaks.forEach((p,k)=>{const ok=D.perpix[p.name+"_ok"][i];
      tr.push({x:XX,y:f.comps[k].map((v,j)=>v+f.base[j]),type:"scatter",mode:"lines",fill:"none",
        line:{color:pal[k%6],width:1.4,dash:ok?"solid":"dot"},name:`${p.name} (${p.shape})${ok?"":" – rejected"}`});});
    if(D.nbase>0)tr.push({x:XX,y:f.base,type:"scatter",mode:"lines",line:{color:"#999",width:1,dash:"dash"},name:"baseline"});
    tr.push({x:XX,y:f.tot,type:"scatter",mode:"lines",line:{color:"#111",width:1.6},name:"total fit"});
    resid={x:XX,y:fitIdx.map((j,k)=>y[j]-f.tot[k]),type:"scatter",mode:"lines",line:{color:"#555",width:0.8},name:"residual",xaxis:"x",yaxis:"y2",showlegend:false};
    tr.push(resid);
  }
  const zoom=document.getElementById("fitonly").checked, fr=eV?[HC/D.fit_range[1],HC/D.fit_range[0]]:D.fit_range;
  const lay={margin:{l:58,r:10,t:46,b:44},legend:{font:{size:11},orientation:"h",x:0,y:1.02,yanchor:"bottom"},
    xaxis:{title:{text:eV?"Energy (eV)":"Wavelength (nm)"},range:zoom?fr:null,autorange:!zoom,anchor:"y2"},
    yaxis:{title:{text:"Counts"},domain:resid?[0.3,1]:[0,1],zeroline:true,zerolinecolor:"#ccc"},
    yaxis2:{title:{text:"resid."},domain:[0,0.22],zeroline:true,zerolinecolor:"#bbb",visible:!!resid},
    shapes:[{type:"rect",xref:"x",yref:"paper",x0:fr[0],x1:fr[1],y0:0,y1:1,fillcolor:"rgba(31,78,140,0.05)",line:{width:0},layer:"below"}]};
  Plotly.react("spec",tr,lay,{responsive:true,displaylogo:false});
  let h="<tr><th>peak</th><th>λ (nm)</th><th>E (meV)</th><th>FWHM (meV)</th><th>FWHM (nm)</th><th>intensity</th><th>height</th><th>SNR</th><th>fit</th></tr>";
  for(const p of D.peaks){const g=k=>D.perpix[p.name+"_"+k][i], ok=g("ok");
    h+=`<tr><td>${p.name} <span class="hint">${p.shape}</span></td><td>${fmt(g("center_nm"),3)} ± ${fmt(g("center_nm_err"),3)}</td><td>${fmt(g("center_meV"),2)}</td>`+
       `<td>${fmt(g("fwhm_meV"),2)} ± ${fmt(g("fwhm_meV_err"),2)}</td><td>${fmt(g("fwhm_nm"),3)}</td><td>${fmt(g("intensity"),0)}</td><td>${fmt(g("height"),1)}</td><td>${fmt(g("snr"),1)}</td>`+
       `<td class="${ok?"good":"bad"}">${ok?"good":"rejected"}</td></tr>`;}
  document.getElementById("ptab").innerHTML=h;
}

// ---------- mean spectrum + comparison
(function(){
  const m=D.mean, xs=m.wl.map(l=>HC/l), f=evalFit(m.theta,xs), tr=[{x:m.wl,y:m.y,mode:"lines",type:"scatter",line:{color:"#8a8a8a",width:1},name:`mean of ${m.n} brightest points`}];
  const pal=["#d62728","#1f77b4","#2ca02c","#9467bd","#ff7f0e","#17becf"];
  D.peaks.forEach((p,k)=>tr.push({x:m.wl,y:f.comps[k].map((v,j)=>v+f.base[j]),mode:"lines",type:"scatter",line:{color:pal[k%6],width:1.4},name:`${p.name} (${p.shape})`}));
  tr.push({x:m.wl,y:f.tot,mode:"lines",type:"scatter",line:{color:"#111",width:1.6},name:"total fit"});
  Plotly.newPlot("meanplot",tr,{margin:{l:58,r:10,t:10,b:44},xaxis:{title:{text:"Wavelength (nm)"}},yaxis:{title:{text:"Counts (bkg-subtracted)"}},legend:{font:{size:11}}},{responsive:true,displaylogo:false});
  let h="<tr><th>peak</th><th>Lorentzian</th><th>Gaussian</th><th>Voigt</th><th>recommended</th><th>used</th></tr>";
  for(const p of D.peaks){const c=D.comparison[p.name];h+=`<tr><td>${p.name}</td>`+["lorentzian","gaussian","voigt"].map(s=>`<td${s===c.recommended?' class="good"':''}>${fmt(c[s],3)}</td>`).join("")+
    `<td>${c.recommended}</td><td${c.recommended===p.shape?"":' class="bad"'}>${p.shape}</td></tr>`;}
  document.getElementById("ctab").innerHTML=h;
  let t="<tr><th>peak (mean spectrum)</th><th>λ (nm)</th><th>E (meV)</th><th>FWHM (meV)</th><th>intensity</th></tr>";
  D.peaks.forEach((p,k)=>{const d=m.derived[k];t+=`<tr><td>${p.name}</td><td>${fmt(d.center_nm,3)}</td><td>${fmt(d.center_meV,2)}</td><td>${fmt(d.fwhm_meV,2)}</td><td>${fmt(d.intensity,0)}</td></tr>`;});
  document.getElementById("mtab").innerHTML=t;
})();

// ---------- wiring
document.getElementById("title").textContent=D.title;
document.getElementById("subtitle").textContent=`${D.nx} × ${D.ny} points, step ${(+D.x_step_um.toFixed(4))} × ${(+D.y_step_um.toFixed(4))} µm · `+
  `${D.present.filter(x=>x).length} of ${D.nx*D.ny} files · fit ${D.fit_range[0].toFixed(1)}–${D.fit_range[1].toFixed(1)} nm, `+
  D.peaks.map(p=>`${p.name}=${p.shape}`).join(", ")+`, baseline ${D.baseline} · rejected if SNR < ${D.min_snr} or at a bound · plmap ${D.version}`;
qsel.value="integrated_counts";
qsel.onchange=()=>{document.getElementById("bandctl").style.display=qsel.value==="__band"?"flex":"none";drawMap(true);};
for(const id of ["mask","flip"])document.getElementById(id).onchange=()=>drawMap(false);
for(const id of ["b0","b1"])document.getElementById(id).onchange=()=>drawMap(true);
for(const id of ["cmin","cmax","cmap"])document.getElementById(id).onchange=()=>drawMap(false);
document.getElementById("auto").onclick=()=>drawMap(true);
for(const el of document.querySelectorAll("input[name=ax]"))el.onchange=drawSpec;
for(const id of ["showraw","fitonly"])document.getElementById(id).onchange=drawSpec;
drawMap(true);drawSpec();
document.getElementById("map").on("plotly_click",ev=>{const pt=ev.points.find(p=>p.data.type==="heatmap");if(!pt)return;
  const c=D.x_um.indexOf(pt.x), r=D.y_um.indexOf(pt.y);if(c<0||r<0)return;sel=r*NX+c;moveMarker();drawSpec();});
document.addEventListener("keydown",e=>{if(["INPUT","SELECT"].includes(document.activeElement.tagName))return;
  let r=Math.floor(sel/NX),c=sel%NX;const fl=document.getElementById("flip").checked?-1:1;
  if(e.key==="ArrowLeft")c-=fl;else if(e.key==="ArrowRight")c+=fl;else if(e.key==="ArrowUp")r--;else if(e.key==="ArrowDown")r++;else return;
  e.preventDefault();r=Math.max(0,Math.min(NY-1,r));c=Math.max(0,Math.min(NX-1,c));sel=r*NX+c;moveMarker();drawSpec();});
</script>
</body></html>
"""
