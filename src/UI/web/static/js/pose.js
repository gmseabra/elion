/* ============================================================================
   Pose-Gen engine — SMILES parsing, rotatable-bond perception, torsion tree,
   2D layout, pose math, PDB parsing.  Pure (no DOM). Node-testable.
   ============================================================================ */
(function(root){
  /* ---- vec math ---- */
  const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
  const add=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
  const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
  const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const nrm=a=>{const m=Math.hypot(a[0],a[1],a[2])||1;return [a[0]/m,a[1]/m,a[2]/m];};
  function matVec(M,v){return [M[0][0]*v[0]+M[0][1]*v[1]+M[0][2]*v[2],M[1][0]*v[0]+M[1][1]*v[1]+M[1][2]*v[2],M[2][0]*v[0]+M[2][1]*v[1]+M[2][2]*v[2]];}
  function matMul(A,B){const C=[[0,0,0],[0,0,0],[0,0,0]];for(let i=0;i<3;i++)for(let j=0;j<3;j++)for(let k=0;k<3;k++)C[i][j]+=A[i][k]*B[k][j];return C;}
  function rotX(t){const c=Math.cos(t),s=Math.sin(t);return [[1,0,0],[0,c,-s],[0,s,c]];}
  function rotY(t){const c=Math.cos(t),s=Math.sin(t);return [[c,0,s],[0,1,0],[-s,0,c]];}
  function quatToMat(q){const a=q[0],b=q[1],c=q[2],d=q[3];
    const aa=a*a,ab=a*b,ac=a*c,ad=a*d,bb=b*b,bc=b*c,bd=b*d,cc=c*c,cd=c*d,dd=d*d;
    return [[aa+bb-cc-dd,2*(-ad+bc),2*(ac+bd)],[2*(ad+bc),aa-bb+cc-dd,2*(-ab+cd)],[2*(-ac+bd),2*(ab+cd),aa-bb-cc+dd]];}
  function normQuat(q){const m=Math.hypot(q[0],q[1],q[2],q[3])||1;return [q[0]/m,q[1]/m,q[2]/m,q[3]/m];}
  function eulerToQuat(rx,ry,rz){const cx=Math.cos(rx/2),sx=Math.sin(rx/2),cy=Math.cos(ry/2),sy=Math.sin(ry/2),cz=Math.cos(rz/2),sz=Math.sin(rz/2);
    return normQuat([cx*cy*cz+sx*sy*sz,sx*cy*cz-cx*sy*sz,cx*sy*cz+sx*cy*sz,cx*cy*sz-sx*sy*cz]);}
  function randomQuat(){const u1=Math.random(),u2=Math.random(),u3=Math.random();const s1=Math.sqrt(1-u1),s2=Math.sqrt(u1),t1=2*Math.PI*u2,t2=2*Math.PI*u3;
    return [s2*Math.cos(t2),s1*Math.sin(t1),s1*Math.cos(t1),s2*Math.sin(t2)];}
  function rotAxis(p,piv,k,th){const v=sub(p,piv),ct=Math.cos(th),st=Math.sin(th),kv=dot(k,v),kc=cross(k,v);
    return [piv[0]+v[0]*ct+kc[0]*st+k[0]*kv*(1-ct),piv[1]+v[1]*ct+kc[1]*st+k[1]*kv*(1-ct),piv[2]+v[2]*ct+kc[2]*st+k[2]*kv*(1-ct)];}
  function q2e(q){q=normQuat(q);const w=q[0],x=q[1],y=q[2],z=q[3];
    const rx=Math.atan2(2*(w*x+y*z),1-2*(x*x+y*y));let sp=2*(w*y-z*x);sp=Math.max(-1,Math.min(1,sp));const ry=Math.asin(sp);
    const rz=Math.atan2(2*(w*z+x*y),1-2*(y*y+z*z));return [rx,ry,rz];}

  /* ---- element table ---- */
  const ELEM={ C:{col:'#cbd5e1',r:9}, N:{col:'#60a5fa',r:9.5}, O:{col:'#fb7185',r:9.5},
    S:{col:'#fbbf24',r:11}, P:{col:'#fb923c',r:11}, F:{col:'#86efac',r:8},
    Cl:{col:'#86efac',r:10}, Br:{col:'#fca5a5',r:11}, I:{col:'#c4b5fd',r:12},
    B:{col:'#fcd34d',r:9.5}, H:{col:'#e2e8f0',r:6}, '*':{col:'#f0abfc',r:9} };
  function elInfo(e){return ELEM[e]||ELEM['*'];}

  /* ---- SMILES parser (organic subset + brackets, rings, branches, aromatics) ---- */
  const ORG2=['Cl','Br'];
  function bondOrder(ch){return ch==='='?2:ch==='#'?3:ch===':'?4:1;}
  function parseSMILES(smi){
    const atoms=[], bonds=[];
    let prev=-1, pend=null; const stack=[]; const ring={};
    const addBond=(a,b,o)=>{ if(a===b)return; bonds.push({a,b,order:o}); };
    const addAtom=(el,arom)=>{
      const idx=atoms.length; atoms.push({el,arom:!!arom});
      if(prev>=0){ const o = pend!=null?pend : ((atoms[prev].arom&&arom)?4:1); addBond(prev,idx,o); }
      pend=null; prev=idx; return idx;
    };
    const doRing=(label)=>{
      if(ring[label]==null){ ring[label]={atom:prev,bond:pend}; pend=null; }
      else { const r=ring[label]; const o = pend!=null?pend : (r.bond!=null?r.bond : ((atoms[prev].arom&&atoms[r.atom].arom)?4:1));
        addBond(r.atom,prev,o); ring[label]=null; pend=null; }
    };
    let i=0;
    while(i<smi.length){
      const ch=smi[i];
      if(ch==='('){ stack.push(prev); i++; continue; }
      if(ch===')'){ prev=stack.pop(); i++; continue; }
      if(ch==='-'||ch==='='||ch==='#'||ch===':'||ch==='/'||ch==='\\'){ pend=bondOrder(ch); i++; continue; }
      if(ch==='.'){ prev=-1; pend=null; i++; continue; }
      if(ch==='%'){ const lab=parseInt(smi.substr(i+1,2),10); i+=3; doRing('%'+lab); continue; }
      if(ch>='0'&&ch<='9'){ doRing(ch); i++; continue; }
      if(ch==='['){
        const j=smi.indexOf(']',i); const body=smi.slice(i+1,j);
        const m=body.match(/^(\d*)([A-Z][a-z]?|se|as|[bcnops])/);
        let el='*', arom=false;
        if(m){ let sym=m[2]; arom=/^[bcnops]$|^se$|^as$/.test(sym); el = arom? (sym==='se'?'Se':sym==='as'?'As':sym.toUpperCase()) : sym; }
        addAtom(el,arom); i=j+1; continue;
      }
      // organic subset
      const two=smi.substr(i,2);
      if(ORG2.indexOf(two)>=0){ addAtom(two,false); i+=2; continue; }
      if(/[A-Z]/.test(ch)){ addAtom(ch,false); i++; continue; }
      if(/[bcnops]/.test(ch)){ addAtom(ch.toUpperCase(),true); i++; continue; }
      i++; // skip unknown (e.g. H inside? @, +, -, stray)
    }
    // keep only the largest connected component (ligand = one molecule)
    return largestComponent(atoms,bonds);
  }
  function buildAdj(n,bonds){ const adj=Array.from({length:n},()=>[]); bonds.forEach((b,id)=>{ adj[b.a].push({to:b.b,id}); adj[b.b].push({to:b.a,id}); }); return adj; }
  function largestComponent(atoms,bonds){
    const n=atoms.length; if(!n) return {atoms,bonds};
    const adj=buildAdj(n,bonds); const comp=new Array(n).fill(-1); let nc=0; const sizes=[];
    for(let s=0;s<n;s++){ if(comp[s]>=0)continue; const q=[s]; comp[s]=nc; let cnt=0; while(q.length){const u=q.pop();cnt++;for(const e of adj[u])if(comp[e.to]<0){comp[e.to]=nc;q.push(e.to);}} sizes.push(cnt); nc++; }
    if(nc===1) return {atoms,bonds};
    let best=0; for(let c=1;c<nc;c++) if(sizes[c]>sizes[best]) best=c;
    const keep=[]; const remap=new Array(n).fill(-1);
    atoms.forEach((a,i)=>{ if(comp[i]===best){ remap[i]=keep.length; keep.push(a);} });
    const kb=bonds.filter(b=>comp[b.a]===best).map(b=>({a:remap[b.a],b:remap[b.b],order:b.order}));
    return {atoms:keep,bonds:kb};
  }

  /* ---- bridges (Tarjan) ---- */
  function findBridges(n,adj){
    const disc=new Array(n).fill(-1), low=new Array(n).fill(0), br={}; let t=0;
    const st=[];
    for(let s=0;s<n;s++){ if(disc[s]>=0)continue;
      st.push({u:s,pe:-1,ix:0}); disc[s]=low[s]=t++;
      while(st.length){ const fr=st[st.length-1]; const u=fr.u;
        if(fr.ix<adj[u].length){ const e=adj[u][fr.ix++]; if(e.id===fr.pe)continue;
          if(disc[e.to]<0){ disc[e.to]=low[e.to]=t++; st.push({u:e.to,pe:e.id,ix:0}); }
          else low[u]=Math.min(low[u],disc[e.to]);
        } else { st.pop(); if(st.length){ const p=st[st.length-1]; low[p.u]=Math.min(low[p.u],low[u]); if(low[u]>disc[p.u]) br[fr.pe]=true; } }
      }
    }
    return br;
  }

  /* ---- rotatable-bond perception ---- */
  function perceive(mol){
    const {atoms,bonds}=mol; const n=atoms.length; const adj=buildAdj(n,bonds);
    const deg=new Array(n).fill(0); bonds.forEach(b=>{deg[b.a]++;deg[b.b]++;});
    const br=findBridges(n,adj);
    // an aromatic-flagged bond that is a bridge cannot be in a ring → it is a single bond
    const orders=bonds.map((b,id)=> (b.order===4 && br[id]) ? 1 : b.order );
    // amide C–N where C has a double bond to O
    const hasCarbonyl=(c)=>bonds.some((b,k)=>orders[k]===2 && (b.a===c||b.b===c) && atoms[(b.a===c?b.b:b.a)].el==='O');
    const rot={};
    bonds.forEach((b,id)=>{
      if(orders[id]!==1) return;
      if(!br[id]) return;
      if(deg[b.a]<2||deg[b.b]<2) return;
      const ea=atoms[b.a].el, eb=atoms[b.b].el;
      // amide: C–N with carbonyl on the C
      if((ea==='C'&&eb==='N'&&hasCarbonyl(b.a))||(eb==='C'&&ea==='N'&&hasCarbonyl(b.b))) return;
      rot[id]=true;
    });
    return {deg,bridges:br,rotatable:rot,orders};
  }

  /* ---- ring detection (simple cycles among non-bridge bonds) for layout ---- */
  function simpleRings(mol,bridges){
    const {atoms,bonds}=mol; const n=atoms.length;
    const radj=Array.from({length:n},()=>[]);
    bonds.forEach((b,id)=>{ if(!bridges[id]){ radj[b.a].push(b.b); radj[b.b].push(b.a); } });
    const seen=new Array(n).fill(false); const rings=[];
    for(let s=0;s<n;s++){ if(seen[s]||radj[s].length===0)continue;
      // BFS component
      const q=[s], comp=[]; seen[s]=true;
      while(q.length){ const u=q.pop(); comp.push(u); for(const v of radj[u]) if(!seen[v]){seen[v]=true;q.push(v);} }
      // simple cycle iff every vertex has ring-degree 2
      if(comp.every(v=>radj[v].length===2)){
        // order around the cycle
        const order=[comp[0]]; let prev=-1, cur=comp[0];
        for(let k=0;k<comp.length-1;k++){ const nx=radj[cur].find(w=>w!==prev); prev=cur; cur=nx; order.push(cur); }
        rings.push(order);
      }
    }
    return rings;
  }

  /* ---- 2D layout: BFS tree init + ring polygon snap + force relax ---- */
  function layout2D(mol,bridges){
    const {atoms,bonds}=mol; const n=atoms.length; const L=1.4;
    const adj=Array.from({length:n},()=>[]); bonds.forEach(b=>{adj[b.a].push(b.b);adj[b.b].push(b.a);});
    const pos=Array.from({length:n},()=>[0,0]);
    // BFS tree init
    const placed=new Array(n).fill(false);
    const ang=new Array(n).fill(0);
    if(n>0){ const q=[0]; placed[0]=true; pos[0]=[0,0]; ang[0]=Math.PI/2;
      while(q.length){ const u=q.shift(); const nb=adj[u].filter(v=>!placed[v]); let base=ang[u]+Math.PI;
        const spread=Math.PI*2/3; nb.forEach((v,k)=>{ const a=base + (k-(nb.length-1)/2)*(spread/Math.max(1,nb.length));
          pos[v]=[pos[u][0]+L*Math.cos(a), pos[u][1]+L*Math.sin(a)]; ang[v]=a; placed[v]=true; q.push(v); }); }
    }
    // ring polygon snap (overwrite ring atoms to a regular polygon at their centroid)
    simpleRings(mol,bridges).forEach(ring=>{
      const k=ring.length; const cx=ring.reduce((s,i)=>s+pos[i][0],0)/k, cy=ring.reduce((s,i)=>s+pos[i][1],0)/k;
      const R=L/(2*Math.sin(Math.PI/k)); const a0=Math.atan2(pos[ring[0]][1]-cy,pos[ring[0]][0]-cx);
      ring.forEach((idx,j)=>{ const a=a0 + j*2*Math.PI/k; pos[idx]=[cx+R*Math.cos(a), cy+R*Math.sin(a)]; });
    });
    // force relax (spring on bonds + repulsion) — light, to relieve overlaps
    const bonded=new Set(bonds.map(b=>Math.min(b.a,b.b)+'_'+Math.max(b.a,b.b)));
    for(let it=0;it<160;it++){ const f=Array.from({length:n},()=>[0,0]); const cool=1-it/200;
      for(let i=0;i<n;i++)for(let j=i+1;j<n;j++){ const dx=pos[i][0]-pos[j][0],dy=pos[i][1]-pos[j][1]; let d2=dx*dx+dy*dy; if(d2<1e-4)d2=1e-4; const d=Math.sqrt(d2);
        const rep=0.55*L*L/d2; f[i][0]+=rep*dx/d; f[i][1]+=rep*dy/d; f[j][0]-=rep*dx/d; f[j][1]-=rep*dy/d; }
      bonds.forEach(b=>{ const dx=pos[b.b][0]-pos[b.a][0],dy=pos[b.b][1]-pos[b.a][1]; const d=Math.hypot(dx,dy)||1e-4; const s=0.10*(d-L);
        f[b.a][0]+=s*dx/d; f[b.a][1]+=s*dy/d; f[b.b][0]-=s*dx/d; f[b.b][1]-=s*dy/d; });
      for(let i=0;i<n;i++){ const m=Math.hypot(f[i][0],f[i][1]); const cap=0.25*L; const sc=m>cap?cap/m:1; pos[i][0]+=f[i][0]*sc*cool; pos[i][1]+=f[i][1]*sc*cool; }
    }
    // center + scale so median bond length == L
    const cx=pos.reduce((s,p)=>s+p[0],0)/Math.max(1,n), cy=pos.reduce((s,p)=>s+p[1],0)/Math.max(1,n);
    const lens=bonds.map(b=>Math.hypot(pos[b.a][0]-pos[b.b][0],pos[b.a][1]-pos[b.b][1])).sort((x,y)=>x-y);
    const med=lens.length?lens[Math.floor(lens.length/2)]:L; const sc=med>1e-3?L/med:1;
    return pos.map(p=>[ (p[0]-cx)*sc, (p[1]-cy)*sc, 0 ]);
  }

  /* ---- torsion tree (generalized): rigid fragments → fragment tree from largest ---- */
  function buildTree(mol,rotatable){
    const {atoms,bonds}=mol; const n=atoms.length;
    const radj=Array.from({length:n},()=>[]);
    bonds.forEach((b,id)=>{ if(!rotatable[id]){ radj[b.a].push(b.b); radj[b.b].push(b.a); } });
    const frag=new Array(n).fill(-1); const fragments=[];
    for(let s=0;s<n;s++){ if(frag[s]>=0)continue; const fid=fragments.length; const q=[s]; frag[s]=fid; const comp=[];
      while(q.length){ const u=q.pop(); comp.push(u); for(const v of radj[u]) if(frag[v]<0){frag[v]=fid;q.push(v);} } fragments.push(comp); }
    let rootFrag=0; for(let i=1;i<fragments.length;i++) if(fragments[i].length>fragments[rootFrag].length) rootFrag=i;
    // fragment adjacency via rotatable bonds
    const fadj=Array.from({length:fragments.length},()=>[]);
    const rotBonds=[]; bonds.forEach((b,id)=>{ if(rotatable[id]){ rotBonds.push({id,a:b.a,b:b.b}); fadj[frag[b.a]].push({to:frag[b.b],bond:{a:b.a,b:b.b}}); fadj[frag[b.b]].push({to:frag[b.a],bond:{a:b.b,b:b.a}}); } });
    // BFS fragment tree from root; record parent
    const par=new Array(fragments.length).fill(-1); const parBond=new Array(fragments.length).fill(null);
    const vis=new Array(fragments.length).fill(false); const orderF=[]; const q=[rootFrag]; vis[rootFrag]=true;
    while(q.length){ const u=q.shift(); orderF.push(u); for(const e of fadj[u]) if(!vis[e.to]){ vis[e.to]=true; par[e.to]=u; parBond[e.to]={from:e.bond.a,to:e.bond.b}; q.push(e.to); } }
    // downstream atoms for a child fragment = all atoms in its fragment-subtree
    function subtreeAtoms(childFrag){ const out=[]; const qq=[childFrag]; const seen=new Set([childFrag]);
      while(qq.length){ const u=qq.pop(); out.push(...fragments[u]); for(const e of fadj[u]) if(!seen.has(e.to)&&e.to!==par[u]){ /*move away from root*/ }
        // proper: traverse children only (those whose par is u)
      }
      return out; }
    // build proper subtree via parent pointers
    const childrenOf=Array.from({length:fragments.length},()=>[]);
    for(let fcur=0;fcur<fragments.length;fcur++) if(par[fcur]>=0) childrenOf[par[fcur]].push(fcur);
    function subtree(f){ const out=[]; const stk=[f]; while(stk.length){ const u=stk.pop(); out.push(...fragments[u]); for(const c of childrenOf[u]) stk.push(c);} return out; }
    const TORS=[];
    orderF.forEach(f=>{ if(par[f]<0)return; const pb=parBond[f]; // from in parent frag, to in child frag
      TORS.push({ from:pb.from, to:pb.to, moves:subtree(f) }); });
    return { ROOT:fragments[rootFrag].slice(), TORS, frag, fragCount:fragments.length };
  }

  /* ---- assemble a full ligand model from SMILES ---- */
  function buildLigand(smi){
    const mol=parseSMILES(smi);
    if(!mol.atoms.length) return {ok:false,err:'No atoms parsed'};
    if(mol.atoms.length>140) return {ok:false,err:'Too large for client preview ('+mol.atoms.length+' heavy atoms)'};
    const per=perceive(mol);
    const ref=layout2D(mol,per.bridges);
    const tree=buildTree(mol,per.rotatable);
    const rotIds=Object.keys(per.rotatable).map(Number);
    // formula
    const counts={}; mol.atoms.forEach(a=>counts[a.el]=(counts[a.el]||0)+1);
    const order=['C','H','N','O','S','P','F','Cl','Br','I'];
    let formula=''; order.forEach(e=>{ if(counts[e]) formula+=e+(counts[e]>1?counts[e]:''); });
    Object.keys(counts).sort().forEach(e=>{ if(order.indexOf(e)<0) formula+=e+(counts[e]>1?counts[e]:''); });
    // bond metadata: type + torsion index
    const torsionOfBond={}; // bondId -> torsion index in TORS
    // map each rotatable bond to its TORS entry (match by endpoints)
    mol.bonds.forEach((b,id)=>{ if(per.rotatable[id]){ const ti=tree.TORS.findIndex(t=>(t.from===b.a&&t.to===b.b)||(t.from===b.b&&t.to===b.a)); torsionOfBond[id]=ti; } });
    const BONDS=mol.bonds.map((b,id)=>({a:b.a,b:b.b,order:per.orders[id],
      type: per.rotatable[id]?'rot':(per.bridges[id]?'rigid':'ring'),
      ti: per.rotatable[id]?torsionOfBond[id]:-1 }));
    const TINFO=tree.TORS.map((t,i)=>({ n: mol.atoms[t.from].el+'–'+mol.atoms[t.to].el, a:'A'+t.from+'–A'+t.to }));
    return {ok:true, smiles:smi, atoms:mol.atoms.map(a=>a.el), arom:mol.atoms.map(a=>a.arom),
      REF:ref, BONDS, ROOT:tree.ROOT, TORS:tree.TORS, TINFO, N:tree.TORS.length, formula,
      counts };
  }

  /* ---- compute pose: torsions (parent-first) then quaternion rotation + translation ---- */
  function computePose(L,quat,trans,thetas){
    const c=L.REF.map(p=>p.slice());
    L.TORS.forEach((t,i)=>{ const ang=thetas[i]||0; if(!ang)return; const k=nrm(sub(c[t.to],c[t.from])), piv=c[t.from];
      t.moves.forEach(id=>{ if(id!==t.from) c[id]=rotAxis(c[id],piv,k,ang); }); });
    const cen=L.ROOT.reduce((a,id)=>add(a,c[id]),[0,0,0]).map(x=>x/Math.max(1,L.ROOT.length));
    const R=quatToMat(normQuat(quat));
    return c.map(p=>add(add(matVec(R,sub(p,cen)),cen),trans));
  }

  /* ---- PDB parser ---- */
  function parsePDB(text){
    const ca=[], het=[], atoms=[]; const chains={}; const resset={}; let natoms=0;
    let mn=[1e9,1e9,1e9], mx=[-1e9,-1e9,-1e9];
    const TWO=['CL','BR','FE','ZN','MG','MN','SE','NA','CA','CU','NI','CO','HG','CD','K','I','F','P','S','B'];
    function elemOf(ln,name,isHet){
      let e=ln.slice(76,78).trim();
      if(e){return e[0].toUpperCase()+e.slice(1).toLowerCase();}
      let nm=name.replace(/[0-9'"]/g,'').trim().toUpperCase();
      if(isHet&&nm.length>=2&&TWO.indexOf(nm.slice(0,2))>=0){const t=nm.slice(0,2);return t[0]+t.slice(1).toLowerCase();}
      return (nm[0]||'C');
    }
    const lines=text.split('\n');
    for(const ln of lines){ const rec=ln.slice(0,6);
      if(rec==='ATOM  '||rec==='HETATM'){
        const x=parseFloat(ln.slice(30,38)), y=parseFloat(ln.slice(38,46)), z=parseFloat(ln.slice(46,54));
        if(isNaN(x)||isNaN(y)||isNaN(z))continue; natoms++;
        if(x<mn[0])mn[0]=x; if(y<mn[1])mn[1]=y; if(z<mn[2])mn[2]=z;
        if(x>mx[0])mx[0]=x; if(y>mx[1])mx[1]=y; if(z>mx[2])mx[2]=z;
        const name=ln.slice(12,16).trim(); const chain=ln[21]||'A'; const resn=ln.slice(17,20).trim(); const resseq=ln.slice(22,26).trim();
        const isHet=rec==='HETATM'; const water=(resn==='HOH'||resn==='WAT');
        const el=elemOf(ln,name,isHet);
        if(!water) atoms.push({x,y,z,el,name,resn,resseq,chain,het:isHet});
        if(rec==='ATOM  '){ chains[chain]=1; resset[chain+'|'+resseq]=1; if(name==='CA') ca.push([x,y,z,chain]); }
        else { if(!water&&resn!=='SO4'&&resn!=='PO4') het.push([x,y,z,resn]); }
      }
    }
    const protCenter=[(mn[0]+mx[0])/2,(mn[1]+mx[1])/2,(mn[2]+mx[2])/2];
    let hetCenter=null, hetSize=null;
    if(het.length){ hetCenter=[0,0,0]; het.forEach(h=>{hetCenter[0]+=h[0];hetCenter[1]+=h[1];hetCenter[2]+=h[2];}); hetCenter=hetCenter.map(v=>v/het.length);
      let hmn=[1e9,1e9,1e9],hmx=[-1e9,-1e9,-1e9]; het.forEach(h=>{for(let k=0;k<3;k++){if(h[k]<hmn[k])hmn[k]=h[k];if(h[k]>hmx[k])hmx[k]=h[k];}});
      hetSize=Math.max(16, Math.ceil(Math.max(hmx[0]-hmn[0],hmx[1]-hmn[1],hmx[2]-hmn[2])+8)); }
    const center=hetCenter||protCenter;
    const size=hetSize||22;
    return { ca, het, atoms, chains:Object.keys(chains), nres:Object.keys(resset).length, natoms, min:mn, max:mx,
      center, size, hasLigand:het.length>0, protCenter };
  }

  const API={ sub,add,dot,cross,nrm,matVec,matMul,rotX,rotY,quatToMat,normQuat,eulerToQuat,randomQuat,rotAxis,q2e,
    elInfo, parseSMILES, perceive, simpleRings, layout2D, buildTree, buildLigand, computePose, parsePDB, findBridges, buildAdj };
  if(typeof module!=='undefined'&&module.exports) module.exports=API; else root.PoseEngine=API;
})(typeof window!=='undefined'?window:globalThis);
/* ════════════════ Pose Generation — UI controller (uses PoseEngine) ════════════════ */
(function(){
  const PE=window.PoseEngine;
  const POSE_LIGAND_API='/pose/ligand';   // RDKit backend; falls back to in-browser engine if unavailable
  const $=id=>document.getElementById(id);
  const CAM=PE.matMul(PE.rotY(0.32),PE.rotX(-0.36));

  /* ---- helpers ---- */
  function molExtent(L){let m=0;for(const p of L.REF){const d=Math.hypot(p[0],p[1],p[2]);if(d>m)m=d;}return m||1;}
  function fitScale(L,px){return px/molExtent(L);}
  function boxParams(L){const ext=molExtent(L),half=Math.max(4,ext*1.3),S=150/half,tmax=Math.max(0.6,(half-ext)*0.85);return {ext,half,S,tmax};}
  function kv(k,v,col){return '<div style="display:flex;justify-content:space-between;font-family:ui-monospace,monospace;font-size:11px;padding:3px 0;border-bottom:1px solid rgba(30,41,59,.5);"><span style="color:#64748b;">'+k+'</span><span style="color:'+(col||'#e2e8f0')+';">'+v+'</span></div>';}

  /* ---- render ---- */
  function project(coords,cam,scale,cx,cy){return coords.map(p=>{const cp=PE.matVec(cam,p);return {x:cx+scale*cp[0],y:cy-scale*cp[1],z:cp[2]};});}
  function svgMol(L,coords,cam,opt){
    const scale=opt.scale,cx=opt.cx,cy=opt.cy,labels=opt.labels,rootHi=opt.rootHi,hot=opt.hot==null?-1:opt.hot,small=opt.small;
    const P=project(coords,cam,scale,cx,cy);let s='';
    const root=new Set(L.ROOT);
    L.BONDS.forEach(bd=>{const A=P[bd.a],B=P[bd.b];const dep=(A.z+B.z)/2;const ds=Math.min(1.4,Math.max(.7,1+dep*0.07));
      let col=bd.type==='rot'?'#a78bfa':bd.type==='ring'?'#5b6b86':'#647089';let w=(small?1.6:3.0)*ds;
      if(bd.type==='rot'&&bd.ti===hot){col='#c4b5fd';w*=1.5;}
      if(bd.order===2){const dx=B.x-A.x,dy=B.y-A.y,Ln=Math.hypot(dx,dy)||1,ox=-dy/Ln*2.2,oy=dx/Ln*2.2;
        s+='<line x1="'+(A.x+ox)+'" y1="'+(A.y+oy)+'" x2="'+(B.x+ox)+'" y2="'+(B.y+oy)+'" stroke="'+col+'" stroke-width="'+(w*.8)+'" stroke-linecap="round"/>';
        s+='<line x1="'+(A.x-ox)+'" y1="'+(A.y-oy)+'" x2="'+(B.x-ox)+'" y2="'+(B.y-oy)+'" stroke="'+col+'" stroke-width="'+(w*.8)+'" stroke-linecap="round"/>';
      } else { s+='<line x1="'+A.x+'" y1="'+A.y+'" x2="'+B.x+'" y2="'+B.y+'" stroke="'+col+'" stroke-width="'+w+'" stroke-linecap="round"/>'; }
      if(bd.type==='rot'&&!small){const mx=(A.x+B.x)/2,my=(A.y+B.y)/2,r=bd.ti===hot?6:4.5;
        s+='<circle cx="'+mx+'" cy="'+my+'" r="'+r+'" fill="none" stroke="'+(bd.ti===hot?'#c4b5fd':'#a78bfa')+'" stroke-width="1.3" stroke-dasharray="2.2 2"/>';}
    });
    const ord=P.map((p,i)=>i).sort((i,j)=>P[i].z-P[j].z);
    ord.forEach(i=>{const p=P[i],el=L.atoms[i],info=PE.elInfo(el),dep=p.z,ds=Math.min(1.45,Math.max(.65,1+dep*0.07));const r=(small?info.r*0.42:info.r)*ds;
      if(rootHi&&root.has(i))s+='<circle cx="'+p.x+'" cy="'+p.y+'" r="'+(r+5)+'" fill="none" stroke="#22d3ee" stroke-width="1.6" opacity=".55"/>';
      s+='<circle cx="'+p.x+'" cy="'+p.y+'" r="'+r+'" fill="'+elcol(el,true)+'" stroke="#05070f" stroke-width="'+(small?1:1.5)+'"/>';
      s+='<circle cx="'+(p.x-r*0.3)+'" cy="'+(p.y-r*0.3)+'" r="'+(r*0.34)+'" fill="#fff" opacity=".3"/>';
      if(labels&&!small&&el!=='C')s+='<text x="'+p.x+'" y="'+(p.y+r+11)+'" font-size="9.5" fill="#94a3b8" text-anchor="middle" font-family="ui-monospace,monospace">'+el+'</text>';
    });
    return s;
  }
  function svgBoxWire(cam,scale,cx,cy,B){
    const c=[[-B,-B,-B],[B,-B,-B],[B,B,-B],[-B,B,-B],[-B,-B,B],[B,-B,B],[B,B,B],[-B,B,B]];
    const P=project(c,cam,scale,cx,cy);const E=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];let s='';
    E.forEach((e,idx)=>{s+='<line x1="'+P[e[0]].x+'" y1="'+P[e[0]].y+'" x2="'+P[e[1]].x+'" y2="'+P[e[1]].y+'" stroke="#22d3ee" stroke-width="1" opacity="'+(idx<4?'.22':'.42')+'" stroke-dasharray="'+(idx<4?'4 4':'0')+'"/>';});
    return s;
  }
  /* ---- Plotly 3D: ball-and-stick pocket + ligand + interactions ---- */
  function layout3D(rev){const ax={showgrid:false,zeroline:false,showticklabels:false};   // matches vina.js _sceneP
    return {paper_bgcolor:'transparent',plot_bgcolor:'transparent',margin:{l:0,r:0,t:0,b:0},showlegend:false,font:{color:'#94a3b8'},uirevision:(rev||'none'),
      scene:{xaxis:ax,yaxis:ax,zaxis:ax,aspectmode:'data',bgcolor:'#05070f',dragmode:'orbit',uirevision:(rev||'none'),camera:{eye:{x:1.35,y:1.35,z:1.05}}}};}
  function boxTrace3D(c,size){const h=size/2;const C=[[-h,-h,-h],[h,-h,-h],[h,h,-h],[-h,h,-h],[-h,-h,h],[h,-h,h],[h,h,h],[-h,h,h]].map(d=>[c[0]+d[0],c[1]+d[1],c[2]+d[2]]);
    const E=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];const x=[],y=[],z=[];
    E.forEach(e=>{x.push(C[e[0]][0],C[e[1]][0],null);y.push(C[e[0]][1],C[e[1]][1],null);z.push(C[e[0]][2],C[e[1]][2],null);});
    return {type:'scatter3d',mode:'lines',x:x,y:y,z:z,line:{color:'#22d3ee',width:3,dash:'dot'},hoverinfo:'skip',showlegend:false};}

  const COV={H:0.31,C:0.76,N:0.71,O:0.66,S:1.05,P:1.07,F:0.57,Cl:1.02,Br:1.20,I:1.39,Se:1.20,B:0.84,Zn:1.22,Fe:1.32,Mg:1.41,Mn:1.39,Ca:1.76,Na:1.66,K:2.03,Cu:1.32,Ni:1.24,Co:1.26};
  function covr(el){return COV[el]||0.76;}
  function elcol(el,ligC,cCol){ if(el==='C')return cCol||(ligC?'#22c55e':'#97a3b0');
    const M={N:'#3b6cf6',O:'#ef4444',S:'#e6b800',P:'#f97316',H:'#e3eaf1',F:'#67e8f9',Cl:'#22c55e',Br:'#b45309',I:'#7c3aed',Se:'#f59e0b'};
    return M[el]||(PE.elInfo(el)||{}).col||'#c4b5fd'; }
  function dist(a,b){return Math.hypot(a[0]-b[0],a[1]-b[1],a[2]-b[2]);}

  // covalent bonds by distance (one-off per pocket)
  function inferBonds(atoms){const b=[];const n=atoms.length;
    for(let i=0;i<n;i++){const a=atoms[i];for(let j=i+1;j<n;j++){const c=atoms[j];
      const dx=a.x-c.x,dy=a.y-c.y,dz=a.z-c.z;const d2=dx*dx+dy*dy+dz*dz;
      const mx=covr(a.el)+covr(c.el)+0.45; if(d2<mx*mx&&d2>0.20)b.push([i,j]); }}
    return b;}
  /* ---- ☰ View › H Hydrogens: drop H atoms and repair the bond indices --------
     Bonds are index pairs INTO the atom array, so removing atoms invalidates
     every index after the first removal. Build a forward map in the same pass,
     then both REMAP the survivors and DROP any bond that touched a hydrogen.
     A plain atoms.filter() leaves bonds pointing at the wrong atoms — which
     renders as sticks joining unrelated atoms, i.e. it looks like a chemistry
     bug rather than an indexing one.

     Two bond shapes are in play here and both must survive:
       ligand  L.BONDS      -> {a, b, order}   (order and any extra fields kept)
       protein inferBonds() -> [i, j]          (plain pairs)
     stickTraces accepts either, so this returns whichever it was given.

     Cosmetic only. Every analytical path already drops H on its own and does
     not consult this flag: _surfaceTraces (pose.js ~1725/1753), bond-order
     perception (~428), the Vina engine's typing, and interactions(), which
     reads L.atoms / world / site.atoms directly rather than the draw arrays. */
  function stripH(atoms,bonds){
    const n=atoms?atoms.length:0,map=new Array(n),keep=[];
    for(let i=0;i<n;i++){
      if(atoms[i]&&atoms[i].el!=='H'){map[i]=keep.length;keep.push(atoms[i]);}else map[i]=-1;
    }
    const nb=[];
    (bonds||[]).forEach(bd=>{
      if(!bd)return;
      const arr=Array.isArray(bd);
      const i0=arr?bd[0]:bd.a, j0=arr?bd[1]:bd.b;
      const ii=(i0>=0&&i0<n)?map[i0]:-1, jj=(j0>=0&&j0<n)?map[j0]:-1;
      if(ii<0||jj<0)return;                                  // touched a hydrogen -> drop
      nb.push(arr?(bd.length>2?[ii,jj,bd[2]]:[ii,jj]):Object.assign({},bd,{a:ii,b:jj}));
    });
    return {atoms:keep,bonds:nb,map:map,removed:n-keep.length};
  }
  /* ---- perceive bond orders for a coordinates-only ligand (PDB/PDBQT) --------
     PDB files carry no bond orders, so a benzene ring or an amide C=O renders as
     plain single sticks. This recovers orders from geometry so stickTraces can
     draw the inner parallel line:
       · aromatic rings: 5-/6-membered rings of C/N/O/S that are planar (all atoms
         within ~0.35 Å of the mean plane) → alternating ring bonds order-2
         (Kekulé), so a 6-ring gets 3 inner lines like benzene.
       · isolated doubles: a non-ring bond noticeably shorter than the single-bond
         reference (carbonyl C=O, imine C=N, alkene C=C) → order-2.
     Heuristic, not a full valence model, but it matches Maestro for the common
     cases without a chemistry toolkit. `bonds` is [i,j] from inferBonds. */
  function perceivePdbBondOrders(atoms, bonds){
    const n=atoms.length;
    const order=bonds.map(()=>1);
    const bidx={}; bonds.forEach((b,i)=>{bidx[(b[0]<b[1]?b[0]+'_'+b[1]:b[1]+'_'+b[0])]=i;});
    const bkey=(i,j)=>i<j?i+'_'+j:j+'_'+i;
    const adj=Array.from({length:n},()=>[]);
    bonds.forEach(b=>{adj[b[0]].push(b[1]);adj[b[1]].push(b[0]);});
    const P=i=>atoms[i], D=(i,j)=>Math.hypot(P(i).x-P(j).x,P(i).y-P(j).y,P(i).z-P(j).z);

    // small rings (size 5,6) via bounded DFS, dedup by sorted membership
    const seen={}, rings=[];
    const dfs=(start,cur,prev,path)=>{
      if(path.length>6)return;
      for(const nx of adj[cur]){
        if(nx===prev)continue;
        if(nx===start&&path.length>=5&&path.length<=6){
          const key=path.slice().sort((a,b)=>a-b).join(','); if(!seen[key]){seen[key]=1;rings.push(path.slice());}
          continue;
        }
        if(path.indexOf(nx)>=0)continue;
        path.push(nx); dfs(start,nx,cur,path); path.pop();
      }
    };
    for(let i=0;i<n;i++){ if(adj[i].length>=2) dfs(i,i,-1,[i]); }

    const aromaticEl=el=>el==='C'||el==='N'||el==='O'||el==='S';
    const usedDbl=new Array(n).fill(false);   // each atom takes at most one double

    rings.forEach(ring=>{
      const m=ring.length;
      if(!ring.every(i=>aromaticEl(P(i).el)))return;
      const a=P(ring[0]),b=P(ring[1]),c=P(ring[2]);
      let ux=b.x-a.x,uy=b.y-a.y,uz=b.z-a.z, vx=c.x-a.x,vy=c.y-a.y,vz=c.z-a.z;
      let nx=uy*vz-uz*vy, ny=uz*vx-ux*vz, nz=ux*vy-uy*vx; const nl=Math.hypot(nx,ny,nz);
      if(nl<1e-3)return; nx/=nl;ny/=nl;nz/=nl;
      let maxdev=0; ring.forEach(i=>{const dv=Math.abs((P(i).x-a.x)*nx+(P(i).y-a.y)*ny+(P(i).z-a.z)*nz); if(dv>maxdev)maxdev=dv;});
      if(maxdev>0.35)return;                                   // non-planar → not aromatic
      const cyc=[ring[0]]; let prev=-1,cur=ring[0];
      for(let k=0;k<m-1;k++){const nxt=adj[cur].find(w=>w!==prev&&ring.indexOf(w)>=0&&cyc.indexOf(w)<0); if(nxt==null)break; prev=cur;cur=nxt;cyc.push(cur);}
      if(cyc.length!==m)return;
      for(let k=0;k<m;k++){
        const i=cyc[k], j=cyc[(k+1)%m];
        if(usedDbl[i]||usedDbl[j])continue;
        const bi=bidx[bkey(i,j)]; if(bi==null)continue;
        if(D(i,j)>1.62)continue;
        order[bi]=2; usedDbl[i]=true; usedDbl[j]=true;
      }
    });

    // isolated (non-ring) doubles by short length
    const REF={'C-C':1.54,'C-N':1.47,'C-O':1.43,'N-O':1.44,'C-S':1.82,'N-N':1.45,'P-O':1.63,'S-O':1.57};
    bonds.forEach((b,i)=>{
      if(order[i]!==1)return;
      const ea=P(b[0]).el, eb=P(b[1]).el; if(ea==='H'||eb==='H')return;
      if(usedDbl[b[0]]||usedDbl[b[1]])return;
      const ref=REF[[ea,eb].sort().join('-')]; if(!ref)return;
      if(D(b[0],b[1]) < ref-0.13){ order[i]=2; usedDbl[b[0]]=true; usedDbl[b[1]]=true; }
    });
    return order;
  }
  /* ---- build a rigid ligand model straight from PDB atoms (for the uploaded-ligand
     gallery): real 3D coordinates, distance-inferred bonds, no torsion tree. It is
     rendered green by the shared pose path exactly like a SMILES-built ligand. ---- */
  function ligandFromPDB(p){
    if(!p||!p.atoms||!p.atoms.length)return null;
    var src=p.atoms.filter(function(a){return a.het;});   // ligand files usually use HETATM…
    if(!src.length)src=p.atoms.slice();                   // …but fall back to ATOM records
    if(!src.length)return null;
    var ctr=(p.center&&p.center.length===3&&p.center.every(function(v){return isFinite(v);}))
      ? p.center
      : (function(){var c=[0,0,0];src.forEach(function(a){c[0]+=a.x;c[1]+=a.y;c[2]+=a.z;});return c.map(function(v){return v/src.length;});})();
    var REF=src.map(function(a){return [a.x-ctr[0],a.y-ctr[1],a.z-ctr[2]];});   // recentre on box centre → renders at true coords
    var els=src.map(function(a){return a.el;});
    var _conn=inferBonds(src);
    var _ord=perceivePdbBondOrders(src,_conn);   // recover double/aromatic bonds from geometry
    var BONDS=_conn.map(function(pr,i){return {a:pr[0],b:pr[1],order:_ord[i],type:'ring',ti:-1};});
    var counts={};els.forEach(function(e){counts[e]=(counts[e]||0)+1;});
    var ORD=['C','H','N','O','S','P','F','Cl','Br','I'];var formula='';
    ORD.forEach(function(e){if(counts[e])formula+=e+(counts[e]>1?counts[e]:'');});
    Object.keys(counts).sort().forEach(function(e){if(ORD.indexOf(e)<0)formula+=e+(counts[e]>1?counts[e]:'');});
    return {ok:true,smiles:'',atoms:els,arom:els.map(function(){return false;}),REF:REF,BONDS:BONDS,
      ROOT:els.map(function(_,i){return i;}),TORS:[],TINFO:[],N:0,formula:formula,counts:counts,source:'pdb',
      _realPose:true,_realCenter:ctr.slice()};   // true docked coordinates → keep the box anchored here across a receptor load
  }
  // two-tone sticks (split each bond at midpoint) grouped by colour + atom spheres
  function stickTraces(atoms,bonds,ligC,width,mksz,cCol){
    const byCol={}; const half=(col,p,m)=>{const g=byCol[col]||(byCol[col]={x:[],y:[],z:[]});g.x.push(p.x,m.x,null);g.y.push(p.y,m.y,null);g.z.push(p.z,m.z,null);};
    // A bond may be [i,j] (connectivity only) or {a,b,order}. Normalise, and for
    // double/triple bonds add extra parallel line(s) offset perpendicular to the
    // bond. For a RING bond the offset points toward that ring's OWN centre, so
    // the inner line always sits inside its ring — not toward the whole-molecule
    // centroid, which for a fused/substituted ring lies off to one side and would
    // push one bond's inner line OUTSIDE the ring (the bug this fixes).
    const seg=(a,c,ox,oy,oz)=>{const A={x:a.x+ox,y:a.y+oy,z:a.z+oz},C={x:c.x+ox,y:c.y+oy,z:c.z+oz};
      const m={x:(A.x+C.x)/2,y:(A.y+C.y)/2,z:(A.z+C.z)/2};half(elcol(a.el,ligC,cCol),A,m);half(elcol(c.el,ligC,cCol),C,m);};

    // local connectivity for ring finding
    const N=atoms.length; const adj=Array.from({length:N},()=>[]);
    bonds.forEach(pr=>{const ia=Array.isArray(pr)?pr[0]:pr.a, ib=Array.isArray(pr)?pr[1]:pr.b; if(ia==null||ib==null)return; adj[ia].push(ib); adj[ib].push(ia);});
    // find small rings (size 3–7), dedup by membership; map each edge → a ring centroid
    const ringSeen={}; const edgeRingCentre={};
    const dfsR=(start,cur,prev,path)=>{
      if(path.length>7)return;
      for(const nx of adj[cur]){
        if(nx===prev)continue;
        if(nx===start&&path.length>=3&&path.length<=7){
          const key=path.slice().sort((a,b)=>a-b).join(',');
          if(!ringSeen[key]){ ringSeen[key]=1;
            let cx=0,cy=0,cz=0; path.forEach(i=>{cx+=atoms[i].x;cy+=atoms[i].y;cz+=atoms[i].z;});
            const L=path.length; cx/=L;cy/=L;cz/=L;
            for(let k=0;k<L;k++){const i=path[k],j=path[(k+1)%L];const ek=i<j?i+'_'+j:j+'_'+i;
              // if an edge is shared by two rings, keep the smaller ring's centre (tighter/inner)
              const pc=edgeRingCentre[ek];
              if(!pc||L<pc.n){edgeRingCentre[ek]={x:cx,y:cy,z:cz,n:L};}}
          }
          continue;
        }
        if(path.indexOf(nx)>=0)continue;
        path.push(nx); dfsR(start,nx,cur,path); path.pop();
      }
    };
    for(let i=0;i<N;i++){ if(adj[i].length>=2) dfsR(i,i,-1,[i]); }

    // molecular centroid — fallback for non-ring (chain) double bonds only
    let gx=0,gy=0,gz=0; atoms.forEach(a=>{gx+=a.x;gy+=a.y;gz+=a.z;}); const nA=atoms.length||1; gx/=nA;gy/=nA;gz/=nA;
    const off=0.28;   // Å between the parallel lines of a multiple bond
    bonds.forEach(pr=>{
      const ia=Array.isArray(pr)?pr[0]:pr.a, ib=Array.isArray(pr)?pr[1]:pr.b;
      const ord=Array.isArray(pr)?1:(pr.order||1);
      const a=atoms[ia],c=atoms[ib]; if(!a||!c)return;
      seg(a,c,0,0,0);                                          // the primary line (always drawn)
      if(ord!==2&&ord!==3)return;
      let dx=c.x-a.x,dy=c.y-a.y,dz=c.z-a.z; const dl=Math.hypot(dx,dy,dz)||1; dx/=dl;dy/=dl;dz/=dl;
      // reference point the inner line should lean toward: this bond's ring centre
      // if it is a ring bond, else the molecular centroid.
      const ek=ia<ib?ia+'_'+ib:ib+'_'+ia; const rc=edgeRingCentre[ek];
      const rx=rc?rc.x:gx, ry=rc?rc.y:gy, rz=rc?rc.z:gz;
      const mx=(a.x+c.x)/2,my=(a.y+c.y)/2,mz=(a.z+c.z)/2;
      let px=rx-mx,py=ry-my,pz=rz-mz; const pd=px*dx+py*dy+pz*dz; px-=pd*dx;py-=pd*dy;pz-=pd*dz;   // project off bond axis → perpendicular, pointing inward
      let pl=Math.hypot(px,py,pz);
      if(pl<1e-3){ // degenerate (e.g. C=O off a chain): any perpendicular will do
        px=dy;py=-dx;pz=0; if(Math.hypot(px,py,pz)<1e-3){px=dz;py=0;pz=-dx;} pl=Math.hypot(px,py,pz)||1; }
      px/=pl;py/=pl;pz/=pl;
      if(ord===2){ seg(a,c,px*off,py*off,pz*off); }           // inner parallel line
      else { seg(a,c,px*off,py*off,pz*off); seg(a,c,-px*off,-py*off,-pz*off); }   // triple: both sides
    });
    const traces=Object.keys(byCol).map(col=>({type:'scatter3d',mode:'lines',x:byCol[col].x,y:byCol[col].y,z:byCol[col].z,line:{color:col,width:width},hoverinfo:'skip',showlegend:false}));
    traces.push({type:'scatter3d',mode:'markers',x:atoms.map(a=>a.x),y:atoms.map(a=>a.y),z:atoms.map(a=>a.z),
      marker:{color:atoms.map(a=>elcol(a.el,ligC,cCol)),size:atoms.map(a=>a.el==='H'?mksz*0.55:(a.el==='C'?mksz:mksz*1.3)),line:{width:0}},
      text:atoms.map(a=>(a.name||a.el)+(a.resn?(' · '+a.resn+a.resseq):'')),hoverinfo:ligC?'skip':'text',showlegend:false});
    return traces;
  }
  // dashed 3D line → on/off sub-segments
  function pushDash(acc,p,q,dash,gap){const dx=q[0]-p[0],dy=q[1]-p[1],dz=q[2]-p[2];const L=Math.hypot(dx,dy,dz)||1;const ux=dx/L,uy=dy/L,uz=dz/L;let t=0,on=true;
    while(t<L-1e-6){const t2=Math.min(L,t+(on?dash:gap));if(on){acc.x.push(p[0]+ux*t,p[0]+ux*t2,null);acc.y.push(p[1]+uy*t,p[1]+uy*t2,null);acc.z.push(p[2]+uz*t,p[2]+uz*t2,null);}t=t2;on=!on;}}
  // smallest 5-/6-ring through each ligand bond
  function smallRings(L){const n=L.atoms.length;const adj=Array.from({length:n},()=>[]);
    L.BONDS.forEach(b=>{adj[b.a].push(b.b);adj[b.b].push(b.a);});const seen={};const rings=[];
    L.BONDS.forEach(b=>{const u=b.a,v=b.b;const ded={},prev={};ded[u]=0;prev[u]=-1;const q=[u];let qi=0;
      while(qi<q.length){const x=q[qi++];for(const w of adj[x]){if((x===u&&w===v)||(x===v&&w===u))continue;if(ded[w]===undefined){ded[w]=ded[x]+1;prev[w]=x;q.push(w);}}}
      if(ded[v]!==undefined){const sz=ded[v]+1;if(sz>=5&&sz<=6){const path=[];let c=v;while(c!==-1&&c!==undefined){path.push(c);c=prev[c];}const key=path.slice().sort((a,b)=>a-b).join(',');if(!seen[key]){seen[key]=1;rings.push(path);}}}});
    return rings;}
  function ringCentroid(coords,idx){const c=[0,0,0];idx.forEach(i=>{c[0]+=coords[i][0];c[1]+=coords[i][1];c[2]+=coords[i][2];});return [c[0]/idx.length,c[1]/idx.length,c[2]/idx.length];}
  // ligand carboxylate O atoms (C bonded to ≥2 O) — anionic
  function ligCarboxylate(L,coords){const out=[];const adj={};L.BONDS.forEach(b=>{(adj[b.a]||(adj[b.a]=[])).push(b.b);(adj[b.b]||(adj[b.b]=[])).push(b.a);});
    L.atoms.forEach((el,i)=>{if(el==='C'){const os=(adj[i]||[]).filter(j=>L.atoms[j]==='O');if(os.length>=2)os.forEach(j=>out.push(coords[j]));}});return out;}
  // aromatic-residue ring centroids in the pocket
  function protAromatics(sel){const byRes={};sel.forEach(a=>{const k=a.chain+'|'+a.resseq;(byRes[k]||(byRes[k]={resn:a.resn,at:{}})).at[a.name]=[a.x,a.y,a.z];});
    const want={PHE:['CG','CD1','CD2','CE1','CE2','CZ'],TYR:['CG','CD1','CD2','CE1','CE2','CZ'],HIS:['CG','ND1','CD2','CE1','NE2'],TRP:['CD2','CE2','CE3','CZ2','CH2','CZ3']};const out=[];
    Object.keys(byRes).forEach(k=>{const r=byRes[k];const names=want[r.resn];if(!names)return;const pts=names.map(n=>r.at[n]).filter(Boolean);if(pts.length>=4){const c=[0,0,0];pts.forEach(p=>{c[0]+=p[0];c[1]+=p[1];c[2]+=p[2];});out.push([c[0]/pts.length,c[1]/pts.length,c[2]/pts.length]);}});return out;}
  // ligand↔pocket interactions for the current pose
  function interactions(L,coords,site){
    const hb={x:[],y:[],z:[]},sb={x:[],y:[],z:[]},pp={x:[],y:[],z:[]};
    const ligNO=[];L.atoms.forEach((el,i)=>{if(el==='N'||el==='O')ligNO.push({el,p:coords[i]});});
    const protNO=site.atoms.filter(a=>a.el==='N'||a.el==='O');
    ligNO.forEach(l=>protNO.forEach(a=>{if(dist(l.p,[a.x,a.y,a.z])<3.5)pushDash(hb,l.p,[a.x,a.y,a.z],0.32,0.22);}));
    const acidicO=site.atoms.filter(a=>(a.resn==='ASP'||a.resn==='GLU')&&a.el==='O'&&a.name[0]==='O'&&(a.name.indexOf('D')>0||a.name.indexOf('E')>0));
    const basicN=site.atoms.filter(a=>((a.resn==='LYS'&&a.name==='NZ')||(a.resn==='ARG'&&a.name[0]==='N'&&a.name!=='N')||(a.resn==='HIS'&&(a.name==='ND1'||a.name==='NE2'))));
    const ligN=ligNO.filter(l=>l.el==='N');const ligAnion=ligCarboxylate(L,coords);
    ligN.forEach(l=>acidicO.forEach(a=>{if(dist(l.p,[a.x,a.y,a.z])<4.2)pushDash(sb,l.p,[a.x,a.y,a.z],0.5,0.3);}));
    ligAnion.forEach(p=>basicN.forEach(a=>{if(dist(p,[a.x,a.y,a.z])<4.2)pushDash(sb,p,[a.x,a.y,a.z],0.5,0.3);}));
    const lrings=smallRings(L).filter(r=>r.every(i=>'CNOS'.indexOf(L.atoms[i])>=0)).map(r=>ringCentroid(coords,r));
    site.aromatics.forEach(pc=>lrings.forEach(rc=>{if(dist(rc,pc)<6.0)pushDash(pp,rc,pc,0.55,0.35);}));
    const traces=[];const mk=(acc,col)=>{if(acc.x.length)traces.push({type:'scatter3d',mode:'lines',x:acc.x,y:acc.y,z:acc.z,line:{color:col,width:4},hoverinfo:'skip',showlegend:false});};
    mk(hb,'#fde047');mk(sb,'#d946ef');mk(pp,'#38bdf8');
    return {traces,n:{hb:hb.x.length/3,sb:sb.x.length/3,pp:pp.x.length/3}};
  }

  function setSvg(id,inner,vb){const el=$(id);if(!el)return;el.setAttribute('viewBox',vb);el.innerHTML=inner;}

  /* ---- state ---- */
  let L=null, protein=null, box={center:[0,0,0],size:20}, cfgBox=null;

  const PG={
    open(){var m=$('poseModal');m.classList.remove('hidden');m.classList.add('flex');PG.stage('random');PG._loadConfig();},
    /* fetch the default receptor .pdb configured in input_TS.yml (pose.default_receptor_pdb) */
    /* fetch pose config: cache the default docking box (applied to EVERY receptor load —
       default OR uploaded), pre-fill the output-dir field, seed the SMILES box from
       pose.default_smile, load the default receptor, then build once. */
    _loadConfig(){var built=false;var doBuild=function(){if(!built&&!L){built=true;PG.build();}};
      fetch('/pose/default_receptor').then(function(r){return r.json();}).then(function(res){
      res=res||{};
      try{
        if(res.box)cfgBox=res.box;
        if(res.out_dir){var od=$('poseDaOutDir');if(od&&!od.value.trim())od.value=res.out_dir;var og=$('poseGignOutDir');if(og&&!og.value.trim())og.value=res.out_dir;}
        if(res.default_smile&&!L){var e=$('poseSmiles');if(e)e.value=res.default_smile;}
        if(!protein&&res.ok&&res.pdb){PG._applyProtein(PE.parsePDB(res.pdb),res.name||'receptor',res.pdb);}
      }catch(e){PG._miniLog('default config error: '+e.message,'#fb7185');}
      doBuild();
    }).catch(function(){doBuild();});},
    close(){PG._stopAnim();var m=$('poseModal');m.classList.remove('flex');m.classList.add('hidden');if(window.Plotly){try{Plotly.purge('poseBox3D');}catch(e){}}},

    setPreset(smi){$('poseSmiles').value=smi;PG.build();},
    build(){const smi=$('poseSmiles').value.trim();
      return fetch(POSE_LIGAND_API,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({smiles:smi})})
        .then(r=>r.ok?r.json():null)
        .then(res=>{ if(res&&res.ok){res.source=res.source||'rdkit';PG._setLigand(res);} else PG._fallback(smi); })
        .catch(()=>PG._fallback(smi));},
    _fallback(smi){const j=PE.buildLigand(smi); if(j.ok)j.source='js'; PG._setLigand(j);},
    _setLigand(res){const e=$('poseSmiles');
      if(!res||!res.ok){e.style.borderColor='#fb7185';PG._derivedErr((res&&res.err)||'parse failed');return;}
      e.style.borderColor='#1e293b';L=res;if(L&&!L._label)L._label='SMILES build';PG._derived();PG.init.cur=null;PG._poseSrc='random';PG._invalidate();PG.stage(PG._stage);},
    _derivedErr(msg){var d=$('poseDerived');if(d)d.innerHTML='<span style="color:#fb7185;">⚠ '+msg+'</span>';},
    _derived(){var d=$('poseDerived');if(!d)return;
      const lab=(t)=>'<span class="text-[10px] font-semibold text-slate-600 uppercase tracking-wider mr-1.5">'+t+'</span>';
      const sep='<div class="w-px h-3 bg-slate-700"></div>';
      const inp=(id,val,step,min,title)=>'<input id="'+id+'" type="number" step="'+step+'"'+(min!=null?' min="'+min+'"':'')+' value="'+val+'" onchange="PoseGen.setCenter()" title="'+title+'" style="width:60px;font-family:ui-monospace,monospace;font-size:11px;background:#0b1120;border:1px solid #1e293b;border-radius:6px;padding:2px 6px;color:#22d3ee;outline:none;text-align:right;">';
      let h='';
      if(L){h+=lab('Formula')+'<span class="font-mono text-slate-200">'+L.formula+'</span>'
        +sep+lab('Rotatable')+'<span class="font-mono text-cyan-300">N = '+L.N+'</span>'
        +sep+lab('Search space')+'<span class="font-mono text-violet-300">6 + '+L.N+' = '+(6+L.N)+' D</span>';}
      if(protein){if(h)h+=sep;h+=lab('Target')+'<span class="font-mono text-slate-200">'+(protein.id||'uploaded')+' · '+protein.chains.length+' ch · '+protein.nres+' res</span>'
        +sep+'<span style="display:inline-flex;align-items:center;gap:5px;">'+lab('Box center')
        +['x','y','z'].map((ax,k)=>inp('poseC'+ax,box.center[k].toFixed(1),'0.5',null,'docking box center '+ax+' (Å)')).join('')
        +'<span style="color:#475569;font-size:10px;">Å</span></span>'
        +sep+'<span style="display:inline-flex;align-items:center;gap:5px;">'+lab('Box length')
        +inp('poseLen',box.size.toFixed(0),'1','6','docking box edge length (Å)')+'<span style="color:#475569;font-size:10px;">Å</span></span>';}
      if(L)h+='<div class="flex-1"></div><span class="text-[9px] px-2 py-0.5 rounded-full" style="'+(L.source==='rdkit'?'background:rgba(52,211,153,.12);border:1px solid rgba(52,211,153,.35);color:#34d399;':'background:rgba(148,163,184,.1);border:1px solid #334155;color:#94a3b8;')+'">'+(L.source==='rdkit'?'RDKit · 3D ETKDG':L.source==='pdb'?'uploaded · 3D coords':'in-browser · 2D layout')+'</span>';
      d.innerHTML=h;},

    /* editable docking box (center + length) → rebuild pocket and re-derive the current stage */
    setCenter(){const v=['poseCx','poseCy','poseCz'].map(id=>{const e=$(id);return e?parseFloat(e.value):NaN;});
      if(v.some(x=>!isFinite(x)))return; box.center=v;
      const le=$('poseLen');if(le){const s=parseFloat(le.value);if(isFinite(s)&&s>=4)box.size=s;}
      if(protein){protein._site=null;protein._siteCenter=null;}     // box moved/resized → rebuild pocket
      PG._invalidate();PG.stage(PG._stage);},
    /* mirror box.center / box.size into the editable inputs (skips a field being typed in) */
    _syncBoxInputs(){var ids=['poseCx','poseCy','poseCz'];for(var k=0;k<3;k++){var e=$(ids[k]);if(e&&e!==document.activeElement)e.value=box.center[k].toFixed(1);}var le=$('poseLen');if(le&&le!==document.activeElement)le.value=box.size.toFixed(0);},

    /* ---- shared drawing (used by every stage) ---- */
    _show(view){$('poseSvgBox').style.display=view==='2d'?'block':'none';$('poseBox3D').style.display=view==='3d'?'block':'none';
      var lg=$('poseLegend3D');if(lg)lg.style.display=view==='3d'?'block':'none';var h=$('poseBox3DHint');if(h)h.style.display=view==='3d'?'block':'none';},
    /* Short stable identity for a ligand object, for the _draw3D debug log below.
       The id is stamped lazily on first sight, so the SAME object keeps the same
       #n across redraws and a genuinely new load shows up as a new number. */
    _ligTag(lig){if(!lig)return '(none)';
      if(!lig._dbgId)lig._dbgId=(PG._ligSeq=(PG._ligSeq||0)+1);
      var bits=[(lig.atoms?lig.atoms.length:0)+' atoms'];
      if(lig._label)bits.push(lig._label);
      if(lig._smiles)bits.push(String(lig._smiles));        // NOT truncated: a cut SMILES has unbalanced parens and can't be pasted anywhere
      return '#'+lig._dbgId+' ['+bits.join(' · ')+']';},

    /* ========================================================================
       THIS is the function that makes a ligand disappear.

       Note there is no hide/remove/clear call anywhere below. `data` is rebuilt
       from EMPTY on every single draw, and the only ligand ever added to it is
       the current value of the module-level `L` — the single ligand slot
       (`let L=null, protein=null, box=…` at the top of this IIFE). Plotly.react()
       at the end then swaps the ENTIRE scene for `data`.

       So a ligand vanishes purely by omission: the moment anything reassigns `L`
         · PG.init.pickLigand()  — clicking a gallery card
         · PG._impLoad()         — the import ◀ ▶ stepper / ▶ Play
         · PG.build()            — building from SMILES
       and triggers a redraw, the previous ligand object is no longer referenced
       by anything, so there is nothing left to draw it from.

       This is why a ligand vanishes on a swap — but pinned cards are the
       exception: their structures are kept in PG._pinLigs and re-added to
       `data` on every redraw (in amber, at their own true coordinates), so a
       pinned ligand stays on screen while the active one keeps changing.

       Watch it live with:   PoseGen._dbg = true    in the browser console.
       ======================================================================== */
    _draw3D(world,opt){opt=opt||{};const site=PG.init.site();
      // DEBUG: the ligand slot changed → whatever was on screen is about to be dropped.
      if(L!==PG._prevLig){
        PG._log('LIGAND SLOT REPLACED → previous ligand will NOT be drawn again:',
                PG._ligTag(PG._prevLig),'→',PG._ligTag(L),
                '· only the new one survives this redraw (single `L` slot)');
        PG._prevLig=L;
      }
      const ligAtoms=world.map((p,i)=>({x:p[0],y:p[1],z:p[2],el:L.atoms[i]}));
      let data=[];
      if(site&&!PG._hideProtein){
        if(opt.proteinAtoms){
          const pf=PG._hideH?stripH(opt.proteinAtoms,site.bonds):{atoms:opt.proteinAtoms,bonds:site.bonds};
          data=data.concat(stickTraces(pf.atoms,pf.bonds,false,5,2));
        }else data=data.concat(PG._siteTraces(site));
      }
      data.push(boxTrace3D(box.center,box.size));
      const _nBefore=data.length;                                                              // DEBUG: mark where ligand traces start
      const _lf=PG._hideH?stripH(ligAtoms,L.BONDS):{atoms:ligAtoms,bonds:L.BONDS};
      data=data.concat(stickTraces(_lf.atoms,_lf.bonds,true,7,3.2));

      /* ---- pinned ligands ------------------------------------------------
         The active ligand above lives in the single `L` slot and is drawn at
         the CURRENT pose. Pinned ones are extra structures held in PG._pinLigs
         and drawn at their own true coordinates (REF + _realCenter, i.e. the
         geometry straight out of their PDB), in amber so they read as
         reference copies rather than the thing being posed. Skipped when a pin
         is also the active ligand, so it isn't drawn twice into z-fighting. */
      var _pinDrawn=[];
      try{
        var _pins=PG._pinLigs||{};
        Object.keys(_pins).forEach(function(nm){
          var pl=_pins[nm];
          if(!pl||!pl.REF||!pl.REF.length||!pl._realCenter)return;
          if(L&&(pl===L||(L._label&&L._label===nm)))return;            // already on screen as the active ligand
          var c=pl._realCenter;
          var at=pl.REF.map(function(p,i){return {x:p[0]+c[0],y:p[1]+c[1],z:p[2]+c[2],el:pl.atoms[i],name:nm};});
          var pf2=PG._hideH?stripH(at,pl.BONDS):{atoms:at,bonds:pl.BONDS};
          data=data.concat(stickTraces(pf2.atoms,pf2.bonds,true,6,2.8,PG._PIN_COL));
          _pinDrawn.push(nm);
        });
      }catch(e){PG._log('pinned ligand draw failed:',e&&e.message);}

      // DEBUG: how many ligands actually made it into the scene this redraw.
      PG._log('scene rebuilt from scratch: ligands drawn='+(1+_pinDrawn.length),PG._ligTag(L),
              _pinDrawn.length?('+ pinned ['+_pinDrawn.join(', ')+']'):'(no pinned ligands)',
              '· ligand traces='+(data.length-_nBefore),
              '· protein traces='+(_nBefore-1),
              '· dropped-by-omission=',PG._prevLigDrawn&&PG._prevLigDrawn!==L?PG._ligTag(PG._prevLigDrawn):'(none)');
      PG._prevLigDrawn=L;
      if(site&&opt.interactions){const ix=interactions(L,world,site);data=data.concat(ix.traces);PG.init._ix=ix.n;}
      PG._last3D={world:world,opt:opt};                                                          // remember last 3D draw so view toggles can replay it
      PG._surfIdx=-1;
      if(PG._showSurface&&site){try{const st=PG._surfaceTraces(site);if(st.length){PG._surfIdx=data.length;data=data.concat(st);}}catch(e){}}   // 1.4 Å water-probe molecular surface of the pocket
      if(PG._showSS){try{data=data.concat(PG._ssTraces());}catch(e){}}                            // DSSP-style backbone: helix / sheet / loop
      const rev=protein?('p:'+protein.id):'none';const lay=layout3D(rev);
      // CAMERA. Snapshot a *copy* of the live camera — never a live reference: Plotly owns and mutates
      // its own camera object during react, so holding the reference meant our "saved" camera turned
      // into the reset one, and writing it back is what snapped the view to its pre-rotation state.
      if(!PG._styling){const c0=PG._camGet();if(c0){PG._cam=c0;PG._log('draw3D captured live cam ->',JSON.stringify(c0.eye));}}
      if(PG._cam)lay.scene.camera=PG._camClone(PG._cam);
      PG._log('draw3D react: traces=',data.length,' savedCam=',PG._cam?JSON.stringify(PG._cam.eye):'null');
      PG._lastRev=rev;
      try{const pr=PG._safe(Plotly.react('poseBox3D',data,lay,{responsive:true,displayModeBar:false,displaylogo:false}),'react');
        PG._camBind();                                                                              // track the user's camera from now on
        if(pr&&pr.then)pr.then(function(){PG._camRestore();PG._dumpTraces('after-draw');});
        else {PG._camRestore();PG._dumpTraces('after-draw');}
      }catch(e){}
      },
    _draw2D(model){const bp=boxParams(L);setSvg('poseSvgBox',svgBoxWire(CAM,bp.S,230,210,bp.half)+svgMol(L,model,CAM,{scale:bp.S,cx:230,cy:210}),'0 0 460 420');},
    _worldFromInternal(model){let c=[0,0,0];model.forEach(p=>{c[0]+=p[0];c[1]+=p[1];c[2]+=p[2];});const n=model.length||1;c=[c[0]/n,c[1]/n,c[2]/n];
      return model.map(p=>[p[0]-c[0]+box.center[0],p[1]-c[1]+box.center[1],p[2]-c[2]+box.center[2]]);},
    /* draw a pose {off,quat,tors} — same convention for ALL stages so they show the same ligand */
    _ligCenter(off){return [box.center[0]+off[0]*box.size*0.32, box.center[1]+off[1]*box.size*0.32, box.center[2]+off[2]*box.size*0.32];},
    _drawPose(pose,opt){const off=pose.off,quat=pose.quat,tors=pose.tors;
      if(PG.init.use3D()){PG._show('3d');const base=PE.computePose(L,quat,[0,0,0],tors);const lc=PG._ligCenter(off);
        PG._draw3D(base.map(p=>[p[0]+lc[0],p[1]+lc[1],p[2]+lc[2]]),opt);}
      else{PG._show('2d');const bp=boxParams(L);PG._draw2D(PE.computePose(L,quat,off.map(v=>v*bp.tmax),tors));}},
    _spark(id,vals,col){const el=$(id);if(!el)return;if(!vals.length){el.innerHTML='';return;}
      const W=184,H=44,pad=4;const mn=Math.min(...vals),mx=Math.max(...vals),rng=(mx-mn)||1;
      const pts=vals.map((v,i)=>[pad+(W-2*pad)*(vals.length===1?1:i/(vals.length-1)),H-pad-(H-2*pad)*((v-mn)/rng)]);
      const d='M'+pts.map(p=>p[0].toFixed(1)+' '+p[1].toFixed(1)).join(' L');
      el.innerHTML='<path d="'+d+'" fill="none" stroke="'+col+'" stroke-width="1.6" stroke-linejoin="round"/><circle cx="'+pts[pts.length-1][0].toFixed(1)+'" cy="'+pts[pts.length-1][1].toFixed(1)+'" r="2.4" fill="'+col+'"/>';},

    /* ---- stage manager ---- */
    /* ONE shared pose (PG.init.cur) flows through all 3 stages; each stage transforms it forward and every
       tab renders it. _poseVer bumps only on DISCRETE pose changes (new random / seed pick / a completed
       minimize or search) so a stage keeps its chart while the pose is unchanged, and re-derives once it moves. */
    _stage:'random', _raf:null, _lastRev:null, _poseVer:0, _poseSrc:'random', _ready:{random:false,min:false,mc:false},
    _bump(){PG._poseVer++;},
    _invalidate(){PG._ready.random=false;PG.min._work=null;PG.mc._work=null;PG._bump();},   // box / protein change: regrid seeds, drop stale min/MC results, force re-derive
    /* connected-trajectory ribbon: shows the shared pose flowing ① → ② → ③ (snapshot metrics, frozen per run) */
    _updateTraj(){
      var r=$('poseTrajR');if(r)r.innerHTML='① start <b style="color:#cbd5e1;">'+(PG.init.cur?'✓':'—')+'</b>';
      var m=$('poseTrajM');if(m)m.innerHTML='② min <b style="color:#34d399;">'+(PG.min._work?(PG.min.E0.toFixed(0)+'→'+PG.min.Efin.toFixed(0)):'—')+'</b>';
      var c=$('poseTrajC');if(c)c.innerHTML='③ MC <b style="color:#38bdf8;">'+(PG.mc._work&&PG.mc._hist.length?(PG.mc._hist[0].toFixed(1)+'→'+PG.mc.bestScore.toFixed(1)):'—')+'</b>';
      ['R','M','C'].forEach(k=>{const seg=$('poseTraj'+k);if(seg)seg.classList.toggle('on',(PG._stage==='random'&&k==='R')||(PG._stage==='min'&&k==='M')||(PG._stage==='mc'&&k==='C'));});
      var a1=$('poseTrajA1'),a2=$('poseTrajA2');if(a1)a1.style.color=PG.min._work?'#475569':'#1e293b';if(a2)a2.style.color=PG.mc._work?'#475569':'#1e293b';},
    _stopAnim(){if(PG._raf){cancelAnimationFrame(PG._raf);PG._raf=null;}if(PG.min)PG.min._running=false;if(PG.mc)PG.mc._running=false;},
    stage(name){PG._stopAnim();PG._stage=name;const S={random:'Random',min:'Min',mc:'MC'};
      Object.keys(S).forEach(nm=>{const on=nm===name;const b=$('poseSt'+S[nm]);if(b)b.classList.toggle('on',on);
        const c=$('poseCtrl'+S[nm]);if(c)c.style.display=on?'flex':'none';const sd=$('poseSide'+S[nm]);if(sd)sd.style.display=on?'flex':'none';});
      var proc=$('poseProcPanel');if(proc)proc.style.display=(name==='random')?'none':'block';
      try{var _imp=$('poseImpScore');if(_imp&&_imp.style.display!=='none'&&PG._impScoreShow)PG._impScoreShow(true);}catch(e){}   // proc panel just toggled → re-stack the import score under it
      const titles={random:'Random placement inside the docking box',min:'Ligand geometry relaxation',mc:'Monte-Carlo + BFGS pose search',vina:'AutoDock Vina scoring function'};
      const hints={random:'Random initialization inside the search box.',min:'Relax the embedded conformer before docking.',mc:'Global search: perturb → BFGS → Metropolis accept/reject.',vina:'Five-term Vina empirical score on the current pose — weights editable.'};
      var vt=$('poseViewTitle');if(vt)vt.textContent=titles[name];var sh=$('poseStageHint');if(sh)sh.textContent=hints[name];
      if(!L)return;
      if(name==='random')PG.init.enter();else if(name==='min')PG.min.enter();else if(name==='mc')PG.mc.enter();
      PG._updateTraj();
      if(PG.init.use3D())setTimeout(()=>{PG._resizeKeepCam();},60);},

    /* ---- stage ②: minimization — PROTEIN HELD FIXED; the ligand pose relaxes against the fixed pocket.
       Strain energy  E = wT·Σθ²  +  wC·Σ max(0, d0−d)²  +  wK·‖c − c_box‖²   (coefficients editable in the sidebar) ---- */
    min:{w:{tors:12,clash:8,d0:3.2,center:2},pocket:null,world0:null,_work:null,iter:0,E0:0,Efin:0,_ver:-1,_running:false,_hist:[],
      enter(){if(PG.min._ver===PG._poseVer&&PG.min._work)PG.min._restore();else PG.min.reset();},
      _restore(){var pt=$('poseProcTitle');if(pt)pt.textContent='Strain energy';PG.min._reflectW();var rb=$('poseMinRunBtn');if(rb)rb.textContent=(PG.min.iter>0?'▶ Continue':'▶ Minimize');PG.min._draw();PG.min._stats();},
      _reflectW(){const set=(id,v)=>{var e=$(id);if(e&&document.activeElement!==e)e.value=v;};set('poseEwTors',PG.min.w.tors);set('poseEwClash',PG.min.w.clash);set('poseEd0',PG.min.w.d0);set('poseEwCenter',PG.min.w.center);},
      _readW(){const get=(id,d)=>{var e=$(id);var v=e?parseFloat(e.value):NaN;return isFinite(v)?v:d;};
        PG.min.w={tors:get('poseEwTors',12),clash:get('poseEwClash',8),d0:Math.max(0.4,get('poseEd0',3.2)),center:get('poseEwCenter',2)};},
      setWeights(){PG.min._readW();PG.min.reset();},                                 // edited formula → re-derive the well from the current pose
      _sync(){const w=PG.min._work,a=PG.init.cur;if(!a||!w)return;a.off=w.off.slice();a.quat=PE.eulerToQuat(w.e[0],w.e[1],w.e[2]);a.tors=w.tors.slice();},
      _pocketAtoms(){const site=PG.init.site();if(!site)return [];const bc=box.center,h=box.size/2+1.5;
        let sel=site.atoms.filter(at=>Math.abs(at.x-bc[0])<=h&&Math.abs(at.y-bc[1])<=h&&Math.abs(at.z-bc[2])<=h);
        if(sel.length>220){const dc=a=>Math.hypot(a.x-bc[0],a.y-bc[1],a.z-bc[2]);sel.sort((a,b)=>dc(a)-dc(b));sel=sel.slice(0,220);}return sel;},
      _world(wk){const base=PE.computePose(L,PE.eulerToQuat(wk.e[0],wk.e[1],wk.e[2]),[0,0,0],wk.tors);const lc=PG._ligCenter(wk.off);return base.map(p=>[p[0]+lc[0],p[1]+lc[1],p[2]+lc[2]]);},
      _energy(wk){wk=wk||PG.min._work;const W=PG.min.w;let e=0;
        for(let i=0;i<wk.tors.length;i++)e+=wk.tors[i]*wk.tors[i]*W.tors;             // internal torsional strain
        const wd=PG.min._world(wk),pk=PG.min.pocket||[],d0=W.d0;                       // ligand ↔ FIXED protein clash
        for(let i=0;i<wd.length;i++){const a=wd[i];for(let j=0;j<pk.length;j++){const dx=a[0]-pk[j].x,dy=a[1]-pk[j].y,dz=a[2]-pk[j].z;const d2=dx*dx+dy*dy+dz*dz;if(d2<d0*d0){const ov=d0-Math.sqrt(d2);e+=ov*ov*W.clash;}}}
        let c=[0,0,0];wd.forEach(p=>{c[0]+=p[0];c[1]+=p[1];c[2]+=p[2];});const n=wd.length||1;c=[c[0]/n,c[1]/n,c[2]/n];   // keep the ligand seated in the pocket
        e+=((c[0]-box.center[0])**2+(c[1]-box.center[1])**2+(c[2]-box.center[2])**2)*W.center;return e;},
      reset(){PG._stopAnim();PG.min._running=false;PG.min._readW();
        const a=PG.init.cur||(PG.init.cur=PG.init.rand());
        PG.min._work={off:a.off.slice(),e:PE.q2e(a.quat),tors:a.tors.slice()};
        PG.min.pocket=PG.min._pocketAtoms();PG.min.world0=PG.min._world(PG.min._work);
        PG.min.iter=0;PG.min.E0=PG.min._energy();PG.min.Efin=PG.min.E0;PG.min._hist=[PG.min.E0];PG.min._ver=PG._poseVer;
        PG.min._reflectW();var pt=$('poseProcTitle');if(pt)pt.textContent='Strain energy';var rb=$('poseMinRunBtn');if(rb)rb.textContent='▶ Minimize';
        PG.min._draw();PG.min._stats();},
      _rmsd(){const wd=PG.min._world(PG.min._work),w0=PG.min.world0||wd;let s=0;for(let i=0;i<wd.length;i++)s+=dist(wd[i],w0[i])**2;return Math.sqrt(s/Math.max(1,wd.length));},
      _draw(){const a=PG.init.cur;PG._drawPose({off:a.off,quat:a.quat,tors:a.tors},{interactions:false});},   // pocket rendered at its fixed coordinates (no override)
      _stats(){const E=PG.min.Efin;var e;
        e=$('poseProcVal');if(e)e.textContent=E.toFixed(1);e=$('poseProcSub');if(e)e.textContent='iter '+PG.min.iter+' · ΔE '+(PG.min.E0-E).toFixed(1);
        PG._spark('poseProcChart',PG.min._hist,'#34d399');
        e=$('poseMinEstart');if(e)e.textContent=PG.min.E0.toFixed(1);e=$('poseMinEcur');if(e)e.textContent=E.toFixed(1);
        e=$('poseMinDelta');if(e)e.textContent='−'+(PG.min.E0-E).toFixed(1);e=$('poseMinIter');if(e)e.textContent=PG.min.iter;
        e=$('poseMinRmsd');if(e)e.textContent=PG.min._rmsd().toFixed(2)+' Å';
        e=$('poseMinAtoms');if(e)e.textContent=L.atoms.length+' lig moved · '+((PG.min.pocket||[]).length)+' pocket fixed';PG._updateTraj();},
      run(){if(PG.min._running){PG.min._running=false;PG._stopAnim();var b=$('poseMinRunBtn');if(b)b.textContent='▶ Minimize';return;}
        PG.min._running=true;var b=$('poseMinRunBtn');if(b)b.textContent='⏸ Pause';PG.min._loop();},
      _loop(){if(!PG.min._running)return;
        const wk=PG.min._work,f=Math.max(0.3,1-PG.min.iter/180),steps={off:0.05*f,e:0.06*f,tors:0.06*f};
        let E=PG.min._energy(wk);const Estart=E;
        const tryDof=(arr,idx,st)=>{const o=arr[idx];arr[idx]=o+st;const ep=PG.min._energy(wk);arr[idx]=o-st;const em=PG.min._energy(wk);arr[idx]=o;
          if(ep<E&&ep<=em){arr[idx]=o+st;E=ep;}else if(em<E){arr[idx]=o-st;E=em;}};                          // coordinate descent: ligand DOFs only
        for(let k=0;k<3;k++)tryDof(wk.off,k,steps.off);
        for(let k=0;k<3;k++)tryDof(wk.e,k,steps.e);
        for(let k=0;k<wk.tors.length;k++)tryDof(wk.tors,k,steps.tors);
        PG.min._sync();PG.min.iter++;PG.min.Efin=E;PG.min._hist.push(E);if(PG.min._hist.length>140)PG.min._hist.shift();
        PG.min._draw();PG.min._stats();
        if((Estart-E)<0.05||PG.min.iter>200){PG.min._running=false;PG._poseSrc='minimized';PG._bump();PG.min._ver=PG._poseVer;var bt=$('poseMinRunBtn');if(bt)bt.textContent='✓ Converged';return;}
        PG._raf=requestAnimationFrame(PG.min._loop);},
    },

    /* ---- stage ③: Monte-Carlo + BFGS (illustrative) ---- */
    mc:{sw:{clash:0.6,d0:2.5,center:0.4},best:null,bestScore:1e9,score:0,nstep:0,acc:0,T:1.2,steps:150,mutAmp:0.3,localIter:8,localAmp:0.12,intraW:0,intraCap:1000,scorer:'surrogate',searchScore:'surrogate',daPk:null,_daBusy:false,gignPk:null,_gignBusy:false,_running:false,_hist:[],_bhist:[],_ver:-1,_work:null,_srcLabel:'random',
      enter(){if(PG.mc._ver===PG._poseVer&&PG.mc._work)PG.mc._restore();else PG.mc.reset();},
      _restore(){var isV=PG.mc.searchScore==='vina';var pt=$('poseProcTitle');if(pt)pt.textContent=isV?'Vina ΔG (lower = better)':'Score (lower = better)';var ss=$('poseSearchScore');if(ss)ss.value=PG.mc.searchScore;var sc=$('poseSurrCard');if(sc)sc.style.display=isV?'none':'block';var vn=$('poseVinaSearchNote');if(vn)vn.style.display=isV?'flex':'none';var sel=$('poseMcScorer');if(sel)sel.value=PG.mc.scorer;PG.mc.setScorer(PG.mc.scorer);PG.mc.setT(PG.mc.T);PG.mc._reflectSW();var rb=$('poseMcRunBtn');if(rb)rb.textContent=(PG.mc.nstep>=PG.mc.steps?'✓ Done':(PG.mc.nstep>0?'▶ Continue':'▶ Run search'));PG.mc._draw();PG.mc._stats();},
      _reflectSW(){const set=(id,v)=>{var e=$(id);if(e&&document.activeElement!==e)e.value=v;};set('poseScwClash',PG.mc.sw.clash);set('poseScd0',PG.mc.sw.d0);set('poseScwCenter',PG.mc.sw.center);},
      _readSW(){const get=(id,d)=>{var e=$(id);var v=e?parseFloat(e.value):NaN;return isFinite(v)?v:d;};
        PG.mc.sw={clash:get('poseScwClash',0.6),d0:Math.max(0.4,get('poseScd0',2.5)),center:get('poseScwCenter',0.4)};},
      setScore(){PG.mc._readSW();PG.mc.reset();},                                    // edited score formula → re-derive from the current pose
      _sync(){const w=PG.mc._work,a=PG.init.cur;if(!a||!w)return;a.off=w.off.slice();a.quat=PG.mc._quat(w);a.tors=w.tors.slice();},   // mirror the working pose → shared pose (live)
      setT(t){PG.mc.T=t;document.querySelectorAll('#poseMcTempToggles [data-t]').forEach(b=>{const on=Math.abs(+b.dataset.t-t)<1e-6;b.style.borderColor=on?'#22d3ee':'#1e293b';b.style.background=on?'rgba(34,211,238,.1)':'#0b1120';b.style.color=on?'#22d3ee':'#94a3b8';});var v=$('poseMcTempVal');if(v)v.textContent=t.toFixed(1);},
      /* scorer selector: live search always uses the fast surrogate; DeepAtom evaluates the best pose on demand (server CNN) */
      setScorer(v){PG.mc.scorer=v;const da=(v==='deepatom');var b=$('poseDaScoreBtn');if(b)b.style.display=da?'inline-block':'none';var c=$('poseDaCard');if(c)c.style.display=da?'flex':'none';const gi=(v==='yupu_gign');var gb=$('poseGignScoreBtn');if(gb)gb.style.display=gi?'inline-block':'none';var gc=$('poseGignCard');if(gc)gc.style.display=gi?'flex':'none';if(gi&&PG.mc.gignPk==null)PG.mc._gignSet('idle - run the search, then score with GIGN',null,null);
        if(da&&!PG.mc.daPk)PG.mc._daSet('idle — run the search, then score the best pose',null,null);},
      _daSet(status,pk,dg){var e;e=$('poseDaPk');if(e)e.textContent=(pk!=null?pk.toFixed(2):'—');e=$('poseDaDg');if(e)e.textContent=(dg!=null?dg.toFixed(2):'—');PG.mc.daPk=pk;if(status)PG._miniLog(status,pk!=null?'#34d399':'#94a3b8');},
      _ligandPdb(coords){const R=(s,n)=>String(s).padStart(n);let out='';
        for(let i=0;i<L.atoms.length;i++){const el=L.atoms[i],p=coords[i];const name=((el.length===1?' '+el:el)+'  ').slice(0,4);
          out+='HETATM'+R(i+1,5)+' '+name+' LIG A'+R(1,4)+'    '+R(p[0].toFixed(3),8)+R(p[1].toFixed(3),8)+R(p[2].toFixed(3),8)+'  1.00  0.00          '+R(el.toUpperCase(),2)+'\n';}
        return out+'END\n';},
      /* build + download the .pdb for the current ligand pose (the same pose DeepAtom scores) */
      /* compact creation timestamp used as the pose file name (no underscore), e.g. 20260623233859 */
      _ts(){var d=new Date(),p=function(n){return String(n).padStart(2,'0');};return ''+d.getFullYear()+p(d.getMonth()+1)+p(d.getDate())+p(d.getHours())+p(d.getMinutes())+p(d.getSeconds());},
      exportPosePdb(){
        if(!L){PG._miniLog('build a ligand first','#fbbf24');return;}
        var p=PG.mc._work||PG.mc.best;
        if(!p){PG._miniLog('enter ③ Monte-Carlo first, then generate','#fbbf24');return;}
        const pdb=PG.mc._ligandPdb(PG.mc._world(p));
        const ts=PG.mc._ts();const fn=ts+'.pdb';
        var od=(($('poseDaOutDir')||{}).value||'').trim();
        if(od){
          PG._miniLog('saving pose .pdb to '+od+'…','#a78bfa');
          fetch('/pose/save_pdb',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pdb:pdb,out_dir:od,name:ts})})
            .then(function(r){return r.json();}).then(function(res){
              if(res&&res.ok)PG._miniLog('⤓ pose .pdb saved on server → '+res.path,'#34d399');
              else{PG.mc._downloadPdb(pdb,fn);PG._miniLog('server save failed: '+((res&&res.err)||'?')+' — downloaded locally','#fbbf24');}
            }).catch(function(e){PG.mc._downloadPdb(pdb,fn);PG._miniLog('save error: '+e.message+' — downloaded locally','#fbbf24');});
        }else{
          PG.mc._downloadPdb(pdb,fn);PG._miniLog('⤓ pose .pdb downloaded → '+fn,'#a78bfa');
        }},
      // auto-save the searched best pose when "▶ Run search" completes. Posts with
      // no out_dir, so the server writes it into pose.search_pose_dir (input_TS.yml);
      // silent no-op if that key is blank or the endpoint isn't deployed.
      _autoSavePose(){
        if(!L)return;var p=PG.mc.best||PG.mc._work;if(!p)return;
        const pdb=PG.mc._ligandPdb(PG.mc._world(p));const ts=PG.mc._ts();
        fetch('/pose/save_pdb',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pdb:pdb,name:ts,out_dir:''})})
          .then(function(r){return r.json();}).then(function(res){
            if(res&&res.ok)PG._miniLog('⤓ searched pose saved on server → '+res.path,'#34d399');
            else if(res&&res.err)PG._miniLog('searched pose not auto-saved ('+res.err+')','#7c6bae',true);
          }).catch(function(){});},
      _downloadPdb(pdb,fn){try{const blob=new Blob([pdb],{type:'chemical/x-pdb'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=fn;document.body.appendChild(a);a.click();setTimeout(function(){URL.revokeObjectURL(url);a.remove();},0);}catch(e){PG._miniLog('could not generate .pdb ('+e.message+')','#fb7185');}},
      scoreDeepAtom(){if(!L)return;if(!protein||!protein.raw){PG.mc._daSet('load a target protein (.pdb) first',null,null);return;}
        if(PG.mc._daBusy)return;PG.mc._daBusy=true;var b=$('poseDaScoreBtn');if(b)b.textContent='⏳ scoring…';PG.mc._daSet('scoring best pose on the server…',null,null);
        const lig=PG.mc._ligandPdb(PG.mc._world(PG.mc.best));
        var od=$('poseDaOutDir');var outDir=od?od.value.trim():'';
        fetch('/pose/deepatom_score',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ligand_pdb:lig,receptor_pdb:protein.raw,name:PG.mc._ts(),smiles:(L.smiles||''),out_dir:outDir})})
          .then(r=>r.ok?r.json():null).then(res=>{PG.mc._daBusy=false;var b=$('poseDaScoreBtn');if(b)b.textContent='⚛ Score best pose';
            if(res&&res.cmd)PG._miniLog('▶ command:  '+res.cmd,'#7dd3fc',true);
            if(res&&res.cwd)PG._miniLog('📂 cwd:  '+res.cwd,'#7dd3fc',true);
            if(res&&res.log)PG._miniLog('📝 debug log → '+res.log,'#a78bfa',true);
            if(res&&res.ok){const pk=(res.pred_pk!=null?res.pred_pk:null);const dg=(res.deltaG!=null?res.deltaG:(pk!=null?-pk*1.36:null));
              const p=res.produced||{};const note=(p.npz&&p.atomtypes)?(' · '+p.atomtypes.length+' atomtypes, '+p.npz.length+' npz'):'';
              PG.mc._daSet('scored ✓ '+(res.elapsed?('('+res.elapsed+')'):'')+note,pk,dg);}
            else{PG.mc._daSet((res&&res.err)||'DeepAtom unavailable on server',null,null);if(res&&res.stdout)PG._miniLog('── script output ──\n'+res.stdout,'#94a3b8',true);}})
          .catch(()=>{PG.mc._daBusy=false;var b=$('poseDaScoreBtn');if(b)b.textContent='⚛ Score best pose';PG.mc._daSet('request failed (is the endpoint deployed?)',null,null);});},
      _gignSet(status,pk,dg){var e;e=$('poseGignPk');if(e)e.textContent=(pk!=null?pk.toFixed(2):'—');e=$('poseGignDg');if(e)e.textContent=(dg!=null?dg.toFixed(2):'—');PG.mc.gignPk=pk;if(status)PG._miniLog(status,pk!=null?'#34d399':'#94a3b8');},
      /* score the live pose with Yupu_GIGN: send the posed ligand + the loaded receptor; the server cuts the pocket, builds the graph, runs the model */
      scoreGign(){if(!L){PG.mc._gignSet('build a ligand first',null,null);return;}
        if(!protein||!protein.raw){PG.mc._gignSet('load a target protein (.pdb) first',null,null);return;}
        if(PG.mc._gignBusy)return;PG.mc._gignBusy=true;var b=$('poseGignScoreBtn');if(b)b.textContent='⏳ scoring…';PG.mc._gignSet('scoring best pose with GIGN (server)…',null,null);
        const lig=PG.mc._ligandPdb(PG.mc._world(PG.mc.best||PG.mc._work));
        var od=$('poseGignOutDir')||$('poseDaOutDir');var outDir=od?od.value.trim():'';
        fetch('/pose/gign_score',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ligand_pdb:lig,receptor_pdb:protein.raw,name:PG.mc._ts(),smiles:(L.smiles||''),out_dir:outDir})})
          .then(r=>r.ok?r.json():null).then(res=>{PG.mc._gignBusy=false;var b=$('poseGignScoreBtn');if(b)b.textContent='⚛ Score (GIGN)';
            if(res&&res.cmd)PG._miniLog('▶ command:  '+res.cmd,'#7dd3fc',true);
            if(res&&res.cwd)PG._miniLog('📂 cwd:  '+res.cwd,'#7dd3fc',true);
            if(res&&res.log)PG._miniLog('📝 debug log → '+res.log,'#a78bfa',true);
            if(res&&res.ok){const pk=(res.pred_pk!=null?res.pred_pk:null);const dg=(res.deltaG!=null?res.deltaG:(pk!=null?-pk*1.36:null));
              PG.mc._gignSet('scored ✓ '+(res.elapsed?('('+res.elapsed+')'):''),pk,dg);}
            else{PG.mc._gignSet((res&&res.err)||'GIGN unavailable on server',null,null);if(res&&res.stdout)PG._miniLog('── script output ──\n'+res.stdout,'#94a3b8',true);}})
          .catch(()=>{PG.mc._gignBusy=false;var b=$('poseGignScoreBtn');if(b)b.textContent='⚛ Score (GIGN)';PG.mc._gignSet('request failed (is the endpoint deployed?)',null,null);});},
      _randPose(){const q=PE.randomQuat();return {off:[Math.random()*2-1,Math.random()*2-1,Math.random()*2-1],e:PE.q2e(q),tors:L.TORS.map(()=>(Math.random()*2-1)*Math.PI)};},
      _quat(p){return PE.eulerToQuat(p.e[0],p.e[1],p.e[2]);},
      _world(p){const base=PE.computePose(L,PG.mc._quat(p),[0,0,0],p.tors);const lc=[box.center[0]+p.off[0]*box.size*0.32,box.center[1]+p.off[1]*box.size*0.32,box.center[2]+p.off[2]*box.size*0.32];return base.map(q=>[q[0]+lc[0],q[1]+lc[1],q[2]+lc[2]]);},
      _score(p){const w=PG.mc._world(p);
        var intra=(PG.mc.intraW>0)?PG.mc.intraW*PG.mc._intraStrain(w):0;
        if(PG.mc.searchScore==='vina')return PG.vina._scoreWorld(w).dg+intra;   // Vina ΔG as the search objective (lower = better)
        const S=PG.mc.sw,d0=S.d0;let clash=0;const site=PG.init.site();
        if(site){const pa=site.atoms;for(let i=0;i<w.length;i++){const a=w[i];for(let j=0;j<pa.length;j++){const dx=a[0]-pa[j].x,dy=a[1]-pa[j].y,dz=a[2]-pa[j].z;const d2=dx*dx+dy*dy+dz*dz;if(d2<d0*d0)clash+=(d0-Math.sqrt(d2));}}}
        let c=[0,0,0];w.forEach(q=>{c[0]+=q[0];c[1]+=q[1];c[2]+=q[2];});const n=w.length;c=[c[0]/n,c[1]/n,c[2]/n];
        return clash*S.clash+dist(c,box.center)*S.center+intra;},
      // intramolecular strain (model::eval_intramolecular): curl-capped repulsion over
      // the ligand's own non-bonded (non-1-2) heavy-atom pairs. Invariant under rigid
      // motion (so 0 effect on rigid ligands); varies with torsions for flexible ones.
      _intraStrain(w){if(!L||!L.atoms)return 0;
        if(!L._adj12){var a={};(L.BONDS||[]).forEach(function(b){(a[b.a]=a[b.a]||{})[b.b]=1;(a[b.b]=a[b.b]||{})[b.a]=1;});L._adj12=a;}
        var n=L.atoms.length,e=0,R=PG.vina.RAD,cap=PG.mc.intraCap;
        for(var i=0;i<n;i++){if(L.atoms[i]==='H')continue;var ri=R[L.atoms[i]]||1.8,wi=w[i];
          for(var j=i+1;j<n;j++){if(L.atoms[j]==='H')continue;if(L._adj12[i]&&L._adj12[i][j])continue;
            var dx=wi[0]-w[j][0],dy=wi[1]-w[j][1],dz=wi[2]-w[j][2],s=Math.sqrt(dx*dx+dy*dy+dz*dz)-ri-(R[L.atoms[j]]||1.8);
            if(s<0){var v=s*s;e+=(cap>0?v*cap/(cap+v):v);}}}
        return e;},
      // Advanced config: read the local-search (Monte-Carlo + BFGS) + intramolecular knobs
      setAdv(){var g=function(id,d){var e=$(id);var v=e?parseFloat(e.value):NaN;return isFinite(v)?v:d;};
        PG.mc.steps=Math.max(1,Math.round(g('poseAdvSteps',150)));
        PG.mc.mutAmp=Math.max(0.01,g('poseAdvMut',0.3));
        PG.mc.localIter=Math.max(0,Math.round(g('poseAdvLocal',8)));
        PG.mc.intraW=Math.max(0,g('poseAdvIntraW',0));
        PG.mc.intraCap=Math.max(1,g('poseAdvIntraCap',1000));
        if(L&&PG._stage==='mc')PG.mc.reset();},
      // choose which objective drives the search: 'surrogate' (clash + restraint) or 'vina'
      setSearchScore(v){PG.mc.searchScore=(v==='vina')?'vina':'surrogate';const isV=PG.mc.searchScore==='vina';
        var sc=$('poseSurrCard');if(sc)sc.style.display=isV?'none':'block';
        var vn=$('poseVinaSearchNote');if(vn)vn.style.display=isV?'flex':'none';
        var sel=$('poseSearchScore');if(sel)sel.value=PG.mc.searchScore;
        if(isV)PG.vina._readW();                                    // pick up the current Vina weights
        var pt=$('poseProcTitle');if(pt)pt.textContent=isV?'Vina ΔG (lower = better)':'Score (lower = better)';
        if(L&&PG._stage==='mc')PG.mc.reset();},                     // re-derive the search from the current pose under the new objective
      _copy(p){return {off:p.off.slice(),e:p.e.slice(),tors:p.tors.slice()};},
      _perturb(p,s){const q=PG.mc._copy(p);const k=Math.floor(Math.random()*3);
        if(k===0){const ax=Math.floor(Math.random()*3);q.off[ax]=Math.max(-1,Math.min(1,q.off[ax]+(Math.random()*2-1)*s));}
        else if(k===1){const ax=Math.floor(Math.random()*3);q.e[ax]+=(Math.random()*2-1)*s;}
        else if(q.tors.length){const ti=Math.floor(Math.random()*q.tors.length);q.tors[ti]+=(Math.random()*2-1)*s*2;}
        return q;},
      _bfgs(p){let best=p,bs=PG.mc._score(p);for(let it=0;it<PG.mc.localIter;it++){const t=PG.mc._perturb(best,PG.mc.localAmp);const s=PG.mc._score(t);if(s<bs){bs=s;best=t;}}return {pose:best,score:bs};},
      reset(){PG._stopAnim();PG.mc._running=false;PG.mc._readSW();
        const a=PG.init.cur||(PG.init.cur=PG.init.rand());        // search starts from the SHARED pose (whatever ① or ② last produced)
        PG.mc._work={off:a.off.slice(),e:PE.q2e(a.quat),tors:a.tors.slice()};PG.mc._srcLabel=PG._poseSrc;
        PG.mc.score=PG.mc._score(PG.mc._work);PG.mc.best=PG.mc._copy(PG.mc._work);PG.mc.bestScore=PG.mc.score;PG.mc.nstep=0;PG.mc.acc=0;PG.mc._hist=[PG.mc.score];PG.mc._bhist=[PG.mc.bestScore];PG.mc._ver=PG._poseVer;
        PG.mc._reflectSW();PG.mc.setT(PG.mc.T);var sel=$('poseMcScorer');if(sel)sel.value=PG.mc.scorer;PG.mc.setScorer(PG.mc.scorer);var isV=PG.mc.searchScore==='vina';var ss=$('poseSearchScore');if(ss)ss.value=PG.mc.searchScore;var sc=$('poseSurrCard');if(sc)sc.style.display=isV?'none':'block';var vn=$('poseVinaSearchNote');if(vn)vn.style.display=isV?'flex':'none';var pt=$('poseProcTitle');if(pt)pt.textContent=isV?'Vina ΔG (lower = better)':'Score (lower = better)';var rb=$('poseMcRunBtn');if(rb)rb.textContent='▶ Run search';PG.mc._draw();PG.mc._stats();},
      _pose(){const w=PG.mc._work;return {off:w.off,quat:PG.mc._quat(w),tors:w.tors};},
      _draw(){PG._drawPose(PG.mc._pose(),{interactions:true});},
      _stats(){if(PG.vina&&PG.vina.eng&&PG.vina.eng._running)return;      // the Vina engine owns the overlay while it runs
        var e;e=$('poseProcVal');if(e)e.textContent=PG.mc.score.toFixed(2);e=$('poseProcSub');if(e)e.textContent='step '+PG.mc.nstep+' · best '+PG.mc.bestScore.toFixed(2);PG._spark('poseProcChart',PG.mc._hist,'#38bdf8');
        e=$('poseMcStep');if(e)e.textContent=PG.mc.nstep;e=$('poseMcScore');if(e)e.textContent=PG.mc.score.toFixed(2);e=$('poseMcBest');if(e)e.textContent=PG.mc.bestScore.toFixed(2);e=$('poseMcAcc');if(e)e.textContent=PG.mc.acc+' / '+PG.mc.nstep;e=$('poseMcTempVal');if(e)e.textContent=PG.mc.T.toFixed(1);
        e=$('poseMcSrc');if(e)e.textContent=PG.mc._srcLabel+' pose';if(PG.mc.searchScore==='vina')PG.vina.evaluate();PG._updateTraj();},
      step(){if(!L)return;PG.mc.nstep++;const A=PG.mc.mutAmp;let prop=PG.mc._perturb(PG.mc._perturb(PG.mc._perturb(PG.mc._work,A),A),A);const opt=PG.mc._bfgs(prop);const dS=opt.score-PG.mc.score;
        if(dS<0||Math.random()<Math.exp(-dS/PG.mc.T)){PG.mc._work=opt.pose;PG.mc.score=opt.score;PG.mc.acc++;if(opt.score<PG.mc.bestScore){PG.mc.bestScore=opt.score;PG.mc.best=PG.mc._copy(opt.pose);}}
        PG.mc._sync();PG.mc._hist.push(PG.mc.score);PG.mc._bhist.push(PG.mc.bestScore);if(PG.mc._hist.length>140){PG.mc._hist.shift();PG.mc._bhist.shift();}PG.mc._draw();PG.mc._stats();},
      run(){if(PG.mc._running){PG.mc._running=false;PG._stopAnim();var b=$('poseMcRunBtn');if(b)b.textContent='▶ Run search';return;}PG.mc._running=true;var b=$('poseMcRunBtn');if(b)b.textContent='⏸ Pause';PG.mc._loop();},
      _loop(){if(!PG.mc._running)return;PG.mc.step();if(PG.mc.nstep>=PG.mc.steps){PG.mc._running=false;var b=$('poseMcRunBtn');if(b)b.textContent='✓ Done';PG.mc._work=PG.mc._copy(PG.mc.best);PG.mc.score=PG.mc.bestScore;PG.mc._sync();PG._poseSrc='searched';PG._bump();PG.mc._ver=PG._poseVer;PG.mc._draw();PG.mc._stats();PG.mc._autoSavePose();if(PG.mc.scorer==='deepatom')PG.mc.scoreDeepAtom();else if(PG.mc.scorer==='yupu_gign')PG.mc.scoreGign();return;}PG._raf=requestAnimationFrame(PG.mc._loop);},
    },

    /* ---- stage: Vina — full AutoDock Vina empirical score on the current pose (weights editable) ----
       Per ligand↔pocket pair, surface gap s = r − Rᵢ − Rⱼ:
         e = w₁·gauss₁(s) + w₂·gauss₂(s) + w₃·repulsion(s) + w₄·hydrophobic(s) + w₅·Hbond(s)
       Affinity:  ΔG = Σe / (1 + w_rot·Nrot).   Defaults = the standard Vina weights (match the server binary). ---- */
    vina:{w:{g1:-0.035579,g2:-0.005156,rep:0.840245,hyd:-0.035069,hb:-0.587439,rot:0.05846},
      _hist:[],inter:0,dg:0,nrot:0,npair:0,
      opts:{nrot:'auto',typing:'vina_1_2_7'},   // calibration vs the AutoDock Vina binary — driven by the Calibration controls in the card
      RAD:{C:1.9,N:1.8,O:1.7,S:2.0,P:2.1,F:1.5,Cl:1.8,Br:2.0,I:2.2,H:1.0,B:1.8},   // element → Vina XS van-der-Waals radius (Å)
      _rad(el){return PG.vina.RAD[el]!=null?PG.vina.RAD[el]:1.8;},
      _isHyd(el){return el==='C'||el==='F'||el==='Cl'||el==='Br'||el==='I';},        // hydrophobic XS types
      _isHB(el){return el==='N'||el==='O';},                                         // donor/acceptor (simplified)
      _g1(s){const t=s/0.5;return Math.exp(-t*t);},
      _g2(s){const t=(s-3.0)/2.0;return Math.exp(-t*t);},
      _rep(s){return s<0?s*s:0;},
      _hyd(s){return s<=0.5?1:(s>=1.5?0:1.5-s);},                                    // linear ramp 1→0 over 0.5..1.5 Å
      _hb(s){return s<=-0.7?1:(s>=0?0:-s/0.7);},                                     // linear ramp 1→0 over −0.7..0 Å
      // ── Calibration (reconcile the in-browser score with the Vina binary) ──────
      _readOpts(){var n=$('poseVinaNrotMode'),t=$('poseVinaTypeMode');
        if(n)PG.vina.opts.nrot=n.value;if(t)PG.vina.opts.typing=t.value;},
      setOpts(){PG.vina._readOpts();PG.vina.eng.invalidate();PG.vina.evaluate();if(PG._stage==='mc'&&PG.mc.searchScore==='vina')PG.mc.reset();},
      // Effective N_rot used in the denominator: manual → the N_rot box, rigid → 0,
      // auto → the rotatable-bond count. Lets the user match Vina's torsion-tree count.
      // NOTE: kept FRACTIONAL on purpose. Vina's conf_independent.cpp accumulates
      // num_tors += 0.5 * atom_rotors(...) per heavy atom, so its effective count is
      // often a half-integer (7.5 for Elion_iag933, not the PDBQT's "TORSDOF 8").
      // Rounding here made an exact match with the binary impossible.
      _nrotEffective(){var m=PG.vina.opts.nrot;
        if(m==='rigid')return 0;
        if(m==='manual'){var nv=$('poseVinaNrotVal');if(nv&&nv.value!==''&&isFinite(+nv.value)&&+nv.value>=0)return +nv.value;}
        return PG.vina._nrotCount();},
      setNrot(){PG.vina.opts.nrot='manual';var s=$('poseVinaNrotMode');if(s)s.value='manual';
        PG.vina.evaluate();if(PG._stage==='mc'&&PG.mc.searchScore==='vina')PG.mc.reset();},
      // show/hide the Advance Config panel (local search + intramolecular + calibration)
      toggleAdv(){var p=$('poseAdvConfig'),c=$('poseAdvChevron');if(!p)return;
        var open=(p.style.display==='none'||p.style.display==='');p.style.display=open?'flex':'none';if(c)c.textContent=open?'▾':'▸';},
      // active rotatable bonds. Flexible (SMILES-built) ligands already carry the
      // torsion tree; for an uploaded .pdb we rebuild it from the derived SMILES so
      // the Nrot entropy penalty matches AutoDock Vina. Cached on the ligand.
      _nrotCount(){if(!L)return 0;
        // Vina's num_tors is NOT the torsion count: conf_independent.cpp sums
        // 0.5*atom_rotors over heavy atoms and skips bonds to terminal groups,
        // so it is often a half-integer (7.5 for Elion_iag933). Ask the engine
        // when it is available; fall back to |TORS| otherwise.
        if(PG.vina.opts.typing==='vina_1_2_7'&&PG.vina.eng.available()){
          var nt=PG.vina.eng.numTors();if(nt!=null)return nt;}
        if(L._nrotAuto!=null)return L._nrotAuto;var n=0;
        if(L.TORS&&L.TORS.length)n=L.TORS.length;
        else if(L._smiles){try{var m=PE.buildLigand(L._smiles);if(m&&m.ok&&m.TORS)n=m.TORS.length;}catch(e){}}
        L._nrotAuto=n;return n;},
      // Vina C_P detection — a carbon bonded to a heteroatom is polar and earns no
      // hydrophobic reward. Ligand: from the bond graph; pocket: by covalent distance.
      // WHICH heteroatoms count depends on the typing mode:
      //   'vina'       → N/O only (the original, narrower rule)
      //   'vina_exact' → ANY non-C/non-H neighbour, matching Vina's
      //                  model.cpp bonded_to_heteroatom() — this is what makes
      //                  halogen-bonded carbons (C–F, C–Cl) polar, as the binary does.
      // Cached per mode, so flipping the dropdown recomputes instead of reusing a
      // stale mask.
      _hetSet(){return PG.vina.opts.typing==='vina_exact'
        ? null                      // null = "any element that isn't C or H"
        : {N:1,O:1};},
      _isHet(el){var h=PG.vina._hetSet();
        return h ? !!h[el] : (el!=='C'&&el!=='H');},
      _ligPolar(){if(!L||!L.atoms)return [];
        var key=PG.vina.opts.typing;
        if(L._polarC&&L._polarCKey===key)return L._polarC;
        var n=L.atoms.length,p=new Array(n).fill(false);
        (L.BONDS||[]).forEach(function(b){var ea=L.atoms[b.a],eb=L.atoms[b.b];
          if(ea==='C'&&PG.vina._isHet(eb))p[b.a]=true;
          if(eb==='C'&&PG.vina._isHet(ea))p[b.b]=true;});
        L._polarC=p;L._polarCKey=key;return p;},
      _sitePolar(site){if(!site||!site.atoms)return [];
        var key=PG.vina.opts.typing;
        if(site._polarC&&site._polarCKey===key)return site._polarC;
        var pa=site.atoms,m=pa.length,p=new Array(m).fill(false),C2=1.85*1.85,het=[],k;
        for(k=0;k<m;k++){if(PG.vina._isHet(pa[k].el))het.push(k);}
        for(var i=0;i<m;i++){if(pa[i].el!=='C')continue;var xi=pa[i].x,yi=pa[i].y,zi=pa[i].z;
          for(var t=0;t<het.length;t++){var j=het[t],dx=xi-pa[j].x,dy=yi-pa[j].y,dz=zi-pa[j].z;
            if(dx*dx+dy*dy+dz*dz<C2){p[i]=true;break;}}}
        site._polarC=p;site._polarCKey=key;return p;},
      /* ═══════════════════════════════════════════════════════════════════════
       * vina_engine.js bridge — real Vina 1.2.7 scoring + Monte-Carlo/BFGS.
       * The engine is DOM-free and lives in static/js/vina_engine.js; the search
       * runs in static/js/vina_worker.js so a full-depth run (globalSteps ~26k
       * x exhaustiveness 8) does not freeze the tab.
       * ═══════════════════════════════════════════════════════════════════════ */
      eng:{
        _model:null,_key:null,_worker:null,_running:false,
        _cfg:{exhaustiveness:8,seed:null,numModes:9,energyRange:3,minRmsd:1.0,
              globalSteps:0,localSteps:0,huntCap:10,temperature:1.2,mutAmp:2.0},
        available(){return typeof window.VinaEngine!=='undefined';},
        ready(){return PG.vina.eng.available()&&!!PG.vina.eng._build();},
        // {els, ref, bonds, tors, root} straight out of PoseEngine's torsion tree
        _ligSpec(){if(!L||!L.REF)return null;
          return {els:L.atoms.slice(),ref:L.REF.map(p=>p.slice()),
                  bonds:(L.BONDS||[]).map(b=>({a:b.a,b:b.b,order:b.order,type:b.type})),
                  tors:(L.TORS||[]).map(t=>({from:t.from,to:t.to,moves:t.moves.slice()})),
                  root:(L.ROOT||[]).slice()};},
        _recAtoms(){const site=PG.init.site();return site?site.atoms:null;},
        _box(){const s=(Array.isArray(box.size)?box.size:[box.size,box.size,box.size]);
          return {center:box.center.slice(),size:s.slice()};},
        // Cache the built model; rebuild when ligand, receptor, box or weights move.
        _build(){if(!PG.vina.eng.available())return null;
          const lig=PG.vina.eng._ligSpec(),rec=PG.vina.eng._recAtoms();
          if(!lig||!rec||!rec.length)return null;
          const W=PG.vina.w;
          const key=[L._label||L.smiles||'lig',lig.els.length,lig.tors.length,rec.length,
                     box.center.map(v=>v.toFixed(3)).join(','),String(box.size),
                     W.g1,W.g2,W.rep,W.hyd,W.hb,W.rot].join('|');
          if(PG.vina.eng._model&&PG.vina.eng._key===key)return PG.vina.eng._model;
          try{
            const E=window.VinaEngine;
            const weights={gauss1:W.g1,gauss2:W.g2,repulsion:W.rep,hydrophobic:W.hyd,hbond:W.hb,rot:W.rot};
            const typed=E.typeReceptor(rec);
            const m=new E.Model(lig,typed,{weights:weights,box:PG.vina.eng._box()});
            PG.vina.eng._model=m;PG.vina.eng._key=key;return m;
          }catch(e){PG._miniLog('vina_engine: could not build model — '+e.message,'#fb7185');return null;}},
        invalidate(){PG.vina.eng._model=null;PG.vina.eng._key=null;},
        /* Score a set of ligand world coords using the engine, returning the same
           shape _scoreWorld's legacy path returns so the card is unchanged. */
        scoreWorld(ligWorld){
          const m=PG.vina.eng._build();if(!m)return null;
          const E=window.VinaEngine,W=PG.vina.w;
          const a=PG.init.cur;if(!a)return null;
          const conf=E.confFromUI(m,{off:a.off,quat:a.quat,tors:a.tors},PG.vina.eng._box());
          const weights={gauss1:W.g1,gauss2:W.g2,repulsion:W.rep,hydrophobic:W.hyd,hbond:W.hb,rot:W.rot};
          const b=E.scorePose(m,conf,weights);
          // Honour the Nrot dropdown: rigid -> 0, manual -> the box value.
          const nrot=PG.vina._nrotEffective();
          const dg=E.confIndependent(b.inter,nrot,W.rot);
          return {inter:b.inter,dg:dg,nrot:nrot,np:b.npair,hasSite:true,
                  cg1:b.cg1,cg2:b.cg2,crp:b.crp,chy:b.chy,chb:b.chb,engine:true};},
        numTors(){const m=PG.vina.eng._build();return m?m.numTors:null;},
        _readCfg(){const g=(id,d)=>{const e=$(id);const v=e?parseFloat(e.value):NaN;return isFinite(v)?v:d;};
          const c=PG.vina.eng._cfg;
          c.exhaustiveness=Math.max(1,Math.round(g('poseVinaExh',8)));
          c.numModes=Math.max(1,Math.round(g('poseVinaModes',9)));
          c.energyRange=Math.max(0,g('poseVinaERange',3));
          c.minRmsd=Math.max(0,g('poseVinaMinRmsd',1.0));
          c.globalSteps=Math.max(0,Math.round(g('poseVinaGSteps',0)));
          c.localSteps=Math.max(0,Math.round(g('poseVinaLSteps',0)));
          c.huntCap=Math.max(0.01,g('poseVinaHuntCap',10));
          c.temperature=Math.max(0.01,g('poseVinaTemp',1.2));
          c.mutAmp=Math.max(0.01,g('poseVinaMutAmp',2.0));
          const se=$('poseVinaSeed');const sv=se?se.value.trim():'';
          c.seed=(sv===''?null:(parseInt(sv,10)|0));
          return c;},
        /* ── Full Vina search, across a pool of Workers ────────────────────
         * parallel_mc splits `exhaustiveness` tasks over `num_threads`; the
         * browser equivalent is one Worker per hardware thread. Each worker
         * runs a slice of the tasks and returns its RAW minima; the main thread
         * merges them with the same 2 A RMSD dedup, then hands the merged set
         * to one worker for the shared refine+rescore pass, so the subtracted
         * intramolecular energy comes from the GLOBAL best pose exactly as
         * vina.cpp:958 does.
         * ───────────────────────────────────────────────────────────────────*/
        _pool:[],_pending:0,_raw:[],_t0:0,_spec:null,
        _dir(){const base=(Array.from(document.scripts).map(s=>s.src).find(s=>/pose\.js/.test(s))||'');
          return base?base.replace(/[^/]*$/,''):'/static/js/';},
        /* ── running-best readout on the 3D overlay (#poseProcPanel) ──────────────
           Every worker reports its best-so-far on each progress tick, but the panel
           only moves when the search actually BREAKS the record. A monotone staircase
           is readable at a glance; a number that jitters between four workers' local
           bests is not. _procBest() therefore ignores anything not strictly lower —
           which also means the cross-worker minimum falls out for free, since a worker
           reporting a worse best simply loses.

           Honesty note, and the reason the title changes at the end: d.best is the
           SEARCH energy — evaluated on the cached affinity grid, with hunt_cap 10 and
           no Nrot division — not the reported affinity. The two differ, sometimes by a
           lot, so the panel says "best so far" during the run and only calls the number
           Vina ΔG once _onRefined has the real, exact-pairwise, Nrot-divided value. */
        _best:Infinity,_dg:null,_bestHist:[],_nImp:0,
        /* The worker's `best` is the SEARCH energy: cached affinity grid, hunt_cap 10,
           intra included, no N_rot division. It reads far more negative than the number
           the run finally reports, which is why the panel used to jump at the end.
           So the gate still uses it — it is what the search actually minimises, and it
           is monotone — but what gets DISPLAYED is this: the running-best pose put
           through the very same scorePose + conf_independent the breakdown card and the
           mode list use. Same pose, same formula, same N_rot ⇒ the live number and the
           final number are the same quantity, and the only movement left at the end is
           the refinement genuinely improving the pose. */
        _scoreConf(conf){
          const EG=PG.vina.eng,E=window.VinaEngine;
          if(!conf||!E)return null;
          const m=EG._build();if(!m)return null;
          try{
            const W=PG.vina.w;
            const c=E.makeConf(m.tors.length);
            c.pos=conf.pos.slice();c.q.set(conf.q);c.tors.set(conf.tors);
            const b=E.scorePose(m,c,{gauss1:W.g1,gauss2:W.g2,repulsion:W.rep,
                                     hydrophobic:W.hyd,hbond:W.hb,rot:W.rot});
            const dg=E.confIndependent(b.inter,PG.vina._nrotEffective(),W.rot);
            return isFinite(dg)?dg:null;
          }catch(e){return null;}},
        _procReset(){
          const EG=PG.vina.eng;
          EG._best=Infinity;EG._dg=null;EG._bestHist=[];EG._nImp=0;EG._liveAt=0;
          const p=$('poseProcPanel');if(p)p.style.display='block';
          const t=$('poseProcTitle');if(t)t.textContent='Vina ΔG · live';
          const v=$('poseProcVal');if(v){v.textContent='—';v.style.color='#64748b';v.style.textShadow='none';}
          const s=$('poseProcSub');if(s)s.textContent='starting…';
          PG._spark('poseProcChart',[],'#38bdf8');
          // the panel may have just appeared — restack the imported-pose readout under it
          try{var im=$('poseImpScore');if(im&&im.style.display!=='none')PG._impScoreShow(true);}catch(e){}},
        _procBest(best,frac,conf){
          const EG=PG.vina.eng;
          if(best==null||!isFinite(best)||!(best<EG._best-1e-9))return false;     // not a new minimum → leave it alone
          EG._best=best;EG._nImp++;
          const dg=EG._scoreConf(conf);                                           // the reportable ΔG
          if(dg!=null){
            EG._dg=dg;EG._bestHist.push(dg);
            if(EG._bestHist.length>240)EG._bestHist.shift();                      // the sparkline is 184 px wide
            const t=$('poseProcTitle');if(t)t.textContent='Vina ΔG · live';
            const v=$('poseProcVal');
            if(v){v.textContent=dg.toFixed(2);v.style.color=dg<0?'#34d399':'#fb7185';
              v.style.transition='none';v.style.textShadow='0 0 11px rgba(52,211,153,.6)';  // flash: a record broke
              setTimeout(function(){v.style.transition='text-shadow .5s ease';v.style.textShadow='none';},20);}
            PG._spark('poseProcChart',EG._bestHist,'#38bdf8');}
          const s=$('poseProcSub');
          if(s)s.textContent=Math.round((frac||0)*100)+'% · '+EG._nImp+' improvement'+(EG._nImp===1?'':'s');
          EG._procPose(conf);                                                     // and move the ligand
          return true;},
        /* The refined affinity is now the SAME quantity the staircase has been plotting,
           so it belongs on the chart as its final step: the drop you see there is the
           refinement pass genuinely improving the pose (exact pairwise instead of the
           grid, out-of-box slope escalated), not a change of units. */
        _procFinal(mode,n){
          const EG=PG.vina.eng;
          const t=$('poseProcTitle');if(t)t.textContent='Vina ΔG (lower = better)';
          const v=$('poseProcVal');
          if(v){v.textContent=mode.affinity.toFixed(2);v.style.color=mode.affinity<0?'#34d399':'#fb7185';
            v.style.transition='none';v.style.textShadow='0 0 13px rgba(52,211,153,.75)';
            setTimeout(function(){v.style.transition='text-shadow .7s ease';v.style.textShadow='none';},20);}
          const s=$('poseProcSub');
          if(s)s.textContent='mode 1 of '+n+' · refined';
          EG._dg=mode.affinity;
          EG._bestHist.push(mode.affinity);
          if(EG._bestHist.length>240)EG._bestHist.shift();
          PG._spark('poseProcChart',EG._bestHist,'#34d399');},
        _procStop(note){
          const s=$('poseProcSub');if(s)s.textContent=note;
          const v=$('poseProcVal');if(v)v.style.textShadow='none';},
        run(){
          const V=PG.vina,EG=V.eng;
          if(!EG.available()){PG._miniLog('vina_engine.js is not loaded — add <script src="…/js/vina_engine.js"> before pose.js','#fb7185');return;}
          if(!L){PG._miniLog('build a ligand first','#fbbf24');return;}
          if(!protein||!protein.raw){PG._miniLog('load a target protein (.pdb) first','#fbbf24');return;}
          if(EG._running){EG.cancel();return;}
          const lig=EG._ligSpec(),rec=EG._recAtoms();
          if(!lig||!rec||!rec.length){PG._miniLog('no receptor atoms in range of the box','#fbbf24');return;}
          const cfg=EG._readCfg(),W=PG.vina.w;
          const spec={lig:lig,
            recAtoms:rec.map(a=>({el:a.el,x:a.x,y:a.y,z:a.z,name:a.name,resn:a.resn,resseq:a.resseq,chain:a.chain,het:a.het})),
            box:EG._box(),
            weights:{gauss1:W.g1,gauss2:W.g2,repulsion:W.rep,hydrophobic:W.hyd,hbond:W.hb,rot:W.rot},
            exhaustiveness:cfg.exhaustiveness,numModes:cfg.numModes,energyRange:cfg.energyRange,
            minRmsd:cfg.minRmsd,seed:(cfg.seed===null?((Math.random()*2147483647)|0):cfg.seed),
            huntCap:cfg.huntCap,temperature:cfg.temperature,mutationAmplitude:cfg.mutAmp,
            numTors:PG.vina._nrotEffective()};   // the panel's N_rot, not the model's, so
                                                 // the reported ΔG matches the breakdown card
          if(cfg.globalSteps>0)spec.globalSteps=cfg.globalSteps;
          if(cfg.localSteps>0)spec.localSteps=cfg.localSteps;
          const hw=(navigator.hardwareConcurrency||4);
          const nw=Math.max(1,Math.min(cfg.exhaustiveness,hw));
          EG._spec=spec;EG._raw=[];EG._pending=nw;EG._running=true;EG._t0=Date.now();
          EG._procReset();                                              // overlay tracks the run from here
          const btn=$('poseVinaRunSearchBtn');if(btn)btn.textContent='⏸ Cancel search';
          const st=$('poseVinaSearchStatus');if(st)st.textContent=`starting ${nw} worker${nw>1?'s':''}…`;
          PG._miniLog(`▶ Vina search — exhaustiveness ${spec.exhaustiveness} over ${nw} worker${nw>1?'s':''}, `+
            `seed ${spec.seed}, ${rec.length} receptor atoms`,'#7dd3fc');
          const dir=EG._dir();
          const prog=new Array(nw).fill(0);
          for(let k=0;k<nw;k++){
            let w;
            try{w=new Worker(dir+'vina_worker.js');}
            catch(e){PG._miniLog('could not start the Vina worker: '+e.message,'#fb7185');EG._finish();return;}
            EG._pool.push(w);
            const from=Math.floor(k*cfg.exhaustiveness/nw), to=Math.floor((k+1)*cfg.exhaustiveness/nw);
            w.onmessage=ev=>{
              const d=ev.data||{};
              if(d.type==='progress'){
                prog[k]=d.frac||0;
                const avg=prog.reduce((a,b)=>a+b,0)/nw;
                EG._procBest(d.best,avg,d.conf);                       // only repaints on a new minimum
                const s2=$('poseVinaSearchStatus');
                if(s2)s2.textContent=d.phase?(`worker ${k+1}: `+d.phase)
                  :`${Math.round(avg*100)}% · ${nw} workers · best ${d.best!=null?d.best.toFixed(3):'—'}`;
                return;}
              if(d.type==='error'){PG._miniLog('Vina worker error: '+d.message,'#fb7185');EG._procStop('worker error');EG._finish();return;}
              if(d.type==='raw'){EG._raw=EG._raw.concat(d.minima);
                if(--EG._pending===0)EG._mergeAndRefine();
                return;}
              if(d.type==='refined'){EG._onRefined(d);return;}
            };
            w.onerror=e=>{PG._miniLog('Vina worker failed: '+(e.message||'unknown'),'#fb7185');EG._finish();};
            w.postMessage({cmd:'dock',spec:Object.assign({},spec,
              {taskFrom:from,taskTo:to,rawOnly:true})});
          }},
        /* Merge every worker's minima (2 A dedup, parallel_mc.cpp:57) then run
           the shared exact-pairwise refinement in one worker. */
        _mergeAndRefine(){
          const EG=PG.vina.eng,E=window.VinaEngine,spec=EG._spec;
          const nH=(EG._build()||{}).nHeavy||0;
          const raw=EG._raw.slice().sort((a,b)=>a.e-b.e);
          const merged=[];
          for(const cand of raw){
            let dup=false;
            for(let i=0;i<merged.length;i++){
              if(E.rmsdLB(Float64Array.from(merged[i].coords),Float64Array.from(cand.coords),nH)<2.0){
                if(cand.e<merged[i].e)merged[i]=cand; dup=true; break;}}
            if(!dup)merged.push(cand);
            if(merged.length>=Math.max(spec.numModes,20))break;
          }
          const st=$('poseVinaSearchStatus');
          if(st)st.textContent=`refining ${merged.length} poses (exact pairwise)…`;
          const w=EG._pool[0];
          if(!w){EG._finish();return;}
          w.postMessage({cmd:'refine',spec:Object.assign({},spec,
            {minima:merged,useCache:false,numModes:spec.numModes})});},
        _onRefined(d){
          const EG=PG.vina.eng;
          const elapsed=Date.now()-EG._t0, spec=EG._spec;
          EG._finish();
          if(!d.modes.length){PG._miniLog('no poses found','#fbbf24');EG._procStop('no poses found');return;}
          PG._miniLog(`✓ Vina search done in ${(elapsed/1000).toFixed(1)} s — `+
            `${d.modes.length} modes · exhaustiveness ${spec.exhaustiveness} · `+
            `Nrot ${d.numTors} · ${d.nReceptorAtoms} receptor atoms · seed ${spec.seed}`,'#34d399');
          EG._procFinal(d.modes[0],d.modes.length);
          EG._showModes(d.modes);
          EG._applyMode(d.modes[0]);
          const st=$('poseVinaSearchStatus');
          if(st)st.textContent=`done · best ${d.modes[0].affinity.toFixed(4)} kcal/mol · ${(elapsed/1000).toFixed(1)} s`;},
        // The workers' dock loops are synchronous, so postMessage cancellation
        // would sit in their queues until each run finished. Terminate instead.
        cancel(){const EG=PG.vina.eng;EG._finish();
          const st=$('poseVinaSearchStatus');if(st)st.textContent='cancelled';
          EG._procStop('cancelled'+(EG._nImp?(' · '+EG._nImp+' improvement'+(EG._nImp===1?'':'s')):''));
          PG._miniLog('Vina search cancelled','#fbbf24');},
        _finish(){const EG=PG.vina.eng;EG._running=false;EG._pending=0;
          EG._pool.forEach(w=>{try{w.terminate();}catch(e){}});EG._pool=[];
          const b=$('poseVinaRunSearchBtn');if(b)b.textContent='▶ Run Vina search';},
        _modes:[],
        _showModes(modes){
          PG.vina.eng._modes=modes;
          const el=$('poseVinaModeList');if(!el)return;
          let h='<div style="font-family:ui-monospace,monospace;font-size:10.5px;">'+
                '<div style="display:flex;color:#475569;padding:2px 0;border-bottom:1px solid #1e293b;">'+
                '<span style="width:34px;">mode</span><span style="width:78px;text-align:right;">affinity</span>'+
                '<span style="width:62px;text-align:right;">rmsd l.b.</span><span style="flex:1;"></span></div>';
          modes.forEach((m,i)=>{
            h+=`<div onclick="PoseGen.vina.eng.pick(${i})" title="click to load this pose" `+
               `style="display:flex;padding:3px 0;border-bottom:1px solid rgba(30,41,59,.6);cursor:pointer;color:#cbd5e1;">`+
               `<span style="width:34px;color:#22d3ee;">${i+1}</span>`+
               `<span style="width:78px;text-align:right;color:${m.affinity<0?'#34d399':'#fb7185'};">${m.affinity.toFixed(4)}</span>`+
               `<span style="width:62px;text-align:right;color:#64748b;">${m.rmsdLB.toFixed(3)}</span>`+
               `<span style="flex:1;"></span></div>`;});
          el.innerHTML=h+'</div>';},
        pick(i){const m=PG.vina.eng._modes[i];if(m)PG.vina.eng._applyMode(m);},
        /* Load an engine pose back into the shared UI pose ①/②/③ all read. */
        _applyMode(mode,live){
          const E=window.VinaEngine,m=PG.vina.eng._build();if(!m||!mode)return;
          const conf=E.makeConf(m.tors.length);
          conf.pos=mode.conf.pos.slice();conf.q.set(mode.conf.q);conf.tors.set(mode.conf.tors);
          const ui=E.confToUI(m,conf,PG.vina.eng._box());
          PG.init.cur={off:ui.off,quat:ui.quat,tors:ui.tors,seed:0};
          PG._poseSrc='vina';
          if(!live)PG._bump();          // a live preview is not a DISCRETE pose change: bumping
          PG.vina._draw();              // _poseVer on every record would make ①/②/③ re-derive
          PG.vina.evaluate();},         // dozens of times mid-search
        /* Draw the running best while the search is still going.
           Rate-limited on top of the worker's 200 ms throttle because the cost here
           is a Plotly react over the whole scene plus an interaction re-scan, not the
           scoring. Records also arrive from several workers at once, and only the ones
           that actually lower the global best get this far.
           Note the pose you watch is the SEARCH pose — grid-scored, hunt_cap 10, not yet
           refined — so expect a small settle when _onRefined lands the exact one. */
        _liveAt:0,_liveMs:350,
        _procPose(conf){
          const EG=PG.vina.eng;
          if(!conf||!EG._running)return false;
          const now=Date.now();
          if(now-EG._liveAt<EG._liveMs)return false;
          EG._liveAt=now;
          try{EG._applyMode({conf:conf},true);}catch(e){return false;}
          return true;}
      },
      enter(){if(!PG.init.cur)PG.init.cur=PG.init.rand();PG.vina._hist=[];PG.vina._reflectW();var pt=$('poseProcTitle');if(pt)pt.textContent='Vina ΔG';PG.vina._draw();PG.vina.evaluate();PG.vina._reflectEngine();},
      /* Engine badge + intra-pair count in the Advance Config panel. */
      _reflectEngine(){
        var st=$('poseVinaEngineState');
        if(st){
          if(!PG.vina.eng.available()){st.textContent='vina_engine.js not loaded';st.style.color='#fb7185';}
          else{var m=PG.vina.eng._build();
            if(!m){st.textContent='loaded · needs a ligand + target';st.style.color='#fbbf24';}
            else{st.textContent=m.nHeavy+' lig heavy · '+m.rec.n+' rec atoms · Nrot '+m.numTors;st.style.color='#34d399';}}}
        var ip=$('poseVinaIntraPairs');
        if(ip){var mm=PG.vina.eng.available()?PG.vina.eng._build():null;
          ip.textContent=mm?String(mm.intraPairs.length/2):'—';}},
      _reflectW(){const set=(id,v)=>{var e=$(id);if(e&&document.activeElement!==e)e.value=v;};
        set('poseVwGauss1',PG.vina.w.g1);set('poseVwGauss2',PG.vina.w.g2);set('poseVwRep',PG.vina.w.rep);set('poseVwHyd',PG.vina.w.hyd);set('poseVwHB',PG.vina.w.hb);set('poseVwRot',PG.vina.w.rot);},
      _readW(){const get=(id,d)=>{var e=$(id);var v=e?parseFloat(e.value):NaN;return isFinite(v)?v:d;};
        PG.vina.w={g1:get('poseVwGauss1',-0.035579),g2:get('poseVwGauss2',-0.005156),rep:get('poseVwRep',0.840245),hyd:get('poseVwHyd',-0.035069),hb:get('poseVwHB',-0.587439),rot:Math.max(0,get('poseVwRot',0.05846))};},
      setWeights(){PG.vina._readW();PG.vina.eng.invalidate();PG.vina.evaluate();if(PG._stage==='mc'&&PG.mc.searchScore==='vina')PG.mc.reset();},          // edited weight → recompute breakdown + re-derive the search
      resetWeights(){PG.vina.w={g1:-0.035579,g2:-0.005156,rep:0.840245,hyd:-0.035069,hb:-0.587439,rot:0.05846};PG.vina._reflectW();PG.vina.eng.invalidate();PG.vina.evaluate();if(PG._stage==='mc'&&PG.mc.searchScore==='vina')PG.mc.reset();},
      _ligWorld(){const a=PG.init.cur;if(!a)return [];const base=PE.computePose(L,a.quat,[0,0,0],a.tors);const lc=PG._ligCenter(a.off);return base.map(p=>[p[0]+lc[0],p[1]+lc[1],p[2]+lc[2]]);},
      _draw(){const a=PG.init.cur;if(a)PG._drawPose({off:a.off,quat:a.quat,tors:a.tors},{interactions:true});},
      // pure Vina score for a set of ligand world coords (no UI) — used by the
      // Vina tab's evaluate() and, when selected, as the Monte-Carlo search objective.
      // Five-term Vina score for a set of ligand world coords.
      //
      // `typing:'vina_1_2_7'` (the new default) routes through vina_engine.js,
      // which does REAL XS typing: donor/acceptor resolved from residue+atom
      // chemistry on the receptor and from the bond graph + implicit-H valence
      // on the ligand, C_P from the bond graph, metals as Met_D (r=1.2, donor).
      // The legacy element-only paths are kept so the difference stays visible:
      // they treat every N and O as simultaneously donor AND acceptor, which
      // pays out full H-bond reward on acceptor-acceptor pairs Vina scores as
      // zero -- worth about +2.5 kcal/mol of spurious binding on a polar ligand.
      _scoreWorld(lig){const W=PG.vina.w,V=PG.vina;const site=PG.init.site();const pa=site?site.atoms:[];
        if(V.opts.typing==='vina_1_2_7'&&V.eng.ready()){
          const r=V.eng.scoreWorld(lig);
          if(r)return r;                                 // falls through if the engine cannot build
        }
        const strict=(PG.vina.opts.typing==='vina'||PG.vina.opts.typing==='vina_exact');
        const lPol=strict?V._ligPolar():null;const pPol=(strict&&site)?V._sitePolar(site):null;
        let sg1=0,sg2=0,srp=0,shy=0,shb=0,np=0;const CUT2=8.0*8.0;
        for(let i=0;i<lig.length;i++){const le=L.atoms[i];if(le==='H')continue;const lr=V._rad(le),lp=lig[i],lHB=V._isHB(le);
          const lHy=V._isHyd(le)&&!(strict&&lPol&&le==='C'&&lPol[i]);
          for(let j=0;j<pa.length;j++){const pe=pa[j].el;if(pe==='H')continue;
            const dx=lp[0]-pa[j].x,dy=lp[1]-pa[j].y,dz=lp[2]-pa[j].z;const r2=dx*dx+dy*dy+dz*dz;if(r2>CUT2)continue;
            const s=Math.sqrt(r2)-lr-V._rad(pe);
            sg1+=V._g1(s);sg2+=V._g2(s);srp+=V._rep(s);
            const pHy=V._isHyd(pe)&&!(strict&&pPol&&pe==='C'&&pPol[j]);
            if(lHy&&pHy)shy+=V._hyd(s);
            if(lHB&&V._isHB(pe))shb+=V._hb(s);
            np++;}}
        const cg1=W.g1*sg1,cg2=W.g2*sg2,crp=W.rep*srp,chy=W.hyd*shy,chb=W.hb*shb;
        const inter=cg1+cg2+crp+chy+chb,nrot=V._nrotEffective(),denom=1+W.rot*nrot,dg=denom!==0?inter/denom:inter;
        return {inter:inter,dg:dg,nrot:nrot,np:np,hasSite:!!site,cg1:cg1,cg2:cg2,crp:crp,chy:chy,chb:chb};},
      evaluate(){if(!L)return;const V=PG.vina,r=V._scoreWorld(V._ligWorld());
        const inter=r.inter,dg=r.dg,nrot=r.nrot,np=r.np;
        V.inter=inter;V.dg=dg;V.nrot=nrot;V.npair=np;
        var _nv=$('poseVinaNrotVal');
        if(_nv){if(V.opts.nrot==='auto')_nv.value=V._nrotCount();
          else if(V.opts.nrot==='rigid')_nv.value=0;
          else if(V.opts.nrot==='manual'&&_nv.value==='')_nv.value=V._nrotCount();}
        const sgn=x=>(x>=0?'+':'')+x.toFixed(3),T=(id,v,col)=>{var e=$(id);if(e){e.textContent=v;if(col)e.style.color=col;}};
        T('poseVinaG1',sgn(r.cg1));T('poseVinaG2',sgn(r.cg2));T('poseVinaRep',sgn(r.crp));T('poseVinaHyd',sgn(r.chy));T('poseVinaHB',sgn(r.chb));
        T('poseVinaInter',inter.toFixed(3));T('poseVinaPairs',np+' · '+nrot);
        if(r.hasSite){T('poseVinaDg',dg.toFixed(3),dg<0?'#34d399':'#fb7185');T('poseVinaPk',dg<0?(-dg/1.36).toFixed(2):'—','#e2e8f0');}
        else{T('poseVinaDg','— no target','#64748b');T('poseVinaPk','—','#64748b');}
        PG.vina._reflectEngine();},
      // "⚛ Compute & log breakdown" -> POST the posed ligand + receptor + current
      // weights to /pose/vina_breakdown. The server recomputes the five-term score
      // over the full receptor (authoritative), writes the per-atom-pair log into
      // pose.vina_breakdown_dir (input_TS.yml), and returns the totals + top
      // pairs, which we reflect into the card so the display matches the log.
      logBreakdown(){
        if(!L){PG._miniLog('build a ligand first','#fbbf24');return;}
        if(!protein||!protein.raw){PG._miniLog('load a target protein (.pdb) first — the Vina breakdown needs a receptor','#fbbf24');return;}
        if(PG.vina._busy)return;PG.vina._busy=true;
        var b=$('poseVinaRunBtn');var lbl=b?b.textContent:'';if(b)b.textContent='⏳ scoring…';
        PG.vina._readW();PG.vina._readOpts();const W=PG.vina.w;
        var _nr=PG.vina._nrotEffective();
        // The server (/pose/vina_breakdown) currently understands only 'vina' | 'element'.
        // Sending an unknown 'vina_exact' would fall through its else-branch to ELEMENT
        // typing — every carbon hydrophobic — i.e. looser than what the client just
        // computed, not tighter. Degrade to 'vina' and say so, rather than silently
        // logging a breakdown that disagrees with the panel in the wrong direction.
        var _typing=PG.vina.opts.typing, _typingSent=_typing;
        if(_typing==='vina_exact'){_typingSent='vina';
          PG._miniLog('note: the server breakdown does not implement "vina_exact" yet — '
            +'sending "vina" (N/O rule). The ⚛ log will read ~0.5 kcal/mol more favourable '
            +'than the panel on halogenated ligands until pose_routes.py is updated.','#fbbf24');}
        PG._miniLog('computing Vina breakdown on the server (full receptor, five terms · N_rot '+_nr+' · '+_typingSent+' typing)'+(L._realPose?' · uploaded pose as-loaded':'')+'…','#f59e0b');
        const lig=PG.mc._ligandPdb(PG.vina._ligWorld());
        const done=function(){PG.vina._busy=false;if(b)b.textContent=lbl||'⚛ Compute & log breakdown';};
        fetch('/pose/vina_breakdown',{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({ligand_pdb:lig,receptor_pdb:protein.raw,name:PG.mc._ts(),smiles:(L._smiles||L.smiles||''),typing:_typingSent,
            nrot:_nr,
            weights:{gauss1:W.g1,gauss2:W.g2,repulsion:W.rep,hydrophobic:W.hyd,hbond:W.hb,rot:W.rot}})})
          .then(function(r){return r.json();})
          .then(function(res){done();
            if(res&&res.ok){
              const T=(id,v,col)=>{var e=$(id);if(e){e.textContent=v;if(col)e.style.color=col;}};
              const sgn=x=>((Number(x)>=0?'+':'')+Number(x).toFixed(3));
              if(res.terms){T('poseVinaG1',sgn(res.terms.gauss1));T('poseVinaG2',sgn(res.terms.gauss2));T('poseVinaRep',sgn(res.terms.repulsion));T('poseVinaHyd',sgn(res.terms.hydrophobic));T('poseVinaHB',sgn(res.terms.hbond));}
              T('poseVinaInter',Number(res.inter).toFixed(3));T('poseVinaPairs',res.npairs+' · '+res.nrot);
              T('poseVinaDg',Number(res.dg).toFixed(3),res.dg<0?'#34d399':'#fb7185');
              T('poseVinaPk',(res.pk!=null?Number(res.pk).toFixed(2):'—'),'#e2e8f0');
              var pv=$('poseProcVal');if(pv){pv.textContent=Number(res.dg).toFixed(2);pv.style.color=res.dg<0?'#f59e0b':'#fb7185';}
              var ps=$('poseProcSub');if(ps)ps.textContent=res.npairs+' pairs · Nrot '+res.nrot+' · server ✓';
              PG._miniLog('⚛ Vina ΔG = '+Number(res.dg).toFixed(3)+' kcal/mol · '+res.npairs+' pairs'+(res.elapsed?(' ('+res.elapsed+')'):''),'#34d399');
              if(res.log)PG._miniLog('📝 per-pair breakdown logged → '+res.log,'#a78bfa',true);
              else PG._miniLog('breakdown computed but not written to disk (pose.vina_breakdown_dir is blank in input_TS.yml)','#7c6bae',true);
              if(res.top_pairs&&res.top_pairs.length){var lines=res.top_pairs.slice(0,8).map(function(p){return '  '+p.lig+' ↔ '+p.rec+'   r='+Number(p.r).toFixed(2)+' Å   pair_e='+Number(p.pair_e).toFixed(4);}).join('\n');PG._miniLog('top contributing pairs:\n'+lines,'#94a3b8',true);}
            } else {
              PG._miniLog('Vina breakdown failed: '+((res&&res.err)||'unknown error'),'#fb7185');
            }
          })
          .catch(function(e){done();PG._miniLog('request failed (is /pose/vina_breakdown deployed?): '+e.message,'#fb7185');});
      },
    },

    init:{ex:8,cur:null,_states:[],_uploads:[],_uploadsLoaded:false,_selRec:null,_selLig:null,_pins:{},_clickTimer:null,_dlBusy:false,
      enter(){if(!PG._ready.random)PG.init.regen();else PG.init.restore();PG.init._renderUploads();},
      restore(){if(!L)return;PG.init.draw(PG.init.cur||PG.init.rand());},
      rand(){return {seed:Math.floor(Math.random()*4294967295),off:[Math.random()*2-1,Math.random()*2-1,Math.random()*2-1],quat:PE.randomQuat(),tors:L.TORS.map(()=>(Math.random()*2-1)*Math.PI)};},
      use3D(){return !!(protein&&window.Plotly);},
      ligCenter(st){return [box.center[0]+st.off[0]*box.size*0.32, box.center[1]+st.off[1]*box.size*0.32, box.center[2]+st.off[2]*box.size*0.32];},
      draw(st){if(!L)return;PG.init.cur=st;
        if(PG.init.use3D())PG.init.render3D(st); else PG.init.draw2D(st);
        const center=PG.init.use3D()?PG.init.ligCenter(st):st.off.map(v=>v*boxParams(L).tmax);
        const T=(id,v)=>{var e=$(id);if(e)e.textContent=v;};
        T('poseCurSeed',st.seed);T('poseCurPos','('+center.map(v=>(+v).toFixed(2)).join(', ')+')');
        T('poseCurQuat',st.quat.map(v=>v.toFixed(2)).join(', '));
        T('poseCurTors',st.tors.length?st.tors.map(v=>(v*180/Math.PI).toFixed(0)+'°').join(' '):'(rigid)');
        PG.init.card();PG._updateTraj();},
      draw2D(st){PG._show('2d');const bp=boxParams(L);const pos=st.off.map(v=>v*bp.tmax);
        PG._draw2D(PE.computePose(L,st.quat,pos,st.tors));},
      // Receptor selection for SCORING (§19 / G70). Vina scores every receptor atom
      // within the grid box plus the 8 A pair cutoff and drops nothing, so this now
      // does the same: box half-width + 8 A, HETATM kept (waters, metals and
      // cofactors are load-bearing on targets like TEAD3), and no atom cap.
      // The old version used an arbitrary max(8, size/2+1) A residue shell, skipped
      // every HETATM residue, and silently sliced to 1400 atoms -- a RENDERING cap
      // that was quietly changing a score. `drawAtoms` keeps that cap for the 3D
      // view, which is the only thing that ever needed it.
      site(){ if(!protein)return null;
        if(protein._site&&protein._siteCenter&&dist(protein._siteCenter,box.center)<0.05)return protein._site;
        const bc=box.center;
        const half=(Array.isArray(box.size)?Math.max.apply(null,box.size):box.size)*0.5;
        const R=half+8.0;                              // grid box + Vina's pair cutoff
        const byRes={};protein.atoms.forEach(a=>{const k=(a.het?'H':'')+a.chain+'|'+a.resseq;(byRes[k]||(byRes[k]=[])).push(a);});
        const dc=a=>Math.hypot(a.x-bc[0],a.y-bc[1],a.z-bc[2]);
        let sel=[];Object.keys(byRes).forEach(k=>{const g=byRes[k];if(g.some(a=>dc(a)<R))sel=sel.concat(g);});
        let draw=sel;
        if(draw.length>1400){draw=sel.slice().sort((a,b)=>dc(a)-dc(b)).slice(0,1400);}
        const bonds=inferBonds(draw);const traces=stickTraces(draw,bonds,false,5,2);const aromatics=protAromatics(draw);
        const nres=Object.keys(byRes).filter(k=>byRes[k].some(a=>dc(a)<R)).length;
        const nhet=sel.filter(a=>a.het).length;
        protein._site={atoms:sel,drawAtoms:draw,bonds,traces,aromatics,nres,nhet};
        protein._siteCenter=box.center.slice();return protein._site;},
      render3D(st){PG._show('3d');
        const base=PE.computePose(L,st.quat,[0,0,0],st.tors);const lc=PG.init.ligCenter(st);
        const coords=base.map(p=>[p[0]+lc[0],p[1]+lc[1],p[2]+lc[2]]);
        PG._draw3D(coords,{interactions:true});
        const cx=$('poseCurIx');if(cx)cx.innerHTML=protein?('<span style="color:#fde047;">'+PG.init._ix.hb+'</span> H-bond · <span style="color:#d946ef;">'+PG.init._ix.sb+'</span> salt · <span style="color:#38bdf8;">'+PG.init._ix.pp+'</span> π'):'—';},
      shuffle(){PG._poseSrc='random';PG.init.draw(PG.init.rand());PG._bump();},
      setEx(n){PG.init.ex=n;document.querySelectorAll('#poseExhToggles [data-ex]').forEach(t=>{const on=+t.dataset.ex===n;t.style.borderColor=on?'#22d3ee':'#1e293b';t.style.background=on?'rgba(34,211,238,.1)':'#0b1120';t.style.color=on?'#22d3ee':'#94a3b8';});PG.init.regen();},
      regen(){if(!L)return;PG._ready.random=true;PG.init.draw(PG.init.cur||PG.init.rand());PG.init._renderUploads();},
      load(i){PG._poseSrc='random';PG.init.draw(PG.init._states[i]);PG._bump();},
      // ---- Uploaded receptors / ligands galleries (DB-backed, drag to reclassify) ----
      // Fetches pose.default_upload_path split into receptors + ligands (persisted in
      // the uploads DB), renders each section into its grid, and lets the user drag a
      // card between/within the two sections. Receptor click loads the receptor;
      // ligand click centers the docking box on the ligand.
      /* The uploads gallery only changes on upload / save / reclassify — NEVER because a pose was
         redrawn. But enter() and regen() both call this, so every ligand build fired 2 HTTP fetches
         + 2 full innerHTML teardowns of both grids. Over a 488-product ▶ Play that was ~976 requests
         and ~976 gallery rebuilds: the un-awaited fetches outran the backend (it parses each PDB to
         report atom counts), the connection pool backed up behind them, and the responses hammered
         the main thread with DOM churn — starving pointer events until the 3D view stopped rotating.
         Load once; pass force=true from the paths where the gallery genuinely changed. */
      _renderUploads(force){
        if(!$('poseRecGrid')&&!$('poseLigGrid'))return;
        if(!force&&PG.init._uploadsLoaded)return;                      // already on screen and unchanged → no fetch, no DOM
        PG.init._uploadsLoaded=true;                                   // set before the fetch so rapid rebuilds can't stack requests
        fetch('/pose/uploaded_pdbs').then(function(r){return r.json();}).then(function(res){
          if(!res||!res.ok){var m=PG.init._uploadMsg('could not read uploads'+((res&&res.err)?(' — '+res.err):''));var gr=$('poseRecGrid'),gl=$('poseLigGrid');if(gr)gr.innerHTML=m;if(gl)gl.innerHTML='';PG.init._uploads=[];PG.init._counts(0,0);return;}
          var recs=res.receptors||[],ligs=res.ligands||[];
          recs.forEach(function(f){f.kind='receptor';});ligs.forEach(function(f){f.kind='ligand';});
          PG.init._uploads=recs.concat(ligs);
          PG.init._paintUploads(recs,ligs);
        }).catch(function(){PG.init._uploadsLoaded=false;var m=PG.init._uploadMsg('upload list unavailable (is /pose/uploaded_pdbs deployed?)');var gr=$('poseRecGrid'),gl=$('poseLigGrid');if(gr)gr.innerHTML=m;if(gl)gl.innerHTML='';PG.init._counts(0,0);});},
      _paintUploads(recs,ligs){
        var gr=$('poseRecGrid'),gl=$('poseLigGrid');
        if(gr)gr.innerHTML=(recs&&recs.length)?recs.map(function(f){return PG.init._uploadCard(f);}).join(''):PG.init._uploadMsg('No receptors — drag a ligand card here, or use ⤓ Upload .pdb');
        if(gl)gl.innerHTML=(ligs&&ligs.length)?ligs.map(function(f){return PG.init._uploadCard(f);}).join(''):PG.init._uploadMsg('No ligands — drag a receptor card here');
        PG.init._counts(recs?recs.length:0,ligs?ligs.length:0);try{PG.init._refreshMarks();}catch(e){}},
      _counts(nr,nl){var a=$('poseRecCount');if(a)a.textContent=nr+(nr===1?' file':' files');var b=$('poseLigCount');if(b)b.textContent=nl+(nl===1?' file':' files');PG.init._dlSync();},
      _uploadMsg(txt){return '<div style="grid-column:1/-1;font-size:11px;color:#64748b;background:#0b1120;border:1px dashed #1e293b;border-radius:11px;padding:12px;text-align:center;line-height:1.5;">'+txt+'</div>';},
      /* ---- ⤓ Download all uploads ---------------------------------------------------
         Saves every archived file back to the user's machine in the format it was
         uploaded in: original basename, original extension, the file's own text
         straight from /pose/uploaded_pdb. No archive container — the browser writes N
         separate .pdb files.
         Two things here are browser behaviour, not ours, and both are load-bearing:
         (1) programmatic downloads fired in a tight loop are silently DROPPED — Chrome
             keeps roughly the first and discards the rest — so the saves run strictly
             sequentially with a gap between them; and
         (2) the first save trips the "Download multiple files?" permission prompt. If
             the user dismisses it, everything after file 1 is blocked with no error to
             catch. The button's tooltip and the opening log line both say so, because
             the failure is otherwise indistinguishable from a broken button.
         A per-file fetch failure is collected and reported at the end rather than
         aborting the run — one unreadable file should not cost you the other forty. */
      _dlGap:220,                                                    // ms between saves — see (1)
      _saveText(txt,fn){
        var blob=new Blob([txt],{type:'chemical/x-pdb'});
        var url=URL.createObjectURL(blob);
        var a=document.createElement('a');a.href=url;a.download=fn;
        document.body.appendChild(a);a.click();a.remove();
        setTimeout(function(){URL.revokeObjectURL(url);},4000);},     // NOT 0 — a same-tick revoke
                                                                      // can cancel the write itself
      /* What ⤓ saves depends on the gallery's selection, because that is what the user
         has just pointed at: with a receptor and/or ligand card selected (the cyan dot)
         it saves exactly those, and with nothing selected it saves the lot. There is one
         selection per grid, so the selected set is 0, 1 or 2 files. Clicking the selected
         card again clears it and the button goes back to "all" — the label and tooltip
         both track it, so the button always states what it is about to do. */
      _dlSel(){
        var out=[],seen={},all=PG.init._uploads||[];
        [PG.init._selRec,PG.init._selLig].forEach(function(nm){
          if(!nm||seen[nm])return;seen[nm]=1;
          var hit=null,i;for(i=0;i<all.length;i++){if(all[i].name===nm){hit=all[i];break;}}
          out.push(hit||{name:nm});});                                 // stale selection → still try it
        return out;},
      _dlLabel(){
        var s=PG.init._dlSel();
        if(s.length)return '⤓ Download selected'+(s.length>1?(' ('+s.length+')'):'');
        var n=(PG.init._uploads||[]).length;
        return '⤓ Download all'+(n?(' ('+n+')'):'');},
      _dlTitle(){
        var s=PG.init._dlSel();
        if(!s.length)return 'Save every archived upload back to this machine — one file each, same name and format as uploaded. Select a card first to download only that one. Your browser asks once to allow multiple downloads; allow it, or only the first file arrives.';
        return 'Save '+(s.length>1?'both selected files':'the selected file')+' — '
             +s.map(function(f){return f.name;}).join(', ')
             +' — same name and format as uploaded. Click the selected card again to clear it and download everything.';},
      _dlSync(){var b=$('poseUplDlBtn');if(b&&!PG.init._dlBusy){b.textContent=PG.init._dlLabel();b.title=PG.init._dlTitle();}
        PG.init._delSync();},                                          // the bin tracks the same selection

      /* ---- Delete selected ----------------------------------------------------------
         Deliberately narrower than the download button: delete ONLY ever touches the
         selection, and the button is disabled when nothing is selected. There is no
         "delete all" — a one-click, no-undo wipe of someone's prepared receptors is not
         a feature.
         Two-step arm rather than a confirm() modal: the first click turns the button red
         and names what it is about to remove, the second does it, and it disarms itself
         after 4 s or if the selection changes underneath it (an armed button pointing at
         a file you have since deselected is exactly how you delete the wrong thing).
         Server-side this is a move into <upload dir>/_trash/, not an unlink, so a
         mis-click is still recoverable on the box. */
      _delArm:null,_delTimer:null,_delBusy:false,
      _delKey(){return PG.init._dlSel().map(function(f){return f.name;}).join(' ');},
      _delDisarm(){
        if(PG.init._delTimer){clearTimeout(PG.init._delTimer);PG.init._delTimer=null;}
        if(PG.init._delArm===null)return;
        PG.init._delArm=null;PG.init._delSync();},
      _delSync(){
        var b=$('poseUplDelBtn');if(!b||PG.init._delBusy)return;
        var s=PG.init._dlSel(),armed=(PG.init._delArm!==null&&PG.init._delArm===PG.init._delKey());
        if(PG.init._delArm!==null&&!armed){PG.init._delArm=null;                  // selection moved -> stale arm
          if(PG.init._delTimer){clearTimeout(PG.init._delTimer);PG.init._delTimer=null;}}
        b.disabled=!s.length;
        b.style.opacity=s.length?'1':'.45';
        b.style.cursor=s.length?'pointer':'not-allowed';
        if(armed){
          var nm=s[0].name.replace(/\.(pdb|ent)$/i,'');
          if(nm.length>17)nm=nm.slice(0,16)+'…';                                  // keep the armed button on one row
          b.textContent=(s.length>1?('\u{1F5D1} Delete '+s.length+' files?'):('\u{1F5D1} Delete '+nm+'?'));
          b.style.borderColor='#9f1239';b.style.background='rgba(244,63,94,.14)';b.style.color='#fda4af';
          b.title='Click again to move '+s.map(function(f){return f.name;}).join(', ')+' to the server’s _trash/ folder. Disarms itself in a moment.';
        }else{
          b.textContent='\u{1F5D1} Delete'+(s.length>1?(' ('+s.length+')'):'');
          b.style.borderColor='#1e293b';b.style.background='#0b1120';b.style.color='#94a3b8';
          b.title=s.length
            ? ('Remove '+s.map(function(f){return f.name;}).join(', ')+' — moved to the server’s _trash/ folder, not erased. Asks for a second click first.')
            : 'Select a receptor or ligand card first — delete only ever removes the selection, never the whole gallery.';}},
      deleteSelected(){
        if(PG.init._delBusy)return;
        var sel=PG.init._dlSel();
        if(!sel.length){PG._miniLog('select a card first — delete only removes the selection','#fbbf24');return;}
        var key=PG.init._delKey();
        if(PG.init._delArm!==key){                                                 // first click -> arm
          PG.init._delArm=key;PG.init._delSync();
          if(PG.init._delTimer)clearTimeout(PG.init._delTimer);
          PG.init._delTimer=setTimeout(function(){PG.init._delTimer=null;PG.init._delArm=null;PG.init._delSync();},4000);
          return;}
        if(PG.init._delTimer){clearTimeout(PG.init._delTimer);PG.init._delTimer=null;}
        PG.init._delArm=null;PG.init._delBusy=true;
        var b=$('poseUplDelBtn');if(b){b.disabled=true;b.textContent='\u{1F5D1} …';b.style.opacity='.6';b.style.cursor='progress';}
        var names=sel.map(function(f){return f.name;});
        var finish=function(){PG.init._delBusy=false;PG.init._delSync();};
        fetch('/pose/delete_upload',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({names:names})})
          .then(function(r){return r.json();})
          .then(function(res){
            var gone=(res&&res.deleted)||[];
            gone.forEach(function(n){                                              // drop the selection + any pin
              if(PG.init._selRec===n)PG.init._selRec=null;
              if(PG.init._selLig===n)PG.init._selLig=null;
              if(PG.init._pins)delete PG.init._pins[n];
              if(PG._pinLigs)delete PG._pinLigs[n];});
            if(gone.length)PG._miniLog('\u{1F5D1} '+gone.join(', ')+' → _trash/ ('+gone.length+' removed)','#fb7185');
            var bad=(res&&res.failed)||[];
            if(bad.length)PG._miniLog('could not remove '+bad.map(function(f){return f.name+' ('+f.err+')';}).join(', '),'#fbbf24',true);
            if(!gone.length&&!bad.length)PG._miniLog('delete failed ('+((res&&res.err)||'?')+')','#fb7185');
            finish();
            if(gone.length)PG.init._renderUploads(true);                           // refetch: the DB row is gone too
          })
          .catch(function(e){PG._miniLog('delete failed ('+e.message+') — is /pose/delete_upload deployed?','#fb7185');finish();});},
      _dlBtn(label,busy){
        var b=$('poseUplDlBtn');if(!b)return;
        b.disabled=!!busy;b.style.opacity=busy?'.6':'1';b.style.cursor=busy?'progress':'pointer';
        if(label!=null)b.textContent=label;},
      downloadAll(){
        if(PG.init._dlBusy)return;                                    // double-click guard
        PG.init._dlBusy=true;PG.init._dlBtn('⤓ …',true);
        var done=function(){PG.init._dlBusy=false;PG.init._dlBtn(null,false);PG.init._dlSync();};
        var go=function(files){
          if(!files.length){PG._miniLog('nothing to download — no archived uploads','#fbbf24');done();return;}
          PG._miniLog('⤓ downloading '+files.length+' file'+(files.length===1?'':'s')
            +(files.length>1?' — allow “download multiple files” if your browser asks':' · '+files[0].name),'#a78bfa');
          var i=0,okN=0,bad=[];
          (function next(){
            if(i>=files.length){
              done();
              if(!bad.length)PG._miniLog('✓ '+okN+' file'+(okN===1?'':'s')+' downloaded','#34d399');
              else PG._miniLog('⤓ '+okN+' downloaded · '+bad.length+' failed: '+bad.slice(0,3).join(', ')+(bad.length>3?' …':''),'#fbbf24',true);
              return;}
            var f=files[i++];
            PG.init._dlBtn('⤓ '+i+'/'+files.length+'…',true);
            fetch('/pose/uploaded_pdb',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:f.name})})
              .then(function(r){return r.json();})
              .then(function(res){
                if(res&&res.ok&&res.pdb!=null){PG.init._saveText(res.pdb,res.name||f.name);okN++;}
                else bad.push(f.name+' ('+((res&&res.err)||'?')+')');})
              .catch(function(e){bad.push(f.name+' ('+e.message+')');})
              .then(function(){setTimeout(next,PG.init._dlGap);});})();};
        var sel=PG.init._dlSel();
        if(sel.length){go(sel);return;}                                 // a selection beats "all"
        var have=PG.init._uploads||[];
        if(have.length){go(have.slice());return;}
        /* Panel opened straight onto Download and the gallery has not painted yet —
           pull the list rather than downloading nothing. */
        fetch('/pose/uploaded_pdbs').then(function(r){return r.json();}).then(function(res){
          if(!res||!res.ok){PG._miniLog('could not read uploads'+((res&&res.err)?(' — '+res.err):''),'#fb7185');done();return;}
          var recs=res.receptors||[],ligs=res.ligands||[];
          recs.forEach(function(f){f.kind='receptor';});ligs.forEach(function(f){f.kind='ligand';});
          PG.init._uploads=recs.concat(ligs);
          go(PG.init._uploads.slice());
        }).catch(function(){PG._miniLog('upload list unavailable (is /pose/uploaded_pdbs deployed?)','#fb7185');done();});},
      _dragStart(name,ev){PG.init._dragName=name;try{ev.dataTransfer.effectAllowed='move';ev.dataTransfer.setData('text/plain',name);}catch(e){}},
      _drop(kind,beforeName,ev){
        if(ev&&ev.preventDefault)ev.preventDefault();
        var name=PG.init._dragName||(ev&&ev.dataTransfer&&ev.dataTransfer.getData?ev.dataTransfer.getData('text/plain'):null);
        PG.init._dragName=null;
        if(!name||name===beforeName)return;
        PG.init._rearrange(name,kind,beforeName||null);},
      _rearrange(name,kind,beforeName){
        var all=PG.init._uploads||[],moved=null,i;
        for(i=0;i<all.length;i++){if(all[i].name===name){moved=all[i];break;}}
        if(!moved)return;
        moved.kind=kind;
        var recs=[],ligs=[];
        for(i=0;i<all.length;i++){var f=all[i];if(f.name===name)continue;(f.kind==='ligand'?ligs:recs).push(f);}
        var tgt=(kind==='ligand')?ligs:recs,at=tgt.length;
        if(beforeName){for(i=0;i<tgt.length;i++){if(tgt[i].name===beforeName){at=i;break;}}}
        tgt.splice(at,0,moved);
        PG.init._uploads=recs.concat(ligs);
        PG.init._paintUploads(recs,ligs);
        fetch('/pose/uploaded_arrange',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({receptors:recs.map(function(f){return f.name;}),ligands:ligs.map(function(f){return f.name;})})})
          .then(function(r){return r.json();}).then(function(res){
            if(res&&res.ok)PG._miniLog('↔ '+name+' → '+kind+'s (saved)','#a78bfa');
            else PG._miniLog('reclassify not saved ('+((res&&res.err)||'?')+')','#7c6bae',true);
          }).catch(function(){PG._miniLog('reclassify not saved (endpoint offline)','#7c6bae',true);});},
      _uploadCard(f){
        var nm=String(f.name||'');var esc=nm.replace(/\\/g,'\\\\').replace(/'/g,"\\'");
        var disp=nm.replace(/\.(pdb|ent)$/i,'');
        var kind=(f.kind==='ligand')?'ligand':'receptor';
        var atoms=(f.atoms!=null)?(f.atoms.toLocaleString()+' atoms'):'—';
        var kb=(f.size!=null)?((f.size/1024).toFixed(f.size<10240?1:0)+' KB'):'';
        var ctr=(f.center&&f.center.length===3&&f.center.every(function(v){return v!=null&&isFinite(v);}))?f.center.map(function(v){return (+v).toFixed(1);}).join(', '):null;
        var glyph='<svg viewBox="0 0 56 56" width="50" height="50">'
          +'<circle cx="28" cy="28" r="21" fill="none" stroke="#1e3a4a" stroke-width="1.5" stroke-dasharray="3 3"></circle>'
          +'<line x1="20" y1="24" x2="30" y2="19" stroke="#5b6b86" stroke-width="2" stroke-linecap="round"></line>'
          +'<line x1="30" y1="19" x2="35" y2="30" stroke="#5b6b86" stroke-width="2" stroke-linecap="round"></line>'
          +'<line x1="35" y1="30" x2="24" y2="34" stroke="#5b6b86" stroke-width="2" stroke-linecap="round"></line>'
          +'<line x1="24" y1="34" x2="20" y2="24" stroke="#5b6b86" stroke-width="2" stroke-linecap="round"></line>'
          +'<circle cx="20" cy="24" r="4.5" fill="#3b6cf6" stroke="#05070f"></circle>'
          +'<circle cx="30" cy="19" r="4.5" fill="#22c55e" stroke="#05070f"></circle>'
          +'<circle cx="35" cy="30" r="4.5" fill="#ef4444" stroke="#05070f"></circle>'
          +'<circle cx="24" cy="34" r="4.5" fill="#22d3ee" stroke="#05070f"></circle>'
          +'</svg>';
        var selK=(kind==='ligand')?PG.init._selLig:PG.init._selRec;
        var isSel=(selK===esc),isPin=!!(PG.init._pins||{})[esc];
        var selDot='<div class="pose-sel-dot" title="selected" style="position:absolute;top:7px;left:7px;width:13px;height:13px;border-radius:50%;background:#22d3ee;box-shadow:0 0 0 2px #05070f,0 0 6px rgba(34,211,238,.75);z-index:3;pointer-events:none;display:'+(isSel?'block':'none')+';"></div>';
        var pinDot='<div class="pose-pin-dot" title="pinned · double-click to unpin" style="position:absolute;top:7px;right:7px;width:13px;height:13px;border-radius:50%;background:#f59e0b;box-shadow:0 0 0 2px #05070f;z-index:3;pointer-events:none;display:'+(isPin?'block':'none')+';"></div>';
        return '<div draggable="true" data-upl="'+esc+'" data-kind="'+kind+'"'
          +' onclick="PoseGen.init._cardClick(\''+kind+'\',\''+esc+'\',event)"'
          +' ondblclick="PoseGen.init._cardDbl(\''+kind+'\',\''+esc+'\',event)"'
          +' ondragstart="PoseGen.init._dragStart(\''+esc+'\',event)"'
          +' ondragover="event.preventDefault();event.stopPropagation()"'
          +' ondrop="event.stopPropagation();PoseGen.init._drop(\''+kind+'\',\''+esc+'\',event)"'
          +' title="'+nm+(f.mtime?(' · '+f.mtime):'')+(ctr?(' · center '+ctr):'')+' · click to select · click again to unselect · double-click to pin · drag to reclassify" style="position:relative;border:1px solid #1e293b;border-radius:11px;background:#0b1120;overflow:hidden;cursor:pointer;" onmouseover="this.style.borderColor=\'#22d3ee\'" onmouseout="this.style.borderColor=\'#1e293b\'">'
          +selDot+pinDot
          +'<div style="height:80px;display:flex;align-items:center;justify-content:center;background:radial-gradient(circle at 50% 42%,rgba(34,211,238,.06),transparent 70%);">'+glyph+'</div>'
          +'<div style="padding:5px 7px;border-top:1px solid #1e293b;font-family:ui-monospace,monospace;font-size:9px;">'
            +'<div style="color:#cbd5e1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-weight:600;">'+disp+'</div>'
            +'<div style="display:flex;justify-content:space-between;margin-top:2px;color:#64748b;gap:4px;"><span style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'+atoms+'</span><span style="color:#22d3ee;flex-shrink:0;">'+kb+'</span></div>'
            +(ctr?'<div style="margin-top:2px;color:#64748b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="box center '+ctr+' Å"><span style="color:#475569;">ctr</span> '+ctr+'</div>':'')
          +'</div>'
        +'</div>';},
      // ---- selection (single click) + pin (double click) markers on gallery cards ----
      // Single click marks the card selected (solid cyan dot, top-left) and loads it, but
      // the load is deferred ~220 ms so a double-click — which pins — doesn't ALSO reload
      // the molecule (a big receptor re-parse would visibly flash the 3D view). Double
      // click toggles a persistent amber pin dot (top-right). State is keyed by the
      // (escaped) filename and lives in PG.init, so it survives the grid re-renders that
      // drag-reclassify / refresh trigger (_uploadCard reads it back on every render).
      _cardClick(kind,name,ev){
        var sel=(kind==='ligand')?PG.init._selLig:PG.init._selRec;
        // ev.detail is the native click count: 1 for a standalone click, 2 for the
        // second click INSIDE a double-click. Only a standalone repeat click clears
        // the selection — otherwise double-clicking to pin would fight the toggle.
        var multi=!!(ev&&ev.detail>1);
        if(sel===name&&!multi){                                               // clicking the selected card again → unselect
          if(kind==='ligand')PG.init._selLig=null; else PG.init._selRec=null;
          if(PG.init._clickTimer){clearTimeout(PG.init._clickTimer);PG.init._clickTimer=null;}
          PG.init._refreshMarks();                                            // dot off; molecule stays loaded, so no re-parse
          return;
        }
        if(kind==='ligand')PG.init._selLig=name; else PG.init._selRec=name;   // one selection per grid
        PG.init._refreshMarks();                                              // instant dot feedback
        if(PG.init._clickTimer)clearTimeout(PG.init._clickTimer);
        PG.init._clickTimer=setTimeout(function(){PG.init._clickTimer=null;
          if(kind==='ligand')PG.init.pickLigand(name); else PG.init.pickUpload(name);},220);},
      _cardDbl(kind,name,ev){
        if(ev&&ev.preventDefault)ev.preventDefault();
        if(PG.init._clickTimer){clearTimeout(PG.init._clickTimer);PG.init._clickTimer=null;}   // cancel the pending single-click load — pinning must not reload the 3D view
        if(kind==='ligand')PG.init._selLig=name; else PG.init._selRec=name;                    // selection follows the double-clicked card
        PG.init._togglePin(name);PG.init._refreshMarks();
        var pinned=!!(PG.init._pins||{})[name];
        PG._miniLog((pinned?'📌 pinned ':'unpinned ')+name.replace(/\.(pdb|ent)$/i,''),pinned?'#f59e0b':'#7c6bae');},
      /* Pinned ligand structures, keyed by gallery file name. _draw3D re-adds
         each of these to the scene on every redraw, so they stay visible while
         the active ligand keeps being replaced. Receptors are marker-only:
         pinning one flags the card but adds nothing to the 3D box. */
      _togglePin(name){
        if(!PG.init._pins)PG.init._pins={};
        if(!PG._pinLigs)PG._pinLigs={};
        if(PG.init._pins[name]){                                   // unpin → drop the structure and redraw without it
          delete PG.init._pins[name];
          if(PG._pinLigs[name]){
            delete PG._pinLigs[name];
            PG._syncPinCenter(false);                              // last pin gone → box is free again
            PG.init._pinRedraw();
          }
          return;
        }
        PG.init._pins[name]=true;
        var kind=PG.init._kindOf(name);
        if(kind!=='ligand')return;                                 // receptors: badge only, nothing to overlay
        if(PG._pinLigs[name]){                                     // already loaded earlier this session
          PG._syncPinCenter(false);                                // re-pin the box on it
          PG.init._pinRedraw();
          return;
        }
        PG.init._pinLoad(name);
      },
      _kindOf(name){
        var u=(PG.init._uploads||[]).filter(function(f){return f.name===name;})[0];
        if(u&&u.kind)return u.kind;
        var card=document.querySelector('[data-upl="'+String(name).replace(/"/g,'\\"')+'"]');
        return (card&&card.getAttribute('data-kind'))||'ligand';
      },
      /* Fetch a pinned ligand's real structure once and keep it. Same endpoint
         and parsing as pickLigand, but it must NOT touch `L`, the box centre or
         the pose — pinning is additive and has to leave the active view alone. */
      _pinLoad(name){
        PG._miniLog('pinning '+name+'…','#f59e0b');
        fetch('/pose/uploaded_pdb',{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({name:name,smiles:false})})
          .then(function(r){return r.json();})
          .then(function(res){
            if(!res||!res.ok||!res.pdb){PG._miniLog('could not pin '+name+' (no structure)','#fb7185');return;}
            var lig=null;
            try{lig=ligandFromPDB(PE.parsePDB(res.pdb));}catch(e){lig=null;}
            if(!lig||!lig.REF||!lig.REF.length){PG._miniLog('could not pin '+name+' (unparsable)','#fb7185');return;}
            lig._label=String(name);
            if(!PG._pinLigs)PG._pinLigs={};
            PG._pinLigs[name]=lig;
            PG._log('PINNED LIGAND ADDED →',PG._ligTag(lig),'· stays on screen across ligand swaps');
            PG._miniLog('📌 '+name.replace(/\.(pdb|ent)$/i,'')+' pinned in view','#f59e0b');
            PG._syncPinCenter(false);        // anchor the docking box on this ligand's real centre
            PG.init._pinRedraw();
          })
          .catch(function(e){PG._miniLog('could not pin '+name+': '+e.message,'#fb7185');});
      },
      /* Redraw the box so the pinned set takes effect, without disturbing the
         active ligand or its pose (unlike stage(), which can regenerate). */
      _pinRedraw(){
        try{ if(L&&PG.init.use3D()&&PG.init.cur)PG.init.render3D(PG.init.cur);
             else if(L&&PG.init.cur)PG.init.draw(PG.init.cur); }catch(e){PG._log('pinRedraw failed:',e&&e.message);}
      },
      _refreshMarks(){
        var pins=PG.init._pins||{},sel={receptor:PG.init._selRec,ligand:PG.init._selLig};
        var cards=document.querySelectorAll('#poseRecGrid [data-upl],#poseLigGrid [data-upl]');
        Array.prototype.forEach.call(cards,function(card){
          var nm=card.getAttribute('data-upl'),kd=card.getAttribute('data-kind');
          var s=card.querySelector('.pose-sel-dot'),p=card.querySelector('.pose-pin-dot');
          if(s)s.style.display=(sel[kd]===nm)?'block':'none';
          if(p)p.style.display=pins[nm]?'block':'none';});
        PG.init._dlSync();},                                           // the ⤓ button follows the selection
      pickUpload(name){
        PG._miniLog('loading receptor '+name+' from server…','#67e8f9');
        fetch('/pose/uploaded_pdb',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:name})})
          .then(function(r){return r.json();}).then(function(res){
            if(res&&res.ok&&res.pdb){
              try{var p=PE.parsePDB(res.pdb);PG._applyProtein(p,String(res.name||name).replace(/\.(pdb|ent)$/i,''),res.pdb,true);
                var c=p.center;PG._miniLog('✓ receptor loaded → '+(res.name||name)+((c&&c.length===3)?(' · box centered on '+c.map(function(v){return (+v).toFixed(1);}).join(', ')):''),'#34d399');}
              catch(e){PG._miniLog('parse error: '+e.message,'#fb7185');}
            }else{PG._miniLog('could not load '+name+': '+((res&&res.err)||'unknown error'),'#fb7185');}
          }).catch(function(e){PG._miniLog('request failed (is /pose/uploaded_pdb deployed?): '+e.message,'#fb7185');});},
      pickLigand(name){
        PG._miniLog('loading ligand '+name+'…','#67e8f9');
        fetch('/pose/uploaded_pdb',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:name,smiles:true})})
          .then(function(r){return r.json();}).then(function(res){
            if(res&&res.ok&&res.pdb){
              try{
                var p=PE.parsePDB(res.pdb);var c=p.center;
                // recentre the docking box on the picked ligand
                if(c&&c.length===3&&c.every(function(v){return isFinite(v);})){
                  box.center=[+c[0],+c[1],+c[2]];
                  if(protein){protein._site=null;protein._siteCenter=null;}
                }
                // swap the displayed green ligand for the picked one's real 3D structure —
                // it renders green through the same shared pose path as the default ligand
                var lig=ligandFromPDB(p);
                if(lig){
                  L=lig;L._label=String(name||'gallery');   // debug identity for _ligTag / the [Pose3D] log
                  L._smiles=(res.smiles||'');   // derived SMILES → lets N_rot (auto) rebuild the torsion tree for the entropy penalty
                  PG.init.cur={seed:0,off:[0,0,0],quat:[1,0,0,0],tors:[]};   // identity pose → show the actual uploaded geometry
                  PG._poseSrc='uploaded';
                  PG._derived();PG._invalidate();PG._syncBoxInputs();PG.stage(PG._stage);
                }else{
                  PG._syncBoxInputs();PG._invalidate();PG.stage(PG._stage);
                }
                var ctr=(c&&c.length===3)?(' · box → '+c.map(function(v){return (+v).toFixed(1);}).join(', ')):'';
                if(res.smiles){var si=$('poseSmiles');if(si)si.value=res.smiles;
                  PG._miniLog('✓ ligand '+(res.name||name)+' shown in the box'+(lig?(' ('+lig.atoms.length+' atoms)'):'')+' · SMILES set → '+res.smiles+ctr+' · press ● Build to dock it flexibly','#34d399');}
                else PG._miniLog('✓ ligand '+(res.name||name)+' shown in the box'+ctr+' — no SMILES yet (add a .smi sidecar, a REMARK SMILES, or RDKit on the server, then ● Build to dock)','#fbbf24');
              }catch(e){PG._miniLog('parse error: '+e.message,'#fb7185');}
            }else{PG._miniLog('could not load '+name+': '+((res&&res.err)||'unknown error'),'#fb7185');}
          }).catch(function(e){PG._miniLog('request failed: '+e.message,'#fb7185');});},
      card(){var bl=$('poseBoxLabel');var pc=$('poseProteinCard');if(!pc)return;if(!protein){if(bl)bl.textContent='20³ Å';
          pc.innerHTML='<div style="font-size:11.5px;color:#64748b;background:#0b1120;border:1px dashed #1e293b;border-radius:11px;padding:11px;">Load a target protein (PDB ID or .pdb) to render it in 3D and auto-center the docking box on its binding pocket. The box center and length stay editable in the bar above.</div>';return;}
        if(bl)bl.textContent=protein.size+'³ Å';
        pc.innerHTML='<div style="background:#0b1120;border:1px solid #1e293b;border-radius:13px;padding:13px;">'
          +'<p style="font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#475569;margin:0 0 8px;">Target — '+(protein.id||'uploaded')+'</p>'
          +kv('chains',protein.chains.join(', ')||'—')+kv('residues',protein.nres)+kv('atoms',protein.natoms)
          +(protein._site?kv('pocket shown',protein._site.nres+' res · sticks','#97a3b0'):'')
          +kv('box center',box.center.map(v=>v.toFixed(1)).join(', '),'#22d3ee')
          +kv('box size',box.size+' Å'+(cfgBox?'  (config)':(protein.hasLigand?'  (from ligand)':'')),'#34d399')
          +(window.Plotly?'':kv('render','Plotly absent — 2D box','#fbbf24'))+'</div>';},
    },

    /* ---- protein intake ---- */
    fetchPdb(){const id=$('posePdbId').value.trim().toUpperCase();if(!/^[0-9A-Za-z]{4}$/.test(id)){PG._status('Enter a 4-character PDB ID.','#fbbf24');return;}
      PG._status('Fetching '+id+' from RCSB…','#67e8f9');
      fetch('https://files.rcsb.org/download/'+id+'.pdb').then(r=>{if(!r.ok)throw new Error('HTTP '+r.status);return r.text();})
        .then(txt=>PG._applyProtein(PE.parsePDB(txt),id,txt))
        .catch(err=>PG._status('Could not fetch '+id+' ('+err.message+'). Upload the .pdb instead.','#fb7185'));},
    loadPdbFile(file){if(!file)return;PG._status('Reading '+file.name+'…','#67e8f9');const fr=new FileReader();var nm=file.name.replace(/\.[^.]+$/,'');
      fr.onload=()=>{try{PG._applyProtein(PE.parsePDB(fr.result),nm,fr.result);PG._saveUpload(nm,fr.result);}catch(e){PG._status('Parse error: '+e.message,'#fb7185');}};
      fr.onerror=()=>PG._status('Could not read file.','#fb7185');fr.readAsText(file);},
    /* archive the uploaded .pdb to pose.default_upload_path so the user can find it on the server */
    _saveUpload(name,pdb){fetch('/pose/save_upload',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:name,pdb:pdb})}).then(function(r){return r.json();}).then(function(res){if(res&&res.ok){PG._miniLog('📁 upload saved on server → '+res.path,'#a78bfa');PG.init._renderUploads(true);}else if(res&&res.err)PG._miniLog('upload not archived ('+res.err+')','#7c6bae');}).catch(function(){});},
    _applyProtein(p,id,raw,useOwnCenter){if(!p.natoms){PG._status('No ATOM/HETATM records found in '+id+'.','#fb7185');return;}
      p.id=id;p.raw=raw||'';protein=p;box={center:p.center,size:p.size};
      if(cfgBox){if(!useOwnCenter&&cfgBox.cx!=null&&cfgBox.cy!=null&&cfgBox.cz!=null)box.center=[+cfgBox.cx,+cfgBox.cy,+cfgBox.cz];if(cfgBox.len!=null&&+cfgBox.len>=4)box.size=+cfgBox.len;}
      // If a real-pose ligand (picked from an uploaded .pdb) is active, keep the box on
      // the ligand's docked centre. Otherwise this receptor load would recentre the box
      // on the protein and drag the ligand tens of Å off its pose, so ⚛ breakdown would
      // score a displaced pose (0 pairs / garbage) instead of the uploaded one.
      if(L&&L._realPose&&Array.isArray(L._realCenter)&&L._realCenter.length===3){box.center=L._realCenter.slice();}
      PG._syncBoxInputs();
      var _od=(($('poseDaOutDir')||{}).value||'').trim().replace(/\/+$/,'');var _stage=(_od||'<out-dir>')+'/Dataset_VS/'+id+'_mc';
      var _anchored=(L&&L._realPose&&Array.isArray(L._realCenter));
      PG._status('✓ '+id+' — '+p.chains.length+' chains, '+p.nres+' residues, '+p.natoms+' atoms. '+(_anchored?('Box kept on the uploaded ligand pose ('+L._realCenter.map(function(v){return (+v).toFixed(1);}).join(', ')+').'):('Box auto-centered'+(p.hasLigand?' on bound ligand.':' on protein.')))+' · saved on server to '+_stage+'/','#34d399');
      PG._derived();PG._invalidate();PG.stage(PG._stage);
      if(PG.init.use3D())setTimeout(()=>{var g=PG._plotReady&&PG._plotReady();if(g){try{PG._safe&&PG._safe(Plotly.Plots.resize(g),'resize-applyProtein');}catch(e){}}},80);},
    _status(msg,col){PG._miniLog(msg,col);},
    /* append a line to the activity mini-chat; flash the toggle button when the panel is hidden */
    _miniLog(msg,col,mono){var out=$('poseMiniChatOutput');if(out){var d=document.createElement('div');d.style.cssText='font-size:11px;line-height:1.5;color:'+(col||'#94a3b8')+';padding:6px 9px;background:#0a0716;border:1px solid #3b2a6b;border-radius:8px;word-break:break-word;'+(mono?'font-family:ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap;':'');d.textContent=msg;out.appendChild(d);out.scrollTop=out.scrollHeight;}
      var chat=$('poseMiniChat'),btn=$('poseMiniBtn');if(btn&&chat&&(chat.style.display==='none'||!chat.style.display)){btn.style.boxShadow='0 0 0 2px rgba(167,139,250,.55)';}},
    /* show / hide the activity mini-chat */
    miniToggle(){var chat=$('poseMiniChat');if(!chat)return;var hidden=(chat.style.display==='none'||!chat.style.display);chat.style.display=hidden?'flex':'none';var btn=$('poseMiniBtn');if(hidden&&btn)btn.style.boxShadow='none';},
    /* lightweight Q&A over the pose tool's own methods (no LLM) */
    miniAsk(){var inp=$('poseMiniInput');if(!inp)return;var q=(inp.value||'').trim();if(!q)return;inp.value='';PG._miniLog('you: '+q,'#67e8f9');PG._miniLog(PG._miniAnswer(q),'#cbd5e1');},
    _miniAnswer(q){q=q.toLowerCase();var KB=[
      {k:['monte','carlo',' mc','metropolis','accept','reject','③'],a:'③ Monte-Carlo + BFGS is a global search. Each step: (1) perturb the pose — random translation, rotation, and torsion change; (2) run a BFGS local optimization to the nearest score minimum; (3) accept or reject by the Metropolis criterion P(accept) = min(1, exp(−ΔS / kT)). Downhill moves are always taken; uphill moves are taken with a probability that falls off as the score worsens, which lets the search escape local minima. The per-step score is the fast surrogate (clash + box-center), not DeepAtom.'},
      {k:['minim','miniz','minia','strain','relax','②','geometry'],a:'② Ligand Minimization relaxes the embedded conformer by gradient-descending a strain energy: E = wT·Σθ² + wC·Σ max(0, d0−d)² + wK·‖c − c_box‖². The terms are torsional strain (wT, default 12), steric clash against the fixed protein pocket (wC, default 8; d0 ≈ 3.2 Å contact distance), and a pull toward the docking-box center (wK, default 2). All weights are editable in the sidebar; lowering E removes clashes and over-stretched torsions before docking.'},
      {k:['bfgs','quasi','newton','hessian'],a:'BFGS is a quasi-Newton local optimizer: it builds an approximate inverse-Hessian from successive gradients to take near-Newton steps toward the nearest minimum of the score. In stage ③ it polishes each perturbed pose before the Metropolis accept/reject decision.'},
      {k:['surrogate','clash','fast score','live search','per-step','search score'],a:'The live search is driven by a fast in-browser surrogate, S = wC·Σ max(0, d0−d) + wK·‖c − c₀‖ (clash penalty + distance from box center), editable in the sidebar. It runs thousands of times per search, so DeepAtom — the slow server CNN — only scores the single best pose on demand.'},
      {k:['deepatom','cnn','shufflenet','affinity','pk','δg','delta g','dg','score best'],a:'DeepAtom is a server-side ShuffleNetV3 CNN. It voxel-grids the single best pose and predicts ΔG; the card shows pK = −ΔG / 1.36. It is far too slow to drive the per-step search, so it scores only the final best pose when you click ⚛ Score best pose.'},
      {k:['box','docking box','center','length','grid'],a:'The docking box (center x/y/z + edge length, editable in the 3D legend) bounds the search. ① Random Start places the ligand uniformly inside it (uniform translation, Shoemake-uniform rotation, torsions uniform in [−π, π]); the minimization and surrogate scores both include a term pulling the ligand center toward the box center.'},
      {k:['random','start','①','placement','embed'],a:'① Random Start initializes the pose inside the docking box: translation uniform in the box, rotation as a uniform random orientation (Shoemake), and each rotatable torsion uniform in [−π, π]. The 3D conformer itself comes from RDKit ETKDG + a quick force-field relax.'}
    ];for(var i=0;i<KB.length;i++)for(var j=0;j<KB[i].k.length;j++)if(q.indexOf(KB[i].k[j])>=0)return KB[i].a;
      return 'I can explain: ① Random Start, ② Minimization, ③ Monte-Carlo, BFGS, Metropolis, the surrogate score, the docking box, or DeepAtom scoring — ask about any of those (e.g. "explain Monte-Carlo formula").';},
  };

  /* ============================================================================
     VIEW LAYERS — molecular Surface (1.4 Å water probe) + Secondary Structure (DSSP-lite)
     Two optional overlays for the 3D pocket view, toggled by the buttons in the
     #poseBox3DHint bar. Both follow the workflow in agent_info.txt:
       • Surface depends only on atom coordinates + van der Waals radii. A water-sized
         probe (1.4 Å) is rolled over the pocket atoms' vdW spheres → the solvent-excluded
         CONTACT surface: the SHAPE of the cavity the ligand sits in, not the fold.
       • Secondary structure comes from the backbone hydrogen-bond pattern (Kabsch–Sander /
         DSSP): i→i+4 H-bonds = α-helix, parallel/antiparallel bridges = β-sheet, rest = loop.
         Coloured red / cyan / grey, like Maestro.
     Both are rigid-receptor properties, cached per (protein, box) and reused across every
     pose / MC frame, so nothing heavy recomputes while the search animates.
     ============================================================================ */
  const VDW={H:1.20,C:1.70,N:1.55,O:1.52,S:1.80,P:1.80,F:1.47,Cl:1.75,Br:1.85,I:1.98,Se:1.90,B:1.92,Zn:1.39,Fe:2.00,Mg:1.73,Mn:2.00,Ca:2.31,Na:2.27,K:2.75,Cu:1.40,Ni:1.63,Co:2.00};
  function vdw(el){return VDW[el]||1.70;}
  const PROBE=1.4;                                   // water radius (Å) — the rolling probe
  function fiboSphere(n){const p=[],inc=Math.PI*(3-Math.sqrt(5)),off=2/n;for(let i=0;i<n;i++){const y=i*off-1+off/2,r=Math.sqrt(Math.max(0,1-y*y)),phi=i*inc;p.push([Math.cos(phi)*r,y,Math.sin(phi)*r]);}return p;}
  // Solvent-excluded CONTACT dots: points on each atom's vdW sphere that (a) are not buried in
  // a neighbour and (b) where a rolled probe does not clash — the part a 1.4 Å water can touch.
  function surfaceDots(atoms,probe,dens){
    const A=atoms.filter(a=>a.el!=='H');const n=A.length;if(!n)return {x:[],y:[],z:[]};
    const R=new Array(n);for(let i=0;i<n;i++)R[i]=vdw(A[i].el);
    let maxR=0;for(let i=0;i<n;i++)if(R[i]>maxR)maxR=R[i];
    const cell=Math.max(2.5,maxR+probe),gk=v=>Math.floor(v/cell),key=(a,b,c)=>a+'_'+b+'_'+c;
    const grid={};for(let i=0;i<n;i++){const a=A[i],k=key(gk(a.x),gk(a.y),gk(a.z));(grid[k]||(grid[k]=[])).push(i);}
    function near(x,y,z){const out=[],cx=gk(x),cy=gk(y),cz=gk(z);for(let dx=-1;dx<=1;dx++)for(let dy=-1;dy<=1;dy++)for(let dz=-1;dz<=1;dz++){const c=grid[key(cx+dx,cy+dy,cz+dz)];if(c)for(let m=0;m<c.length;m++)out.push(c[m]);}return out;}
    const sph=fiboSphere(dens),X=[],Y=[],Z=[];
    for(let i=0;i<n;i++){const a=A[i],ri=R[i],nb=near(a.x,a.y,a.z);
      for(let s=0;s<sph.length;s++){const u=sph[s];
        const px=a.x+u[0]*ri,py=a.y+u[1]*ri,pz=a.z+u[2]*ri;                                   // point on the vdW surface
        const cx=a.x+u[0]*(ri+probe),cy=a.y+u[1]*(ri+probe),cz=a.z+u[2]*(ri+probe);           // probe centre if rolled here
        let hide=false;
        for(let t=0;t<nb.length;t++){const j=nb[t];if(j===i)continue;const b=A[j],rj=R[j];
          const ax=px-b.x,ay=py-b.y,az=pz-b.z;if(ax*ax+ay*ay+az*az<(rj-0.02)*(rj-0.02)){hide=true;break;}            // buried in a neighbour
          const ex=cx-b.x,ey=cy-b.y,ez=cz-b.z;if(ex*ex+ey*ey+ez*ez<(rj+probe-0.02)*(rj+probe-0.02)){hide=true;break;}} // probe would clash → unreachable
        if(!hide){X.push(px);Y.push(py);Z.push(pz);}
      }
    }
    return {x:X,y:Y,z:Z};
  }
  /* ---- solvent-excluded surface as a REAL mesh (so it can go fully solid, Maestro-style) ----
     Grid → mark the probe-forbidden zone (atoms inflated by the probe) → flood-fill the free
     probe-centre space that connects to the outside → exact Euclidean distance transform from
     that space → the SES is the level set at distance == probe (morphological erosion of the
     solvent-accessible volume by the probe ball). Surface-nets then turns the field into a
     smooth watertight mesh whose opacity the slider drives (translucent → solid).
     Sanity check: for one isolated atom the level set lands exactly on its vdW radius.        */
  function sesMesh(atoms,probe,maxCells){
    const A=atoms.filter(a=>a.el!=='H');const n=A.length;if(!n)return null;
    const R=new Float64Array(n);let maxR=0;
    for(let t=0;t<n;t++){R[t]=vdw(A[t].el)+probe;if(R[t]>maxR)maxR=R[t];}   // atom inflated by the probe
    let mnx=1e9,mny=1e9,mnz=1e9,mxx=-1e9,mxy=-1e9,mxz=-1e9;
    for(let t=0;t<n;t++){const a=A[t],r=R[t];
      if(a.x-r<mnx)mnx=a.x-r;if(a.y-r<mny)mny=a.y-r;if(a.z-r<mnz)mnz=a.z-r;
      if(a.x+r>mxx)mxx=a.x+r;if(a.y+r>mxy)mxy=a.y+r;if(a.z+r>mxz)mxz=a.z+r;}
    const pad=probe+1.2;mnx-=pad;mny-=pad;mnz-=pad;mxx+=pad;mxy+=pad;mxz+=pad;   // border must lie in free space
    const ex=mxx-mnx,ey=mxy-mny,ez=mxz-mnz;
    let h=0.5;const cnt=q=>(Math.ceil(ex/q)+1)*(Math.ceil(ey/q)+1)*(Math.ceil(ez/q)+1);
    while(cnt(h)>maxCells)h*=1.08;                                            // coarsen until the grid fits the budget
    const NX=Math.ceil(ex/h)+1,NY=Math.ceil(ey/h)+1,NZ=Math.ceil(ez/h)+1,NXY=NX*NY,TOT=NXY*NZ;
    if(NX<4||NY<4||NZ<4)return null;
    const ac=maxR+1.0;                                                       // atom lookup grid (flat int index — no string keys)
    const GX=Math.max(1,Math.ceil(ex/ac)+1),GY=Math.max(1,Math.ceil(ey/ac)+1),GZ=Math.max(1,Math.ceil(ez/ac)+1),GXY=GX*GY;
    const gi_=(px,py,pz)=>Math.min(GX-1,Math.max(0,Math.floor((px-mnx)/ac)))
                        +Math.min(GY-1,Math.max(0,Math.floor((py-mny)/ac)))*GX
                        +Math.min(GZ-1,Math.max(0,Math.floor((pz-mnz)/ac)))*GXY;
    const head=new Int32Array(GX*GY*GZ).fill(-1),nxt=new Int32Array(n).fill(-1);   // bucket linked-lists
    for(let t=0;t<n;t++){const a=A[t],c=gi_(a.x,a.y,a.z);nxt[t]=head[c];head[c]=t;}
    let nrT=-1,nrD=0,nrV=0;                                                  // out-params (avoids allocating an object per call)
    function nearest(px,py,pz){                                              // sphere minimising (|p-c| - R)
      const ci=Math.min(GX-1,Math.max(0,Math.floor((px-mnx)/ac))),
            cj=Math.min(GY-1,Math.max(0,Math.floor((py-mny)/ac))),
            ck=Math.min(GZ-1,Math.max(0,Math.floor((pz-mnz)/ac)));
      nrT=-1;nrV=1e9;nrD=0;
      const i0=Math.max(0,ci-1),i1=Math.min(GX-1,ci+1),j0=Math.max(0,cj-1),j1=Math.min(GY-1,cj+1),k0=Math.max(0,ck-1),k1=Math.min(GZ-1,ck+1);
      for(let k=k0;k<=k1;k++)for(let j=j0;j<=j1;j++)for(let i=i0;i<=i1;i++){
        for(let t=head[i+j*GX+k*GXY];t>=0;t=nxt[t]){const a=A[t];
          const dx=px-a.x,dy=py-a.y,dz=pz-a.z,d=Math.sqrt(dx*dx+dy*dy+dz*dz),v=d-R[t];
          if(v<nrV){nrV=v;nrT=t;nrD=d;}}}
      return nrT>=0;}
    // 1) voxels a probe CENTRE cannot occupy
    const inS=new Uint8Array(TOT);
    for(let t=0;t<n;t++){const a=A[t],r=R[t],r2=r*r;
      const k0=Math.max(0,Math.floor((a.z-r-mnz)/h)),k1=Math.min(NZ-1,Math.ceil((a.z+r-mnz)/h));
      const j0=Math.max(0,Math.floor((a.y-r-mny)/h)),j1=Math.min(NY-1,Math.ceil((a.y+r-mny)/h));
      for(let k=k0;k<=k1;k++){const dz=mnz+k*h-a.z,dz2=dz*dz;if(dz2>r2)continue;
        for(let j=j0;j<=j1;j++){const dy=mny+j*h-a.y,d2=dz2+dy*dy;if(d2>r2)continue;
          const rem=Math.sqrt(r2-d2),base=j*NX+k*NXY;
          const ia=Math.max(0,Math.ceil((a.x-rem-mnx)/h)),ib=Math.min(NX-1,Math.floor((a.x+rem-mnx)/h));
          for(let i=ia;i<=ib;i++)inS[base+i]=1;}}}
    // 2) free probe-centre space that actually reaches the outside (flood fill from the border)
    const ext=new Uint8Array(TOT),stk=new Int32Array(TOT);let sp=0;
    const push=c=>{if(!inS[c]&&!ext[c]){ext[c]=1;stk[sp++]=c;}};
    for(let k=0;k<NZ;k++)for(let j=0;j<NY;j++)for(let i=0;i<NX;i++)
      if(i===0||j===0||k===0||i===NX-1||j===NY-1||k===NZ-1)push(i+j*NX+k*NXY);
    while(sp>0){const c=stk[--sp],i=c%NX,j=((c/NX)|0)%NY,k=(c/NXY)|0;
      if(i>0)push(c-1);if(i<NX-1)push(c+1);
      if(j>0)push(c-NX);if(j<NY-1)push(c+NX);
      if(k>0)push(c-NXY);if(k<NZ-1)push(c+NXY);}
    // 3) sample the reachable SAS: project each boundary voxel onto its nearest sphere. These are
    //    EXACT surface points, so distances measured from them are sub-voxel accurate — that is what
    //    puts the level set exactly one probe radius in, with no grid fattening.
    const SX=[],SY=[],SZ=[];
    for(let k=1;k<NZ-1;k++)for(let j=1;j<NY-1;j++)for(let i=1;i<NX-1;i++){
      const c=i+j*NX+k*NXY;
      if(!inS[c])continue;                                                    // sample from the solid side of the boundary
      if(!(ext[c-1]||ext[c+1]||ext[c-NX]||ext[c+NX]||ext[c-NXY]||ext[c+NXY]))continue;   // ...that touches reachable free space
      const px=mnx+i*h,py=mny+j*h,pz=mnz+k*h;
      if(!nearest(px,py,pz)||nrD<1e-6)continue;
      const a=A[nrT],f=R[nrT]/nrD;
      const qx=a.x+(px-a.x)*f,qy=a.y+(py-a.y)*f,qz=a.z+(pz-a.z)*f;            // exact point on that sphere
      if(nearest(qx,qy,qz)&&nrV<-0.06)continue;                               // buried in another sphere → not on ∂S
      SX.push(qx);SY.push(qy);SZ.push(qz);}
    if(!SX.length)return null;
    // 4) distance from every voxel to that sampled SAS (splat each sample into its neighbourhood)
    const DMAX=probe+0.9,DM2=DMAX*DMAX;                                      // only needs to bracket the level set (D=probe)
    const D2=new Float32Array(TOT).fill(DM2);
    for(let q=0;q<SX.length;q++){const qx=SX[q],qy=SY[q],qz=SZ[q];
      const i0=Math.max(0,Math.ceil((qx-DMAX-mnx)/h)),i1=Math.min(NX-1,Math.floor((qx+DMAX-mnx)/h));
      const j0=Math.max(0,Math.ceil((qy-DMAX-mny)/h)),j1=Math.min(NY-1,Math.floor((qy+DMAX-mny)/h));
      const k0=Math.max(0,Math.ceil((qz-DMAX-mnz)/h)),k1=Math.min(NZ-1,Math.floor((qz+DMAX-mnz)/h));
      for(let k=k0;k<=k1;k++){const dz=mnz+k*h-qz,dz2=dz*dz;if(dz2>DM2)continue;
        for(let j=j0;j<=j1;j++){const dy=mny+j*h-qy,d2=dz2+dy*dy;if(d2>DM2)continue;
          const base=j*NX+k*NXY;
          for(let i=i0;i<=i1;i++){const c=base+i;if(!inS[c])continue;         // D is only ever read for solid voxels
            const dx=mnx+i*h-qx,dd=d2+dx*dx;
            if(dd<D2[c])D2[c]=dd;}}}}
    // 5) F<0 inside the solvent-excluded solid, F>0 in solvent, F=0 IS the surface (SES = SAS eroded by the probe)
    const F=new Float32Array(TOT);
    for(let t=0;t<TOT;t++)F[t]=ext[t]?probe:(inS[t]?(probe-Math.sqrt(D2[t])):(probe-DMAX));
    // 6) surface nets: one vertex per sign-changing cell, quads around sign-changing grid edges
    const CX=NX-1,CY=NY-1,CZ=NZ-1,CXY=CX*CY;
    const cv=new Int32Array(CX*CY*CZ).fill(-1);
    const VX=[],VY=[],VZ=[];
    const EG=[[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]];
    const g=new Float32Array(8);
    for(let k=0;k<CZ;k++)for(let j=0;j<CY;j++)for(let i=0;i<CX;i++){
      let mask=0;
      for(let c=0;c<8;c++){const val=F[(i+(c&1))+(j+((c>>1)&1))*NX+(k+((c>>2)&1))*NXY];g[c]=val;if(val<0)mask|=1<<c;}
      if(mask===0||mask===255)continue;
      let sx=0,sy=0,sz=0,ne=0;
      for(let e=0;e<12;e++){const a=EG[e][0],b=EG[e][1],ga=g[a],gb=g[b];
        if((ga<0)===(gb<0))continue;
        const den=ga-gb;if(Math.abs(den)<1e-9)continue;const t=ga/den;
        const ax=a&1,ay=(a>>1)&1,az=(a>>2)&1,bx=b&1,by=(b>>1)&1,bz=(b>>2)&1;
        sx+=ax+(bx-ax)*t;sy+=ay+(by-ay)*t;sz+=az+(bz-az)*t;ne++;}
      if(!ne)continue;
      cv[i+j*CX+k*CXY]=VX.length;
      VX.push(mnx+(i+sx/ne)*h);VY.push(mny+(j+sy/ne)*h);VZ.push(mnz+(k+sz/ne)*h);}
    const TI=[],TJ=[],TK=[];
    // Wind each quad so its normal already points outward (F rises outward): when the edge runs
    // inside->outside, list the 4 cells counter-clockwise seen from the outward end; else reverse.
    const quad=(fwd,a,b,c,d)=>{if(a<0||b<0||c<0||d<0)return;
      if(fwd){TI.push(a,a);TJ.push(b,c);TK.push(c,d);}else{TI.push(a,a);TJ.push(d,c);TK.push(c,b);}};
    for(let k=1;k<CZ;k++)for(let j=1;j<CY;j++)for(let i=0;i<CX;i++){          // grid edges along x
      const p=i+j*NX+k*NXY,in0=F[p]<0;if(in0===(F[p+1]<0))continue;           // CCW seen from +x: (y,z)
      quad(in0,cv[i+(j-1)*CX+(k-1)*CXY],cv[i+j*CX+(k-1)*CXY],cv[i+j*CX+k*CXY],cv[i+(j-1)*CX+k*CXY]);}
    for(let k=1;k<CZ;k++)for(let j=0;j<CY;j++)for(let i=1;i<CX;i++){          // grid edges along y
      const p=i+j*NX+k*NXY,in0=F[p]<0;if(in0===(F[p+NX]<0))continue;          // CCW seen from +y: (z,x)
      quad(in0,cv[(i-1)+j*CX+(k-1)*CXY],cv[(i-1)+j*CX+k*CXY],cv[i+j*CX+k*CXY],cv[i+j*CX+(k-1)*CXY]);}
    for(let k=0;k<CZ;k++)for(let j=1;j<CY;j++)for(let i=1;i<CX;i++){          // grid edges along z
      const p=i+j*NX+k*NXY,in0=F[p]<0;if(in0===(F[p+NXY]<0))continue;         // CCW seen from +z: (x,y)
      quad(in0,cv[(i-1)+(j-1)*CX+k*CXY],cv[i+(j-1)*CX+k*CXY],cv[i+j*CX+k*CXY],cv[(i-1)+j*CX+k*CXY]);}
    if(!VX.length||!TI.length)return null;
    /* 6.5) keep only the largest connected component (drops stray surface islands — the little
       free-floating blobs from micro-cavities the flood-fill can't reach) and Laplacian-smooth the
       vertex positions a couple of passes so the surface reads as smooth as a rendered SES.       */
    {
      const NV0=VX.length;
      const par=new Int32Array(NV0);for(let v=0;v<NV0;v++)par[v]=v;
      const find=x=>{while(par[x]!==x){par[x]=par[par[x]];x=par[x];}return x;};
      const uni=(a,b)=>{a=find(a);b=find(b);if(a!==b)par[a]=b;};
      for(let t=0;t<TI.length;t++){uni(TI[t],TJ[t]);uni(TJ[t],TK[t]);}
      const cnt={};for(let v=0;v<NV0;v++){const r=find(v);cnt[r]=(cnt[r]||0)+1;}
      let big=-1,bn=-1;for(const r in cnt)if(cnt[r]>bn){bn=cnt[r];big=+r;}
      if(bn<NV0){                                                                   // there ARE islands → rebuild keeping the big one
        const remap=new Int32Array(NV0).fill(-1);const nx=[],ny=[],nz=[];
        for(let v=0;v<NV0;v++)if(find(v)===big){remap[v]=nx.length;nx.push(VX[v]);ny.push(VY[v]);nz.push(VZ[v]);}
        const ni=[],nj=[],nk=[];
        for(let t=0;t<TI.length;t++){const a=remap[TI[t]],b=remap[TJ[t]],c=remap[TK[t]];if(a>=0&&b>=0&&c>=0){ni.push(a);nj.push(b);nk.push(c);}}
        VX.length=0;VY.length=0;VZ.length=0;TI.length=0;TJ.length=0;TK.length=0;
        for(let q=0;q<nx.length;q++){VX.push(nx[q]);VY.push(ny[q]);VZ.push(nz[q]);}
        for(let q=0;q<ni.length;q++){TI.push(ni[q]);TJ.push(nj[q]);TK.push(nk[q]);}
      }
      const NV=VX.length;                                                           // Taubin smoothing: λ shrink + μ expand → smooth WITHOUT deflating the surface
      const LAM=0.5,MU=-0.53;
      const smoothPass=(f)=>{const sx=new Float64Array(NV),sy=new Float64Array(NV),sz=new Float64Array(NV),dg=new Uint16Array(NV);
        for(let t=0;t<TI.length;t++){const a=TI[t],b=TJ[t],c=TK[t];
          sx[a]+=VX[b]+VX[c];sy[a]+=VY[b]+VY[c];sz[a]+=VZ[b]+VZ[c];dg[a]+=2;
          sx[b]+=VX[a]+VX[c];sy[b]+=VY[a]+VY[c];sz[b]+=VZ[a]+VZ[c];dg[b]+=2;
          sx[c]+=VX[a]+VX[b];sy[c]+=VY[a]+VY[b];sz[c]+=VZ[a]+VZ[b];dg[c]+=2;}
        for(let v=0;v<NV;v++)if(dg[v]){VX[v]+=f*(sx[v]/dg[v]-VX[v]);VY[v]+=f*(sy[v]/dg[v]-VY[v]);VZ[v]+=f*(sz[v]/dg[v]-VZ[v]);}};
      for(let it=0;it<2;it++){smoothPass(LAM);smoothPass(MU);}
    }
    /* 7) AMBIENT OCCLUSION. Plotly casts no shadows, so a pit is lit exactly like a bump and the
       pocket reads flat. We already have the field, so march a hemisphere of rays off each vertex
       and see how many are blocked by the protein: enclosed vertices go dark, exposed ones stay
       bright. Baked into vertex colours, so the depth cue holds from every camera angle.        */
    const ND=32,DIRS=new Float64Array(ND*3);
    for(let q=0;q<ND;q++){const y=(q+0.5)/ND,r=Math.sqrt(Math.max(0,1-y*y)),ph=q*2.399963229728653;
      DIRS[q*3]=r*Math.cos(ph);DIRS[q*3+1]=r*Math.sin(ph);DIRS[q*3+2]=y;}          // hemisphere, +z = normal
    const RMAX=11.0,STEP=0.9,NS=12,invh=1/h;                                        // reach ~11 Å so AO actually senses pocket enclosure
    const gAt=(i,j,k)=>F[Math.max(0,Math.min(NX-1,i))+Math.max(0,Math.min(NY-1,j))*NX+Math.max(0,Math.min(NZ-1,k))*NXY];
    const AO=new Float32Array(VX.length);
    const WT=new Float64Array(NS+1);for(let sIdx=1;sIdx<=NS;sIdx++)WT[sIdx]=1-0.72*(sIdx*STEP/RMAX);
    for(let q=0;q<VX.length;q++){
      const px=VX[q],py=VY[q],pz=VZ[q];
      const bi=(px-mnx)*invh+0.5|0,bj=(py-mny)*invh+0.5|0,bk=(pz-mnz)*invh+0.5|0;
      let nx=gAt(bi+1,bj,bk)-gAt(bi-1,bj,bk),ny=gAt(bi,bj+1,bk)-gAt(bi,bj-1,bk),nz=gAt(bi,bj,bk+1)-gAt(bi,bj,bk-1);
      let L=Math.sqrt(nx*nx+ny*ny+nz*nz);
      if(L<1e-9){AO[q]=1;continue;}
      nx/=L;ny/=L;nz/=L;                                                            // outward normal (F rises outward)
      let tx,ty,tz;
      if(Math.abs(nx)<0.9){tx=0;ty=-nz;tz=ny;}else{tx=-nz;ty=0;tz=nx;}
      L=Math.sqrt(tx*tx+ty*ty+tz*tz)||1;tx/=L;ty/=L;tz/=L;
      const bx=ny*tz-nz*ty,by=nz*tx-nx*tz,bz=nx*ty-ny*tx;                            // tangent frame
      const ox=px+nx*0.45-mnx,oy=py+ny*0.45-mny,oz=pz+nz*0.45-mnz;                   // ray origin, already grid-relative
      let occ=0;
      for(let dq=0;dq<ND;dq++){const q3=dq*3,d0=DIRS[q3],d1=DIRS[q3+1],d2=DIRS[q3+2];
        const dx=(tx*d0+bx*d1+nx*d2)*invh,dy=(ty*d0+by*d1+ny*d2)*invh,dz=(tz*d0+bz*d1+nz*d2)*invh;
        const sx=ox*invh+0.5,sy=oy*invh+0.5,sz=oz*invh+0.5;                          // march directly in voxel units
        for(let sIdx=1;sIdx<=NS;sIdx++){const tt=sIdx*STEP;
          const i=sx+dx*tt|0,j=sy+dy*tt|0,k=sz+dz*tt|0;
          if((i>>>0)>=NX||(j>>>0)>=NY||(k>>>0)>=NZ)break;                            // left the grid = open solvent
          if(F[i+j*NX+k*NXY]<0){occ+=WT[sIdx];break;}                                // nearer blockers darken more
        }
      }
      AO[q]=1-occ/ND;
    }
    const NV=VX.length;                                                             // 2 smoothing passes over the mesh graph
    for(let it=0;it<2;it++){                                                        // (kills ray-sampling mottle, keeps the shape)
      const sum=new Float32Array(NV),cn=new Uint16Array(NV);
      for(let t=0;t<TI.length;t++){const a=TI[t],b=TJ[t],c=TK[t];
        sum[a]+=AO[b]+AO[c];cn[a]+=2;sum[b]+=AO[a]+AO[c];cn[b]+=2;sum[c]+=AO[a]+AO[b];cn[c]+=2;}
      for(let v=0;v<NV;v++)if(cn[v])AO[v]=0.45*AO[v]+0.55*(sum[v]/cn[v]);
    }
    // UNIFY WINDING. Surface-nets can emit triangles with mixed orientation; gl-mesh3d derives its
    // shading normals from the winding, so backwards facets get inward normals and render lit-from-behind
    // (dark) — the cause of 'some views are dark' that no light tweak could fix. Reorder each triangle so
    // its geometric normal agrees with the field gradient (the true outward direction) at its centroid.
    {
      var flipped=0,ih=1/h;                                                          // local invh (the AO block's is out of scope here)
      var gAtF=function(ix,iy,iz){return F[Math.max(0,Math.min(NX-1,ix))+Math.max(0,Math.min(NY-1,iy))*NX+Math.max(0,Math.min(NZ-1,iz))*NXY];};
      for(var t=0;t<TI.length;t++){
        var a=TI[t],b=TJ[t],c=TK[t];
        var ux=VX[b]-VX[a],uy=VY[b]-VY[a],uz=VZ[b]-VZ[a],wx=VX[c]-VX[a],wy=VY[c]-VY[a],wz=VZ[c]-VZ[a];
        var nx=uy*wz-uz*wy,ny=uz*wx-ux*wz,nz=ux*wy-uy*wx;                            // facet normal from current winding
        var mx=(VX[a]+VX[b]+VX[c])/3,my=(VY[a]+VY[b]+VY[c])/3,mz=(VZ[a]+VZ[b]+VZ[c])/3;
        var gi=(mx-mnx)*ih+0.5|0,gj=(my-mny)*ih+0.5|0,gk=(mz-mnz)*ih+0.5|0;         // outward = grad(F) at the centroid (LOCAL, correct for pockets)
        var ox=gAtF(gi+1,gj,gk)-gAtF(gi-1,gj,gk),oy=gAtF(gi,gj+1,gk)-gAtF(gi,gj-1,gk),oz=gAtF(gi,gj,gk+1)-gAtF(gi,gj,gk-1);
        if(nx*ox+ny*oy+nz*oz<0){ TJ[t]=c; TK[t]=b; flipped++; }                      // wound inward -> swap two indices
      }
      if(typeof console!=='undefined'&&console.log)console.log('[sesMesh] winding unified: flipped '+flipped+'/'+TI.length+' facets outward');
    }
    return {x:VX,y:VY,z:VZ,i:TI,j:TJ,k:TK,ao:AO,h:h,nv:NV,nt:TI.length};
  }
  PG._dbg=true;                                                                  // set PoseGen._dbg=true in console for verbose camera/light logs
  PG._log=function(){ if(PG._dbg&&typeof console!=='undefined'){try{console.log.apply(console,['[Pose3D]'].concat([].slice.call(arguments)));}catch(e){}} };

  /* ==== full debug log → config.POSE_DEBUG_DIR (data/pose/debug) ============
     The console truncates long runs, collapses repeats and is awkward to share,
     so every [Pose3D] line is also buffered here with a timestamp and shipped to
     POST /pose/debug_log. Buffering happens even when PoseGen._dbg is false, so
     you can keep the console quiet and still get a complete file.
       PoseGen.dumpLog()        → flush now, prints the path it wrote
       PoseGen._logAuto = false → stop the periodic auto-flush
       PoseGen._logDir = '…'    → write somewhere else
     One file per page load: pose_<session>.log, truncated on the first flush and
     appended to afterwards, so it always holds the whole run.
     ======================================================================== */
  PG._logBuf=[]; PG._logSent=0; PG._logSession=null; PG._logTimer=null;
  PG._logAuto=true; PG._logDir=null; PG._logMax=20000;
  (function(){
    var _raw=PG._log;
    PG._log=function(){
      try{
        var p=[];
        for(var i=0;i<arguments.length;i++){var a=arguments[i];
          p.push(typeof a==='string'?a:(function(){try{return JSON.stringify(a);}catch(e){return String(a);}})());}
        PG._logBuf.push(new Date().toISOString()+'  '+p.join(' '));
        if(PG._logBuf.length>PG._logMax){                       // ring-buffer: drop the oldest, keep _logSent aligned
          var cut=PG._logBuf.length-PG._logMax;
          PG._logBuf.splice(0,cut); PG._logSent=Math.max(0,PG._logSent-cut);}
        if(PG._logAuto&&!PG._logTimer)
          PG._logTimer=setTimeout(function(){PG._logTimer=null;PG.dumpLog(true);},4000);
      }catch(e){}
      return _raw.apply(PG,arguments);
    };
  })();

  /* Flush the un-sent tail to disk. quiet=true for the auto-flush (stays silent
     unless it fails). Uses console.log directly, never PG._log — logging inside
     the flusher would feed the buffer it is draining. */
  PG.dumpLog=function(quiet){
    var say=function(m){ if(typeof console!=='undefined'){try{console.log('[Pose3D] '+m);}catch(e){}} };
    if(PG._logSent>=PG._logBuf.length){ if(!quiet)say('dumpLog: nothing new to write'); return Promise.resolve(); }
    if(!PG._logSession)PG._logSession=new Date().toISOString().replace(/[:.]/g,'-');
    var from=PG._logSent, chunk=PG._logBuf.slice(from);
    var body={text:chunk.join('\n'),session:PG._logSession,reset:(from===0)};
    if(PG._logDir)body.dir=PG._logDir;
    return fetch('/pose/debug_log',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)})
      .then(function(r){return r.json();})
      .then(function(res){
        if(res&&res.ok){ PG._logSent=from+chunk.length;
          if(!quiet)say('dumpLog → '+res.path+'  ('+chunk.length+' new lines, '+res.bytes+' bytes total)'); }
        else say('dumpLog FAILED: '+((res&&res.err)||'unknown')); })
      .catch(function(e){ say('dumpLog FAILED: '+e.message); });
  };
  try{ window.addEventListener('beforeunload',function(){try{PG.dumpLog(true);}catch(e){}}); }catch(e){}   // last flush on navigate away
  // Plotly throws "Resize must be passed a displayed plot div element" when the div is hidden or 0-sized.
  // Only touch the plot when it's actually on-screen, and always swallow the async rejection.
  PG._plotReady=function(){
    var gd=$('poseBox3D');
    if(!gd||!window.Plotly){PG._log('plotReady: no gd/Plotly');return null;}
    if(gd.offsetParent===null){PG._log('plotReady: div not displayed (offsetParent null)');return null;}   // display:none anywhere up the tree
    var r=(gd.getBoundingClientRect?gd.getBoundingClientRect():{width:1,height:1});
    if(!gd._fullLayout){PG._log('plotReady: not plotted yet');return null;}                                  // never restyle/resize before first plot
    if(r.width<2||r.height<2){PG._log('plotReady: zero-size',r.width,r.height);return null;}
    return gd;
  };
  PG._safe=function(p,tag){ if(p&&p.then){p.then(function(){PG._log(tag,'ok');},function(err){PG._log(tag,'rejected:',err&&err.message||err);}); } return p; };
  // Plotly's camera `eye` lives in the NORMALISED scene frame (|eye|~1), but mesh3d `lightposition`
  // is interpreted in the trace's DATA frame (Angstroms). If the scene's axis spans differ, a direction
  // in one frame is SKEWED in the other — which is why a freshly-computed light could point somewhere
  // useless while a stale one looked fine. Read the spans and convert explicitly.
  PG._spans=function(){
    var gd=$('poseBox3D');
    try{var sc=gd&&gd._fullLayout&&gd._fullLayout.scene;
      if(!sc)return null;
      var rx=sc.xaxis&&sc.xaxis.range, ry=sc.yaxis&&sc.yaxis.range, rz=sc.zaxis&&sc.zaxis.range;
      if(!rx||!ry||!rz)return null;
      var sp=[Math.abs(rx[1]-rx[0]),Math.abs(ry[1]-ry[0]),Math.abs(rz[1]-rz[0])];
      if(!(sp[0]>0&&sp[1]>0&&sp[2]>0))return null;
      return sp;
    }catch(e){return null;}
  };
  PG._n2d=function(v){                      // normalised-frame direction -> data-frame direction
    var sp=PG._spans();
    if(!sp)return {x:v.x,y:v.y,z:v.z};
    var d={x:v.x*sp[0], y:v.y*sp[1], z:v.z*sp[2]};
    var m=Math.hypot(d.x,d.y,d.z)||1;
    return {x:d.x/m,y:d.y/m,z:d.z/m};
  };
  // THE decisive diagnostic: what does Plotly ACTUALLY have on the traces? If _surfIdx does not point
  // at the mesh3d, our light updates land on the wrong trace and the surface keeps whatever light it
  // was born with (or Plotly's default (1e5,1e5,0), a fixed world light) -> lit at some angles only.
  PG._pixelStats=function(tag){
    if(!PG._dbg)return;
    try{
      var gd=$('poseBox3D');if(!gd)return;
      // find the WebGL canvas Plotly renders the 3D scene into
      var cvs=gd.querySelectorAll?gd.querySelectorAll('canvas'):[];
      if(!cvs||!cvs.length){PG._log('PIXELS('+tag+'): no canvas');return;}
      var best=null,bestA=0;
      for(var ci=0;ci<cvs.length;ci++){var cc=cvs[ci],ar=(cc.width||0)*(cc.height||0);if(ar>bestA){bestA=ar;best=cc;}}
      if(!best){PG._log('PIXELS('+tag+'): no sized canvas');return;}
      // draw the gl canvas into a 2D canvas so we can read pixels back
      var w=Math.min(200,best.width||200),h=Math.min(150,best.height||150);
      var tmp=document.createElement('canvas');tmp.width=w;tmp.height=h;
      var ctx=tmp.getContext('2d',{willReadFrequently:true});if(!ctx){PG._log('PIXELS('+tag+'): no 2d ctx');return;}
      ctx.drawImage(best,0,0,w,h);
      var data=ctx.getImageData(0,0,w,h).data;
      var n=0,sum=0,lit=0,mid=0,dark=0,bg=0,mn=255,mx=0;
      for(var p=0;p<data.length;p+=4){
        var r=data[p],g=data[p+1],b=data[p+2];
        // skip the near-black page background (very dark bluish) so we measure the SURFACE only
        if(r<12&&g<14&&b<24){bg++;continue;}
        var lum=(0.299*r+0.587*g+0.114*b);
        n++;sum+=lum;if(lum<mn)mn=lum;if(lum>mx)mx=lum;
        if(lum>180)lit++;else if(lum>70)mid++;else dark++;
      }
      if(!n){PG._log('PIXELS('+tag+'): all background');return;}
      PG._log('PIXELS('+tag+'): meanLum='+(sum/n).toFixed(1)+'/255  min='+mn.toFixed(0)+' max='+mx.toFixed(0)+
              '  | bright(>180)='+(100*lit/n).toFixed(0)+'%  mid='+(100*mid/n).toFixed(0)+'%  dark(<70)='+(100*dark/n).toFixed(0)+'%'+
              '  surfacePx='+n+' bgPx='+bg+'  <<< ACTUAL rendered brightness');
    }catch(e){PG._log('PIXELS('+tag+') threw',e&&e.message);}
  };
  PG._dumpTraces=function(tag){
    if(!PG._dbg)return;
    var gd=$('poseBox3D');
    if(!gd||!gd.data){PG._log('TRACES('+tag+'): gd.data missing');return;}
    var out=[];
    for(var i=0;i<gd.data.length;i++){
      var t=gd.data[i]||{},lp=t.lightposition;
      out.push(i+':'+(t.type||'?')+(lp?('[L='+Math.round(lp.x)+','+Math.round(lp.y)+','+Math.round(lp.z)+']'):''));
    }
    PG._log('TRACES('+tag+') n='+gd.data.length+' :',out.join('  '));
    var st=gd.data[PG._surfIdx];
    PG._log('   _surfIdx='+PG._surfIdx+' -> type='+((st&&st.type)||'MISSING')+
            ((st&&st.type==='mesh3d')?'  OK':'  *** WRONG TRACE: light updates are going nowhere ***'));
    if(st&&st.lightposition){var L=st.lightposition,m=Math.hypot(L.x,L.y,L.z)||1;
      var e=(PG._cam&&PG._cam.eye)||{x:0,y:0,z:1},EL=Math.hypot(e.x,e.y,e.z)||1;
      var dot=(L.x/m)*(e.x/EL)+(L.y/m)*(e.y/EL)+(L.z/m)*(e.z/EL);
      var deg=Math.acos(Math.max(-1,Math.min(1,dot)))*180/Math.PI;
      PG._log('   ACTUAL light on the mesh: dir=('+(L.x/m).toFixed(2)+','+(L.y/m).toFixed(2)+','+(L.z/m).toFixed(2)+
              ')  angle-to-view='+deg.toFixed(1)+'°  '+(deg<60?'lit':'*** LIT FROM BEHIND -> DARK ***'));}
    // Measure the ACTUAL brightness of the surface we can SEE: average diffuse (ambient + diff*N·L)
    // over the mesh facets whose normal faces the camera. If this is ~equal between two views that
    // LOOK different in brightness, the difference is FRAMING/ZOOM (how much black surrounds the
    // surface), not the lighting — the surface itself is lit the same.
    if(st&&st.type==='mesh3d'&&st.x&&st.i&&st.lightposition){
      try{
        var Lp=st.lightposition,lm=Math.hypot(Lp.x,Lp.y,Lp.z)||1,Lx=Lp.x/lm,Ly=Lp.y/lm,Lz=Lp.z/lm;
        var ev=(PG._cam&&PG._cam.eye)||{x:0,y:0,z:1},em=Math.hypot(ev.x,ev.y,ev.z)||1,Vx=ev.x/em,Vy=ev.y/em,Vz=ev.z/em;
        var amb=0.24,dif=0.78,X=st.x,Y=st.y,Z=st.z,I=st.i,J=st.j,Kk=st.k;
        var spanx=1,spany=1,spanz=1,sp=PG._spans();if(sp){spanx=sp[0];spany=sp[1];spanz=sp[2];}
        var sumB=0,sumNL=0,vis=0,step=Math.max(1,(I.length/4000)|0);
        for(var t=0;t<I.length;t+=step){
          var a=I[t],b=J[t],c=Kk[t];
          var ux=X[b]-X[a],uy=Y[b]-Y[a],uz=Z[b]-Z[a],wx=X[c]-X[a],wy=Y[c]-Y[a],wz=Z[c]-Z[a];
          var nx=uy*wz-uz*wy,ny=uz*wx-ux*wz,nz=ux*wy-uy*wx,nn=Math.hypot(nx,ny,nz)||1;nx/=nn;ny/=nn;nz/=nn;
          if(nx*Vx+ny*Vy+nz*Vz<=0)continue;                                  // facet faces away from camera → not visible
          vis++;var nl=nx*Lx+ny*Ly+nz*Lz;if(nl<0)nl=0;sumNL+=nl;sumB+=amb+dif*nl;
        }
        if(vis>0)PG._log('   SURFACE BRIGHTNESS (visible facets): meanN·L='+(sumNL/vis).toFixed(3)+
                         ' meanDiffuse='+(sumB/vis).toFixed(3)+' over '+vis+' facets'+
                         '  <-- compare this between light/dark views: if ~equal, it is FRAMING not lighting');
      }catch(_e){PG._log('   brightness calc threw',_e&&_e.message);}
    }
  };
  PG._lightViaRedraw=true;         // Plotly's restyle updates gd.data but does NOT reliably push a new
                                   // lightposition into the mesh3d shader — the GPU keeps the STALE light
                                   // (world-fixed), which is why the log said a perfect 33° while the render
                                   // was lit at some angles and dark at others. A full react rebuilds the
                                   // trace, so the light genuinely reaches the shader.
  PG._flipWinding=false;   // no longer needed: winding is unified at mesh-build time           // EXPERIMENT: PoseGen._flipWinding=true; PoseGen._redraw3D();
                                   // gl-mesh3d computes normals from the triangle winding. If ours is the
                                   // opposite of what it expects, the shaded normals are inverted and the
                                   // lit side renders DARK — which would explain why every quantity I can
                                   // measure (light angle, diffuse, specular, AO) is identical between a
                                   // bright view and a dark one, yet they render completely differently.
  PG._lightMode='fixed';           // fixed world light: the direction you picked (see _fixedLight)          // 'camera' = pinned to the viewer (PyMOL-style: consistent at EVERY angle)
  PG._lightUp=0.52;                // seed offset ABOVE the view axis   (with _lightRight ⇒ ≈33° off-axis)
  PG._lightRight=-0.40;            // seed offset: negative = LEFT of the viewer
  PG._fixedLight={x:0.21,y:0.36,z:0.91};                // <-- the light direction you chose (image1's reflection)
  PG._lightW=null;                 // the light's world direction (carried along with the camera)
  PG._lightView=null;              // the view direction _lightW was last synced to
  // rotate vector v by the shortest-arc rotation that carries unit a onto unit b (Rodrigues)
  PG._rotAtoB=function(a,b,v){
    var kx=a.y*b.z-a.z*b.y, ky=a.z*b.x-a.x*b.z, kz=a.x*b.y-a.y*b.x;
    var sn=Math.hypot(kx,ky,kz), cs=a.x*b.x+a.y*b.y+a.z*b.z;
    if(sn<1e-9)return (cs>=0)?{x:v.x,y:v.y,z:v.z}:{x:-v.x,y:-v.y,z:-v.z};            // no move, or a rare exact 180°
    kx/=sn;ky/=sn;kz/=sn;
    var cx=ky*v.z-kz*v.y, cy=kz*v.x-kx*v.z, cz2=kx*v.y-ky*v.x;                        // k × v
    var kv=kx*v.x+ky*v.y+kz*v.z;
    return {x:v.x*cs+cx*sn+kx*kv*(1-cs), y:v.y*cs+cy*sn+ky*kv*(1-cs), z:v.z*cs+cz2*sn+kz*kv*(1-cs)};
  };
  PG._seedLight=function(V){                                                          // 33° up-and-left of the viewer
    var ex=V.x,ey=V.y,ez=V.z,s2=ex*ex+ey*ey,px,py,pz;
    if(s2<1e-8){ px=0; py=(ez>=0?1:-1); pz=0; }
    else { var kk=(1-ez)/s2; px=-ex*ey*kk; py=ez+ex*ex*kk; pz=-ey; }
    var PL=Math.hypot(px,py,pz)||1;px/=PL;py/=PL;pz/=PL;
    var rx=ey*pz-ez*py,ry=ez*px-ex*pz,rz=ex*py-ey*px,RL=Math.hypot(rx,ry,rz)||1;rx/=RL;ry/=RL;rz/=RL;
    var A=PG._lightUp,B=PG._lightRight;
    var L={x:ex+A*px+B*rx,y:ey+A*py+B*ry,z:ez+A*pz+B*rz},m=Math.hypot(L.x,L.y,L.z)||1;
    return {x:L.x/m,y:L.y/m,z:L.z/m};
  };
  PG._lightPos=function(){
    var K=1e5,L;
    if(PG._lightMode==='fixed'){ L=PG._fixedLight; }
    else{
      // CAMERA-PINNED KEY LIGHT, by PARALLEL TRANSPORT.
      // A world-fixed light leaves half the views unlit. A light rebuilt from a fixed frame lurches at
      // the pole (any global frame on a sphere must have a singularity). So instead we CARRY the light:
      // each time the camera turns, we rotate the light by exactly the same rotation. That preserves the
      // light-to-view angle EXACTLY (rotations preserve angles) and is smooth everywhere — no pole, no
      // lurch — so the face you are looking at is always lit the same way from every direction.
      var e=(PG._cam&&PG._cam.eye)?PG._cam.eye:{x:1.35,y:1.35,z:1.05};
      var EL=Math.hypot(e.x,e.y,e.z)||1,Vn={x:e.x/EL,y:e.y/EL,z:e.z/EL};    // view dir, NORMALISED frame
      var V=PG._n2d(Vn);                                                     // view dir, DATA frame  <-- the frame the light lives in
      if(!PG._lightW||!PG._lightView){ PG._lightW=PG._seedLight(V); PG._lightView=V; PG._log('light SEEDED'); }
      else { PG._lightW=PG._rotAtoB(PG._lightView,V,PG._lightW); PG._lightView=V; }
      L=PG._lightW;
      if(PG._dbg){
        var sp=PG._spans();
        var dot=L.x*V.x+L.y*V.y+L.z*V.z, deg=Math.acos(Math.max(-1,Math.min(1,dot)))*180/Math.PI;
        PG._log('LIGHT  spans=',sp?sp.map(function(q){return q.toFixed(1);}).join('/'):'n/a',
                '| eye(norm)=',Vn.x.toFixed(3)+','+Vn.y.toFixed(3)+','+Vn.z.toFixed(3),
                '| view(data)=',V.x.toFixed(3)+','+V.y.toFixed(3)+','+V.z.toFixed(3),
                '| light(data)=',L.x.toFixed(3)+','+L.y.toFixed(3)+','+L.z.toFixed(3),
                '| light-to-view=',deg.toFixed(1)+'°',(deg>25&&deg<45)?'OK':'*** BAD ***');
      }
    }
    var m=Math.hypot(L.x,L.y,L.z)||1;
    return {x:L.x/m*K, y:L.y/m*K, z:L.z/m*K};
  };
  PG.setLightMode=function(mode){                                     // console: PoseGen.setLightMode('fixed'|'camera')
    PG._lightMode=(mode==='fixed')?'fixed':'camera';
    PG._lightW=null;PG._lightView=null;                               // re-seed on next use
    PG._log('light mode ->',PG._lightMode); PG._redraw3D();
  };
  PG._surfaceTraces=function(site){
    if(!protein||!site)return [];
    const kkey=box.center.map(v=>(+v).toFixed(2)).join(',')+'|'+box.size+'|'+site.atoms.length+'|mesh';
    if(!protein._surf||protein._surf.key!==kkey){                                // geometry is opacity-independent → cache it once
      let m=null;try{m=sesMesh(site.atoms,PROBE,320000);}catch(e){m=null;}
      protein._surf={key:kkey,mesh:m,dots:m?null:surfaceDots(site.atoms,PROBE,54)};
    }
    const S=protein._surf,op=(PG._surfOpacity==null?0.45:PG._surfOpacity);
    if(S.mesh){const m=S.mesh;
      if(!S.vc&&m.ao){                                                              // bake AO into vertex colours (once)
        const R0=230,G0=233,B0=237,vc=new Array(m.nv);
        for(let q=0;q<m.nv;q++){const sh=0.06+0.94*Math.pow(m.ao[q],2.2);           // deep crevices → near-black, exposed rims → bright (strong depth cue)
          vc[q]='rgb('+(R0*sh|0)+','+(G0*sh|0)+','+(B0*sh|0)+')';}
        S.vc=vc;
      }
      const tr={type:'mesh3d',x:m.x,y:m.y,z:m.z,i:m.i,j:m.j,k:m.k,opacity:op,flatshading:false,
        lighting:{ambient:0.24,diffuse:0.78,specular:1.15,roughness:0.18,fresnel:0.15},  // low ambient (no self-glow) + STRONG diffuse: with an off-axis key light the
                                                                                        // N·L gradient sweeps light→dark across every curve, which is what reads as
                                                                                        // depth. Specular still gives the glossy glints; baked AO darkens the crevices.
        lightposition:PG._lightPos(),hoverinfo:'skip',showlegend:false};
      if(S.vc)tr.vertexcolor=S.vc;else tr.color='#c9d3df';
      if(PG._flipWinding){var _i=tr.i;tr.i=tr.j;tr.j=_i;PG._log('surface: winding FLIPPED (normals inverted)');}
      return [tr];}
    const d=S.dots;let x=d.x,y=d.y,z=d.z;const N=x.length,CAP=14000;             // fallback: the old dot cloud
    if(N>CAP){const st=N/CAP,xx=[],yy=[],zz=[];for(let i=0;i<CAP;i++){const q=Math.floor(i*st);xx.push(x[q]);yy.push(y[q]);zz.push(z[q]);}x=xx;y=yy;z=zz;}
    return [{type:'scatter3d',mode:'markers',x:x,y:y,z:z,marker:{color:'#9fb4c9',size:2.1,opacity:Math.max(0.06,op*0.6),line:{width:0}},hoverinfo:'skip',showlegend:false}];
  };

  /* ---- DSSP-lite: secondary structure from the backbone H-bond pattern ---- */
  function buildResidues(atoms){
    const map={};for(const a of atoms){if(a.het)continue;const k=a.chain+'|'+a.resseq;let r=map[k];
      if(!r){r={chain:a.chain,seq:parseInt(a.resseq,10),resn:a.resn,key:k,N:null,CA:null,C:null,O:null};map[k]=r;}
      if(a.name==='N')r.N=[a.x,a.y,a.z];else if(a.name==='CA')r.CA=[a.x,a.y,a.z];else if(a.name==='C')r.C=[a.x,a.y,a.z];else if(a.name==='O')r.O=[a.x,a.y,a.z];}
    const byChain={};Object.keys(map).forEach(k=>{const r=map[k];(byChain[r.chain]||(byChain[r.chain]=[])).push(r);});
    Object.keys(byChain).forEach(c=>byChain[c].sort((a,b)=>a.seq-b.seq));return byChain;
  }
  function dsspAssign(byChain){
    // Pool ALL residues (H-bonds span the whole structure, so inter-chain sheets are found).
    // Sequence neighbours are looked up by chain|seq so helix turns / bridge patterns stay chain-aware.
    const all=[];Object.keys(byChain).forEach(ch=>byChain[ch].forEach(r=>all.push(r)));
    const n=all.length,ss={},Q=0.084*332;                                                        // Kabsch–Sander electrostatic factor
    const idx={};for(let i=0;i<n;i++)idx[all[i].chain+'|'+all[i].seq]=i;                          // chain|seq → global index
    const nb=(i,d)=>idx[all[i].chain+'|'+(all[i].seq+d)];                                         // same-chain sequence neighbour
    for(let i=0;i<n;i++){const r=all[i];r.H=null;const pi=nb(i,-1);                               // amide H ~1 Å off N, anti to prev C=O
      if(pi!==undefined){const p=all[pi];if(r.N&&p.C&&p.O){const dx=p.C[0]-p.O[0],dy=p.C[1]-p.O[1],dz=p.C[2]-p.O[2],L2=Math.hypot(dx,dy,dz)||1;r.H=[r.N[0]+dx/L2,r.N[1]+dy/L2,r.N[2]+dz/L2];}}}
    const D=(p,q)=>Math.hypot(p[0]-q[0],p[1]-q[1],p[2]-q[2])||1e-3;
    function hb(i,j){if(i==null||j==null||i<0||j<0||i>=n||j>=n)return false;const A=all[i],B=all[j];   // CO(i)···NH(j) H-bond?
      if(A===B||!A.C||!A.O||!B.N||!B.H)return false;
      if(A.CA&&B.CA&&Math.hypot(A.CA[0]-B.CA[0],A.CA[1]-B.CA[1],A.CA[2]-B.CA[2])>9)return false;      // cheap CA reject
      const E=Q*(1/D(A.O,B.N)+1/D(A.C,B.H)-1/D(A.O,B.H)-1/D(A.C,B.N));return E<-0.5;}
    const hel=new Array(n).fill(false),sht=new Array(n).fill(false);
    for(let i=0;i<n;i++)for(const t of [3,4,5]){const j=nb(i,t);if(j!==undefined&&hb(i,j)){for(let k=1;k<t;k++){const m=nb(i,k);if(m!==undefined)hel[m]=true;}}}   // 3/4/5-turns → helix
    const cell=7.0,gk=v=>Math.floor(v/cell),key=(a,b,c)=>a+'_'+b+'_'+c,grid={};                   // β-bridges: only test spatially close pairs
    for(let i=0;i<n;i++){const p=all[i].CA;if(!p)continue;const k=key(gk(p[0]),gk(p[1]),gk(p[2]));(grid[k]||(grid[k]=[])).push(i);}
    for(let i=0;i<n;i++){const p=all[i].CA;if(!p)continue;const cx=gk(p[0]),cy=gk(p[1]),cz=gk(p[2]);
      for(let dx=-1;dx<=1;dx++)for(let dy=-1;dy<=1;dy++)for(let dz=-1;dz<=1;dz++){const cellArr=grid[key(cx+dx,cy+dy,cz+dz)];if(!cellArr)continue;
        for(const j of cellArr){if(j<=i)continue;
          if(all[i].chain===all[j].chain&&Math.abs(all[i].seq-all[j].seq)<3)continue;             // sequence-adjacent = turn, not a bridge
          const im=nb(i,-1),ip=nb(i,1),jm=nb(j,-1),jp=nb(j,1);
          const anti=(hb(i,j)&&hb(j,i))||(hb(im,jp)&&hb(jm,ip));
          const para=(hb(im,j)&&hb(j,ip))||(hb(jm,i)&&hb(i,jp));
          if(anti||para){sht[i]=true;sht[j]=true;}
        }}}
    for(let i=0;i<n;i++)ss[all[i].key]=hel[i]?'H':(sht[i]?'E':'C');                               // helix wins (DSSP priority)
    return ss;
  }
  function catmull(p0,p1,p2,p3,t){const t2=t*t,t3=t2*t;return [0,1,2].map(k=>0.5*(2*p1[k]+(-p0[k]+p2[k])*t+(2*p0[k]-5*p1[k]+4*p2[k]-p3[k])*t2+(-p0[k]+3*p1[k]-3*p2[k]+p3[k])*t3));}
  const SSCOL={H:'#ef4444',E:'#22d3ee',C:'#8ca0b3'};                                             // helix red · sheet cyan · loop grey
  PG._ssTraces=function(){
    if(!protein)return [];
    const byChain=protein._res||(protein._res=buildResidues(protein.atoms));
    const ss=protein._ss||(protein._ss=dsspAssign(byChain));
    const kkey=box.center.map(v=>(+v).toFixed(2)).join(',')+'|'+box.size+'|cartoon';
    if(protein._ssTr&&protein._ssTr.key===kkey)return protein._ssTr.tr;
    const bc=box.center,Rss=Math.max(16,box.size*0.6+10);                                         // render the backbone that frames the pocket
    // ---- little vector helpers ----
    const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]],dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2],
      cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],
      scl=(a,s)=>[a[0]*s,a[1]*s,a[2]*s],mag=a=>Math.hypot(a[0],a[1],a[2]),
      norm=a=>{const L=mag(a)||1;return [a[0]/L,a[1]/L,a[2]/L];},
      dist=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1],a[2]-b[2]);
    // ---- one combined mesh, coloured per-vertex by secondary structure ----
    const G={x:[],y:[],z:[],i:[],j:[],k:[],col:[]};
    const HT=0.26;                                                                                // ribbon half-thickness (the flat "edge")
    const rectProf=(hw,ht)=>[[hw,ht],[0,ht],[-hw,ht],[-hw,0],[-hw,-ht],[0,-ht],[hw,-ht],[hw,0]];  // 8 pts round a flat rectangle
    const circProf=r=>{const p=[];for(let m=0;m<8;m++){const t=m*Math.PI/4;p.push([r*Math.cos(t),r*Math.sin(t)]);}return p;};
    function ring(P,side,up,prof,color){const base=G.x.length;
      for(let m=0;m<8;m++){const a=prof[m][0],c=prof[m][1];
        G.x.push(P[0]+a*side[0]+c*up[0]);G.y.push(P[1]+a*side[1]+c*up[1]);G.z.push(P[2]+a*side[2]+c*up[2]);G.col.push(color);}
      return base;}
    function stitch(a0,b0){for(let m=0;m<8;m++){const n=(m+1)%8;G.i.push(a0+m,a0+m);G.j.push(a0+n,b0+n);G.k.push(b0+n,b0+m);}}
    function cap(base,ctr,color,rev){const ci=G.x.length;G.x.push(ctr[0]);G.y.push(ctr[1]);G.z.push(ctr[2]);G.col.push(color);
      for(let m=0;m<8;m++){const n=(m+1)%8;if(rev){G.i.push(ci);G.j.push(base+n);G.k.push(base+m);}else{G.i.push(ci);G.j.push(base+m);G.k.push(base+n);}}}

    let total=0;Object.keys(byChain).forEach(ch=>byChain[ch].forEach(r=>{if(r.CA)total++;}));
    const S=total>1200?2:(total>600?3:(total>300?4:6));                                           // spline samples per residue span (scale with size)

    Object.keys(byChain).forEach(chain=>{
      const R=byChain[chain].filter(r=>r.CA);const n=R.length;if(n<2)return;
      // per-residue frame: Carson–Bugg peptide-plane side vector (flips consistent to avoid twisting)
      const fr=[];let prevSide=null;
      for(let i=0;i<n;i++){
        const ca=R[i].CA,caP=(R[i-1]&&R[i-1].CA)||ca,caN=(R[i+1]&&R[i+1].CA)||ca;
        const tan=norm(sub(caN,caP));
        let side;const co=(R[i].O&&R[i].C)?norm(sub(R[i].O,R[i].C)):null;                          // carbonyl C=O gives the ribbon "width" direction
        side=co?sub(co,scl(tan,dot(co,tan))):[0,0,0];
        if(mag(side)<0.1){const t=Math.abs(tan[0])<0.9?[1,0,0]:[0,1,0];side=cross(tan,t);}
        side=norm(side);if(prevSide&&dot(side,prevSide)<0)side=scl(side,-1);
        let up=norm(cross(tan,side));side=norm(cross(up,tan));
        fr.push({ca,tan,side,up,ss:(ss[R[i].key]||'C'),seq:R[i].seq});prevSide=side;
      }
      // one light smoothing pass over side vectors (helps helices read cleanly)
      const sm=fr.map((f,i)=>{const a=fr[Math.max(0,i-1)],c=fr[Math.min(n-1,i+1)];
        let side=norm([a.side[0]+2*f.side[0]+c.side[0],a.side[1]+2*f.side[1]+c.side[1],a.side[2]+2*f.side[2]+c.side[2]]);
        let up=norm(cross(f.tan,side));side=norm(cross(up,f.tan));return {ca:f.ca,tan:f.tan,side,up,ss:f.ss,seq:f.seq};});
      for(let i=0;i<n;i++)fr[i]=sm[i];
      const bonded=(a,b)=>b>=0&&b<n&&Math.abs(fr[b].seq-fr[a].seq)===1&&dist(fr[b].ca,fr[a].ca)<4.6;
      const bondedE=(a,b)=>bonded(a,b)&&fr[a].ss==='E'&&fr[b].ss==='E';
      // per-residue width + cross-section type + colour (β-strands get an arrowhead at their C-terminal end)
      const W=new Array(n),T=new Array(n),COL=new Array(n);
      for(let i=0;i<n;i++){const s=fr[i].ss;
        if(s==='H'){W[i]=1.15;T[i]='rect';COL[i]=SSCOL.H;}
        else if(s==='E'){T[i]='rect';COL[i]=SSCOL.E;
          if(!bondedE(i,i+1))W[i]=0.05;                        // last strand residue = arrow tip
          else if(bondedE(i,i+1)&&!bondedE(i+1,i+2))W[i]=1.95;  // next is the tip → this is the barb (widest)
          else W[i]=1.25;}                                      // strand body
        else {W[i]=0.34;T[i]='circ';COL[i]=SSCOL.C;}}           // loop = thin round tube
      // walk bonded stretches; render those that touch the pocket
      let a=0;
      while(a<n){let b=a;while(bonded(b,b+1))b++;
        let near=false;for(let q=a;q<=b;q++){if(dist(fr[q].ca,bc)<Rss){near=true;break;}}
        if(near&&b>a){
          const CA=q=>fr[Math.max(a,Math.min(b,q))].ca;
          let prevBase=null,firstBase=null,firstP=null,firstCol=null,lastP=null,lastCol=null;
          for(let seg=a;seg<b;seg++){
            const P0=CA(seg-1),P1=CA(seg),P2=CA(seg+1),P3=CA(seg+2),sd1=fr[seg].side,sd2=fr[seg+1].side;
            for(let t=(seg===a?0:1);t<=S;t++){
              const tt=t/S,ri=(t<S)?seg:seg+1;
              const P=catmull(P0,P1,P2,P3,tt);
              let tan=sub(catmull(P0,P1,P2,P3,Math.min(1,tt+0.02)),catmull(P0,P1,P2,P3,Math.max(0,tt-0.02)));
              if(mag(tan)<1e-6)tan=fr[ri].tan;tan=norm(tan);
              let side=norm([sd1[0]*(1-tt)+sd2[0]*tt,sd1[1]*(1-tt)+sd2[1]*tt,sd1[2]*(1-tt)+sd2[2]*tt]);
              let up=norm(cross(tan,side));side=norm(cross(up,tan));
              const w=W[seg]*(1-tt)+W[seg+1]*tt,col=COL[ri],prof=(T[ri]==='circ')?circProf(w):rectProf(w,HT);
              const base=ring(P,side,up,prof,col);
              if(firstBase===null){firstBase=base;firstP=P;firstCol=col;}
              if(prevBase!==null)stitch(prevBase,base);
              prevBase=base;lastP=P;lastCol=col;
            }
          }
          if(firstBase!==null){cap(firstBase,firstP,firstCol,true);cap(prevBase,lastP,lastCol,false);}
        }
        a=b+1;
      }
    });
    const tr=G.x.length?[{type:'mesh3d',x:G.x,y:G.y,z:G.z,i:G.i,j:G.j,k:G.k,vertexcolor:G.col,flatshading:false,
      lighting:{ambient:0.55,diffuse:0.8,specular:0.5,roughness:0.35,fresnel:0.15},lightposition:{x:0,y:0,z:1e5},
      hoverinfo:'skip',showlegend:false}]:[];
    protein._ssTr={key:kkey,tr:tr};return tr;
  };
  PG._ssComposition=function(){                                                                  // helix/sheet/loop split for the POCKET residues
    const out={H:0,E:0,C:0};if(!protein)return out;
    const byChain=protein._res||(protein._res=buildResidues(protein.atoms));
    const ss=protein._ss||(protein._ss=dsspAssign(byChain));
    const site=PG.init.site&&PG.init.site();if(!site)return out;
    const seen={};site.atoms.forEach(a=>{if(a.het)return;const k=a.chain+'|'+a.resseq;if(seen[k])return;seen[k]=1;out[ss[k]||'C']++;});
    return out;
  };

  /* ---- view-layer state + toggles (buttons live in the #poseBox3DHint bar) ---- */
  PG._showSurface=false;PG._showSS=false;PG._hideProtein=false;PG._hideH=false;PG._viewMenuOpen=false;PG._cam=null;PG._styling=false;PG._surfOpacity=0.45;PG._surfIdx=-1;
  /* ---- camera helpers: keep the user's rotation across every redraw ----
     Plotly mutates its internal camera object, so we only ever keep deep COPIES. _camGet reads the
     freshest camera (the live gl scene if it's there, else the layout), _camBind tracks it on every
     user interaction, and _camRestore puts it back ONLY if an operation actually moved it.        */
  PG._camClone=function(c){if(!c||!c.eye)return null;var o={eye:{x:c.eye.x,y:c.eye.y,z:c.eye.z}};
    if(c.center)o.center={x:c.center.x,y:c.center.y,z:c.center.z};
    if(c.up)o.up={x:c.up.x,y:c.up.y,z:c.up.z};
    if(c.projection&&c.projection.type)o.projection={type:c.projection.type};
    return o;};
  PG._camGet=function(){var gd=$('poseBox3D');if(!gd){PG._log('camGet: no gd');return null;}
    try{var sc=gd._fullLayout&&gd._fullLayout.scene;if(!sc){PG._log('camGet: no scene');return null;}
      if(sc._scene&&typeof sc._scene.getCamera==='function'){var lc=PG._camClone(sc._scene.getCamera());
        if(lc){PG._log('camGet <- _scene.getCamera()',JSON.stringify(lc.eye));return lc;}
        PG._log('camGet: getCamera() returned unusable shape');}
      var cc=PG._camClone(sc.camera);
      PG._log('camGet <- fullLayout.scene.camera',cc?JSON.stringify(cc.eye):'null');
      return cc;
    }catch(e){PG._log('camGet threw',e&&e.message);return null;}};
  PG._camBind=function(){var gd=$('poseBox3D');
    if(!gd){PG._log('camBind: NO DIV');return;}
    if(gd._pgCamBound){return;}
    if(typeof gd.on!=='function'){PG._log('camBind: gd.on NOT AVAILABLE YET (will retry next draw)');return;}
    try{gd.on('plotly_relayout',function(e){                                                        // fires whenever the user orbits/zooms
      PG._log('relayout event keys=',Object.keys(e||{}).join(','),'| _styling=',PG._styling);
      if(PG._styling){PG._log('  -> ignored (our own restyle echo)');return;}
      var cam=e&&e['scene.camera'];                                                                 // ONLY a genuine camera change is authoritative.
      if(!cam){PG._log('  -> no scene.camera in event; _cam untouched');return;}
      var c=PG._camClone(cam);if(!c){PG._log('  -> scene.camera unusable');return;}
      PG._log('  -> _cam UPDATED from event:',JSON.stringify(c.eye));
      PG._cam=c;
      if(PG._lightMode==='camera'&&PG._showSurface&&PG._surfIdx>=0&&window.Plotly){
        clearTimeout(PG._lightT);PG._lightT=setTimeout(function(){                    // debounced: fires once the drag settles
          if(!PG._plotReady()){PG._log('light update skipped: plot not ready');return;}
          if(PG._lightViaRedraw){                                                       // <-- the reliable path
            PG._styling=true;
            PG._log('light -> FULL REDRAW (restyle does not reach the mesh3d shader)');
            try{PG._redraw3D();}catch(_e){PG._log('light redraw threw',_e&&_e.message);}
            setTimeout(function(){PG._styling=false;PG._dumpTraces('after-light-redraw');},120);
            return;
          }
          var gdx=$('poseBox3D');
          var tgt=gdx&&gdx.data&&gdx.data[PG._surfIdx];
          if(!tgt||tgt.type!=='mesh3d'){                                                // the index drifted -> find the mesh
            var found=-1;
            if(gdx&&gdx.data)for(var q=0;q<gdx.data.length;q++)if(gdx.data[q]&&gdx.data[q].type==='mesh3d'&&gdx.data[q].i){found=q;break;}
            PG._log('light: _surfIdx',PG._surfIdx,'was not a mesh3d ->',found>=0?('using trace '+found):'NO MESH FOUND');
            if(found<0){PG._dumpTraces('no-mesh');return;}
            PG._surfIdx=found;
          }
          var lp=PG._lightPos(),camNow=PG._camClone(PG._cam);
          PG._styling=true;PG._log('light follows view; pinning camera',camNow?JSON.stringify(camNow.eye):'null');
          var fin=function(){try{PG._camRestore();}catch(_e){}
            PG._dumpTraces('after-light');                                              // read back what Plotly really has
            setTimeout(function(){PG._styling=false;},60);};                            // hold the guard past Plotly's async echo
          try{
            var pr=Plotly.update('poseBox3D',                                            // light + camera set ATOMICALLY, so the
              {lightposition:[{x:lp.x,y:lp.y,z:lp.z}]},                                  // FULL object (dotted paths may not apply
              camNow?{'scene.camera':camNow}:{},                                         // to gl traces); layout keeps the camera
              [PG._surfIdx]);
            if(pr&&pr.then)PG._safe(pr.then(fin,fin),'light');else fin();
          }catch(_e){PG._log('light update threw',_e&&_e.message);PG._styling=false;}
        },140);
      } else {
        PG._log('rotate: light fixed, no restyle');
        clearTimeout(PG._dumpT);                                                     // read-only diagnostic, debounced
        PG._dumpT=setTimeout(function(){PG._dumpTraces('rotate');PG._pixelStats('rotate');},200);   // read-only; pixelStats = ACTUAL screen brightness
      }
    });
      gd._pgCamBound=true;PG._log('camBind: LISTENER BOUND OK');}catch(e){PG._log('camBind threw',e&&e.message);}};
  PG._camRestore=function(){
    if(!PG._cam){PG._log('camRestore: NO SAVED CAM (nothing to restore)');return;}
    if(!window.Plotly)return;
    var c=PG._camGet();if(!c){PG._log('camRestore: cannot read live cam');return;}
    var a=PG._cam.eye,b=c.eye;
    PG._log('camRestore: saved=',JSON.stringify(a),' live=',JSON.stringify(b));
    if(Math.abs(a.x-b.x)<1e-6&&Math.abs(a.y-b.y)<1e-6&&Math.abs(a.z-b.z)<1e-6){PG._log('  -> identical, no action');return;}   // nothing moved it → don't touch
    PG._log('  -> DIFFERENT: forcing camera back to saved');
    if(!PG._plotReady()){PG._log('camRestore skipped: plot not ready');return;}
    PG._log('camRestore ->',PG._cam.eye);
    try{PG._safe(Plotly.relayout('poseBox3D',{'scene.camera':PG._camClone(PG._cam)}),'camRestore');}catch(e){PG._log('camRestore threw',e&&e.message);}};
  PG.toggleSurface=function(){PG._showSurface=!PG._showSurface;PG._syncViewBtns();PG._redraw3D();};
  PG.toggleSS=function(){PG._showSS=!PG._showSS;PG._syncViewBtns();PG._ssReadout();PG._redraw3D();};
  PG.toggleProtein=function(){PG._hideProtein=!PG._hideProtein;PG._syncViewBtns();PG._redraw3D();};   // show / hide the pocket atom sticks
  PG.toggleHydrogens=function(){PG._hideH=!PG._hideH;PG._syncViewBtns();PG._redraw3D();};             // show / hide H atoms (ligand, pinned ligands, pocket)
  /* Pocket sticks are precomputed once per site (site.traces). Build the H-free
     variant lazily and cache it on the same object, so toggling is a pointer
     swap rather than a re-stick of up to 1400 atoms. The cache dies with the
     site object, which PG.init.site() rebuilds whenever the box moves. */
  PG._siteTraces=function(site){
    if(!PG._hideH)return site.traces;
    if(!site._tracesNoH){
      const f=stripH(site.drawAtoms||site.atoms,site.bonds);
      site._tracesNoH=stickTraces(f.atoms,f.bonds,false,5,2);
    }
    return site._tracesNoH;
  };
  PG.toggleViewMenu=function(){PG._viewMenuOpen=!PG._viewMenuOpen;var m=$('poseViewMenu');if(m)m.style.display=PG._viewMenuOpen?'block':'none';
    var t=$('poseViewToggle');if(t){t.style.borderColor=PG._viewMenuOpen?'#0e7490':'#1e293b';t.style.color=PG._viewMenuOpen?'#67e8f9':'#94a3b8';}
    if(PG._viewMenuOpen)PG._syncViewBtns();};
  PG._syncViewBtns=function(){
    const ON={bd:'#0e7490',bg:'rgba(34,211,238,.16)',fg:'#67e8f9'},OFF={bd:'#1e293b',bg:'rgba(11,17,32,.7)',fg:'#94a3b8'};
    const set=(id,active)=>{const e=$(id);if(e){const a=active?ON:OFF;e.style.borderColor=a.bd;e.style.background=a.bg;e.style.color=a.fg;}};
    set('poseSurfBtn',PG._showSurface);set('poseSSBtn',PG._showSS);set('poseProtBtn',!PG._hideProtein);   // protein item is lit while atoms are shown
    set('poseHBtn',!PG._hideH);                                                                          // H item is lit while hydrogens are shown
    PG._ensureSurfUI();
    const sn=$('poseSurfNote');if(sn)sn.style.display=PG._showSurface?'block':'none';
    const lg=$('poseSSLegend');if(lg)lg.style.display=PG._showSS?'block':'none';
  };
  PG._ssReadout=function(){if(!PG._showSS)return;try{const c=PG._ssComposition(),set=(id,v)=>{const e=$(id);if(e)e.textContent=v;};set('poseSSa',c.H);set('poseSSb',c.E);set('poseSSl',c.C);}catch(e){}};
  // replay the last 3D draw (surface/SS traces get folded in by _draw3D when their flags are on)
  PG._redraw3D=function(){if(!protein||!window.Plotly||!PG._last3D)return;try{PG._draw3D(PG._last3D.world,PG._last3D.opt);}catch(e){}};



  /* ---- fullscreen toggle for the 3D viewer pane (button sits at its top-right) ---- */
  PG.toggleFullscreen=function(){
    var b3=$('poseBox3D'),el=b3&&b3.parentElement;if(!el)return;                                 // the pane = the container holding the 3D box + overlays
    var d=document,fs=d.fullscreenElement||d.webkitFullscreenElement||d.mozFullScreenElement||d.msFullscreenElement;
    try{
      if(!fs){var rq=el.requestFullscreen||el.webkitRequestFullscreen||el.mozRequestFullScreen||el.msRequestFullscreen;if(rq)rq.call(el);}
      else{var ex=d.exitFullscreen||d.webkitExitFullscreen||d.mozCancelFullScreen||d.msExitFullscreen;if(ex)ex.call(d);}
    }catch(e){}
  };
  PG._onFsChange=function(){
    var d=document,fs=d.fullscreenElement||d.webkitFullscreenElement||d.mozFullScreenElement||d.msFullscreenElement;
    var b3=$('poseBox3D'),el=b3&&b3.parentElement,on=!!(fs&&el&&fs===el);
    var b=$('poseFsBtn');if(b){b.innerHTML=on?'\u2715':'\u26F6';b.title=on?'Exit fullscreen':'Fullscreen';   // \u26F6 ⛶ enter · \u2715 ✕ exit
      b.style.color=on?'#67e8f9':'#94a3b8';b.style.borderColor=on?'#0e7490':'#1e293b';}
    if(el)el.style.background=on?'#05070f':'';                                                    // dark backdrop while fullscreen
    setTimeout(function(){PG._resizeKeepCam();},60);                                              // let fullscreen sizing settle, then resize
    setTimeout(function(){PG._resizeKeepCam();},280);
  };
  // resize the plot WITHOUT losing the camera: capture live eye, resize, then write it back.
  PG._resizeKeepCam=function(){
    var gd=PG._plotReady();if(!gd){PG._log('resize skipped: plot not ready');return;}                // guard: avoids "Resize must be passed a displayed plot div element"
    var c=PG._camGet();if(c)PG._cam=c;                                                              // copy, not a live reference
    PG._log('resizeKeepCam');
    try{var pr=Plotly.Plots.resize(gd);                                                             // pass the ELEMENT, not the id string
      if(pr&&pr.then)PG._safe(pr.then(PG._camRestore),'resize');else PG._camRestore();
    }catch(e){PG._log('resize threw',e&&e.message);}
  };
  ['fullscreenchange','webkitfullscreenchange','mozfullscreenchange','MSFullscreenChange'].forEach(function(ev){try{document.addEventListener(ev,function(){PG._onFsChange();});}catch(e){}});

  /* ========================================================================
     Reaction builder — "⌬ React" button, lower-right of the 3D pane.
     Popover: ligand center xyz · reaction SMARTS · building-block CSV · cap.
     Confirm → POST /pose/react, which reacts the current #poseSmiles with
     every building block in the CSV (up to the cap) via the SMARTS. Each
     product SMILES is loaded into #poseSmiles and built exactly as if ⌬ Build
     were clicked. ◀ / ▶ step through them; ▶ Play walks 0 → cap on its own.

     LIGAND CENTER: when x/y/z are filled they PIN the ligand's start center —
     PG.init.rand() is wrapped so every start state produced from then on sits
     at that center instead of a random xyz (orientation + torsions stay
     random, as in conf::randomize()). Because every stage (Random Start, New
     random pose, Min, MC, Vina) derives its pose from PG.init.rand(), the pin
     is consistent across all of them. Clear the three fields and Confirm again
     to go back to a random center.
     ======================================================================== */
  PG._rx = { products: [], idx: 0, center: null, playing: false, timer: null, dwell: 700 };
  PG._rx.poses = {};          // idx → remembered pose, so ◀/▶ return to the SAME pose instead of a fresh random one
  PG._rx.built = -1;          // product idx currently built in the box (-1 = none, or a foreign ligand)

  /* world xyz → the normalized `off` every stage actually stores.
     Exact inverse of PG.init.ligCenter():  world = box.center + off·box.size·0.32 */
  PG._worldToOff = function (w) {
    var s = ((box.size || 20) * 0.32) || 1;
    return [0, 1, 2].map(function (k) { return (w[k] - box.center[k]) / s; });
  };

  /* --------------------------------------------------------------------------
     TRUE ligand center vs. the translation anchor.

     PG.init.ligCenter(st) is NOT the center of the molecule — it is the pose's
     translation anchor. render3D draws:

         world = PE.computePose(L, quat, [0,0,0], tors) + ligCenter(off)

     and computePose rotates about the centroid of the ROOT fragment, leaving
     REF's own origin offset in place. So the atoms sit at `anchor + mean(base)`,
     where mean(base) swings around with every random orientation / torsion set.
     Pinning `off` alone therefore pins the anchor and lets the molecule itself
     drift — which is exactly the "center xyz doesn't match the ligand" bug.
     -------------------------------------------------------------------------- */
  PG._baseCentroid = function (quat, tors) {          // mean of computePose(...) at zero translation = anchor→centroid offset
    var base = PE.computePose(L, quat, [0, 0, 0], tors);
    if (!base || !base.length) return [0, 0, 0];
    var s = [0, 0, 0];
    base.forEach(function (p) { s[0] += p[0]; s[1] += p[1]; s[2] += p[2]; });
    return [s[0] / base.length, s[1] / base.length, s[2] / base.length];
  };

  PG._poseCentroid = function (st) {                  // world xyz of the atoms as actually drawn, or null
    if (!L || !st || !st.off || !st.quat) return null;
    try {
      var bc = PG._baseCentroid(st.quat, st.tors);
      var lc = PG.init.ligCenter(st);
      return [lc[0] + bc[0], lc[1] + bc[1], lc[2] + bc[2]];
    } catch (e) { return null; }
  };

  PG._rxClonePose = function (st) {
    return st ? { seed: st.seed, off: st.off.slice(), quat: st.quat.slice(),
                  tors: (st.tors || []).slice(), pinned: !!st.pinned } : null;
  };

  /* Pin the start center by wrapping the one random-pose generator.
     Solve for the anchor that puts the ATOMS' centroid on the pin:
         anchor = pin − mean(base)   ⇒   centroid = anchor + mean(base) = pin  (exact). */
  (function () {
    var _rand = PG.init.rand.bind(PG.init);
    PG.init.rand = function () {
      var st = _rand();
      if (PG._rx.center) {
        try {
          var bc = PG._baseCentroid(st.quat, st.tors);
          st.off = PG._worldToOff([PG._rx.center[0] - bc[0], PG._rx.center[1] - bc[1], PG._rx.center[2] - bc[2]]);
        } catch (e) {
          st.off = PG._worldToOff(PG._rx.center);     // fallback: legacy anchor-only pin
        }
        st.pinned = true;
      }
      return st;
    };
  })();

  PG._reactInject = function () {
    var b3 = $('poseBox3D'); var pane = b3 && b3.parentElement;
    if (!pane || document.getElementById('poseReactBtn')) return;          // mount once
    try { var _pp = document.getElementById('poseProcPanel'); if (_pp) _pp.style.pointerEvents = 'auto'; } catch (e) {}   // let right-click / text-selection land on the score panels, not the canvas behind them
    var num  = 'width:52px;font-family:ui-monospace,monospace;font-size:10px;background:#0b1120;border:1px solid #1e293b;border-radius:6px;padding:2px 6px;color:#22d3ee;outline:none;text-align:right;';
    var txt  = 'width:100%;box-sizing:border-box;font-family:ui-monospace,monospace;font-size:10px;background:#0b1120;border:1px solid #1e293b;border-radius:7px;padding:6px 8px;color:#cbd5e1;outline:none;';
    var lbl  = 'font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;color:#64748b;';
    var step = 'padding:2px 9px;border-radius:6px;border:1px solid #1e293b;background:rgba(11,17,32,.7);color:#94a3b8;font-size:12px;cursor:pointer;line-height:1;';
    var html =
      '<button id="poseReactBtn" onclick="PoseGen.reactToggle()" title="React this ligand with building blocks (SMARTS + CSV)" ' +
        'style="position:absolute;bottom:8px;right:10px;z-index:6;pointer-events:auto;padding:5px 11px;border-radius:8px;border:1px solid #0e4f63;background:rgba(11,17,32,.85);color:#67e8f9;font-family:ui-monospace,monospace;font-size:11px;font-weight:600;cursor:pointer;">&#9004; React</button>' +
      /* Glass panel, matching #poseLegend3D: the same rgba(8,12,20,.42) fill,
         rgba(30,41,59,.6) hairline and 2px backdrop blur. The blur is what makes
         42% opacity readable over the rendered pocket — without it the ligand
         sticks show straight through the text. Keep the two in sync. */
      '<div id="poseReactPop" style="display:none;position:absolute;bottom:42px;right:10px;z-index:7;pointer-events:auto;width:336px;max-height:calc(50% - 29px);overflow-y:auto;background:rgba(8,12,20,.42);border:1px solid rgba(30,41,59,.6);border-radius:12px;padding:12px 13px;box-shadow:0 12px 34px rgba(0,0,0,.6);backdrop-filter:blur(2px);-webkit-backdrop-filter:blur(2px);font-family:\'Inter\',system-ui,sans-serif;">' +
        '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:9px;">' +
          '<span style="font-size:11px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:#67e8f9;">&#9004; React with building blocks</span>' +
          '<button onclick="PoseGen.reactToggle(false)" style="background:none;border:none;color:#475569;font-size:16px;cursor:pointer;line-height:1;">&times;</button>' +
        '</div>' +

        /* ── Pin center ───────────────────────────────────────────────────────
           The anchor the docking box is held at. Filled automatically from a
           pinned ligand's true docked centre (📌 on a gallery card), or typed in
           by hand. While it is set, stepping through imported poses no longer
           drags the box around — see PG._boxPinned / _impLoad. Clear all three
           fields to release the pin and let the box follow each loaded ligand
           again, which is the old behaviour. */
        '<div style="display:flex;align-items:center;gap:5px;margin-bottom:8px;">' +
          '<span title="The docking box is held at this point. Pinning a ligand (📌 on its gallery card) fills it with that ligand\'s real docked centre and locks the box there, so stepping through imported poses keeps the same frame instead of the box jumping to every new ligand. Clear all three to unpin." style="' + lbl + 'width:62px;flex-shrink:0;cursor:help;">Pin center</span>' +
          '<input id="poseRxCx" type="number" step="0.5" placeholder="x" onchange="PoseGen.pinCenterEdit()" style="' + num + '">' +
          '<input id="poseRxCy" type="number" step="0.5" placeholder="y" onchange="PoseGen.pinCenterEdit()" style="' + num + '">' +
          '<input id="poseRxCz" type="number" step="0.5" placeholder="z" onchange="PoseGen.pinCenterEdit()" style="' + num + '">' +
          '<span style="color:#475569;font-size:9px;">&#197;</span>' +
        '</div>' +

        /* The react product stepper (#poseRxResult) and its status line
           (#poseRxStatus) used to sit here. Both belonged to react-by-SMARTS,
           which no longer has any inputs, so they could never show anything but
           "0 / 0" and an empty line. Removed. #poseFindStatus at the bottom of
           the Import section is the one status line now. */

        /* Find pose by file name lives at the BOTTOM of the Import section, after
           #poseImpStatus — see below. It reads PG._imp.all, which only exists once
           a folder has been scanned, so it belongs after the scan controls that
           populate it rather than above them. */

        /* ── Import: scan a folder (recursively) for .pdbqt and step through the
           matches with the same ◀ ▶ ▶ Play stepper the reacted products use. Kept
           in its own state (PG._imp) so it can't clobber a live reaction list. ── */
        '<div style="margin-top:10px;padding-top:10px;border-top:1px solid #1e293b;">' +
          '<div style="' + lbl + 'margin-bottom:6px;">Import from disk</div>' +
          '<div style="margin-bottom:8px;">' +
            '<div style="' + lbl + 'margin-bottom:3px;">Folder path</div>' +
            '<input id="poseImpPath" type="text" spellcheck="false" placeholder="/&#8230;/vina_results" title="Scanned recursively for .pdbqt, .pdb and .ent files. This is also the folder the Find box searches when nothing has been scanned yet." style="' + txt + '">' +
          '</div>' +

          /* score filter — Vina affinity is negative, so "better" means LOWER */
          '<div style="display:flex;align-items:center;gap:6px;margin-bottom:9px;">' +
            '<span title="Keep only files whose best Vina affinity is at most this. Vina scores are negative and more negative = stronger binding, so -9 is a stricter cut than -7. Leave blank to keep every file." style="' + lbl + 'cursor:help;flex-shrink:0;">Max score</span>' +
            '<input id="poseImpMaxScore" type="number" step="0.5" placeholder="any" onchange="PoseGen.importFilter()" style="' + num + 'width:60px;">' +
            '<span style="font-size:9px;color:#475569;flex-shrink:0;">kcal/mol</span>' +
            '<label title="Skip prepared receptors (more than 300 atoms) and keep only docked ligand poses." style="margin-left:auto;display:flex;align-items:center;gap:4px;font-size:9px;color:#64748b;cursor:pointer;white-space:nowrap;">' +
              '<input id="poseImpLigOnly" type="checkbox" checked onchange="PoseGen.importFilter()" style="accent-color:#22d3ee;margin:0;cursor:pointer;">ligands only' +
            '</label>' +
          '</div>' +

          '<button id="poseImportBtn" onclick="PoseGen.importConfirm()" title="Recursively scans the folder above (including subfolders) for .pdbqt / .pdb / .ent files and steps through them in the box, one at a time. Unzipped batches from the download button below are readable straight back in." style="width:100%;padding:8px;border-radius:9px;border:1px solid transparent;background:linear-gradient(135deg,#0891b2,#7c3aed);color:#fff;font-size:12px;font-weight:700;cursor:pointer;font-family:inherit;">Import &#8594; scan folder for poses</button>' +

          '<div id="poseImpSummary" style="display:none;margin-top:7px;font-size:9.5px;color:#64748b;font-family:ui-monospace,monospace;line-height:1.55;"></div>' +

          '<button id="poseImpDownload" onclick="PoseGen.importDownload()" title="Zip every file currently passing the filter, converted to PDB with only its best pose kept, plus a _manifest.csv ranked by score and _best_poses.pdb holding them all as one multi-model file." style="display:none;width:100%;margin-top:7px;padding:7px;border-radius:9px;border:1px solid #14532d;background:rgba(11,17,32,.7);color:#34d399;font-family:ui-monospace,monospace;font-size:10.5px;font-weight:700;cursor:pointer;">&#11015; Download filtered batch (.pdb .zip)</button>' +

          '<div id="poseImpResult" style="display:none;margin-top:9px;padding-top:9px;border-top:1px solid #1e293b;">' +
            '<div style="display:flex;align-items:center;gap:6px;">' +
              '<button onclick="PoseGen.importStep(-1)" title="previous file" style="' + step + '">&#9664;</button>' +
              '<span id="poseImpCount" style="font-family:ui-monospace,monospace;font-size:11px;color:#67e8f9;min-width:54px;text-align:center;">0 / 0</span>' +
              '<button onclick="PoseGen.importStep(1)" title="next file" style="' + step + '">&#9654;</button>' +
              '<button id="poseImpPlay" onclick="PoseGen.importPlay()" title="play through every imported file, first &#8594; last" ' +
                'style="margin-left:auto;padding:3px 11px;border-radius:7px;border:1px solid #14532d;background:rgba(11,17,32,.7);color:#34d399;font-family:ui-monospace,monospace;font-size:10px;font-weight:700;cursor:pointer;">&#9654; Play</button>' +
            '</div>' +
            '<div id="poseImpCur" style="margin-top:5px;font-family:ui-monospace,monospace;font-size:9.5px;color:#64748b;word-break:break-all;line-height:1.4;max-height:46px;overflow:auto;"></div>' +
          '</div>' +

          '<div id="poseImpStatus" style="margin-top:8px;font-size:10px;color:#64748b;font-family:ui-monospace,monospace;min-height:13px;line-height:1.4;"></div>' +

          /* ── Find a pose by file name ────────────────────────────────────────
             Sits at the bottom of the Import section, directly under
             #poseImpStatus, because it operates on what the scan above produced:
             the datalist and every match come from PG._imp.all, which is empty
             until "Import → scan folder for poses" (or this box's own auto-scan)
             has run. Typing a name locates the file, clears whatever filter would
             hide it, and draws it in the 3D box. ──────────────────────────── */
          '<div style="margin-top:10px;padding-top:10px;border-top:1px solid #1e293b;">' +
            '<div style="' + lbl + 'margin-bottom:3px;">Find pose by file name</div>' +
            '<input id="poseFindName" type="text" spellcheck="false" autocomplete="off" list="poseFindList" ' +
                   'placeholder="1424889_20260720103540_out.pdb" ' +
                   'oninput="PoseGen._findSuggest()" onkeydown="if(event.key===\'Enter\'){event.preventDefault();PoseGen.findPose();}" ' +
                   'title="File name, or a fragment of one. A full path (anything containing /) is loaded directly without a scan. Enter to search." ' +
                   'style="' + txt + 'margin-bottom:8px;">' +
            '<datalist id="poseFindList"></datalist>' +

            '<button id="poseFindBtn" onclick="PoseGen.findPose()" ' +
              'title="Finds the file, clears any filter that would hide it, and shows its real docked geometry in the 3D box." ' +
              'style="width:100%;padding:8px;border-radius:9px;border:1px solid transparent;background:linear-gradient(135deg,#0891b2,#7c3aed);color:#fff;font-size:12px;font-weight:700;cursor:pointer;font-family:inherit;">&#128269; Find &#8594; show pose in 3D</button>' +

            '<div id="poseFindStatus" style="margin-top:7px;font-size:9.5px;color:#64748b;font-family:ui-monospace,monospace;min-height:12px;line-height:1.5;word-break:break-all;"></div>' +
          '</div>' +
        '</div>' +
      '</div>' +

      /* Vina score readout for the imported .pdbqt — mirrors #poseProcPanel's look.
         Own IDs so it never clobbers the Min/MC/Vina panel, which owns those. */
      '<div id="poseImpScore" style="display:none;position:absolute;top:8px;right:10px;width:208px;background:rgba(8,12,20,.85);border:1px solid #1e293b;border-radius:11px;padding:10px 12px;pointer-events:auto;">' +
        '<div style="font-size:9.5px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#64748b;">Vina score (lower = better)</div>' +
        '<div id="poseImpScoreVal" style="font-family:ui-monospace,monospace;font-size:19px;font-weight:700;color:#34d399;line-height:1.1;margin-top:1px;">&#8212;</div>' +
        '<svg id="poseImpScoreChart" viewBox="0 0 184 44" style="width:100%;height:44px;margin-top:5px;display:block;"></svg>' +
        '<div id="poseImpScoreSub" style="font-size:9.5px;color:#475569;font-family:ui-monospace,monospace;margin-top:2px;">&#8212;</div>' +
      '</div>';
    var wrap = document.createElement('div');
    wrap.innerHTML = html;
    while (wrap.firstChild) pane.appendChild(wrap.firstChild);
    PG._reactSyncCenter();
  };

  /* Reflect the current pin in the Pin center fields.

     If a ligand is pinned, those fields ARE the pin and must show it — the panel
     is built lazily on first open, so a ligand pinned before the popover was
     ever opened would otherwise leave the boxes blank while the box was in fact
     pinned. Falling back to mirroring the docking-box centre only when nothing
     is pinned keeps the old "sensible default" behaviour. */
  PG._reactSyncCenter = function () {
    var pinned = PG._pinnedCenter ? PG._pinnedCenter() : null;
    if (pinned) { PG._rx.center = pinned.slice(); PG._pinInputs(pinned); return; }
    if (PG._rx.center) { PG._pinInputs(PG._rx.center); return; }   // hand-typed pin — don't clobber
    [['poseCx', 'poseRxCx'], ['poseCy', 'poseRxCy'], ['poseCz', 'poseRxCz']].forEach(function (p) {
      var s = $(p[0]), d = $(p[1]);
      if (d && !d.value && s && s.value !== '') d.value = s.value;
    });
  };

  /* ========================================================================
     PIN CENTER — hold the docking box at one point
     ------------------------------------------------------------------------
     Problem this solves: _impLoad recentres box.center on EVERY imported file,
     because for a lone docked pose that is the only sane framing. Stepping
     through 573 results therefore drags the box (and the camera's sense of
     where "here" is) to a new place 573 times, which makes comparing two poses
     in the same pocket almost impossible — the thing you are looking at keeps
     moving under you.

     A pinned ligand is exactly the statement "this is my reference". So pinning
     one now also pins the box: its true docked centre (lig._realCenter, set by
     ligandFromPDB from the file's own coordinates) is written into the Pin
     center fields AND into box.center, and _impLoad stops recentring while the
     pin is live. Unpinning the last ligand releases it.

     PG._rx.center is reused as the single source of truth for "is the box
     pinned, and where" — it already existed for exactly this idea (it used to
     pin the start centre of reaction products) and nothing else reads it now.
     ======================================================================== */

  /* Mean of every pinned ligand's real centre, or null when nothing is pinned.
     Averaging matters when two ligands are pinned in the same pocket: the box
     should sit between them, not on whichever happened to be pinned first. */
  PG._pinnedCenter = function () {
    var pins = PG._pinLigs || {}, acc = [0, 0, 0], n = 0;
    Object.keys(pins).forEach(function (nm) {
      var c = pins[nm] && pins[nm]._realCenter;
      if (c && c.length === 3 && c.every(function (v) { return isFinite(v); })) {
        acc[0] += +c[0]; acc[1] += +c[1]; acc[2] += +c[2]; n++;
      }
    });
    return n ? [acc[0] / n, acc[1] / n, acc[2] / n] : null;
  };

  /* Write a centre into the Pin center inputs (null clears them). */
  PG._pinInputs = function (c) {
    ['poseRxCx', 'poseRxCy', 'poseRxCz'].forEach(function (id, i) {
      var e = $(id); if (!e || e === document.activeElement) return;
      e.value = c ? (+c[i]).toFixed(1) : '';
    });
  };

  PG._boxPinned = function () { return !!PG._rx.center; };

  /* Recompute the pin from the current pinned set and apply it to the box.
     Called after any pin/unpin. `redraw` is false during a load that is going
     to redraw anyway, so the scene is not rebuilt twice. */
  PG._syncPinCenter = function (redraw) {
    var c = PG._pinnedCenter();
    PG._rx.center = c ? c.slice() : null;
    PG._pinInputs(c);
    if (c) {
      box.center = c.slice();
      if (protein) { protein._site = null; protein._siteCenter = null; }   // pocket selection follows the new centre
      PG._syncBoxInputs();
      PG._log('box PINNED to', c.map(function (v) { return v.toFixed(1); }).join(', '),
              '· imported poses will no longer move it');
      try {
        PG._miniLog('📌 box pinned at ' + c.map(function (v) { return v.toFixed(1); }).join(', ') + ' Å', '#f59e0b');
      } catch (e) {}
    } else {
      PG._log('box UNPINNED — imported poses recentre it again');
      try { PG._miniLog('box unpinned — it follows each loaded pose again', '#7c6bae'); } catch (e) {}
    }
    if (redraw !== false) {
      try { PG._invalidate(); PG._syncBoxInputs(); PG.stage(PG._stage); } catch (e) {
        PG._log('pin redraw failed:', e && e.message);
      }
    }
  };

  /* Typing in the Pin center fields pins (or, when cleared, unpins) by hand.
     Hand edits win over the pinned-ligand average until the pin set changes. */
  PG.pinCenterEdit = function () {
    var v = ['poseRxCx', 'poseRxCy', 'poseRxCz'].map(function (id) {
      var e = $(id); return e ? parseFloat(e.value) : NaN;
    });
    var all = v.every(function (x) { return isFinite(x); });
    var none = ['poseRxCx', 'poseRxCy', 'poseRxCz'].every(function (id) {
      var e = $(id); return !e || e.value === '';
    });
    if (all) {
      PG._rx.center = v;
      box.center = v.slice();
      if (protein) { protein._site = null; protein._siteCenter = null; }
      PG._syncBoxInputs();
      PG._log('box pinned by hand to', v.join(', '));
    } else if (none) {
      PG._rx.center = null;
      PG._log('box unpinned by hand');
    } else {
      return;                                    // partial entry — wait for all three
    }
    try { PG._invalidate(); PG.stage(PG._stage); } catch (e) {}
  };

  PG.reactToggle = function (show) {
    var p = $('poseReactPop'); if (!p) return;
    var open = (show === undefined) ? (!p.style.display || p.style.display === 'none') : !!show;
    p.style.display = open ? 'block' : 'none';
    if (open) PG._reactSyncCenter();
  };

  PG._reactStatus = function (t, c) {
    var st = $('poseRxStatus'); if (st) { st.textContent = t; st.style.color = c || '#64748b'; }
  };

  /* ========================================================================
     FIND A POSE BY FILE NAME
     ------------------------------------------------------------------------
     Replaces the Reaction SMARTS / building-block CSV / cap / Confirm controls.
     Type a name from a results folder — 1424889_20260720103540_out.pdb — and
     this locates the file, makes sure the current filter cannot hide it, and
     draws its real docked geometry in the 3D box.

     Three things had to be true for that to work end to end, and only the first
     is the search itself:

       1. The list has to exist. PG._imp.all is only populated by a folder scan,
          so a cold open has nothing to search. Rather than silently doing
          nothing, this auto-scans the folder in #poseImpPath first. A value
          containing "/" is treated as a full path and loaded directly, no scan.

       2. The score filter has to let the file through. #poseImpMaxScore keeps
          only files whose best affinity is AT MOST the value shown, so a folder
          filtered to -8.5 hides a -7.7 pose completely: _impApplyFilter drops it
          from PG._imp.files and the stepper can never reach it. Searching for a
          file the filter excludes has to relax the filter or it is a no-op. The
          relaxation is minimal — the box is set to the found file's own score,
          rounded up to its 0.5 step, so everything at least that good stays
          visible instead of the filter being thrown away entirely.
          ("ligands only" gets the same treatment: a >300-atom file is invisible
          while it is ticked, so finding one unticks it.)

       3. The camera has to be looking at it. _draw3D re-applies PG._cam on every
          redraw, so a camera the user left pointing at a previous molecule is
          carried over onto the new one. Find explicitly reframes.
     ======================================================================== */
  var _POSE_FIND_MAX_SUGGEST = 60;    // datalist entries; more just slows the browser down

  /* _findStatus writes innerHTML so it can bold the parts that matter, which
     means every value interpolated into it must be escaped first. File names and
     paths come off the server's directory scan, i.e. they are attacker-influenced
     the moment anyone can write into a scanned folder: a file called
     `"><img src=x onerror=alert(1)>.pdb` would otherwise execute here. Same
     defect the codebase already has in its other data-driven innerHTML paths —
     not repeating it. */
  PG._esc = function (s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  };

  PG._findStatus = function (t, c) {
    var st = $('poseFindStatus'); if (st) { st.innerHTML = t; st.style.color = c || '#64748b'; }
  };

  /* basename, lowercased — the unit every match below is done in */
  PG._findBase = function (p) {
    return String(p || '').split('/').pop().toLowerCase();
  };

  /* Live autocomplete over the last scan. Cheap: pure string work on a list the
     scan already built, no fetch. */
  PG._findSuggest = function () {
    var dl = $('poseFindList'), inp = $('poseFindName');
    if (!dl || !inp) return;
    var q = inp.value.trim().toLowerCase();
    var all = PG._imp.all || [];
    if (!all.length) { dl.innerHTML = ''; return; }
    var hits = [];
    for (var i = 0; i < all.length && hits.length < _POSE_FIND_MAX_SUGGEST; i++) {
      var b = PG._findBase(all[i].path);
      if (!q || b.indexOf(q) !== -1) hits.push(all[i].path.split('/').pop());
    }
    dl.innerHTML = hits.map(function (h) {
      return '<option value="' + PG._esc(h) + '"></option>';
    }).join('');
  };

  /* name (or fragment) → the PG._imp.all entry, or a list of near misses.
     Exact basename wins; then exact basename ignoring the extension (so
     "..._out" finds "..._out.pdb"); then a unique substring. */
  PG._findMatch = function (q) {
    var all = PG._imp.all || [];
    var needle = q.toLowerCase();
    var noExt = function (s) { return s.replace(/\.(pdbqt|pdb|ent)$/i, ''); };
    var exact = [], stem = [], loose = [];
    all.forEach(function (f) {
      var b = PG._findBase(f.path);
      if (b === needle) exact.push(f);
      else if (noExt(b) === noExt(needle)) stem.push(f);
      else if (b.indexOf(needle) !== -1) loose.push(f);
    });
    if (exact.length) return { hit: exact[0], n: exact.length };
    if (stem.length)  return { hit: stem[0],  n: stem.length  };
    if (loose.length === 1) return { hit: loose[0], n: 1 };
    return { hit: null, n: loose.length, near: loose.slice(0, 6) };
  };

  /* ── Hold the camera across a Find ─────────────────────────────────────────
     Find used to snap the view back to a default framing (eye 1.25,1.25,1.25)
     on the theory that "show me this file" means "point at it". That was wrong
     in practice: you orbit to the angle that shows the pocket properly, search
     for a pose to compare, and the view you just set up is thrown away. Every
     other way of loading a pose — ◀ ▶, ▶ Play, picking from the gallery —
     preserves the camera, so Find was also the odd one out.

     Snapshot-and-restore rather than simply not touching the camera: _impLoad
     can change box.center, which changes the scene's axis ranges, and a
     Plotly.react across a range change is not guaranteed to leave the camera
     untouched. Capturing the eye before the load and writing it back after
     makes "the view does not move" true by construction instead of by luck.

     _camSnap reads the LIVE camera (falling back to the saved one), never a
     reference — Plotly mutates its own camera object in place, so holding the
     reference would mean the "snapshot" mutates into the post-load value and
     restores nothing. That is the same trap _draw3D documents. */
  PG._camSnap = function () {
    return PG._camClone(PG._camGet() || PG._cam) || null;
  };

  /* Put a snapshot back. PG._cam is set first so _draw3D and _camRestore agree
     with the snapshot instead of fighting it; the relayout only fires if the
     view actually drifted, so the common case costs nothing. _styling is held
     across the call so the plotly_relayout echo is not read as a user orbit. */
  PG._camApply = function (snap) {
    if (!snap || !snap.eye) return;
    PG._cam = PG._camClone(snap);
    if (!window.Plotly || !PG._plotReady || !PG._plotReady()) return;
    var live = PG._camGet();
    if (live && live.eye &&
        Math.abs(live.eye.x - snap.eye.x) < 1e-6 &&
        Math.abs(live.eye.y - snap.eye.y) < 1e-6 &&
        Math.abs(live.eye.z - snap.eye.z) < 1e-6) {
      PG._log('find: camera unchanged by the load, nothing to restore');
      return;
    }
    PG._log('find: restoring camera to', JSON.stringify(snap.eye));
    PG._styling = true;
    try {
      var pr = Plotly.relayout('poseBox3D', { 'scene.camera': PG._camClone(snap) });
      var done = function () { setTimeout(function () { PG._styling = false; }, 60); };
      if (pr && pr.then) pr.then(done, done); else done();
    } catch (e) {
      PG._styling = false;
      PG._log('camApply threw', e && e.message);
    }
  };

  PG.findPose = function () {
    var inp = $('poseFindName');
    var q = inp ? inp.value.trim() : '';
    if (!q) { PG._findStatus('type a file name, e.g. 1424889_20260720103540_out.pdb', '#fb7185'); if (inp) inp.focus(); return; }

    PG._importStop();
    PG._reactStop();

    /* A full path skips the scan entirely — nothing to search, just load it. */
    if (q.indexOf('/') !== -1) { PG._findLoadPath(q); return; }

    if ((PG._imp.all || []).length) { PG._findIn(q); return; }

    /* Nothing scanned yet — scan the Import folder, then search it. */
    var folder = ($('poseImpPath') ? $('poseImpPath').value : '').trim();
    if (!folder) {
      PG._findStatus('nothing scanned yet — put the results folder in <b>Folder path</b> below, or paste a full file path here', '#fbbf24');
      return;
    }
    var btn = $('poseFindBtn');
    if (btn) btn.disabled = true;
    PG._findStatus('scanning ' + PG._esc(folder) + '…', '#94a3b8');
    fetch('/pose/scan_pdbqt', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: folder })
    })
      .then(function (r) { return r.json(); })
      .then(function (res) {
        if (btn) btn.disabled = false;
        if (!res || !res.ok) { PG._findStatus('✗ ' + PG._esc((res && res.err) || 'scan failed'), '#fb7185'); return; }
        PG._imp.dir = res.dir || folder;
        PG._imp.all = (res.files || []).map(function (f) {
          return (typeof f === 'string') ? { path: f, score: null, atoms: null, kind: 'ligand' } : f;
        });
        PG._imp.hist = []; PG._imp.best = null;
        PG._impApplyFilter();
        PG._findSuggest();
        PG._findIn(q);
      })
      .catch(function (e) {
        if (btn) btn.disabled = false;
        PG._findStatus('✗ scan request failed: ' + PG._esc(e.message), '#fb7185');
      });
  };

  /* Load one file by absolute path, whether or not it came from a scan. */
  PG._findLoadPath = function (path) {
    var all = PG._imp.all || [];
    var known = -1;
    for (var i = 0; i < all.length; i++) if (all[i].path === path) { known = i; break; }
    if (known < 0) {                                   // not in the scan → adopt it as a one-file list
      PG._imp.all = all.concat([{ path: path, score: null, atoms: null, kind: 'ligand' }]);
      PG._imp.files = [path];
      PG._imp.idx = 0;
      PG._importUpdateStepper();
      PG._impDownloadLabel();
      var camBefore = PG._camSnap();                   // hold the user's view across the load
      PG._findStatus('loading ' + PG._esc(path.split('/').pop()) + '…', '#94a3b8');
      Promise.resolve(PG._impLoad(0)).then(function () {
        PG._camApply(camBefore);
        PG._findStatus('✓ loaded <span style="color:#67e8f9;">' + PG._esc(path.split('/').pop()) + '</span> by path', '#34d399');
      });
      return;
    }
    PG._findShow(PG._imp.all[known]);
  };

  PG._findIn = function (q) {
    var m = PG._findMatch(q);
    if (!m.hit) {
      if (m.near && m.near.length) {
        PG._findStatus('✗ no exact match — did you mean:<br>' +
          m.near.map(function (f) { return '&nbsp;&nbsp;' + PG._esc(f.path.split('/').pop()); }).join('<br>') +
          (m.n > m.near.length ? ('<br>&nbsp;&nbsp;… and ' + (m.n - m.near.length) + ' more') : ''), '#fbbf24');
      } else {
        PG._findStatus('✗ no file matching "' + PG._esc(q) + '" among ' + PG._imp.all.length +
                       ' scanned under ' + PG._esc(PG._imp.dir || '?'), '#fb7185');
      }
      return;
    }
    PG._findShow(m.hit, m.n > 1 ? m.n : 0);
  };

  /* The payoff: clear whatever would hide this file, jump the stepper to it,
     draw it WITHOUT moving the camera, and say exactly what was changed. */
  PG._findShow = function (f, ambiguous) {
    var name = f.path.split('/').pop();
    var notes = [];

    /* Snapshot the camera here, not in findPose(): the auto-scan path awaits a
       fetch first, and the user can orbit while it runs. Taking it immediately
       before the load captures the view they are actually looking at. */
    var camBefore = PG._camSnap();

    /* (1) ligands-only would drop a receptor-sized file before the score cut
           even runs, so it has to go first. */
    var lEl = $('poseImpLigOnly');
    if (lEl && lEl.checked && f.kind === 'receptor') {
      lEl.checked = false;
      notes.push('unticked <b>ligands only</b> (' + (f.atoms != null ? f.atoms + ' atoms' : 'receptor-sized') + ')');
    }

    /* (2) the score cut. Vina scores are negative and LOWER is better, so a file
           survives when score <= max. Raise max to this file's own score (rounded
           up to the input's 0.5 step) rather than clearing the box, so the rest of
           the filter still means something. A file with no score at all cannot
           clear any numeric cut, so that one does clear the box. */
    var mEl = $('poseImpMaxScore');
    if (mEl && mEl.value !== '') {
      var max = parseFloat(mEl.value);
      if (isFinite(max)) {
        if (typeof f.score !== 'number') {
          mEl.value = '';
          notes.push('cleared <b>max score</b> (this file carries no Vina score)');
        } else if (f.score > max) {
          var relaxed = Math.ceil(f.score * 2) / 2;        // next 0.5 step at or above the score
          mEl.value = String(relaxed);
          notes.push('relaxed <b>max score</b> ' + max.toFixed(1) + ' &#8594; ' + relaxed.toFixed(1));
        }
      }
    }

    /* (3) re-filter, then jump the stepper to this exact path.
           Only re-filter when something actually changed or the file is not in
           the current list — _impApplyFilter resets the score sparkline history,
           and a plain search should not wipe the trend the user built up. */
    var idx = PG._imp.files.indexOf(f.path);
    if (idx < 0 || notes.length) {
      PG._impApplyFilter();
      idx = PG._imp.files.indexOf(f.path);
    }
    if (idx < 0) {                                        // should be unreachable after (1) and (2)
      PG._findStatus('✗ ' + PG._esc(name) + ' is still being filtered out — clear the filters and retry', '#fb7185');
      return;
    }

    PG._findStatus('loading ' + PG._esc(name) + '…', '#94a3b8');
    Promise.resolve(PG._impLoad(idx)).then(function () {
      PG._camApply(camBefore);                            // (4) put the view back exactly as it was
      var bits = ['✓ <span style="color:#67e8f9;">' + PG._esc(name) + '</span>'];
      if (typeof f.score === 'number') bits.push(f.score.toFixed(1) + ' kcal/mol');
      bits.push('file ' + (PG._imp.idx + 1) + ' / ' + PG._imp.files.length);
      var msg = bits.join(' · ');
      if (ambiguous) msg += ' <span style="color:#fbbf24;">(' + ambiguous + ' matched, showing the first)</span>';
      if (notes.length) msg += '<br><span style="color:#fbbf24;">↳ ' + notes.join(' · ') + '</span>';
      PG._findStatus(msg, '#34d399');
    }).catch(function (e) {
      PG._findStatus('✗ could not load ' + PG._esc(name) + ': ' + PG._esc(e.message), '#fb7185');
    });
  };

  /* React-by-SMARTS is retired: its four inputs (#poseRxSmarts, #poseRxCsv,
     #poseRxCap, #poseRxConfirm) were replaced by the Find box above. The rest of
     the reaction machinery — _reactLoad / reactStep / reactPlay and the product
     stepper — is left intact and simply has no products, so nothing throws and a
     future caller could still drive it. Kept as an explicit stub rather than
     deleted so a stale onclick or an external caller gets a clear answer instead
     of "PoseGen.reactConfirm is not a function". */
  PG.reactConfirm = function () {
    PG._reactStop();
    // Reports through _findStatus, not _reactStatus: #poseRxStatus was removed
    // along with the product stepper, so _reactStatus now writes to nothing.
    PG._findStatus('react-by-SMARTS was replaced by the Find box — ' +
                   'search a results folder by file name instead', '#fbbf24');
  };

  PG._reactConfirmLegacy = function () {
    PG._reactStop();
    try { PG._impScoreShow(false); } catch (e) {}   // reaction products own the ligand now → drop the imported file's score
    var lig    = ($('poseSmiles')   ? $('poseSmiles').value   : '').trim();
    var smarts = ($('poseRxSmarts') ? $('poseRxSmarts').value : '').trim();
    var csv    = ($('poseRxCsv')    ? $('poseRxCsv').value    : '').trim();
    var capEl  = $('poseRxCap');
    var cap    = capEl ? parseInt(capEl.value, 10) : NaN;
    if (!isFinite(cap) || cap < 1) { PG._reactStatus('cap must be a positive integer', '#fb7185'); return; }
    if (!lig)    { PG._reactStatus('enter a ligand SMILES in the box above first', '#fb7185'); return; }
    if (!smarts) { PG._reactStatus('enter a reaction SMARTS', '#fb7185'); return; }
    if (!csv)    { PG._reactStatus('enter a building-block CSV path', '#fb7185'); return; }

    /* pin (or clear) the ligand start center */
    var c = ['poseRxCx', 'poseRxCy', 'poseRxCz'].map(function (id) {
      var e = $(id); return e ? parseFloat(e.value) : NaN;
    });
    PG._rx.center = c.every(function (v) { return isFinite(v); }) ? c : null;
    var pinMsg = PG._rx.center
      ? ' · center pinned (' + PG._rx.center.map(function (v) { return v.toFixed(1); }).join(', ') + ')'
      : ' · random center';

    var btn = $('poseRxConfirm');
    PG._reactStatus('reacting with up to ' + cap + ' building blocks\u2026', '#94a3b8');
    if (btn) btn.disabled = true;

    fetch('/pose/react', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ smiles: lig, smarts: smarts, csv_path: csv, limit: cap })
    })
      .then(function (r) { return r.json(); })
      .then(function (res) {
        if (btn) btn.disabled = false;
        if (!res || !res.ok) { PG._reactStatus('\u2717 ' + ((res && res.err) || 'reaction failed'), '#fb7185'); return; }
        PG._rx.products = res.products || []; PG._rx.idx = 0;
        PG._rx.poses = {}; PG._rx.built = -1;              // new product list → drop remembered poses
        if (!PG._rx.products.length) {
          PG._reactStatus('no products \u2014 0 of ' + (res.n_bb_total || 0) + ' building blocks reacted with this SMARTS', '#fbbf24');
          PG._reactUpdateStepper(); return;
        }
        var msg = '\u2713 ' + PG._rx.products.length + ' product(s) from ' + (res.n_bb || 0) + ' BB(s)';
        if (res.truncated) msg += ' · capped at ' + res.limit + ' of ' + res.n_bb_total;
        PG._reactStatus(msg + pinMsg, '#34d399');
        PG._reactLoad(0);
      })
      .catch(function () { if (btn) btn.disabled = false; PG._reactStatus('\u2717 request failed', '#fb7185'); });
  };

  /* load product i into #poseSmiles and build it — returns the build promise.

     PG.build() → _setLigand() nulls PG.init.cur and re-enters the stage, which
     calls PG.init.rand() → a BRAND-NEW random orientation + torsions. So without
     a cache, stepping ▶ then ◀ rebuilds the same SMILES with a different pose and
     you never get the same ligand back. Remember the pose per product index and
     restore it on revisit. */
  PG._reactLoad = function (i) {
    var arr = PG._rx.products;
    if (!arr.length) return Promise.resolve();

    /* remember the pose we're leaving — but only if the box still holds that
       product's SMILES (guards against the user hand-editing/building another ligand) */
    var sm = $('poseSmiles');
    if (PG._rx.built >= 0 && PG.init.cur && sm && sm.value.trim() === arr[PG._rx.built]) {
      PG._rx.poses[PG._rx.built] = PG._rxClonePose(PG.init.cur);
    }

    PG._rx.idx = ((i % arr.length) + arr.length) % arr.length;
    var k = PG._rx.idx;
    if (sm) sm.value = arr[k];
    PG._reactUpdateStepper();

    return Promise.resolve(PG.build()).then(function () {   // same path as clicking ⌬ Build
      PG._rx.built = k;
      var cached = PG._rx.poses[k];
      if (cached && L && L.TORS && cached.tors.length === L.TORS.length) {
        PG.init.cur = PG._rxClonePose(cached);              // same product → same pose as last time
        try { PG.stage(PG._stage); } catch (e) { try { PG.init.draw(PG.init.cur); } catch (_e) {} }
      } else if (PG.init.cur) {
        PG._rx.poses[k] = PG._rxClonePose(PG.init.cur);     // first visit → remember what we got
      }
    });
  };

  /* ======================================================================
     Debug: ligand center (xyz) → browser console + the #poseMiniChat log.
     Fired by the three React controls: ◀ prev, ▶ next, ▶ Play / ⏸ stop.

       · live   — the world center of the pose ACTUALLY in the box right now,
                  via PG.init.ligCenter(PG.init.cur) (the exact off→world inverse).
       · pinned — PG._rx.center, the xyz every product's start pose is pinned to.
                  null → each product gets a random center (conf::randomize()).

     The step/play calls log AFTER the build promise settles: PG.build() nulls
     PG.init.cur and re-derives the pose, so at click time `cur` is still the
     OUTGOING product. Logging post-build reports the product now on screen.
     ====================================================================== */
  PG._rxXYZ = function (c) {                                   // [x,y,z] → "155.6, 143.1, 148.4" (same idiom as the box-center log line)
    return (c && c.length === 3 && c.every(function (v) { return isFinite(v); }))
      ? c.map(function (v) { return (+v).toFixed(1); }).join(', ')
      : null;
  };

  PG._rxLiveCenter = function () {                             // TRUE centroid of the atoms on screen, or null if nothing is built
    return PG._poseCentroid(PG.init && PG.init.cur);
  };

  PG._rxLogCenter = function (tag) {
    try {
      var cur   = PG.init && PG.init.cur;
      var live  = PG._rxLiveCenter();                          // real centroid of the drawn atoms
      var anch  = (cur && cur.off) ? PG.init.ligCenter(cur) : null;   // translation anchor (what ligCenter returns)
      var sLive = PG._rxXYZ(live);
      var sPin  = PG._rxXYZ(PG._rx.center);
      var sBox  = PG._rxXYZ(box && box.center);
      var n     = PG._rx.products.length;
      var pos   = n ? (PG._rx.idx + 1) + ' / ' + n : '0 / 0';

      /* console — full precision; anchor shown too, so anchor≠centroid drift is visible */
      console.log('[React ' + tag + '] product ' + pos + ' · ligand center (Å):',
        live ? { x: +live[0].toFixed(3), y: +live[1].toFixed(3), z: +live[2].toFixed(3) } : '(no pose built yet)');
      console.log('[React ' + tag + '] pinned start center:', sPin ? sPin + ' Å' : '(none — random center)',
        '· anchor:', anch ? PG._rxXYZ(anch) + ' Å' : '(n/a)',
        '· docking-box center:', sBox ? sBox + ' Å' : '(n/a)');

      /* mini-chat — one concise activity line */
      PG._miniLog('⌬ ' + tag + ' · product ' + pos + ' · ligand center → ' + (sLive ? sLive + ' Å' : '—') +
        (sPin ? ' · pinned ' + sPin : ' · random center'), '#67e8f9');
    } catch (e) {
      try { console.warn('[React ' + tag + '] center debug failed:', e); } catch (_e) {}
    }
  };

  PG.reactStep = function (d) {
    PG._reactStop();
    var tag = ((d || 0) < 0) ? '◀ prev' : '▶ next';
    var p = PG._reactLoad(PG._rx.idx + (d || 0));
    Promise.resolve(p).catch(function () {}).then(function () { PG._rxLogCenter(tag); });   // after the build → xyz of the product now in the box
  };

  /* play 0 → cap: build each product, wait for it to render, dwell, next */
  PG.reactPlay = function () {
    if (PG._rx.playing) { PG._rxLogCenter('⏸ stop'); PG._reactStop(); return; }
    var arr = PG._rx.products;
    if (!arr.length) { PG._reactStatus('nothing to play — react first', '#fbbf24'); return; }
    PG._rx.playing = true; PG._reactPlayLabel();
    var i = 0;                                   // from the first product through the last
    var next = function () {
      if (!PG._rx.playing) return;
      if (i >= arr.length) { PG._reactStop(); PG._reactStatus('\u2713 played all ' + arr.length + ' product(s)', '#34d399'); return; }
      var first = (i === 0);                     // log once per Play click, once product 1 is actually built
      var p = PG._reactLoad(i); i++;
      Promise.resolve(p).catch(function () {}).then(function () {
        if (first) PG._rxLogCenter('▶ Play');
        if (!PG._rx.playing) return;
        PG._rx.timer = setTimeout(next, PG._rx.dwell);
      });
    };
    next();
  };

  PG._reactStop = function () {
    PG._rx.playing = false;
    if (PG._rx.timer) { clearTimeout(PG._rx.timer); PG._rx.timer = null; }
    PG._reactPlayLabel();
  };

  PG._reactPlayLabel = function () {
    var b = $('poseRxPlay'); if (!b) return;
    b.innerHTML = PG._rx.playing ? '&#10074;&#10074; Stop' : '&#9654; Play';
    b.style.color = PG._rx.playing ? '#fbbf24' : '#34d399';
    b.style.borderColor = PG._rx.playing ? '#78350f' : '#14532d';
  };

  PG._reactUpdateStepper = function () {
    var arr = PG._rx.products;
    var row = $('poseRxResult'); if (row) row.style.display = arr.length ? 'block' : 'none';
    var lab = $('poseRxCount');  if (lab) lab.textContent = arr.length ? (PG._rx.idx + 1) + ' / ' + arr.length : '0 / 0';
    var cur = $('poseRxCur');    if (cur) cur.textContent = arr.length ? arr[PG._rx.idx] : '';
  };

  /* ========================================================================
     Import — "Import → scan folder for .pdbqt" button, bottom of the React
     popover. Confirm → POST /pose/scan_pdbqt, which recursively lists every
     .pdbqt under a folder (subfolders included). ◀ / ▶ step through the list;
     ▶ Play walks first → last on its own — same stepper pattern as the
     reacted products above, kept in its own PG._imp state so the two never
     clobber each other.

     Each step fetches that one file's text via POST /pose/read_pdbqt, then
     renders it as an ACTUAL structure (not a rebuilt/random pose, unlike the
     SMILES-driven reaction products): atom count decides how, mirroring the
     same heuristic the receptors/ligands upload gallery uses server-side —
       ≤ IMPORT_LIGAND_MAX_ATOMS  → shown as the ligand (real docked geometry,
                                     via ligandFromPDB — same path as picking
                                     an uploaded ligand from the gallery)
       >  IMPORT_LIGAND_MAX_ATOMS → shown as the target receptor, via
                                     PG._applyProtein — same path as picking
                                     an uploaded receptor from the gallery
     So a folder mixing prepared receptor.pdbqt files with ligand poses (e.g.
     the output of the PDB → PDBQT converter) renders each correctly instead
     of forcing every file down one path.
     ======================================================================== */
  var IMPORT_LIGAND_MAX_ATOMS = 300;   // mirrors the server's _POSE_LIGAND_MAX_ATOMS gallery heuristic

  PG._imp = { files: [], all: [], idx: 0, dir: '', playing: false, timer: null, dwell: 700, hist: [], best: null, zipping: false };

  /* Pinned ligands kept on screen across ligand swaps: {fileName: ligandObj}.
     Amber carbons (matching the pin dot) so an overlay never reads as the
     active green ligand. _draw3D re-adds these on every redraw. */
  PG._pinLigs = PG._pinLigs || {};
  PG._PIN_COL = '#f59e0b';

  PG._importStatus = function (t, c) {
    var st = $('poseImpStatus'); if (st) { st.textContent = t; st.style.color = c || '#64748b'; }
  };

  /* Re-filter the scanned list in place (no re-scan — the scan already probed
     every file's score and atom count) and jump to the first survivor. */
  PG.importFilter = function () {
    if (!PG._imp.all.length) return;
    PG._importStop();
    PG._impApplyFilter();
    if (PG._imp.files.length) { PG._impLoad(0); }
    else { PG._impScoreShow(false); PG._importStatus('no files match the filter', '#fbbf24'); }
  };

  /* all → files, honouring "max score" and "ligands only". */
  PG._impApplyFilter = function () {
    var mEl = $('poseImpMaxScore'), lEl = $('poseImpLigOnly');
    var max = (mEl && mEl.value !== '') ? parseFloat(mEl.value) : NaN;
    var ligOnly = lEl ? !!lEl.checked : false;

    var kept = PG._imp.all.filter(function (f) {
      if (ligOnly && f.kind !== 'ligand') return false;
      if (isFinite(max)) {
        if (typeof f.score !== 'number') return false;   // unscored can't clear a score cut
        if (f.score > max) return false;                 // Vina: lower = stronger
      }
      return true;
    });

    PG._imp.files = kept.map(function (f) { return f.path; });
    PG._imp.idx = 0;
    PG._imp.hist = []; PG._imp.best = null;
    PG._importUpdateStepper();
    PG._impSummary(kept.length, isFinite(max) ? max : null, ligOnly);
    PG._impDownloadLabel();
  };

  /* keep the download button showing exactly what the filter kept */
  PG._impDownloadLabel = function () {
    var b = $('poseImpDownload'); if (!b) return;
    var n = PG._imp.files.length;
    b.style.display = n ? 'block' : 'none';
    if (n && !PG._imp.zipping) {
      b.innerHTML = '\u2B07 Download ' + n + ' filtered file' + (n === 1 ? '' : 's') + ' (.pdb .zip)';
    }
  };

  /* Zip the filtered set server-side and hand the browser the blob. Sends the
     exact path list the stepper is showing, so download and view never diverge. */
  PG.importDownload = function () {
    if (PG._imp.zipping) return;
    var paths = PG._imp.files;
    if (!paths.length) { PG._importStatus('nothing to download \u2014 scan a folder first', '#fbbf24'); return; }

    var b = $('poseImpDownload');
    var label = (PG._imp.dir || 'filtered').replace(/\/+$/, '').split('/').pop() || 'filtered';
    PG._imp.zipping = true;
    if (b) { b.disabled = true; b.innerHTML = '\u22EF converting ' + paths.length + ' file' + (paths.length === 1 ? '' : 's') + '\u2026'; }
    PG._importStatus('converting ' + paths.length + ' file(s) to PDB \u2014 large batches take a moment\u2026', '#94a3b8');

    var done = function (msg, col) {
      PG._imp.zipping = false;
      if (b) b.disabled = false;
      PG._impDownloadLabel();
      PG._importStatus(msg, col);
    };

    fetch('/pose/download_pdbqt', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ paths: paths, name: label })
    })
      .then(function (r) {
        var ct = r.headers.get('content-type') || '';
        if (!r.ok || ct.indexOf('application/json') === 0) {      // server reported a problem
          return r.json().then(
            function (j) { throw new Error((j && j.err) || ('HTTP ' + r.status)); },
            function () { throw new Error('HTTP ' + r.status); }
          );
        }
        var meta = {
          n: r.headers.get('X-Pose-Zip-Count'),
          skipped: r.headers.get('X-Pose-Zip-Skipped'),
          poses: r.headers.get('X-Pose-Zip-Poses'),
          trunc: r.headers.get('X-Pose-Zip-Truncated') === '1'
        };
        return r.blob().then(function (blob) { return { blob: blob, meta: meta }; });
      })
      .then(function (o) {
        var url = URL.createObjectURL(o.blob);
        var a = document.createElement('a');
        a.href = url; a.download = label + '_' + (o.meta.n || paths.length) + '_pdb.zip';
        document.body.appendChild(a); a.click(); a.remove();
        setTimeout(function () { URL.revokeObjectURL(url); }, 4000);

        var mb = (o.blob.size / 1048576).toFixed(1);
        var msg = '\u2713 downloaded ' + (o.meta.n || paths.length) + ' file(s) \u00b7 ' + mb + ' MB';
        if (o.meta.poses && +o.meta.poses) msg += ' \u00b7 ' + o.meta.poses + ' best poses combined';
        if (o.meta.skipped && +o.meta.skipped) msg += ' \u00b7 ' + o.meta.skipped + ' unreadable, skipped';
        if (o.meta.trunc) msg += ' \u00b7 stopped at the size cap';
        done(msg, o.meta.trunc ? '#fbbf24' : '#34d399');
      })
      .catch(function (e) { done('\u2717 download failed: ' + e.message, '#fb7185'); });
  };

  /* "4 987 ligands · 13 receptors" + the best score seen + what the filter kept */
  PG._impSummary = function (kept, max, ligOnly) {
    var el = $('poseImpSummary'); if (!el) return;
    var all = PG._imp.all;
    if (!all.length) { el.style.display = 'none'; return; }

    var nLig = 0, nRec = 0, nScored = 0, best = null;
    all.forEach(function (f) {
      if (f.kind === 'ligand') nLig++; else nRec++;
      if (typeof f.score === 'number') {
        nScored++;
        if (best === null || f.score < best) best = f.score;
      }
    });

    var rows = ['<span style="color:#34d399;font-weight:700;">' + nLig + '</span> ligand' + (nLig === 1 ? '' : 's') +
                (nRec ? (' \u00b7 <span style="color:#94a3b8;">' + nRec + '</span> receptor' + (nRec === 1 ? '' : 's')) : '') +
                ' detected'];
    rows.push(nScored + ' scored' + (best !== null ? (' \u00b7 best <span style="color:#67e8f9;">' + best.toFixed(1) + '</span> kcal/mol') : ''));
    if (max !== null || ligOnly) {
      var cond = [];
      if (ligOnly) cond.push('ligands');
      if (max !== null) cond.push('\u2264 ' + max.toFixed(1));
      rows.push('filter ' + cond.join(' \u00b7 ') + ' \u2192 <span style="color:' + (kept ? '#67e8f9' : '#fbbf24') + ';font-weight:700;">' + kept + '</span> shown');
    }
    el.innerHTML = rows.join('<br>');
    el.style.display = 'block';
  };

  /* Show/hide the score panel. #poseProcPanel (Min/MC/Vina) owns top:8px whenever
     it's visible, so stack underneath it rather than on top of it. */
  PG._impScoreShow = function (on) {
    var el = $('poseImpScore'); if (!el) return;
    el.style.display = on ? 'block' : 'none';
    if (!on) return;
    var proc = $('poseProcPanel');
    var procOn = proc && proc.style.display && proc.style.display !== 'none';
    el.style.top = procOn ? ((proc.offsetHeight || 110) + 14) + 'px' : '8px';
  };

  /* Render the Vina readout for the file just loaded. `score` is pose 1's
     affinity (Vina sorts best-first); the sparkline tracks the best score of
     every scored file visited this session, so playing through a results
     folder draws the screen's trend. */
  PG._impScore = function (res) {
    var val = $('poseImpScoreVal'), sub = $('poseImpScoreSub');
    var n = PG._imp.files.length, pos = (PG._imp.idx + 1) + '/' + n;

    if (!res || typeof res.score !== 'number' || !isFinite(res.score)) {
      PG._impScoreShow(true);                                  // scored-file-free entries still get a panel
      if (val) { val.textContent = '\u2014'; val.style.color = '#64748b'; }
      if (sub) sub.textContent = 'file ' + pos + ' \u00b7 no Vina score';
      PG._spark('poseImpScoreChart', PG._imp.hist, '#38bdf8');
      return;
    }

    PG._imp.hist.push(res.score);
    if (PG._imp.hist.length > 200) PG._imp.hist.shift();       // keep the sparkline bounded
    if (PG._imp.best === null || res.score < PG._imp.best) PG._imp.best = res.score;

    PG._impScoreShow(true);
    if (val) {
      val.textContent = res.score.toFixed(1);
      val.style.color = res.score <= -9 ? '#34d399' : (res.score <= -7 ? '#38bdf8' : '#f59e0b');
    }
    if (sub) {
      sub.textContent = 'file ' + pos +
        (res.n_models > 1 ? (' \u00b7 ' + res.n_models + ' poses') : '') +
        ' \u00b7 best ' + PG._imp.best.toFixed(1);
    }
    PG._spark('poseImpScoreChart', PG._imp.hist, '#38bdf8');
  };

  PG.importConfirm = function () {
    PG._importStop();
    var inp = $('poseImpPath');
    var folder = inp ? inp.value.trim() : '';
    if (!folder) { PG._importStatus('enter a folder path above', '#fb7185'); if (inp) inp.focus(); return; }

    var btn = $('poseImportBtn');
    PG._importStatus('scanning ' + folder + '\u2026', '#94a3b8');
    if (btn) btn.disabled = true;

    fetch('/pose/scan_pdbqt', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: folder })
    })
      .then(function (r) { return r.json(); })
      .then(function (res) {
        if (btn) btn.disabled = false;
        if (!res || !res.ok) { PG._importStatus('\u2717 ' + ((res && res.err) || 'scan failed'), '#fb7185'); return; }
        PG._imp.dir = res.dir || folder;
        PG._imp.all = (res.files || []).map(function (f) {          // tolerate a bare-path server
          return (typeof f === 'string')
            ? { path: f, score: null, atoms: null, kind: 'ligand' }
            : f;
        });
        PG._imp.hist = []; PG._imp.best = null;                     // new scan → fresh score history
        PG._impScoreShow(false);
        if (!PG._imp.all.length) {
          PG._imp.files = [];
          PG._impDownloadLabel();
          PG._importStatus('no .pdbqt files found under ' + PG._imp.dir, '#fbbf24');
          PG._importUpdateStepper(); return;
        }
        PG._impApplyFilter();                                       // sets PG._imp.files + the summary
        try { PG._findSuggest(); } catch (e) {}                     // feed the Find box's autocomplete
        var msg = '\u2713 scanned ' + PG._imp.all.length + ' pose file(s) under ' + PG._imp.dir;
        if (res.truncated) msg += ' \u00b7 stopped early at the scan cap';
        if (!PG._imp.files.length) {
          PG._importStatus(msg + ' \u2014 none match the filter', '#fbbf24'); return;
        }
        PG._importStatus(msg, '#34d399');
        PG._impLoad(0);
      })
      .catch(function () { if (btn) btn.disabled = false; PG._importStatus('\u2717 request failed', '#fb7185'); });
  };

  /* load imported file i: fetch its text, then render it as a real structure
     (ligand swap or receptor load) rather than rebuilding from a SMILES. */
  PG._impLoad = function (i) {
    var arr = PG._imp.files;
    if (!arr.length) return Promise.resolve();
    PG._imp.idx = ((i % arr.length) + arr.length) % arr.length;
    var k = PG._imp.idx, path = arr[k];
    PG._importUpdateStepper();

    return fetch('/pose/read_pdbqt', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: path })
    })
      .then(function (r) { return r.json(); })
      .then(function (res) {
        if (!res || !res.ok || !res.pdb) {
          PG._importStatus('\u2717 ' + ((res && res.err) || ('could not read ' + path)), '#fb7185'); return;
        }
        var label = res.name || path.split('/').pop();
        var p;
        try { p = PE.parsePDB(res.pdb); } catch (e) { PG._importStatus('\u2717 parse error in ' + label + ': ' + e.message, '#fb7185'); return; }
        if (!p.natoms) { PG._importStatus('\u2717 no ATOM/HETATM records in ' + label, '#fb7185'); return; }

        // Vina *_out.pdbqt* files pack many docked poses into successive MODEL
        // blocks; the server returns only pose 1 and tells us how many there were.
        var poseNote = (res.n_models && res.n_models > 1) ? (' \u00b7 pose 1 of ' + res.n_models) : '';
        var scoreNote = (typeof res.score === 'number' && isFinite(res.score))
          ? (' \u00b7 ' + res.score.toFixed(1) + ' kcal/mol') : '';
        PG._impScore(res);

        if (p.natoms > IMPORT_LIGAND_MAX_ATOMS) {
          PG._applyProtein(p, label.replace(/\.pdbqt$/i, ''), res.pdb, true);
          PG._importStatus('\u2713 ' + label + ' \u2014 receptor, ' + p.natoms + ' atoms' + poseNote + scoreNote, '#34d399');
          try { PG.init._refreshMarks(); } catch (e) {}   // loading never clears gallery marks — reassert in case any redraw path touched them
          return;
        }
        // Recentre the docking box on the imported ligand's real position (same
        // as picking one from the gallery) — UNLESS the box is pinned. A pinned
        // ligand is the user saying "hold the frame here"; recentring on every
        // file would move the box out from under the pinned reference on each
        // ▶ step, which is precisely what pinning is meant to stop.
        var c = p.center;
        var pinned = PG._boxPinned();
        if (!pinned && c && c.length === 3 && c.every(function (v) { return isFinite(v); })) {
          box.center = [+c[0], +c[1], +c[2]];
          if (protein) { protein._site = null; protein._siteCenter = null; }   // stale pocket selection → recompute around the new center
        }
        var lig = ligandFromPDB(p);
        if (!lig) { PG._importStatus('\u2717 could not build a ligand from ' + label, '#fb7185'); return; }
        L = lig; L._smiles = ''; L._label = label;                       // debug identity for _ligTag / the [Pose3D] log

        /* ── Where the pose actually gets drawn ──────────────────────────────
           The renderer places the active ligand at
               _ligCenter(off) = box.center + off · box.size · 0.32
           and ligandFromPDB returns REF relative to the molecule's OWN centre.
           So with off = [0,0,0] the pose is drawn centred on box.center — which
           is only its true docked position while box.center equals that centre.

           Unpinned that holds, because the branch above just set box.center = c.
           PINNED it does not: the box is deliberately left on the pinned ligand,
           so off = [0,0,0] silently TRANSLATES every searched pose by
           (pin centre − its own centre). The geometry stays intact, so it still
           looks like a plausible pose — it is simply in the wrong place against
           the protein surface, sunk into it or pushed out of the pocket. For the
           two ligands in the pinned session that is ~2.2 Å of drift.

           Fix: invert the placement. off = (c − box.center)/(box.size·0.32) is
           exactly _worldToOff, and makes _ligCenter return c regardless of where
           the box is pinned. The box stays put, the pose stays true.

           box.size·0.32 must be non-zero for that inverse to exist; if it ever
           is not, fall back to recentring the box, because drawing the pose in
           the wrong place is worse than moving the frame. */
        var off = [0, 0, 0];
        if (pinned && c && c.length === 3 && c.every(function (v) { return isFinite(v); })) {
          var scale = (box.size || 0) * 0.32;
          if (isFinite(scale) && Math.abs(scale) > 1e-9) {
            off = PG._worldToOff(c);
            PG._log('pinned draw: box at', box.center.map(function (v) { return (+v).toFixed(1); }).join(', '),
                    '· pose true centre', c.map(function (v) { return (+v).toFixed(1); }).join(', '),
                    '· off', off.map(function (v) { return v.toFixed(3); }).join(', '));
          } else {
            box.center = [+c[0], +c[1], +c[2]];                 // degenerate box → keep the pose honest
            if (protein) { protein._site = null; protein._siteCenter = null; }
            pinned = false;
            PG._log('pinned draw: box.size is degenerate — recentred instead of offsetting');
          }
        }
        PG.init.cur = { seed: 0, off: off, quat: [1, 0, 0, 0], tors: [] };   // identity rotation/torsions → real docked geometry
        PG._poseSrc = 'uploaded';
        PG._derived(); PG._invalidate(); PG._syncBoxInputs(); PG.stage(PG._stage);
        PG._impScoreShow(true);          // PG.stage() may toggle #poseProcPanel → re-place the panel
        // Say which of the two framings happened, and \u2014 when pinned \u2014 where the
        // pose itself landed. Reporting only the box centre while the pose is
        // drawn somewhere else is how the placement drift stayed invisible: the
        // status line looked correct because it was describing the box, not the
        // molecule. These two numbers now differ openly whenever a pin is held.
        var boxNote = pinned
          ? (' \u00b7 \ud83d\udccc box held at ' + box.center.map(function (v) { return (+v).toFixed(1); }).join(', ') +
             (c ? (' \u00b7 pose at ' + c.map(function (v) { return (+v).toFixed(1); }).join(', ')) : ''))
          : (c ? (' \u00b7 box \u2192 ' + c.map(function (v) { return (+v).toFixed(1); }).join(', ')) : '');
        PG._importStatus('\u2713 ' + label + ' \u2014 ligand, ' + p.natoms + ' atoms' + poseNote + scoreNote +
          boxNote, '#34d399');
        try { PG.init._refreshMarks(); } catch (e) {}   // stepping/Play must never drop a pin or the selection dot — reassert from state
      })
      .catch(function (e) { PG._importStatus('\u2717 request failed: ' + e.message, '#fb7185'); });
  };

  PG.importStep = function (d) {
    PG._importStop();
    Promise.resolve(PG._impLoad(PG._imp.idx + (d || 0))).catch(function () {});
  };

  /* play first → last: load each file, wait for it to render, dwell, next */
  PG.importPlay = function () {
    if (PG._imp.playing) { PG._importStop(); return; }
    var arr = PG._imp.files;
    if (!arr.length) { PG._importStatus('nothing to play \u2014 import a folder first', '#fbbf24'); return; }
    PG._imp.playing = true; PG._importPlayLabel();
    var i = 0;
    var next = function () {
      if (!PG._imp.playing) return;
      if (i >= arr.length) { PG._importStop(); PG._importStatus('\u2713 played all ' + arr.length + ' file(s)', '#34d399'); return; }
      var p = PG._impLoad(i); i++;
      Promise.resolve(p).catch(function () {}).then(function () {
        if (!PG._imp.playing) return;
        PG._imp.timer = setTimeout(next, PG._imp.dwell);
      });
    };
    next();
  };

  PG._importStop = function () {
    PG._imp.playing = false;
    if (PG._imp.timer) { clearTimeout(PG._imp.timer); PG._imp.timer = null; }
    PG._importPlayLabel();
  };

  PG._importPlayLabel = function () {
    var b = $('poseImpPlay'); if (!b) return;
    b.innerHTML = PG._imp.playing ? '&#10074;&#10074; Stop' : '&#9654; Play';
    b.style.color = PG._imp.playing ? '#fbbf24' : '#34d399';
    b.style.borderColor = PG._imp.playing ? '#78350f' : '#14532d';
  };

  PG._importUpdateStepper = function () {
    var arr = PG._imp.files;
    var row = $('poseImpResult'); if (row) row.style.display = arr.length ? 'block' : 'none';
    var lab = $('poseImpCount');  if (lab) lab.textContent = arr.length ? (PG._imp.idx + 1) + ' / ' + arr.length : '0 / 0';
    var cur = $('poseImpCur');    if (cur) cur.textContent = arr.length ? arr[PG._imp.idx] : '';
  };

  /* mount on open (idempotent); stop playback on close */
  (function () {
    var _open = PG.open.bind(PG);
    PG.open = function () { PG.init._uploadsLoaded = false; _open(); try { PG._reactInject(); } catch (e) {} };   // re-sync the gallery once per open
    var _close = PG.close.bind(PG);
    PG.close = function () { try { PG._reactStop(); } catch (e) {} try { PG._importStop(); } catch (e) {} _close(); };
  })();

  /* ======================================================================
     Debug: track the ligand center across a docking-box edit.

     Editing the box edge-length (#poseLen) or center (#poseCx/Cy/Cz) calls
     PG.setCenter(), which sets box.size / box.center, then PG._invalidate()
     (drops the Min/MC working pose) and re-derives the active stage. Because
     the world center is  box.center + off·box.size·0.32 + mean(pose),  a change
     in box.size moves a pose whose `off` is non-zero — which is why the ligand
     appears to jump when you retype the edge length. This logs the ligand
     center just BEFORE the edit and again AFTER, into #poseMiniChatOutput.

     We measure PG._last3D.world — the exact atom coordinates last drawn — so the
     number matches whatever is actually on screen regardless of stage (Random /
     Min / MC each draw a different pose object). setCenter's redraw is
     synchronous (stage → *.enter() → _draw3D sets _last3D before Plotly.react),
     so reading _last3D right after the original returns reflects the new pose.
     ====================================================================== */
  PG._drawnCenter = function () {                 // centroid of the atoms actually on screen, or null
    var w = PG._last3D && PG._last3D.world;
    if (w && w.length) {
      var s = [0, 0, 0];
      for (var i = 0; i < w.length; i++) { s[0] += w[i][0]; s[1] += w[i][1]; s[2] += w[i][2]; }
      return [s[0] / w.length, s[1] / w.length, s[2] / w.length];
    }
    try { return PG._poseCentroid(PG.init && PG.init.cur); } catch (e) { return null; }   // 2D / pre-draw fallback
  };

  (function () {
    var _setCenter = PG.setCenter.bind(PG);
    PG.setCenter = function () {
      var beforeC   = PG._drawnCenter();
      var beforeLen = box.size;
      var beforeCtr = box.center.slice();

      var out = _setCenter.apply(PG, arguments);      // apply the box edit + re-derive the stage (synchronous redraw)

      try {
        var afterC   = PG._drawnCenter();
        var afterLen = box.size;
        var afterCtr = box.center.slice();

        var sB = PG._rxXYZ(beforeC), sA = PG._rxXYZ(afterC);
        var ctrMoved = beforeCtr.some(function (v, k) { return Math.abs(v - afterCtr[k]) > 1e-6; });
        var lenChg   = Math.abs(beforeLen - afterLen) > 1e-6;

        var head = lenChg
          ? ('box edge ' + beforeLen.toFixed(0) + ' \u2192 ' + afterLen.toFixed(0) + ' \u00c5')
          : (ctrMoved ? 'box center moved' : 'box unchanged');
        if (lenChg && ctrMoved) head += ' · center ' + PG._rxXYZ(beforeCtr) + ' \u2192 ' + PG._rxXYZ(afterCtr);

        var lig;
        if (sB && sA) {
          var d = Math.sqrt(
            Math.pow(afterC[0] - beforeC[0], 2) +
            Math.pow(afterC[1] - beforeC[1], 2) +
            Math.pow(afterC[2] - beforeC[2], 2));
          lig = 'ligand center ' + sB + ' \u2192 ' + sA + ' \u00c5 · moved ' + d.toFixed(2) + ' \u00c5';
        } else if (sB || sA) {
          lig = 'ligand center ' + (sB ? sB + ' \u2192 (none)' : '(none) \u2192 ' + sA) + ' \u00c5';
        } else {
          lig = 'no ligand pose to measure';
        }

        PG._miniLog('\ud83d\udce6 ' + head + ' · ' + lig, '#fbbf24');
        console.log('[setCenter] box edge ' + beforeLen.toFixed(2) + ' \u2192 ' + afterLen.toFixed(2) +
          ' | box center', beforeCtr.map(function (v) { return +v.toFixed(2); }),
          '\u2192', afterCtr.map(function (v) { return +v.toFixed(2); }));
        console.log('[setCenter] ligand center BEFORE:',
          beforeC ? { x: +beforeC[0].toFixed(3), y: +beforeC[1].toFixed(3), z: +beforeC[2].toFixed(3) } : '(none)',
          '| AFTER:',
          afterC ? { x: +afterC[0].toFixed(3), y: +afterC[1].toFixed(3), z: +afterC[2].toFixed(3) } : '(none)');
      } catch (e) {
        try { console.warn('[setCenter] center debug failed:', e); } catch (_e) {}
      }
      return out;
    };
  })();

  try { document.addEventListener('DOMContentLoaded', function () { try { if ($('poseBox3D')) PG._reactInject(); } catch (e) {} }); } catch (e) {}


  /* ---- surface opacity slider (built inside the existing #poseSurfNote panel — no HTML change) ---- */
  PG._ensureSurfUI=function(){
    const host=$('poseSurfNote');if(!host||$('poseSurfOpacity'))return;
    const pct=Math.round((PG._surfOpacity==null?0.45:PG._surfOpacity)*100);
    host.style.pointerEvents='auto';
    host.innerHTML='<div style="display:flex;align-items:center;gap:6px;white-space:nowrap;">'+
      '<span style="color:#64748b;">Opacity</span>'+
      '<input id="poseSurfOpacity" type="range" min="0" max="100" step="1" value="'+pct+'" '+
      'oninput="PoseGen.setSurfOpacity(this.value)" title="Surface transparency: 0% = invisible, 100% = solid" '+
      'style="width:86px;accent-color:#22d3ee;cursor:pointer;vertical-align:middle;">'+
      '<b id="poseSurfOpacityVal" style="color:#e2e8f0;min-width:30px;text-align:right;">'+pct+'%</b></div>'+
      '<div style="margin-top:3px;color:#475569;">SES \u00b7 1.4 \u00c5 water probe</div>';
  };
  PG.setSurfOpacity=function(v){
    PG._surfOpacity=Math.max(0,Math.min(1,(+v||0)/100));
    const lab=$('poseSurfOpacityVal');if(lab)lab.textContent=Math.round(PG._surfOpacity*100)+'%';
    if(!PG._showSurface)return;
    if(window.Plotly&&PG._surfIdx>=0){                                            // geometry is cached: only the opacity changes
      PG._log('--- setSurfOpacity',PG._surfOpacity,'surfIdx=',PG._surfIdx,'styling=',PG._styling,'savedCam=',PG._cam?JSON.stringify(PG._cam.eye):'null');
      if(!PG._styling){                                                          // user-event time: the live camera is stable and CURRENT — refresh from it.
        var live=PG._camGet();                                                   // (skip while OUR restyle is in flight: that read can return the default)
        if(live){PG._cam=live;PG._log('  slider captured live cam ->',JSON.stringify(live.eye));}
      } else PG._log('  restyle in flight: keeping saved cam (no re-read)');
      PG._styling=true;
      var done=function(){
        try{PG._camRestore();}catch(_e){}
        setTimeout(function(){PG._styling=false;PG._log('styling=false (opacity)');},60);   // the echo lands AFTER the promise; keep ignoring it
      };
      if(!PG._plotReady()){PG._log('opacity restyle skipped: plot not ready');PG._styling=false;PG._redraw3D();return;}
      PG._log('opacity restyle',PG._surfOpacity);
      try{var camNow=PG._camClone(PG._cam);
        var pr=Plotly.update('poseBox3D',{opacity:PG._surfOpacity},camNow?{'scene.camera':camNow}:{},[PG._surfIdx]);
        if(pr&&pr.then)PG._safe(pr.then(done,done),'opacity');else done();
        return;
      }catch(e){PG._log('opacity update threw',e&&e.message);PG._styling=false;}
    }
    PG._redraw3D();
  };

  window.PoseGen=PG;
})();