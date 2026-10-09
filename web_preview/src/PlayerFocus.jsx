import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Crosshair, Download, Film, Play } from "lucide-react";
import "./player-focus.css";

export default function PlayerFocus({ asset, videoRef, onSeek }) {
  const [enabled,setEnabled] = useState(false);
  const [data,setData] = useState(null);
  const [reviews,setReviews] = useState([]);
  const [frame,setFrame] = useState(null);
  const [player,setPlayer] = useState(null);
  const [start,setStart] = useState(0);
  const [end,setEnd] = useState(10);
  const [busy,setBusy] = useState(false);
  const [error,setError] = useState("");
  const [report,setReport] = useState(null);
  const [area,setArea] = useState(null);
  const activeRequest = useRef(0);

  useEffect(() => {
    if (!enabled || data || !asset.playerTrackingUrl) return;
    const abort = new AbortController();
    setError("");
    fetch(asset.playerTrackingUrl,{signal:abort.signal}).then((r) => {if (!r.ok) throw Error("跟踪数据加载失败");return r.json();})
      .then(setData).catch((e) => {if(e.name !== "AbortError")setError(e.message);});
    return () => abort.abort();
  },[enabled,data,asset.playerTrackingUrl]);

  useEffect(() => {
    if (!enabled || !asset.playerReviewsUrl) return;
    const abort = new AbortController();
    fetch(asset.playerReviewsUrl,{signal:abort.signal}).then((r) => r.ok ? r.json() : [])
      .then((rows) => setReviews(rows.filter((r) => r.tracking_sha256 === asset.playerTrackingHash))).catch(() => {});
    return () => abort.abort();
  },[enabled,asset.playerReviewsUrl,asset.playerTrackingHash]);

  useEffect(() => {
    const video = videoRef.current;
    if (!enabled || !data || !video) return;
    let raf, last=-1;
    const update = () => {
      const index = Math.min(data.frames.length-1,Math.max(0,Math.floor(video.currentTime*data.fps+.001)));
      if(index !== last) {last=index;setFrame(data.frames[index]);}
      raf=requestAnimationFrame(update);
    };
    const resize = () => {
      const w=video.clientWidth,h=video.clientHeight,scale=Math.min(w/data.width,h/data.height);
      setArea({left:(w-data.width*scale)/2,top:(h-data.height*scale)/2,width:data.width*scale,height:data.height*scale});
    };
    const observer=new ResizeObserver(resize);observer.observe(video);resize();update();
    return () => {cancelAnimationFrame(raf);observer.disconnect();};
  },[enabled,data,videoRef]);

  const select = (p) => {
    videoRef.current?.pause();setPlayer(p.id);setReport(null);setError("");activeRequest.current++;
    const now=videoRef.current?.currentTime || 0;
    const sceneStart=[0,...data.cuts].filter((t) => t <= now).at(-1);
    const sceneEnd=data.cuts.find((t) => t > now) ?? data.duration;
    setStart(Math.round(Math.max(sceneStart,now-3)*1000)/1000);
    setEnd(Math.floor(Math.min(sceneEnd,now+5)*1000)/1000);
  };
  const exportClip = async () => {
    const request = ++activeRequest.current;
    setBusy(true);setError("");setReport(null);
    try {
      const response=await fetch("/api/player-review",{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({trackingPath:asset.playerTrackingPath,playerId:player,start:Number(start),end:Number(end)})});
      const result=await response.json();if(!response.ok)throw Error(result.error);
      if(request===activeRequest.current)setReport(result);
    } catch(e) {if(request===activeRequest.current)setError(e.message);}
    finally {setBusy(false);}
  };
  const updateRange = (setter,value) => {setter(value);setReport(null);activeRequest.current++;};
  if (!asset.playerTrackingUrl) return null;
  return <section className="player-focus">
    <div className="focus-heading"><h2><Crosshair size={18}/>球员片段分析</h2><label><input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)}/>球员编号</label></div>
    {enabled && <>
      {!data && !error && <p role="status">正在加载跟踪数据…</p>}
      {data && <>
        <div className="focus-form">
          <label>球员<select aria-label="选择球员编号" value={player ?? ""} onChange={(e) => select({id:Number(e.target.value)})}>
            <option value="" disabled>未选择</option>
            {[...new Set([...(frame?.players || []).map((p) => p.id),...(player ? [player] : [])])].map((id) => <option key={id} value={id}>#{id}{frame?.players.some((p) => p.id===id) ? "" : "（当前不可见）"}</option>)}
          </select></label>
          <label>开始（秒）<input aria-label="片段开始秒" type="number" min="0" step="0.1" max={data.duration} value={start} onChange={(e) => updateRange(setStart,e.target.value)}/></label>
          <label>结束（秒）<input aria-label="片段结束秒" type="number" min="0" step="0.1" max={data.duration} value={end} onChange={(e) => updateRange(setEnd,e.target.value)}/></label>
          <button onClick={exportClip} disabled={!player || busy || Number(end)<=Number(start) || Number(end)-Number(start)>30}><Film size={16}/>{busy ? "正在导出" : "生成片段分析"}</button>
        </div>
        {reviews.length>0 && <div className="focus-examples">{reviews.map((r) => <button key={`${r.player_id}-${r.start}`} onClick={() => {
          activeRequest.current++;setReport(null);setPlayer(r.player_id);setStart(r.start);setEnd(r.end);onSeek(r.start);videoRef.current?.pause();
        }}><Play size={14}/>{r.title} · #{r.player_id}</button>)}</div>}
        {report && <div className="focus-result" role="status">
          <h3>#{report.player_id} · {report.start.toFixed(1)}–{report.end.toFixed(1)} 秒</h3>
          <dl><div><dt>可见时长</dt><dd>{report.observed_seconds}s</dd></div><div><dt>跟踪覆盖</dt><dd>{Math.round(report.coverage*100)}%</dd></div><div><dt>脚边球框邻近</dt><dd>{report.ball_near_feet_frames} 帧</dd></div></dl>
          {report.review ? <>{["observation","analysis","advice","limitations"].map((key,i) => <p key={key}><strong>{["观察","技战术解读","训练建议","证据边界"][i]}：</strong>{report.review[key]}</p>)}</>
            : <p>待画面复核。当前仅输出可见性与图像邻近指标，不自动确认触球、成功传球、射门或越位。</p>}
          <p className="focus-caution">{report.limitations.join(" ")}</p>
          <video src={report.clipUrl} poster={report.posterUrl} controls playsInline preload="metadata"/>
          <div className="focus-links"><a href={report.clipUrl} download><Download size={15}/>片段</a><a href={report.reportUrl} download><Download size={15}/>分析报告</a><a href={report.evidenceUrl} target="_blank" rel="noreferrer">证据帧</a></div>
        </div>}
      </>}
      {error && <p role="alert">{error}</p>}
    </>}
    {enabled && frame && area && videoRef.current && createPortal(
      <div className="player-id-overlay" style={area}>{frame.players.map((p) => <button key={p.id}
        className={`player-id ${player===p.id ? "selected" : ""}`} aria-label={`选择球员 ${p.id}`} title={`临时编号 #${p.id}`}
        style={{left:`${p.box[0]*100}%`,top:`${p.box[1]*100}%`,width:`${p.box[2]*100}%`,height:`${p.box[3]*100}%`}}
        onClick={(e) => {e.stopPropagation();select(p);}}><span>#{p.id}</span></button>)}</div>,videoRef.current.parentElement)}
  </section>;
}
