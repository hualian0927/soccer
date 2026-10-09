import { useState } from "react";
import { X, ScanLine, Image, Film } from "lucide-react";
import "./formation-evidence.css";

export default function FormationEvidence({ item, onClose }) {
  const [mode, setMode] = useState("distribution");
  const [error, setError] = useState(false);
  const evidence = item.formationEvidence;
  const stamp = `${Math.floor(evidence.time / 60)}:${String(Math.floor(evidence.time % 60)).padStart(2,"0")}`;
  return <section className="formation-evidence" aria-label="阵型画面证据">
    <header><span>{item.title} · {stamp}</span><button title="关闭阵型标注" aria-label="关闭阵型标注" onClick={onClose}><X size={18}/></button></header>
    <div className="formation-media">
      {mode === "clip" ? <video key={item.id} src={evidence.clipUrl} controls playsInline preload="metadata" onError={() => setError(true)}/>
        : <img src={mode === "original" ? evidence.originalUrl : evidence.imageUrl} alt={mode === "original" ? "阵型代表帧原图" : "真实球员方框与站位连线"} onError={() => setError(true)}/>}
      {error && <p role="alert">证据文件加载失败，请重新打开此片段。</p>}
    </div>
    <footer role="group" aria-label="阵型证据显示模式">
      {[["distribution","分布标注",ScanLine],["original","原始画面",Image],["clip","战术片段",Film]].map(([value,label,Icon]) =>
        <button key={value} aria-pressed={mode === value} onClick={() => {setMode(value);setError(false);}}><Icon size={16}/>{label}</button>)}
    </footer>
  </section>;
}
