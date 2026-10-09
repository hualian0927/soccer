import { useState } from "react";
import { Play, ScanSearch } from "lucide-react";
import { analysisLevelLabels } from "./data";

const stamp = (value) => `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2, "0")}`;

export function LayeredAnalysisPanel({ level, data, onSeek }) {
  const [filter, setFilter] = useState("全部");
  if (!data) return null;
  const items = (data.items || []).filter((item) => filter === "全部"
    || (filter === "采纳" && item.publishedForStatistics)
    || (filter === "待核实" && item.decision === "uncertain")
    || (filter === "未采纳" && item.decision === "rejected"));
  return (
    <section className="layer-analysis" aria-label={analysisLevelLabels[level] || "战术分析"}>
      <h2>{data.title}</h2>
      <p className="layer-note">{data.note}</p>
      {data.items?.some((item) => item.decision) && <div className="layer-filter"><label>复核结论<select aria-label="战术复核筛选" value={filter} onChange={(event) => setFilter(event.target.value)}>{["全部", "采纳", "待核实", "未采纳"].map((option) => <option key={option}>{option}</option>)}</select></label><span>{items.length} 个片段</span></div>}
      {level === "L2" ? (
        <>
          <table className="layer-counts">
            <thead><tr>{(data.columns || ["事件", "候选", "送审", "支持", "否定"]).map((column) => <th key={column}>{column}</th>)}</tr></thead>
            <tbody>{data.counts.map((row) => <tr key={row.label}><th>{row.label}</th><td>{row.candidates}</td><td>{row.reviewed}</td><td>{row.supported}</td><td>{row.rejected}</td></tr>)}</tbody>
          </table>
          {data.reviewedMetrics && <><h3>已采纳动作分类</h3><p className="layer-note">{data.reviewedMetrics.scope} · {data.reviewedMetrics.acceptedCount} 个节点</p><table className="layer-counts"><thead><tr><th>动作</th><th>次数</th></tr></thead><tbody>{Object.entries(data.reviewedMetrics.subtypes).map(([type, count]) => <tr key={type}><th>{({pass: "传球", cross: "传中", carry: "持球推进", aerial_clearance: "争顶解围", foot_pass: "门将脚下短传", foot_clearance: "门将大脚出球", throw_in: "界外球", recycle: "后场回传组织", back_pass: "回传", switch_pass: "转移传球", touchline: "边线出界", unknown: "类型待确认"})[type] || "其他动作"}</th><td>{count}</td></tr>)}</tbody></table></>}
          <h3>尚不能计算的指标</h3>
          <ul>{data.unavailable.map((item) => <li key={item}>{item}</li>)}</ul>
        </>
      ) : (
        <div className="layer-sequences">
          {items.length ? items.map((item) => (
            <article key={item.id}>
              <div className="layer-sequence-title"><strong>{item.title}</strong><button title="回看视频证据" aria-label={`回看${stamp(item.time)}`} onClick={() => onSeek(item.time, item)}><Play size={16} /></button></div>
              <span>{stamp(item.time)} · {item.status}</span>
              {item.formationEvidence && <div className="formation-open"><button className="secondary-button" onClick={() => onSeek(item.formationEvidence.time,item)} aria-label={`查看阵型分布 ${item.id}`}><ScanSearch size={16}/>查看阵型分布</button></div>}
              {item.thumbnailUrl && <button className="layer-evidence-cover" onClick={() => onSeek(item.time, item)} aria-label={`播放${item.title}`}><img src={item.thumbnailUrl} alt={item.title} loading="lazy" /></button>}
              <p>{item.summary}</p>
              {item.commentary && <p className="tactical-interpretation">{item.commentary.interpretation}</p>}
              {item.quality && <small>空间数据：{item.quality.reason}</small>}
              {item.actions && <div className="sequence-actions">{item.actions.map((action, index) => <span key={`${action.time_sec}-${index}`}>{stamp(action.time_sec)} {({shot: "射门候选", carry: "推进", short_pass: "短传", long_pass: "长传", through_ball: "直塞候选", cross: "传中候选"})[action.type] || "传球"}</span>)}</div>}
              {item.limitations?.length > 0 && <small>{item.limitations.join("；")}</small>}
            </article>
          )) : <p className="layer-note">当前素材没有满足条件的片段。</p>}
        </div>
      )}
    </section>
  );
}

export function TacticalCommentary({ item, onSeek }) {
  if (!item?.commentary) return null;
  return <section className="ai-review-detail tactical-commentary" aria-label="战术评语">
    <div className="ai-review-heading"><h2>{item.title}</h2><span>{item.status || item.visionStatusLabel}</span></div>
    <small>智能分析参考 · 基于切片画面 · 非专家评分</small>
    <dl>{[["observation", "画面观察"], ["interpretation", "战术解读"], ["advice", "复盘建议"], ["limitation", "判断边界"]].map(([key, label]) => <div key={key}><dt>{label}</dt><dd>{item.commentary[key]}</dd></div>)}</dl>
    {item.inspectedFrames?.length > 0 && <details className="evidence-details"><summary>复核依据 · {item.inspectedFrames.length} 张时序画面</summary>
      <div className="evidence-filmstrip">{item.inspectedFrames.map((frame, index) => <button key={index} onClick={() => onSeek(frame.time)} title={`回看${stamp(frame.time)}`}><img src={frame.url} alt={`时序证据${index + 1}`} loading="lazy" /><span>{stamp(frame.time)}</span></button>)}</div>
    </details>}
    {item.reviewClipUrl && <details className="evidence-details"><summary>完整上下文片段 · {stamp(item.evidenceStart)}–{stamp(item.evidenceEnd)}</summary><video key={item.id} src={item.reviewClipUrl} poster={item.thumbnailUrl} controls preload="none" playsInline /></details>}
  </section>;
}

export function EventReviewDetail({ event, onSeek }) {
  if (!event) return null;
  return (
    <section className="ai-review-detail" aria-label="关键节点审核详情">
      <div className="ai-review-heading"><h2>{event.title}</h2><span>{event.visionStatusLabel}</span></div>
      {event.model && <small>{event.model} · {event.frameCount}张全景图{event.focusFrameCount > 0 ? ` + ${event.focusFrameCount}张门将局部图` : ""}</small>}
      {event.reviewerLabel && <small>智能复核 · 查验{event.reviewedFrameCount}张来源抽帧 · 非人工专家标注</small>}
      <dl>
        {event.reviewerLabel && <div><dt>原始候选</dt><dd>{event.originalCandidate ? `${event.originalCandidate.title} · ${event.originalCandidate.time.toFixed(2)}秒` : "从既有切片中补充的遗漏事件"}</dd></div>}
        <div><dt>候选观察</dt><dd>{event.observation || event.summary}</dd></div>
        {event.analysis && <div><dt>战术解读</dt><dd>{event.analysis}</dd></div>}
        {event.commentary?.advice && <div><dt>复盘建议</dt><dd>{event.commentary.advice}</dd></div>}
        {event.limitation && <div><dt>待确认部分</dt><dd>{event.limitation}</dd></div>}
      </dl>
      {event.inspectedFrames?.length > 0 && <details className="evidence-details">
        <summary>本次实际查验的来源帧 · {event.inspectedFrames.length}张</summary>
        <div className="evidence-filmstrip">{event.inspectedFrames.map((frame, index) => <button key={`${frame.url}-${index}`} onClick={() => onSeek(frame.time)} title={`回看 ${frame.time.toFixed(2)}秒`}><img src={frame.url} alt={`已查验帧${index + 1}`} loading="lazy" /><span>{frame.time.toFixed(2)}s</span></button>)}</div>
      </details>}
      {event.modelOpinions?.length > 0 && <details className="evidence-details">
        <summary>未采纳的模型原始意见</summary>
        {event.modelOpinions.map((opinion) => <section key={opinion.stage}><h3>{opinion.stage}</h3><p>{opinion.observation}</p><p>{opinion.analysis}</p></section>)}
      </details>}
      {event.reviewClipUrl && <details key={event.id} className="evidence-details">
        <summary><ScanSearch size={16} />审核片段与逐帧证据 · {stamp(event.evidenceStart)}–{stamp(event.evidenceEnd ?? event.end)}</summary>
        <video src={event.reviewClipUrl} poster={event.thumbnailUrl} controls preload="metadata" playsInline />
        <div className="evidence-filmstrip">{event.evidenceFrames.map((frame, index) => (
          <button key={frame.url} title={`回看原片 ${frame.time.toFixed(2)}秒`} onClick={() => onSeek(frame.time)}>
            <img src={frame.url} alt={`证据帧${index + 1}`} loading="lazy" /><span>{frame.time.toFixed(2)}s</span>
          </button>
        ))}</div>
      </details>}
    </section>
  );
}
