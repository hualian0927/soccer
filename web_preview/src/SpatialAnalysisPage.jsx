import { useEffect, useRef, useState } from "react";
import { ArrowLeft, BarChart3, Download, Info, Maximize, Pause, Play, SkipBack, SkipForward, Video, Volume2 } from "lucide-react";
import SwitchableCameraStage from "./SwitchableCameraStage";


const EVENT_GROUP_NAMES = {
  "L1-01": "射门与结果",
  "L1-02": "定位球",
  "L1-03": "接球、持球与传球",
  "L1-04": "球权转换",
  "L1-05": "防守动作",
  "L1-06": "门将事件",
  "L1-07": "比赛重启",
};


const formatTime = (seconds) => {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
};

const metric = (value, unit = "m") => (Number.isFinite(value) ? `${value.toFixed(1)} ${unit}` : "—");

function ZoneDistribution({ team, color, title, labels, values }) {
  const total = values?.reduce((sum, value) => sum + (Number(value) || 0), 0) || 0;
  return (
    <div className={`spatial-distribution ${color}`}>
      <span>{title}</span>
      <div className="spatial-distribution-track" aria-label={`${team}${title}`}>
        {labels.map((label, index) => (
          <span
            key={label}
            title={`${label} ${values?.[index] ?? 0} 人`}
            style={{ width: `${total ? ((values?.[index] || 0) / total) * 100 : 0}%` }}
          />
        ))}
      </div>
      <small>{labels.map((label, index) => `${label} ${values?.[index] ?? 0}`).join(" · ")}</small>
    </div>
  );
}


export default function SpatialAnalysisPage({ project, asset, onBack }) {
  const videoRef = useRef(null);
  const focusStageRef = useRef(null);
  const pendingSeekRef = useRef(null);
  const resumeAfterLoadRef = useRef(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [videoError, setVideoError] = useState(false);
  const [videoMode, setVideoMode] = useState(asset.analysisSrc ? "analysis" : "source");
  const [mainCamera, setMainCamera] = useState("A");
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(1);
  const [sideView, setSideView] = useState("events");
  const [eventGroup, setEventGroup] = useState("all");
  const report = asset.spatialAnalysis;
  const hasBall = Boolean(report?.summary?.ball_observed_frames);
  const l1Events = report?.l1?.events || [];
  const hasL1 = Boolean(report?.l1);
  const windows = report?.windows || [];
  const duration = report?.summary?.duration_sec || asset.duration || 60;
  const visibleEvents = eventGroup === "all" ? l1Events : l1Events.filter((event) => event.l1_group === eventGroup);
  const currentEvent = l1Events.find((event) => Math.abs(event.timestamp_sec - currentTime) < 0.6);
  const selectedWindow = windows.find((window) => currentTime >= window.start_sec && currentTime < window.end_sec)
    || windows[windows.length - 1];

  useEffect(() => {
    pendingSeekRef.current = null;
    resumeAfterLoadRef.current = false;
    setCurrentTime(0);
    setVideoError(false);
    setVideoMode(asset.analysisSrc ? "analysis" : "source");
    setMainCamera("A");
    setIsPlaying(false);
    setSideView("events");
    setEventGroup("all");
  }, [asset.id]);

  const seek = (seconds) => {
    if (videoRef.current) videoRef.current.currentTime = Math.min(Math.max(seconds, 0), duration);
    setCurrentTime(seconds);
  };

  const switchVideoMode = (mode) => {
    if (mode === videoMode) return;
    const oldSource = videoMode === "analysis" ? asset.analysisSrc : asset.src;
    const newSource = mode === "analysis" ? asset.analysisSrc : asset.src;
    if (oldSource !== newSource) {
      pendingSeekRef.current = videoRef.current?.currentTime ?? currentTime;
      resumeAfterLoadRef.current = Boolean(videoRef.current && !videoRef.current.paused);
    }
    setVideoMode(mode);
    setVideoError(false);
  };

  const togglePlayback = () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) video.play().catch(() => setVideoError(true));
    else video.pause();
  };

  if (!report) {
    return <main className="spatial-page"><div className="spatial-empty">二维空间报告尚未载入。</div></main>;
  }

  const blue = selectedWindow?.teams?.blue;
  const red = selectedWindow?.teams?.red;
  const vision = selectedWindow?.vision;

  return (
    <main className="spatial-page">
      <div className="analysis-header spatial-page-header">
        <button className="back-button" onClick={onBack}><ArrowLeft size={18} />返回项目</button>
        <div className="analysis-title">
          <span className="spatial-header-icon"><BarChart3 size={21} /></span>
          <div><h1>{project.title}</h1><p>{asset.title} · {formatTime(duration)} · 双机位原画面与球场投影</p></div>
        </div>
        <a className="spatial-download" href={asset.reportUrl} download="spatial_analysis_report.json"><Download size={16} />下载 JSON</a>
      </div>

      <div className="spatial-kpi-band">
        <div className="spatial-kpi-label"><strong>空间结构总览</strong><span>{hasBall ? "双机位视频 · 球员与足球坐标" : "算法量化为主 · 模型解释为辅"}</span></div>
        <div><small>有效轨迹</small><strong>{report.summary.source_frames} 帧</strong></div>
        <div><small>{hasL1 ? "关键事件候选" : "分析窗口"}</small><strong>{hasL1 ? `${l1Events.length} 个` : `${report.summary.windows} 段`}</strong></div>
        <div><small>蓝队中位宽度</small><strong>{metric(report.summary.teams.blue.width_m)}</strong></div>
        <div><small>红队中位宽度</small><strong>{metric(report.summary.teams.red.width_m)}</strong></div>
        <div><small>{hasBall ? "球观测覆盖" : "模型复核"}</small><strong>{hasBall ? `${Math.round(report.summary.ball_coverage_ratio * 100)}%` : `${report.summary.vision_reviewed_windows}/${report.summary.windows}`}</strong></div>
      </div>

      <div className="spatial-workspace">
        <div className="spatial-main-column">
          <div className="spatial-evidence-heading"><Video size={17} /><strong>{hasBall ? "原画面与二维投影" : "二维俯视证据"}</strong><span>{hasBall ? "左右原始机位 + 右侧投影；球位置以逐帧坐标复核" : "红蓝点为可见球员；画面中没有足球轨迹"}</span></div>
          {asset.analysisSrc && <div className="spatial-video-modes" role="group" aria-label="视频版本">
            <button type="button" className={videoMode === "analysis" ? "active" : ""} onClick={() => switchVideoMode("analysis")}>全景分析版</button>
            {asset.cameraLayout && <button type="button" className={videoMode === "focus" ? "active" : ""} onClick={() => switchVideoMode("focus")}>切换视角版</button>}
            <button type="button" className={videoMode === "source" ? "active" : ""} onClick={() => switchVideoMode("source")}>原始拼接</button>
            {videoMode === "focus" && asset.focusPreviewSrc && <a href={asset.focusPreviewSrc} download="主机位分析预览.mp4"><Download size={14} />下载主机位预览</a>}
          </div>}
          <div className={`spatial-video-stage${videoMode === "focus" ? " spatial-video-stage-focus" : videoMode === "analysis" ? " spatial-video-stage-analysis" : hasBall ? " spatial-video-stage-wide" : ""}`}>
            <video
              ref={videoRef}
              src={videoMode === "analysis" ? asset.analysisSrc : asset.src}
              poster={videoMode === "analysis" ? asset.analysisPoster : asset.poster}
              controls={videoMode !== "focus"}
              className={videoMode === "focus" ? "spatial-focus-source-video" : ""}
              playsInline
              preload="metadata"
              onLoadedMetadata={(event) => {
                if (pendingSeekRef.current !== null) {
                  event.currentTarget.currentTime = pendingSeekRef.current;
                  pendingSeekRef.current = null;
                }
                if (resumeAfterLoadRef.current) {
                  resumeAfterLoadRef.current = false;
                  event.currentTarget.play().catch(() => setIsPlaying(false));
                }
              }}
              onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              onError={() => setVideoError(true)}
            />
            {videoMode === "focus" && <SwitchableCameraStage videoRef={videoRef} posterUrl={asset.poster} stageRef={focusStageRef}
              mainCamera={mainCamera} onSwap={() => setMainCamera((camera) => camera === "A" ? "B" : "A")}
              currentTime={currentTime} duration={duration} isPlaying={isPlaying}
              onPlayPause={togglePlayback} onSeek={seek} />}
            {videoError && <div className="spatial-video-error">视频未能加载，请检查本地媒体接口。</div>}
          </div>
          <div className="spatial-controls">
            <div className="spatial-time-row"><span>{formatTime(currentTime)} / {formatTime(duration)}</span><span>点击下方节点可跳转</span></div>
            {videoMode === "focus" && <div className="spatial-focus-controls">
              <button type="button" title="后退 5 秒" aria-label="后退 5 秒" onClick={() => seek(currentTime - 5)}><SkipBack size={18} /></button>
              <button type="button" title={isPlaying ? "暂停" : "播放"} aria-label={isPlaying ? "暂停" : "播放"} onClick={togglePlayback}>{isPlaying ? <Pause size={20} /> : <Play size={20} />}</button>
              <button type="button" title="前进 5 秒" aria-label="前进 5 秒" onClick={() => seek(currentTime + 5)}><SkipForward size={18} /></button>
              <span className="spatial-focus-controls-spacer" />
              <Volume2 size={17} aria-hidden="true" />
              <input type="range" min="0" max="1" step="0.05" value={volume} aria-label="音量"
                onChange={(event) => { const next = Number(event.target.value); setVolume(next); if (videoRef.current) videoRef.current.volume = next; }} />
              <button type="button" title="全屏" aria-label="全屏" onClick={() => focusStageRef.current?.requestFullscreen?.()}><Maximize size={18} /></button>
            </div>}
            <div className="spatial-timeline">
              <input
                type="range"
                min="0"
                max={duration}
                step="0.04"
                value={Math.min(currentTime, duration)}
                onChange={(event) => seek(Number(event.target.value))}
                aria-label="视频时间轴"
              />
              <div className="spatial-ticks">
                {windows.map((window) => (
                  <button
                    key={window.id}
                    type="button"
                    className={selectedWindow?.id === window.id ? "active" : ""}
                    onClick={() => seek(window.start_sec + 0.1)}
                    title={`${formatTime(window.start_sec)} 至 ${formatTime(window.end_sec)}`}
                    aria-label={`跳转到 ${formatTime(window.start_sec)}`}
                  >
                    <span>{formatTime(window.start_sec)}</span>
                  </button>
                ))}
              </div>
              {hasL1 && <div className="spatial-event-markers" aria-label="关键事件节点">
                {l1Events.map((event) => (
                  <button
                    key={event.event_id}
                    type="button"
                    className={`spatial-event-marker ${["L1-02", "L1-07"].includes(event.l1_group) ? "set-piece" : "on-ball"}${currentEvent?.event_id === event.event_id ? " active" : ""}`}
                    style={{ left: `${Math.min(98, Math.max(2, event.timestamp_sec / duration * 100))}%` }}
                    title={`${formatTime(event.timestamp_sec)} ${event.label_zh} · 待复核`}
                    aria-label={`跳转到 ${formatTime(event.timestamp_sec)} ${event.label_zh}候选`}
                    onClick={() => seek(event.timestamp_sec)}
                  />
                ))}
              </div>}
            </div>
          </div>

          {hasL1 && <section className="spatial-l1-now" aria-live="polite">
            <span className="spatial-l1-now-label">当前关键事件</span>
            {currentEvent ? <div><strong>{currentEvent.label_zh}</strong><span>{formatTime(currentEvent.timestamp_sec)} · {EVENT_GROUP_NAMES[currentEvent.l1_group]} · 待复核</span><p>{currentEvent.summary_zh}</p></div>
              : <p>当前时间没有事件候选；可点击时间轴标记或右侧候选查看对应画面。</p>}
          </section>}

          {selectedWindow && (
            <section className="spatial-current" aria-live="polite">
              <div className="spatial-section-heading">
                <div><span>当前时段</span><h2>{formatTime(selectedWindow.start_sec)}–{formatTime(selectedWindow.end_sec)} 站位对比</h2></div>
                <span>球场坐标 · 米制近似</span>
              </div>
              <p className="spatial-algorithm-finding">{selectedWindow.algorithm_finding_zh}</p>
              {hasBall && <div className="spatial-ball-summary"><span>足球观测 <strong>{selectedWindow.ball.observed_ball_frames} 帧</strong></span><span>主要区域 <strong>{selectedWindow.ball.top_zone || "未知"}</strong></span><span>近球蓝队 <strong>{selectedWindow.ball.nearest_team_within_3m_frames.blue || 0} 帧</strong></span><span>近球红队 <strong>{selectedWindow.ball.nearest_team_within_3m_frames.red || 0} 帧</strong></span><small>{selectedWindow.ball_finding_zh}</small></div>}
              <div className="spatial-team-comparison">
                {[["蓝队", blue, "blue"], ["红队", red, "red"]].map(([name, team, color]) => (
                  <div key={color} className={`spatial-team-metrics ${color}`}>
                    <div><span className="spatial-team-swatch" /> <strong>{name}</strong><small>中位可见 {team?.visible_players ?? "—"} 人</small></div>
                    <dl>
                      <div><dt>横向宽度</dt><dd>{metric(team?.width_m)}</dd></div>
                      <div><dt>纵向纵深</dt><dd>{metric(team?.depth_m)}</dd></div>
                      <div><dt>纵向重心</dt><dd>{metric(team?.centroid_y_m)}</dd></div>
                      <div><dt>最大可见线距</dt><dd>{metric(team?.largest_y_gap_m)}</dd></div>
                    </dl>
                  </div>
                ))}
              </div>
              <div className="spatial-distribution-section">
                <h3>可见球员分布 <small>人数中位数，区域按球场坐标等分</small></h3>
                <div className="spatial-distribution-grid">
                  {[['蓝队', blue, 'blue'], ['红队', red, 'red']].map(([name, team, color]) => (
                    <div key={color}>
                      <strong>{name}</strong>
                      <ZoneDistribution team={name} color={color} title="纵向三区" labels={['1区', '2区', '3区']} values={team?.third_counts} />
                      <ZoneDistribution team={name} color={color} title="横向通道" labels={['左', '中', '右']} values={team?.channel_counts} />
                    </div>
                  ))}
                </div>
              </div>
              <div className="spatial-aux-metrics">
                <span>3 米内跨队近邻对中位数 <strong>{selectedWindow.median_close_opponent_pairs_3m}</strong></span>
                <span>有效观测 <strong>{selectedWindow.observed_frames} 帧</strong></span>
                <span>最近对手距离中位数 <strong>{metric(selectedWindow.median_nearest_opponent_m)}</strong></span>
              </div>
            </section>
          )}

          <section className="spatial-highlights">
            <div className="spatial-section-heading"><div><span>阶段变化</span><h2>本段可复核的空间结论</h2></div></div>
            {(report.highlights || []).map((highlight) => (
              <button key={highlight.id} onClick={() => seek(highlight.start_sec + 0.1)}>
                <span className="spatial-highlight-time">{formatTime(highlight.start_sec)}–{formatTime(highlight.end_sec)}</span>
                <span><strong>{highlight.title_zh}</strong><small>{highlight.summary_zh}</small></span>
                <Play size={17} />
              </button>
            ))}
          </section>

          <section className="spatial-limits">
            <div className="spatial-section-heading"><div><span>证据边界</span><h2>本段不能据此确认的事件</h2></div></div>
            <p>{hasBall ? `已提供逐帧足球坐标。本片可信连续出界证据 ${report.summary.out_of_bounds_evidence_count} 段，中圈开球候选 ${report.summary.set_piece_restart_candidates} 段，确认 ${report.summary.set_piece_restarts_confirmed} 次；近球距离不等于控球，中圈开球不计作角球或任意球。` : "没有足球轨迹与原始双机位比赛画面，不能统计传球、射门、角球、进球、球权转换或门将扑救；也不输出确定的 4-3-3、4-2-3-1 阵型。"}</p>
            <details><summary>查看数据质量与球场标定说明</summary><ul>{report.limitations_zh.map((item) => <li key={item}>{item}</li>)}</ul></details>
          </section>
        </div>

        <aside className="spatial-side-panel">
          <div className="spatial-side-head"><h2>{hasL1 && sideView === "events" ? "关键事件候选" : "时段分析"}</h2><span>{hasL1 && sideView === "events" ? `${l1Events.length} 个待复核` : `${windows.length} 个窗口`}</span></div>
          {hasL1 && <div className="spatial-side-tabs" role="tablist" aria-label="分析视图">
            <button type="button" role="tab" aria-selected={sideView === "events"} className={sideView === "events" ? "active" : ""} onClick={() => setSideView("events")}>关键事件</button>
            <button type="button" role="tab" aria-selected={sideView === "windows"} className={sideView === "windows" ? "active" : ""} onClick={() => setSideView("windows")}>空间时段</button>
          </div>}
          {hasL1 && sideView === "events" && <>
            <div className="spatial-event-filter">
              <label htmlFor="spatial-l1-group">事件类别</label>
              <select id="spatial-l1-group" value={eventGroup} onChange={(event) => setEventGroup(event.target.value)}>
                <option value="all">全部候选 ({l1Events.length})</option>
                {Object.entries(report.l1.summary.group_counts).map(([group, count]) => (
                  <option key={group} value={group}>{EVENT_GROUP_NAMES[group]} ({count})</option>
                ))}
              </select>
            </div>
            <div className="spatial-l1-event-list">
              {visibleEvents.length ? visibleEvents.map((event) => (
                <button type="button" key={event.event_id} className={currentEvent?.event_id === event.event_id ? "active" : ""} onClick={() => seek(event.timestamp_sec)}>
                  <span className="spatial-l1-event-time">{formatTime(event.timestamp_sec)}</span>
                  <span className="spatial-l1-event-content"><span><small>{EVENT_GROUP_NAMES[event.l1_group]}</small><em>待复核</em></span><strong>{event.label_zh}</strong><span className="spatial-l1-event-summary">{event.summary_zh}</span></span>
                </button>
              )) : <p className="spatial-l1-empty">这类事件在本片没有足够证据生成候选。</p>}
            </div>
            <div className="spatial-source-note"><Info size={16} /><span>这些是二维轨迹产生的候选，不是已确认比赛事件。射门、防守动作和门将扑救需原画面逐段核验；跨机位 ID 不用于传球关联。</span></div>
          </>}
          {(!hasL1 || sideView === "windows") && <>
          {selectedWindow && !hasBall && (
            <div className="spatial-vision-note">
              <div><span className="spatial-vision-indicator" /><strong>模型视觉解释</strong><small>待人工复核</small></div>
              <p>{vision?.summary_zh || "该时段尚未生成视觉解释。"}</p>
              {vision?.observations_zh?.length > 0 && <ul>{vision.observations_zh.map((item, index) => <li key={index}>{item}</li>)}</ul>}
              <small>{vision?.uncertainty_zh || "仅能观察可见的球员点位。"}</small>
            </div>
          )}
          <div className="spatial-window-list">
            {windows.map((window) => {
              const blueWidth = window.teams.blue.width_m;
              const redWidth = window.teams.red.width_m;
              const widthLead = blueWidth > redWidth + 2 ? "蓝队更宽" : redWidth > blueWidth + 2 ? "红队更宽" : "宽度接近";
              return (
                <button key={window.id} className={selectedWindow?.id === window.id ? "active" : ""} onClick={() => seek(window.start_sec + 0.1)}>
                  <span className="spatial-window-time">{formatTime(window.start_sec)}–{formatTime(window.end_sec)}</span>
                  <span className="spatial-window-copy"><strong>{widthLead}</strong><small>蓝 {metric(blueWidth)} · 红 {metric(redWidth)}</small></span>
                  <span className="spatial-window-dot" />
                </button>
              );
            })}
          </div>
          <div className="spatial-source-note"><Info size={16} /><span>{hasBall ? "足球位置、球员站位和两个原始机位同步；所有触球与定位球结论仍须原画面复核。" : "模型只读取了各时段的俯视抽帧。场上距离以坐标计算为准；模型文字不用于确认比赛事件。"}</span></div>
          </>}
        </aside>
      </div>
    </main>
  );
}
