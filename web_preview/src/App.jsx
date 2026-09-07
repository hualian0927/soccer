import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeft,
  BarChart3,
  Bell,
  CalendarDays,
  Check,
  ChevronDown,
  ChevronRight,
  CircleDot,
  Clock3,
  Database,
  FileBarChart,
  FileVideo,
  FolderOpen,
  Grid3X3,
  List,
  LoaderCircle,
  Maximize2,
  Menu,
  Pause,
  Play,
  Plus,
  Search,
  Settings,
  SkipBack,
  SkipForward,
  Sparkles,
  UploadCloud,
  Video,
  Volume2,
  X,
} from "lucide-react";
import { buildGeneratedEvents, demoAssets, demoEvents, demoProjects } from "./data";


const navItems = [
  { id: "assets", label: "素材库", icon: FolderOpen },
  { id: "projects", label: "项目", icon: Sparkles },
  { id: "reports", label: "报告", icon: FileBarChart },
];

const colorLabels = {
  green: "#34c987",
  orange: "#f59f45",
  blue: "#4c8dff",
  violet: "#a977ed",
  red: "#ee626d",
  cyan: "#39bfd2",
  yellow: "#e2bd45",
};


function formatTime(value) {
  const total = Math.max(0, Math.round(Number(value) || 0));
  const minutes = Math.floor(total / 60);
  const seconds = String(total % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}


function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "--";
  const units = ["B", "KB", "MB", "GB"];
  let size = bytes;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size.toFixed(index > 1 ? 1 : 0)} ${units[index]}`;
}


function readVideo(file) {
  return new Promise((resolve) => {
    const src = URL.createObjectURL(file);
    const video = document.createElement("video");
    video.preload = "metadata";
    video.muted = true;
    video.playsInline = true;
    video.src = src;
    let settled = false;

    const finish = (duration, poster = "") => {
      if (settled) return;
      settled = true;
      resolve({ src, duration: Number.isFinite(duration) ? duration : 60, poster });
    };

    video.onerror = () => finish(60);
    video.onloadedmetadata = () => {
      const duration = Number.isFinite(video.duration) ? video.duration : 60;
      video.currentTime = Math.min(Math.max(duration * 0.12, 0.1), Math.max(duration - 0.1, 0.1));
      video.onseeked = () => {
        try {
          const canvas = document.createElement("canvas");
          canvas.width = 960;
          canvas.height = 540;
          const context = canvas.getContext("2d");
          context.drawImage(video, 0, 0, canvas.width, canvas.height);
          finish(duration, canvas.toDataURL("image/jpeg", 0.82));
        } catch {
          finish(duration);
        }
      };
      window.setTimeout(() => finish(duration), 1800);
    };
  });
}


function Brand() {
  return (
    <div className="brand" aria-label="赛析足球技战术分析">
      <span className="brand-mark">析</span>
      <span className="brand-name">赛析</span>
      <span className="brand-product">足球技战术分析</span>
    </div>
  );
}


function AppHeader({ onMenu }) {
  return (
    <header className="app-header">
      <button className="icon-button mobile-menu" onClick={onMenu} aria-label="打开导航">
        <Menu size={21} />
      </button>
      <Brand />
      <div className="header-actions">
        <div className="pipeline-state"><span />视觉分析引擎就绪</div>
        <button className="icon-button notification-button" aria-label="通知">
          <Bell size={20} />
          <span className="notification-dot">2</span>
        </button>
        <button className="icon-button" aria-label="设置"><Settings size={20} /></button>
        <div className="avatar">LH</div>
      </div>
    </header>
  );
}


function Sidebar({ view, open, onNavigate, onClose }) {
  return (
    <>
      <aside className={`sidebar ${open ? "is-open" : ""}`}>
        <div className="sidebar-items">
          {navItems.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={`nav-button ${view === id || (id === "projects" && ["project", "analysis"].includes(view)) ? "active" : ""}`}
              onClick={() => {
                onNavigate(id);
                onClose();
              }}
            >
              <Icon size={22} />
              <span>{label}</span>
            </button>
          ))}
        </div>
        <div className="sidebar-version">LOCAL<br />PREVIEW</div>
      </aside>
      {open && <button className="sidebar-scrim" onClick={onClose} aria-label="关闭导航" />}
    </>
  );
}


function PageHeading({ icon: Icon, title, subtitle, action }) {
  return (
    <div className="page-heading">
      <div>
        <h1><Icon size={28} />{title}</h1>
        <p>{subtitle}</p>
      </div>
      {action}
    </div>
  );
}


function AssetCard({ asset, selected, inProject, onSelect, onPreview, onAdd }) {
  return (
    <article className={`asset-card ${selected ? "selected" : ""}`} onClick={() => onSelect(asset.id)}>
      <div className="asset-poster">
        {asset.poster ? (
          <img src={asset.poster} alt="" />
        ) : (
          <div className="poster-fallback"><Video size={36} /></div>
        )}
        <span className="asset-kind"><CircleDot size={13} />足球</span>
        <span className="asset-duration"><Clock3 size={13} />{formatTime(asset.duration)}</span>
        {asset.status && (
          <span className={`asset-status ${asset.status === "分析中" ? "processing" : ""}`}>
            {asset.status === "分析中" ? <LoaderCircle size={13} /> : <Check size={13} />}
            {asset.status}
          </span>
        )}
      </div>
      <div className="asset-card-body">
        <div className="asset-title-row">
          <h3>{asset.title}</h3>
          <button className="compact-icon" aria-label="素材详情"><ChevronRight size={18} /></button>
        </div>
        <p className="filename">{asset.filename}</p>
        <div className="asset-meta">
          <span>{asset.createdAt}</span>
          <span>{asset.size}</span>
        </div>
        <div className="asset-actions" onClick={(event) => event.stopPropagation()}>
          <button className="secondary-button" onClick={() => onPreview(asset)}><Play size={16} />预览</button>
          <button className="primary-button" onClick={() => onAdd(asset.id)}>
            {inProject ? <Check size={17} /> : <Plus size={17} />}
            {inProject ? "已加入项目" : "添加到项目"}
          </button>
        </div>
      </div>
    </article>
  );
}


function UploadZone({ onFiles }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  return (
    <button
      className={`upload-zone ${dragging ? "dragging" : ""}`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        onFiles(event.dataTransfer.files);
      }}
    >
      <span className="upload-icon"><UploadCloud size={28} /></span>
      <strong>上传比赛视频</strong>
      <span>拖拽文件到此处，或点击选择 MP4 / MOV / AVI</span>
      <small>文件保存在本机，并提交 SoccerNetGSR 与 L1 事件分析流水线</small>
      <input
        ref={inputRef}
        type="file"
        accept="video/mp4,video/quicktime,video/x-msvideo,video/*"
        multiple
        hidden
        onChange={(event) => onFiles(event.target.files)}
      />
    </button>
  );
}


function AssetInspector({ asset, projects, onProjectChange, projectId, onAdd, onCreate }) {
  if (!asset) {
    return (
      <aside className="inspector empty-inspector">
        <CircleDot size={28} />
        <h2>素材候选区</h2>
        <p>选择一段视频后，可将它加入现有项目或新建分析项目。</p>
      </aside>
    );
  }
  return (
    <aside className="inspector">
      <div className="inspector-heading">
        <div>
          <span className="eyebrow">当前素材</span>
          <h2>{asset.title}</h2>
        </div>
        <span className="score-chip">{asset.status}</span>
      </div>
      <div className="inspector-preview">
        {asset.poster ? <img src={asset.poster} alt="" /> : <Video size={38} />}
      </div>
      <dl className="metadata-list">
        <div><dt>时长</dt><dd>{formatTime(asset.duration)}</dd></div>
        <div><dt>大小</dt><dd>{asset.size}</dd></div>
        <div><dt>类型</dt><dd>{asset.tags?.join(" · ")}</dd></div>
        {asset.analysisStage && <div><dt>处理阶段</dt><dd>{asset.analysisStage}</dd></div>}
      </dl>
      {asset.status === "分析中" && (
        <div className="analysis-progress" aria-label={`分析进度 ${asset.analysisProgress || 0}%`}>
          <i style={{ width: `${asset.analysisProgress || 0}%` }} />
          <span>{asset.analysisProgress || 0}%</span>
        </div>
      )}
      <label className="field-label" htmlFor="project-select">添加至项目</label>
      <div className="select-wrap">
        <select id="project-select" value={projectId} onChange={(event) => onProjectChange(event.target.value)}>
          {projects.map((project) => <option key={project.id} value={project.id}>{project.title}</option>)}
        </select>
        <ChevronDown size={17} />
      </div>
      <button className="primary-button full-button" onClick={() => onAdd(asset.id, projectId)}>
        <Plus size={18} />添加到所选项目
      </button>
      <button className="secondary-button full-button" onClick={() => onCreate(asset.id)}>
        <FolderOpen size={18} />新建项目并添加
      </button>
      <div className="inspector-note">
        <Sparkles size={17} />
        <span>进入项目后可查看 L1 事件时间轴、定位球分类和独立证据片段。</span>
      </div>
    </aside>
  );
}


function AssetsPage({ assets, projects, onUpload, onOpenAsset, onAddToProject, onCreateProject }) {
  const [selectedId, setSelectedId] = useState(assets[0]?.id);
  const [query, setQuery] = useState("");
  const [layout, setLayout] = useState("grid");
  const [targetProjectId, setTargetProjectId] = useState(projects[0]?.id || "");
  const selectedAsset = assets.find((asset) => asset.id === selectedId) || assets[0];
  const filtered = assets.filter((asset) => `${asset.title} ${asset.filename}`.toLowerCase().includes(query.toLowerCase()));

  const selectForAdd = (assetId) => {
    setSelectedId(assetId);
    document.querySelector(".inspector")?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div className="workspace-with-inspector">
      <main className="page-main assets-main">
        <PageHeading
          icon={Database}
          title="比赛素材库"
          subtitle="上传原始或处理后视频，并组织到技战术分析项目中"
        />
        <UploadZone onFiles={onUpload} />
        <section className="section-block">
          <div className="section-heading">
            <div><h2>视频素材</h2><span>{filtered.length} 个素材</span></div>
            <div className="view-switch">
              <button className={layout === "grid" ? "active" : ""} onClick={() => setLayout("grid")} aria-label="网格视图"><Grid3X3 size={18} /></button>
              <button className={layout === "list" ? "active" : ""} onClick={() => setLayout("list")} aria-label="列表视图"><List size={19} /></button>
            </div>
          </div>
          <div className="toolbar">
            <label className="search-box">
              <Search size={18} />
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索比赛或文件名" />
            </label>
            <span className="toolbar-status"><span />本地素材可用</span>
          </div>
          <div className={`asset-grid ${layout === "list" ? "list-layout" : ""}`}>
            {filtered.map((asset) => (
              <AssetCard
                key={asset.id}
                asset={asset}
                selected={selectedAsset?.id === asset.id}
                inProject={projects.some((project) => project.assetIds.includes(asset.id))}
                onSelect={setSelectedId}
                onPreview={onOpenAsset}
                onAdd={selectForAdd}
              />
            ))}
          </div>
        </section>
      </main>
      <AssetInspector
        asset={selectedAsset}
        projects={projects}
        projectId={targetProjectId}
        onProjectChange={setTargetProjectId}
        onAdd={onAddToProject}
        onCreate={onCreateProject}
      />
    </div>
  );
}


function ProjectCard({ project, assets, onOpen }) {
  const projectAssets = project.assetIds.map((id) => assets.find((asset) => asset.id === id)).filter(Boolean);
  const cover = projectAssets[0];
  return (
    <button className="project-card" onClick={() => onOpen(project.id)}>
      <div className="project-cover">
        {cover?.poster ? <img src={cover.poster} alt="" /> : <div className="poster-fallback"><FolderOpen size={40} /></div>}
        <span className="project-type">{project.type}</span>
        <div className="project-count"><FileVideo size={15} />{projectAssets.length} 个视频</div>
      </div>
      <div className="project-card-body">
        <div>
          <h3>{project.title}</h3>
          <p>{project.subtitle}</p>
        </div>
        <ChevronRight size={20} />
      </div>
      <div className="project-footer"><CalendarDays size={15} />创建于 {project.createdAt}</div>
    </button>
  );
}


function ProjectsPage({ projects, assets, onOpen, onNew }) {
  const [filter, setFilter] = useState("全部项目");
  const tabs = ["全部项目", "技战术分析", "个人技术"];
  const visible = filter === "全部项目" ? projects : projects.filter((item) => item.type === filter);
  return (
    <main className="page-main projects-main">
      <PageHeading
        icon={FolderOpen}
        title="项目列表"
        subtitle="按比赛组织素材、技战术节点和分析成果"
        action={<button className="primary-button heading-action" onClick={onNew}><Plus size={18} />新建项目</button>}
      />
      <div className="segmented-tabs" role="tablist">
        {tabs.map((tab) => <button key={tab} className={filter === tab ? "active" : ""} onClick={() => setFilter(tab)}>{tab}</button>)}
      </div>
      <div className="project-grid">
        {visible.map((project) => <ProjectCard key={project.id} project={project} assets={assets} onOpen={onOpen} />)}
      </div>
    </main>
  );
}


function ProjectPage({ project, assets, onBack, onAnalyze, onPending }) {
  const videos = project.assetIds.map((id) => assets.find((asset) => asset.id === id)).filter(Boolean);
  return (
    <main className="page-main project-detail-main">
      <button className="back-button" onClick={onBack}><ArrowLeft size={18} />返回项目列表</button>
      <div className="project-detail-header">
        <div>
          <span className="eyebrow">L1 事件分析项目</span>
          <h1>{project.title}</h1>
          <p>{project.subtitle}</p>
        </div>
        <div className="project-kpis">
          <div><strong>{videos.length}</strong><span>比赛视频</span></div>
          <div><strong>{videos.reduce((sum, item) => sum + (item.events?.length || (item.status === "分析完成" ? 9 : 0)), 0)}</strong><span>关键节点</span></div>
          <div><strong>{videos.filter((item) => item.status === "分析完成").length}</strong><span>分析完成</span></div>
        </div>
      </div>
      <div className="project-section-title">
        <div><h2>项目视频</h2><p>点击视频进入 L1 事件分析工作台</p></div>
      </div>
      <div className="project-video-list">
        {videos.map((asset) => (
          <button
            key={asset.id}
            className="project-video-row"
            onClick={() => asset.status === "分析完成" ? onAnalyze(asset.id) : onPending(asset)}
          >
            <div className="project-video-thumb">
              {asset.poster ? <img src={asset.poster} alt="" /> : <Video size={30} />}
              <span><Play size={18} fill="currentColor" /></span>
            </div>
            <div className="project-video-info">
              <h3>{asset.title}</h3>
              <p>{asset.filename}</p>
              <div><span><Clock3 size={14} />{formatTime(asset.duration)}</span><span><Database size={14} />{asset.size}</span></div>
              {asset.status === "分析中" && (
                <div className="row-progress"><i style={{ width: `${asset.analysisProgress || 0}%` }} /><span>{asset.analysisStage}</span></div>
              )}
            </div>
            <span className={`analysis-state ${asset.status === "分析完成" ? "done" : ""} ${asset.status === "分析中" ? "processing" : ""}`}>{asset.status}</span>
            <ChevronRight size={22} />
          </button>
        ))}
        {!videos.length && <div className="empty-state"><FileVideo size={38} /><h3>项目中还没有视频</h3><p>返回素材库选择视频并添加到此项目。</p></div>}
      </div>
    </main>
  );
}


function EventTag({ event }) {
  return <span className={`event-tag ${event.color}`}>{event.type}</span>;
}


function AnalysisTimeline({ events, duration, currentTime, onSeek, activeEvent }) {
  const timelineRef = useRef(null);
  const safeDuration = Math.max(duration, 1);
  return (
    <div
      ref={timelineRef}
      className="analysis-timeline"
      onClick={(click) => {
        const rect = timelineRef.current.getBoundingClientRect();
        onSeek(((click.clientX - rect.left) / rect.width) * safeDuration);
      }}
    >
      <div className="timeline-track" />
      <div className="timeline-progress" style={{ width: `${Math.min(100, currentTime / safeDuration * 100)}%` }} />
      {events.map((event) => (
        <button
          key={event.id}
          className={`timeline-marker ${event.color} ${activeEvent?.id === event.id ? "active" : ""}`}
          style={{ left: `${Math.min(99.3, event.time / safeDuration * 100)}%` }}
          onClick={(click) => {
            click.stopPropagation();
            onSeek(event.time, event);
          }}
          aria-label={`${formatTime(event.time)} ${event.title}`}
          title={`${formatTime(event.time)} ${event.title}`}
        />
      ))}
      <span className="timeline-handle" style={{ left: `${Math.min(100, currentTime / safeDuration * 100)}%` }} />
    </div>
  );
}


const setPieceTypeOrder = ["corner", "free_kick", "goal_kick", "throw_in", "kickoff", "penalty"];
const setPieceTypeLabels = {
  corner: "角球",
  free_kick: "任意球",
  goal_kick: "球门球",
  throw_in: "界外球",
  kickoff: "中圈开球",
  penalty: "点球",
  unknown: "未分类",
};

const outcomeLabels = {
  retained: "开球队控制",
  opponent_control: "对方控制",
  controlled: "门将控制",
  success: "成功",
  possession_won: "夺回成功",
  unknown: "待确认",
};


function coordinateText(position) {
  if (!Array.isArray(position) || position.some((value) => value === null || value === undefined || value === "")) return "待确认";
  return `(${Number(position[0]).toFixed(1)}, ${Number(position[1]).toFixed(1)})`;
}


function SetPieceClipsView({ clips, events, poster, onOpenEvent }) {
  const setPieceEvents = events.filter((event) => event.category === "定位球");
  const counts = Object.fromEntries(setPieceTypeOrder.map((type) => [type, setPieceEvents.filter((event) => event.subtype === type).length]));
  return (
    <div className="set-piece-view">
      <div className="set-piece-intro">
        <div><span className="eyebrow">L1-02</span><h2>定位球分类与证据片段</h2><p>每次重启单独成片，保留开出前准备、足球运行和首次稳定落点。</p></div>
        <div className="set-piece-total"><strong>{setPieceEvents.length}</strong><span>定位球候选</span></div>
      </div>
      <div className="set-piece-stats">
        {setPieceTypeOrder.map((type) => (
          <div key={type} className={counts[type] ? "has-events" : ""}><span>{setPieceTypeLabels[type]}</span><strong>{counts[type]}</strong></div>
        ))}
      </div>
      {clips.length ? (
        <div className="set-piece-grid">
          {clips.map((clip, index) => {
            const event = setPieceEvents.find((item) => item.id === clip.event_id);
            return (
              <article className="set-piece-card" key={clip.clip_id || clip.event_id || index}>
                <div className="set-piece-video">
                  <video src={clip.clipUrl} poster={poster} controls preload="metadata" playsInline />
                  <span>{clip.typeLabel || setPieceTypeLabels[clip.event_subtype] || "定位球"}</span>
                </div>
                <div className="set-piece-card-body">
                  <div className="set-piece-card-title"><div><span>#{String(index + 1).padStart(2, "0")}</span><h3>{clip.label_zh || "定位球片段"}</h3></div><strong>{formatTime(clip.timestamp_sec)}</strong></div>
                  <p>{clip.summary_zh || "保留该次定位球的开出与落点证据，供后续专项分析。"}</p>
                  <dl className="set-piece-meta">
                    <div><dt>开出位置</dt><dd>{coordinateText(clip.startPosition)}</dd></div>
                    <div><dt>落点位置</dt><dd>{coordinateText(clip.endPosition)}</dd></div>
                    <div><dt>落点结果</dt><dd>{outcomeLabels[clip.outcome] || "待确认"}</dd></div>
                    <div><dt>片段时长</dt><dd>{Number(clip.clip_duration_sec || 0).toFixed(1)} 秒</dd></div>
                  </dl>
                  {event && <button className="secondary-button" onClick={() => onOpenEvent(event)}><Play size={16} />回到完整视频</button>}
                </div>
              </article>
            );
          })}
        </div>
      ) : (
        <div className="set-piece-empty"><FileVideo size={36} /><h3>暂未导出定位球片段</h3><p>没有检测到定位球，或该素材仍在生成独立片段。</p></div>
      )}
    </div>
  );
}


function AnalysisPage({ project, asset, onBack }) {
  const videoRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [videoReady, setVideoReady] = useState(false);
  const [playbackError, setPlaybackError] = useState("");
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(asset.duration || 60);
  const [category, setCategory] = useState("全部");
  const [activeEvent, setActiveEvent] = useState(null);
  const [tab, setTab] = useState("L1事件时间轴");
  const [reviewStatuses, setReviewStatuses] = useState({});
  const baseEvents = useMemo(() => {
    if (Array.isArray(asset.events)) return asset.events;
    return asset.id === "match-overview" ? demoEvents : buildGeneratedEvents(duration || asset.duration);
  }, [asset.events, asset.id, asset.duration, duration]);
  const events = useMemo(
    () => baseEvents.map((event) => ({ ...event, reviewStatus: reviewStatuses[event.id] || event.reviewStatus || "candidate" })),
    [baseEvents, reviewStatuses],
  );
  const categories = ["全部", "射门", "定位球", "传接带", "球权转换", "防守干预", "门将事件"];
  const visibleEvents = category === "全部" ? events : events.filter((event) => event.category === category);
  const setPieceClips = Array.isArray(asset.setPieces) ? asset.setPieces : [];
  const setPieceCounts = useMemo(() => {
    const setPieceEvents = events.filter((event) => event.category === "定位球");
    return Object.fromEntries(
      setPieceTypeOrder.map((type) => [type, setPieceEvents.filter((event) => event.subtype === type).length]),
    );
  }, [events]);

  useEffect(() => {
    setCurrentTime(0);
    setActiveEvent(events[0] || null);
    setPlaying(false);
    setVideoReady(false);
    setPlaybackError("");
    setReviewStatuses({});
  }, [asset.id]);

  const seek = (time, event = null) => {
    if (videoRef.current) videoRef.current.currentTime = Math.min(Math.max(time, 0), duration || time);
    setCurrentTime(time);
    if (event) setActiveEvent(event);
  };

  const togglePlay = async () => {
    if (!videoRef.current) return;
    try {
      setPlaybackError("");
      if (videoRef.current.paused) {
        await videoRef.current.play();
      } else {
        videoRef.current.pause();
      }
    } catch {
      setPlaying(false);
      setPlaybackError("浏览器无法播放当前视频编码，请转换为 H.264 MP4 后重试。");
    }
  };

  const reloadVideo = () => {
    if (!videoRef.current) return;
    setPlaybackError("");
    setVideoReady(false);
    videoRef.current.load();
  };

  const toggleFullscreen = () => {
    videoRef.current?.requestFullscreen?.();
  };

  const reviewEvent = async (event, reviewStatus) => {
    setReviewStatuses((current) => ({ ...current, [event.id]: reviewStatus }));
    if (!asset.jobId) return;
    try {
      const response = await fetch(`/api/analysis/${asset.jobId}/events/${encodeURIComponent(event.id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reviewStatus }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "事件审核保存失败");
    } catch (error) {
      setReviewStatuses((current) => ({ ...current, [event.id]: event.reviewStatus || "candidate" }));
      window.alert(error.message);
    }
  };

  const nearestEvent = events.reduce((best, event) => (
    Math.abs(event.time - currentTime) < Math.abs((best?.time ?? Number.POSITIVE_INFINITY) - currentTime) ? event : best
  ), null);
  const displayEvent = events.find((event) => event.id === activeEvent?.id) || nearestEvent;

  return (
    <main className="analysis-page">
      <div className="analysis-header">
        <button className="back-button" onClick={onBack}><ArrowLeft size={18} />返回项目</button>
        <div className="analysis-title">
          <span className="football-symbol"><CircleDot size={28} /></span>
          <div><h1>{project.title}</h1><p>{asset.title} · {formatTime(duration)}</p></div>
        </div>
        <div className="analysis-score"><span>视频证据</span><strong>L1 事件层</strong><span>人工可复核</span></div>
      </div>
      <div className="analysis-tabs">
        {["L1事件时间轴", "定位球片段"].map((item) => (
          <button key={item} className={tab === item ? "active" : ""} onClick={() => setTab(item)}>
            {item === "L1事件时间轴" ? <BarChart3 size={17} /> : <FileVideo size={17} />}{item}
          </button>
        ))}
      </div>
      {tab === "L1事件时间轴" ? (
        <div className="analysis-workspace">
          <section className="player-column">
            <div className="video-stage">
              <video
                ref={videoRef}
                src={asset.src}
                poster={asset.poster}
                preload="metadata"
                playsInline
                onLoadedMetadata={(event) => setDuration(Number.isFinite(event.currentTarget.duration) ? event.currentTarget.duration : asset.duration)}
                onCanPlay={() => {
                  setVideoReady(true);
                  setPlaybackError("");
                }}
                onError={() => {
                  setVideoReady(false);
                  setPlaying(false);
                  setPlaybackError("视频加载失败，请检查文件是否存在或是否采用 H.264 编码。");
                }}
                onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
                onEnded={() => setPlaying(false)}
                onClick={togglePlay}
              />
              <button className="center-play" onClick={togglePlay} aria-label={playing ? "暂停" : "播放"}>
                {playing ? <Pause size={27} fill="currentColor" /> : <Play size={29} fill="currentColor" />}
              </button>
              {!videoReady && !playbackError && <span className="video-loading">正在准备视频...</span>}
              {playbackError && (
                <div className="playback-error" role="alert">
                  <FileVideo size={24} />
                  <strong>视频暂时无法播放</strong>
                  <span>{playbackError}</span>
                  <button onClick={reloadVideo}>重新加载</button>
                </div>
              )}
              <span className="video-analysis-badge"><Sparkles size={14} />L1 视觉事件候选</span>
              {displayEvent && (
                <div className="event-overlay">
                  <EventTag event={displayEvent} />
                  <strong>{displayEvent.title}</strong>
                  <span>{displayEvent.detail}</span>
                </div>
              )}
            </div>
            <div className="player-controls">
              <AnalysisTimeline events={events} duration={duration} currentTime={currentTime} onSeek={seek} activeEvent={activeEvent} />
              <div className="control-row">
                <div className="playback-controls">
                  <button onClick={() => seek(Math.max(0, currentTime - 5))} aria-label="后退五秒"><SkipBack size={19} /></button>
                  <button className="play-control" onClick={togglePlay} aria-label={playing ? "暂停" : "播放"}>
                    {playing ? <Pause size={20} fill="currentColor" /> : <Play size={20} fill="currentColor" />}
                  </button>
                  <button onClick={() => seek(Math.min(duration, currentTime + 5))} aria-label="前进五秒"><SkipForward size={19} /></button>
                  <span>{formatTime(currentTime)} / {formatTime(duration)}</span>
                </div>
                <div className="volume-controls"><Volume2 size={18} /><span className="volume-line"><i /></span><button onClick={toggleFullscreen} aria-label="全屏"><Maximize2 size={19} /></button></div>
              </div>
              <div className="timeline-legend">
                {categories.map((item) => (
                  <button key={item} className={category === item ? "active" : ""} onClick={() => setCategory(item)}>{item}{item !== "全部" && <span>{events.filter((event) => event.category === item).length}</span>}</button>
                ))}
              </div>
              <div className="timeline-set-piece-counts" aria-label="定位球分类次数">
                <strong>定位球分类次数</strong>
                <div>
                  {setPieceTypeOrder.map((type) => (
                    <span key={type} className={setPieceCounts[type] ? "has-events" : ""}>
                      {setPieceTypeLabels[type]}：<b>{setPieceCounts[type]}</b>
                    </span>
                  ))}
                </div>
              </div>
            </div>
            <div className="insight-strip">
              <div><span>L1 事件</span><strong>{events.length}</strong><small>可定位到视频证据</small></div>
              <div><span>定位球</span><strong>{events.filter((item) => item.category === "定位球").length}</strong><small>按六种重启类型统计</small></div>
              <div><span>事件组覆盖</span><strong>{new Set(events.map((item) => item.group).filter(Boolean)).size}/6</strong><small>射门至门将事件</small></div>
              <div><span>平均置信度</span><strong>{events.length ? Math.round(events.reduce((sum, item) => sum + item.confidence, 0) / events.length) : 0}%</strong><small>候选排序指标</small></div>
            </div>
          </section>
          <aside className="segments-panel">
            <div className="segments-header">
              <div><h2>L1 事件节点</h2><span><Sparkles size={14} />视觉候选</span></div>
              <p>按“谁、何时、何地、做了什么、结果如何”组织</p>
            </div>
            <div className="segment-filters">
              <div className="select-wrap">
                <select value={category} onChange={(event) => setCategory(event.target.value)}>{categories.map((item) => <option key={item}>{item}</option>)}</select>
                <ChevronDown size={16} />
              </div>
              <span>{visibleEvents.length} 个节点</span>
            </div>
            {displayEvent && (
              <div className="event-review-bar">
                <div><span>当前事件</span><strong>{displayEvent.reviewStatus === "confirmed" ? "已确认" : displayEvent.reviewStatus === "rejected" ? "已拒绝" : "待审核"}</strong></div>
                <button className={displayEvent.reviewStatus === "confirmed" ? "active confirm" : "confirm"} onClick={() => reviewEvent(displayEvent, "confirmed")}><Check size={15} />确认</button>
                <button className={displayEvent.reviewStatus === "rejected" ? "active reject" : "reject"} onClick={() => reviewEvent(displayEvent, "rejected")}><X size={15} />拒绝</button>
              </div>
            )}
            <div className="segment-list">
              {visibleEvents.map((event) => (
                <button key={event.id} className={`segment-row ${activeEvent?.id === event.id ? "active" : ""}`} onClick={() => seek(event.time, event)}>
                  <div className="segment-thumb">{asset.poster ? <img src={asset.poster} alt="" /> : <Video size={23} />}<span>{formatTime(event.time)}</span></div>
                  <div className="segment-copy">
                    <div><EventTag event={event} /><span>{formatTime(event.time)}–{formatTime(event.end)}</span></div>
                    <strong>{event.title}</strong>
                    <p>{event.summary}</p>
                    <div className="confidence"><i style={{ width: `${event.confidence}%` }} /><span>{event.confidence}%</span></div>
                  </div>
                  <ChevronRight size={18} />
                </button>
              ))}
            </div>
          </aside>
        </div>
      ) : (
        <SetPieceClipsView
          clips={setPieceClips}
          events={events}
          poster={asset.poster}
          onOpenEvent={(event) => { setTab("L1事件时间轴"); window.setTimeout(() => seek(event.time, event), 0); }}
        />
      )}
    </main>
  );
}


function ReportsPage({ projects }) {
  return (
    <main className="page-main reports-main">
      <PageHeading icon={FileBarChart} title="分析报告" subtitle="统一管理结构化数据、比赛摘要和高光清单" />
      <div className="report-table">
        <div className="report-row report-head"><span>报告名称</span><span>项目</span><span>格式</span><span>状态</span><span /></div>
        {projects.map((project) => (
          <div className="report-row" key={project.id}>
            <span><FileBarChart size={19} />{project.title} - 综合报告</span>
            <span>{project.title}</span><span>L1 JSON · CSV · 定位球清单</span><span className="report-ready">已生成</span><button className="secondary-button">查看</button>
          </div>
        ))}
      </div>
    </main>
  );
}


function NewProjectDialog({ open, onClose, onSubmit, defaultAsset }) {
  const [title, setTitle] = useState("");
  useEffect(() => {
    if (open) setTitle(defaultAsset ? `${defaultAsset.title}项目` : "新建足球技战术项目");
  }, [open, defaultAsset]);
  if (!open) return null;
  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={onClose}>
      <div className="dialog" role="dialog" aria-modal="true" aria-labelledby="dialog-title" onMouseDown={(event) => event.stopPropagation()}>
        <div className="dialog-title"><div><span className="dialog-icon"><FolderOpen size={22} /></span><div><h2 id="dialog-title">新建分析项目</h2><p>为比赛素材建立独立工作空间</p></div></div><button className="icon-button" onClick={onClose}><X size={19} /></button></div>
        <label className="field-label" htmlFor="project-title">项目名称</label>
        <input id="project-title" className="text-input" value={title} onChange={(event) => setTitle(event.target.value)} autoFocus />
        <label className="field-label" htmlFor="project-description">分析说明</label>
        <textarea id="project-description" className="text-area" defaultValue="识别射门、定位球、传接带、球权转换、防守干预和门将事件。" />
        <div className="dialog-actions"><button className="secondary-button" onClick={onClose}>取消</button><button className="primary-button" disabled={!title.trim()} onClick={() => onSubmit(title.trim())}><Plus size={17} />创建项目</button></div>
      </div>
    </div>
  );
}


function PreviewDialog({ asset, onClose, onAnalyze }) {
  if (!asset) return null;
  return (
    <div className="dialog-backdrop" onMouseDown={onClose}>
      <div className="preview-dialog" onMouseDown={(event) => event.stopPropagation()}>
        <div className="preview-dialog-header"><div><h2>{asset.title}</h2><p>{asset.filename}</p></div><button className="icon-button" onClick={onClose}><X size={20} /></button></div>
        <video src={asset.src} poster={asset.poster} controls autoPlay />
        <div className="preview-dialog-footer"><span>{formatTime(asset.duration)} · {asset.size}</span><button className="primary-button" onClick={() => onAnalyze(asset)}><BarChart3 size={17} />进入分析</button></div>
      </div>
    </div>
  );
}


export default function App() {
  const [view, setView] = useState("assets");
  const [assets, setAssets] = useState(demoAssets);
  const [projects, setProjects] = useState(demoProjects);
  const [activeProjectId, setActiveProjectId] = useState(demoProjects[0].id);
  const [activeAssetId, setActiveAssetId] = useState(demoAssets[0].id);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [dialogAssetId, setDialogAssetId] = useState(null);
  const [previewAsset, setPreviewAsset] = useState(null);
  const [toast, setToast] = useState("");
  const activeProject = projects.find((project) => project.id === activeProjectId) || projects[0];
  const activeAsset = assets.find((asset) => asset.id === activeAssetId) || assets[0];

  useEffect(() => {
    if (!toast) return undefined;
    const timer = window.setTimeout(() => setToast(""), 2600);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const navigate = (target) => {
    setView(target);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const updateAsset = (assetId, changes) => {
    setAssets((current) => current.map((asset) => asset.id === assetId ? { ...asset, ...changes } : asset));
  };

  const waitForAnalysis = async (assetId, jobId) => {
    while (true) {
      await new Promise((resolve) => window.setTimeout(resolve, 1200));
      const response = await fetch(`/api/analysis/${jobId}`);
      const job = await response.json();
      if (!response.ok) throw new Error(job.error || "无法读取本地分析任务");
      updateAsset(assetId, {
        status: job.status === "completed" ? "分析完成" : job.status === "failed" ? "分析失败" : "分析中",
        analysisStage: job.stage,
        analysisProgress: job.progress,
      });
      if (job.status === "completed") return job.result;
      if (job.status === "failed") throw new Error(job.error || "本地分析失败");
    }
  };

  const startLocalAnalysis = async (file, assetId) => {
    try {
      const response = await fetch("/api/analysis", {
        method: "POST",
        headers: {
          "Content-Type": file.type || "application/octet-stream",
          "X-File-Name": encodeURIComponent(file.name),
        },
        body: file,
      });
      const job = await response.json();
      if (!response.ok) throw new Error(job.error || "无法提交本地分析任务");
      updateAsset(assetId, {
        jobId: job.id,
        status: "分析中",
        analysisStage: job.stage,
        analysisProgress: job.progress,
      });
      const result = await waitForAnalysis(assetId, job.id);
      updateAsset(assetId, {
        status: "分析完成",
        analysisStage: result.cached ? "已复用本地完整分析结果" : "本地完整流水线处理完成",
        analysisProgress: 100,
        src: result.videoUrl,
        events: result.events,
        l1Summary: result.l1Summary,
        setPieces: result.setPieces,
        reportUrl: result.reportUrl,
        pipeline: result.pipeline,
        size: formatBytes(result.sizeBytes),
        tags: ["足球", "L1事件输出"],
      });
      setToast(result.cached ? "已匹配本地分析结果，可进入项目查看" : "视频技战术分析已完成");
    } catch (error) {
      updateAsset(assetId, {
        status: "分析失败",
        analysisStage: error.message,
        analysisProgress: 0,
      });
      setToast(`分析失败：${error.message}`);
    }
  };

  const handleUpload = async (fileList) => {
    const files = [...fileList].filter((file) => file.type.startsWith("video/") || /\.(mp4|mov|avi)$/i.test(file.name));
    if (!files.length) {
      setToast("请选择有效的视频文件");
      return;
    }
    setToast(`正在读取 ${files.length} 个视频...`);
    const uploaded = [];
    for (const file of files) {
      const metadata = await readVideo(file);
      const assetId = `upload-${Date.now()}-${uploaded.length}`;
      uploaded.push({
        id: assetId,
        title: file.name.replace(/\.[^.]+$/, ""),
        filename: file.name,
        createdAt: new Date().toLocaleString("zh-CN", { hour12: false }),
        duration: metadata.duration,
        size: formatBytes(file.size),
        status: "分析中",
        analysisStage: "正在上传到本机分析服务",
        analysisProgress: 1,
        poster: metadata.poster,
        src: metadata.src,
        tags: ["足球", "本地上传"],
        sourceFile: file,
      });
    }
    setAssets((current) => [...uploaded, ...current]);
    setToast(`已添加 ${uploaded.length} 个视频，正在执行本地分析`);
    for (const asset of uploaded) startLocalAnalysis(asset.sourceFile, asset.id);
  };

  const addToProject = (assetId, projectId) => {
    setProjects((current) => current.map((project) => (
      project.id === projectId && !project.assetIds.includes(assetId)
        ? { ...project, assetIds: [...project.assetIds, assetId] }
        : project
    )));
    setActiveProjectId(projectId);
    setToast("素材已加入项目");
  };

  const openCreateProject = (assetId = null) => {
    setDialogAssetId(assetId);
    setDialogOpen(true);
  };

  const createProject = (title) => {
    const id = `project-${Date.now()}`;
    const project = {
      id,
      title,
      subtitle: "本地上传比赛 · L1 视觉事件分析",
      createdAt: new Date().toLocaleString("zh-CN", { hour12: false }),
      assetIds: dialogAssetId ? [dialogAssetId] : [],
      type: "技战术分析",
    };
    setProjects((current) => [project, ...current]);
    setActiveProjectId(id);
    setDialogOpen(false);
    setToast("项目已创建");
    navigate("project");
  };

  const openProject = (projectId) => {
    setActiveProjectId(projectId);
    navigate("project");
  };

  const openAnalysis = (assetId, projectId = activeProjectId) => {
    let targetProjectId = projectId;
    const containingProject = projects.find((project) => project.assetIds.includes(assetId));
    if (!projects.find((project) => project.id === targetProjectId)?.assetIds.includes(assetId)) {
      targetProjectId = containingProject?.id || projects[0]?.id;
    }
    setActiveProjectId(targetProjectId);
    setActiveAssetId(assetId);
    setPreviewAsset(null);
    navigate("analysis");
  };

  return (
    <div className="app-shell">
      <AppHeader onMenu={() => setSidebarOpen(true)} />
      <Sidebar view={view} open={sidebarOpen} onNavigate={navigate} onClose={() => setSidebarOpen(false)} />
      <div className="app-content">
        {view === "assets" && (
          <AssetsPage
            assets={assets}
            projects={projects}
            onUpload={handleUpload}
            onOpenAsset={setPreviewAsset}
            onAddToProject={addToProject}
            onCreateProject={openCreateProject}
          />
        )}
        {view === "projects" && <ProjectsPage projects={projects} assets={assets} onOpen={openProject} onNew={() => openCreateProject()} />}
        {view === "project" && activeProject && (
          <ProjectPage
            project={activeProject}
            assets={assets}
            onBack={() => navigate("projects")}
            onAnalyze={openAnalysis}
            onPending={(asset) => setToast(asset.status === "分析失败" ? asset.analysisStage : `${asset.analysisStage || "正在分析"}，完成后可进入工作台`)}
          />
        )}
        {view === "analysis" && activeProject && activeAsset && <AnalysisPage project={activeProject} asset={activeAsset} onBack={() => navigate("project")} />}
        {view === "reports" && <ReportsPage projects={projects} />}
      </div>
      <NewProjectDialog open={dialogOpen} onClose={() => setDialogOpen(false)} onSubmit={createProject} defaultAsset={assets.find((item) => item.id === dialogAssetId)} />
      <PreviewDialog asset={previewAsset} onClose={() => setPreviewAsset(null)} onAnalyze={(asset) => openAnalysis(asset.id)} />
      {toast && <div className="toast"><Check size={17} />{toast}</div>}
    </div>
  );
}
