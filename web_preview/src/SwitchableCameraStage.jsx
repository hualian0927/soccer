import { useEffect, useRef } from "react";
import { ArrowLeftRight, Minimize, Pause, Play, SkipBack, SkipForward } from "lucide-react";


const CAMERA_A = [0, 0, 472, 272];
const CAMERA_B = [472, 0, 472, 272];
const PITCH = [944, 0, 336, 272];

function drawRegion(canvas, source, region) {
  const context = canvas?.getContext("2d", { alpha: false });
  if (!context || !source) return;
  const [sourceX, sourceY, sourceWidth, sourceHeight] = region;
  const displayWidth = source instanceof HTMLVideoElement ? source.videoWidth : source.naturalWidth;
  const horizontalScale = displayWidth / 1280;
  const displayRegionWidth = sourceWidth * horizontalScale;
  const scale = Math.min(canvas.width / displayRegionWidth, canvas.height / sourceHeight);
  const width = displayRegionWidth * scale;
  const height = sourceHeight * scale;
  context.fillStyle = "#101416";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(source, sourceX * horizontalScale, sourceY, sourceWidth * horizontalScale, sourceHeight,
    (canvas.width - width) / 2, (canvas.height - height) / 2, width, height);
}

export default function SwitchableCameraStage({ videoRef, posterUrl, mainCamera, onSwap, stageRef,
  currentTime, duration, isPlaying, onPlayPause, onSeek }) {
  const mainRef = useRef(null);
  const insetRef = useRef(null);
  const pitchRef = useRef(null);

  useEffect(() => {
    let frame;
    const poster = new Image();
    poster.src = posterUrl;
    const render = () => {
      const video = videoRef.current;
      const source = video?.readyState >= 2 ? video : poster.complete && poster.naturalWidth >= 1280 ? poster : null;
      if (source) {
        drawRegion(mainRef.current, source, mainCamera === "A" ? CAMERA_A : CAMERA_B);
        drawRegion(insetRef.current, source, mainCamera === "A" ? CAMERA_B : CAMERA_A);
        drawRegion(pitchRef.current, source, PITCH);
      }
      frame = requestAnimationFrame(render);
    };
    frame = requestAnimationFrame(render);
    return () => cancelAnimationFrame(frame);
  }, [videoRef, posterUrl, mainCamera]);

  return (
    <div ref={stageRef} className="spatial-switch-stage" aria-label="可切换双机位分析画面">
      <div className="spatial-switch-main">
        <div className="spatial-switch-label">主画面 · 机位{mainCamera === "A" ? "一" : "二"}</div>
        <canvas ref={mainRef} width="960" height="540" aria-label={`机位${mainCamera === "A" ? "一" : "二"}主画面`} />
      </div>
      <div className="spatial-switch-side">
        <button type="button" className="spatial-switch-inset" onClick={onSwap}
          aria-label={`将机位${mainCamera === "A" ? "二" : "一"}切换到主画面`}
          title="点击切换主画面">
          <span>机位{mainCamera === "A" ? "二" : "一"} <ArrowLeftRight size={15} /></span>
          <canvas ref={insetRef} width="480" height="272" aria-hidden="true" />
        </button>
        <div className="spatial-switch-pitch">
          <span>二维球场</span>
          <canvas ref={pitchRef} width="336" height="272" aria-label="同步二维球场图" />
        </div>
      </div>
      <div className="spatial-switch-fullscreen-controls">
        <button type="button" aria-label="后退 5 秒" onClick={() => onSeek(currentTime - 5)}><SkipBack size={19} /></button>
        <button type="button" aria-label={isPlaying ? "暂停" : "播放"} onClick={onPlayPause}>{isPlaying ? <Pause size={21} /> : <Play size={21} />}</button>
        <button type="button" aria-label="前进 5 秒" onClick={() => onSeek(currentTime + 5)}><SkipForward size={19} /></button>
        <span>{Math.floor(currentTime)} / {Math.floor(duration)} 秒</span>
        <button type="button" aria-label="退出全屏" onClick={() => document.exitFullscreen?.()}><Minimize size={19} /></button>
      </div>
    </div>
  );
}
