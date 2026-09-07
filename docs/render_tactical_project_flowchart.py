#!/usr/bin/env python3
"""Render the football tactical-analysis project flowchart as a PNG image."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "football_tactical_analysis_project_flowchart.png"

WIDTH = 2800
HEIGHT = 1900
FONT_REGULAR = "/mnt/c/Windows/Fonts/msyh.ttc"
FONT_BOLD = "/mnt/c/Windows/Fonts/msyhbd.ttc"

BG = "#F5F7FA"
INK = "#17212B"
MUTED = "#5B6773"
LINE = "#A9B4BF"
WHITE = "#FFFFFF"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size=size)


def text_center(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    lines: list[str],
    size: int,
    color: str = INK,
    bold_first: bool = True,
    spacing: int = 11,
) -> None:
    x1, y1, x2, y2 = box
    fonts = [font(size, bold_first and index == 0) for index in range(len(lines))]
    heights = [draw.textbbox((0, 0), line, font=fnt)[3] for line, fnt in zip(lines, fonts)]
    total = sum(heights) + spacing * (len(lines) - 1)
    y = y1 + (y2 - y1 - total) / 2
    for line, fnt, height in zip(lines, fonts, heights):
        bbox = draw.textbbox((0, 0), line, font=fnt)
        x = x1 + (x2 - x1 - (bbox[2] - bbox[0])) / 2
        draw.text((x, y), line, font=fnt, fill=color)
        y += height + spacing


def card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    outline: str,
    title: str,
    lines: list[str],
    accent: str | None = None,
    title_size: int = 34,
    body_size: int = 27,
) -> None:
    draw.rounded_rectangle(box, radius=24, fill=fill, outline=outline, width=3)
    x1, y1, x2, y2 = box
    if accent:
        draw.rounded_rectangle((x1, y1, x1 + 16, y2), radius=8, fill=accent)
    draw.text((x1 + 38, y1 + 26), title, font=font(title_size, True), fill=INK)
    y = y1 + 82
    for line in lines:
        draw.text((x1 + 40, y), line, font=font(body_size), fill=MUTED)
        y += body_size + 17


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = LINE,
    width: int = 7,
) -> None:
    draw.line((start, end), fill=color, width=width)
    x, y = end
    if abs(end[1] - start[1]) >= abs(end[0] - start[0]):
        direction = 1 if end[1] > start[1] else -1
        points = [(x, y), (x - 14, y - direction * 24), (x + 14, y - direction * 24)]
    else:
        direction = 1 if end[0] > start[0] else -1
        points = [(x, y), (x - direction * 24, y - 14), (x - direction * 24, y + 14)]
    draw.polygon(points, fill=color)


def pill(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    label: str,
    fill: str,
    color: str,
) -> int:
    x, y = xy
    fnt = font(25, True)
    bbox = draw.textbbox((0, 0), label, font=fnt)
    w = bbox[2] - bbox[0] + 38
    draw.rounded_rectangle((x, y, x + w, y + 48), radius=22, fill=fill)
    draw.text((x + 19, y + 8), label, font=fnt, fill=color)
    return w


def main() -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)

    # Header
    draw.text((120, 62), "足球技战术分析项目总流程", font=font(62, True), fill=INK)
    draw.text(
        (122, 143),
        "共享视觉底座 + 多任务分析器 + 证据化报告与视频交付",
        font=font(31),
        fill=MUTED,
    )
    x = 2060
    x += pill(draw, (x, 76), "单目视频", "#E6F2FF", "#175A9C") + 16
    pill(draw, (x, 76), "无 GPS", "#FDEEDB", "#9A5315")

    # Layer labels
    layer_font = font(25, True)
    layers = [
        ("01  输入与门控", 244),
        ("02  共享视觉底座（SoccerNetGSR）", 430),
        ("03  统一比赛状态", 695),
        ("04  多任务分析器", 900),
        ("05  证据融合与交付", 1390),
    ]
    for label, y in layers:
        draw.text((120, y), label, font=layer_font, fill="#52616F")
        draw.line((340, y + 20, 2680, y + 20), fill="#D6DDE4", width=2)

    # Input and gate
    card(
        draw,
        (430, 230, 1240, 400),
        WHITE,
        "#8FB7DD",
        "比赛视频输入",
        ["远景比赛画面 / 近景动作 / 回放镜头", "720p 或更高分辨率，保留原始时间轴"],
        "#2C7BB6",
    )
    card(
        draw,
        (1560, 230, 2370, 400),
        WHITE,
        "#D6A86B",
        "镜头与数据质量门控",
        ["远近景、回放、观众席及非比赛镜头分类", "足球覆盖率、可见人数、标定置信度检查"],
        "#D18B2C",
    )
    arrow(draw, (1240, 305), (1560, 305), "#7693AD")

    # Perception backbone
    backbone_boxes = [
        ((190, 455, 760, 635), "目标检测", ["足球域 YOLOX", "球员 / 足球 / 裁判 / 门将"], "#2C7BB6", "#EAF4FC"),
        ((810, 455, 1380, 635), "跨帧跟踪", ["Deep-EIoU + ReID", "IDATR 轨迹合并、拆分与修正"], "#248A72", "#E8F6F1"),
        ((1430, 455, 2000, 635), "球队与角色身份", ["上半身球衣颜色 + 多帧投票", "用户颜色先验 + CLIP 原型复核"], "#8A5AAE", "#F3ECF8"),
        ((2050, 455, 2620, 635), "球场标定", ["球场关键点 + Homography", "人物脚底点映射至 105 x 68 米"], "#C66A31", "#FBEFE8"),
    ]
    for box, title, lines, accent, fill in backbone_boxes:
        card(draw, box, fill, accent, title, lines, accent, title_size=32, body_size=25)

    # Shared context
    context_box = (310, 710, 2490, 835)
    draw.rounded_rectangle(context_box, radius=28, fill="#162B3A", outline="#162B3A")
    draw.text(
        (360, 739),
        "统一比赛状态  AnalysisContext（参考 SoccerMaster 统一契约）",
        font=font(35, True),
        fill=WHITE,
    )
    draw.text(
        (360, 790),
        "帧与时间  |  track ID  |  球队与角色  |  球场 XY  |  足球轨迹  |  速度与方向  |  置信度",
        font=font(27),
        fill="#D9E6EF",
    )
    for x_center in (475, 1095, 1715, 2335):
        arrow(draw, (x_center, 635), (x_center, 710), "#6E899B", width=5)

    # Task branches
    task_cards = [
        (
            (120, 930, 755, 1315),
            "P0  时序事件与基础交付",
            [
                "稳定球权状态机与时序窗口",
                "触球 / 接球 / 传球候选",
                "持球推进 / 球权转换候选",
                "射门 / 角球候选与射门结构",
                "事件时间轴 / 区域统计 / 高光",
                "参考：PathCRF、SoccerNet",
            ],
            "#2C7BB6",
            "#EAF4FC",
        ),
        (
            (795, 930, 1430, 1315),
            "P1  空间与队形战术",
            [
                "二维球场坐标跨帧聚合",
                "平均站位与阵型倾向",
                "宽度 / 纵深 / 三线高度 / 线距",
                "有球与无球队形、紧凑度",
                "定位球落点 / 门将干预候选",
                "参考：FIFA Team Shape、BASKET",
            ],
            "#248A72",
            "#E8F6F1",
        ),
        (
            (1470, 930, 2105, 1315),
            "P2  进攻组织与防守风险",
            [
                "传球方向、距离与类型",
                "Packing / 穿线 / 三区进入",
                "射门前进攻链路回溯",
                "防守三区风险与通道来源",
                "传球网络 / 中心性 / 传球熵",
                "参考：TacticAI、Passing Network",
            ],
            "#8A5AAE",
            "#F3ECF8",
        ),
        (
            (2145, 930, 2780, 1315),
            "P3 与近景专项分析",
            [
                "多打少 / 高位逼抢 / PPDA",
                "防守空档 / 攻防转换速度",
                "人体姿态与多帧动作阶段",
                "射门发力 / 门将扑救候选",
                "关键帧回看与分级暂停",
                "参考：TacticAI、Skor-xG",
            ],
            "#C66A31",
            "#FBEFE8",
        ),
    ]
    branch_centers = []
    for box, title, lines, accent, fill in task_cards:
        card(draw, box, fill, accent, title, lines, accent, title_size=30, body_size=25)
        branch_centers.append((box[0] + box[2]) // 2)
    for x_center in branch_centers:
        arrow(draw, (x_center, 835), (x_center, 930), "#6E899B", width=5)

    # Evidence fusion and outputs
    fusion = (250, 1430, 1275, 1605)
    card(
        draw,
        fusion,
        "#FFF6DF",
        "#D3A42E",
        "证据融合与可信度控制",
        ["事件时间 + 代表帧 + 指标 + 置信度", "多帧支持、限制说明、人工复核入口"],
        "#D3A42E",
        title_size=34,
        body_size=27,
    )
    outputs = (1525, 1430, 2550, 1605)
    card(
        draw,
        outputs,
        "#E9F3ED",
        "#4C8B62",
        "最终交付",
        ["JSON / CSV / 中文 Markdown 报告", "完整分析视频 / 简洁战术板 / 自动高光片段"],
        "#4C8B62",
        title_size=34,
        body_size=27,
    )
    for x_center in branch_centers:
        draw.line((x_center, 1315, x_center, 1360), fill="#8297A5", width=5)
    draw.line((branch_centers[0], 1360, branch_centers[-1], 1360), fill="#8297A5", width=5)
    arrow(draw, (760, 1360), (760, 1430), "#8297A5", width=5)
    arrow(draw, (1275, 1518), (1525, 1518), "#8297A5", width=7)

    # Status boundary
    status_box = (160, 1675, 2640, 1835)
    draw.rounded_rectangle(status_box, radius=24, fill=WHITE, outline="#C8D0D8", width=3)
    draw.text((205, 1705), "当前成熟度边界", font=font(31, True), fill=INK)
    pill(draw, (205, 1760), "已落地", "#DDF3E5", "#267A46")
    draw.text((365, 1768), "P0-P3 可运行候选基线、证据时间轴与可视化闭环", font=font(25), fill=MUTED)
    pill(draw, (1280, 1760), "尚未等同", "#FCE3DF", "#A53D33")
    draw.text((1477, 1768), "论文模型完整复现、官方比赛真值或职业级准确率", font=font(25), fill=MUTED)

    image.save(OUTPUT, quality=95, dpi=(180, 180))
    print(OUTPUT)


if __name__ == "__main__":
    main()
