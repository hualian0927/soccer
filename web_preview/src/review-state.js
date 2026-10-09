export function isAccepted(event) {
  if (event.reviewStatus === "rejected") return false;
  if (event.reviewStatus === "confirmed") return true;
  return event.publishedForStatistics === true;
}

export function deriveStatistics(layer, events) {
  const accepted = events.filter(isAccepted);
  const subtypes = {};
  for (const event of accepted) subtypes[event.subtype || "unknown"] = (subtypes[event.subtype || "unknown"] || 0) + 1;
  return {...layer, columns:["事件","展示","已审","采纳","排除"],
    counts:[...new Set(events.map(e=>e.category))].map(label => {
      const group = events.filter(e=>e.category === label);
      return {label,candidates:group.length,
        reviewed:group.filter(e=>["confirmed","rejected"].includes(e.reviewStatus) || e.reviewerLabel || e.reviewDecision || e.assistantDecision).length,
        supported:group.filter(isAccepted).length,
        rejected:group.filter(e=>e.reviewStatus === "rejected" || (e.reviewStatus !== "confirmed" && (e.reviewDecision || e.assistantDecision) === "rejected")).length};
    }),reviewedMetrics:{...layer?.reviewedMetrics,acceptedCount:accepted.length,subtypes,
      scope:"当前展示节点，随人工确认与拒绝同步更新；不是整场完整统计"}};
}

export function refreshOrganization(layer, events) {
  const byId = new Map(events.map(event=>[event.id,event]));
  return {...layer,items:(layer?.items || []).map(item=>{
    if (!item.sourceEventIds?.length) return item;
    const sources = item.sourceEventIds.map(id=>byId.get(id));
    const changed = sources.some(event=>!event || !isAccepted(event));
    const crossesCut = sources.length>1 && (sources.some(event=>event?.sceneId == null)
      || new Set(sources.map(event=>event?.sceneId)).size>1);
    if (!changed && !crossesCut) return item;
    return {...item,decision:"uncertain",publishedForStatistics:false,
      status:changed ? "来源节点已变更，需重新复核" : "镜头边界不连续，仅保留片段观察",
      limitations:[...(item.limitations || []),"以下为历史片段解读，不能继续作为已确认的连续进攻链。"]};
  })};
}
