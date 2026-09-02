const TASK_META = {
  single_image_vqa: {
    label: "VQA",
    color: "bg-blue-500/15 text-blue-300 border-blue-500/30",
  },
  visual_grounding: {
    label: "Grounding",
    color: "bg-purple-500/15 text-purple-300 border-purple-500/30",
  },
  change_vqa: {
    label: "Change Detection",
    color: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  },
  optical_sar_fusion: {
    label: "Optical–SAR Fusion",
    color: "bg-orange-500/15 text-orange-300 border-orange-500/30",
  },
};

export default function TaskBadge({ task }) {
  const meta = TASK_META[task] ?? {
    label: task,
    color: "bg-slate-500/15 text-slate-300 border-slate-500/30",
  };
  return (
    <span
      className={`text-[11px] font-medium px-2 py-0.5 rounded-full border ${meta.color}`}
    >
      {meta.label}
    </span>
  );
}
