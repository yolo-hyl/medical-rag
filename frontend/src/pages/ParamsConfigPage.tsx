import { useState } from 'react';
import { SlidersHorizontal, CheckCircle } from 'lucide-react';

interface Params {
  temperature: number;
  top_k: number;
  max_tokens: number;
  retrieval_count: number;
}

export default function ParamsConfigPage() {
  const [params, setParams] = useState<Params>({
    temperature: 0.7,
    top_k: 50,
    max_tokens: 2048,
    retrieval_count: 5,
  });
  const [toast, setToast] = useState(false);

  const handleSave = () => {
    setToast(true);
    setTimeout(() => setToast(false), 2000);
  };

  const SliderRow = ({
    label,
    desc,
    field,
    min,
    max,
    step,
    format,
  }: {
    label: string;
    desc: string;
    field: keyof Params;
    min: number;
    max: number;
    step: number;
    format?: (v: number) => string;
  }) => (
    <div className="bg-white border border-slate-200 rounded-xl p-4">
      <div className="flex items-start justify-between mb-3">
        <div>
          <p className="text-sm font-semibold text-slate-800">{label}</p>
          <p className="text-xs text-slate-400 mt-0.5">{desc}</p>
        </div>
        <span className="text-sm font-bold text-blue-700 min-w-[3rem] text-right">
          {format ? format(params[field]) : params[field]}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={params[field]}
        onChange={(e) => setParams((p) => ({ ...p, [field]: parseFloat(e.target.value) }))}
        className="w-full accent-blue-600"
      />
      <div className="flex justify-between text-xs text-slate-400 mt-1">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-slate-800 flex items-center gap-2">
          <SlidersHorizontal size={22} className="text-blue-600" />
          参数配置
        </h2>
        <p className="text-sm text-slate-500 mt-1">调整模型推理与检索的超参数</p>
      </div>

      <div className="space-y-4">
        <SliderRow
          label="Temperature（温度）"
          desc="控制生成多样性。值越高输出越随机，越低越保守"
          field="temperature"
          min={0}
          max={2}
          step={0.1}
          format={(v) => v.toFixed(1)}
        />
        <SliderRow
          label="Top-K"
          desc="每步仅从概率最高的K个token中采样"
          field="top_k"
          min={1}
          max={100}
          step={1}
        />
        <SliderRow
          label="Max Tokens（最大生成长度）"
          desc="限制单次回答的最大token数"
          field="max_tokens"
          min={256}
          max={8192}
          step={256}
        />
        <SliderRow
          label="检索文档数量"
          desc="每次检索返回的参考文档数量"
          field="retrieval_count"
          min={1}
          max={20}
          step={1}
        />
      </div>

      <div className="mt-6">
        <button
          onClick={handleSave}
          className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors shadow-sm"
        >
          保存配置
        </button>
      </div>

      {toast && (
        <div className="fixed bottom-6 right-6 flex items-center gap-2 px-4 py-3 bg-emerald-600 text-white rounded-xl shadow-lg text-sm font-medium">
          <CheckCircle size={16} />
          保存成功
        </div>
      )}
    </div>
  );
}
