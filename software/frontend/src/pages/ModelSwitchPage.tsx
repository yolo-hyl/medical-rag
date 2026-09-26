import { useState } from 'react';
import { Cpu, CheckCircle } from 'lucide-react';

const MODELS = [
  { id: 'qwen3:32b', name: 'Qwen3 32B', desc: '千问3代旗舰模型，推理能力强，适合复杂医疗问答', badge: '推荐' },
  { id: 'deepseek-r1:7b', name: 'DeepSeek-R1 7B', desc: '轻量级推理模型，速度快，资源占用低', badge: '' },
  { id: 'deepseek-r1:14b', name: 'DeepSeek-R1 14B', desc: '中等规模推理模型，性能与速度平衡', badge: '' },
  { id: 'llama3.1:8b', name: 'Llama 3.1 8B', desc: 'Meta开源模型，多语言支持良好', badge: '' },
];

export default function ModelSwitchPage() {
  const [current, setCurrent] = useState('qwen3:32b');
  const [pending, setPending] = useState('qwen3:32b');
  const [toast, setToast] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleSwitch = () => {
    if (pending === current) return;
    setLoading(true);
    setTimeout(() => {
      setCurrent(pending);
      setLoading(false);
      setToast(true);
      setTimeout(() => setToast(false), 2000);
    }, 1500);
  };

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-slate-800 flex items-center gap-2">
          <Cpu size={22} className="text-blue-600" />
          模型切换
        </h2>
        <p className="text-sm text-slate-500 mt-1">选择用于问答生成的大语言模型</p>
      </div>

      <div className="space-y-3">
        {MODELS.map((m) => (
          <label
            key={m.id}
            className={`flex items-start gap-4 p-4 border rounded-xl cursor-pointer transition-all ${
              pending === m.id
                ? 'border-blue-500 bg-blue-50 shadow-sm'
                : 'border-slate-200 bg-white hover:border-blue-200'
            }`}
          >
            <input
              type="radio"
              name="model"
              value={m.id}
              checked={pending === m.id}
              onChange={() => setPending(m.id)}
              className="mt-0.5"
            />
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-slate-800">{m.name}</span>
                {m.badge && (
                  <span className="text-xs bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded font-medium">{m.badge}</span>
                )}
                {current === m.id && (
                  <span className="text-xs bg-emerald-100 text-emerald-700 px-1.5 py-0.5 rounded font-medium">当前</span>
                )}
              </div>
              <p className="text-xs text-slate-500 mt-0.5">{m.desc}</p>
            </div>
          </label>
        ))}
      </div>

      <div className="mt-6">
        <button
          onClick={handleSwitch}
          disabled={pending === current || loading}
          className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 text-white rounded-lg text-sm font-medium transition-colors shadow-sm"
        >
          {loading ? '切换中...' : '确认切换'}
        </button>
      </div>

      {toast && (
        <div className="fixed bottom-6 right-6 flex items-center gap-2 px-4 py-3 bg-emerald-600 text-white rounded-xl shadow-lg text-sm font-medium">
          <CheckCircle size={16} />
          模型切换成功
        </div>
      )}
    </div>
  );
}
