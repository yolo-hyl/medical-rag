import { useState } from 'react';
import { FileSearch, Search } from 'lucide-react';

const MOCK_CASE = {
  id: 'CASE-2024-003821',
  name: '张某某',
  age: 58,
  gender: '男',
  admissionDate: '2024-03-10',
  chiefComplaint: '反复胸闷、气短3年，加重2周',
  diagnosis: ['冠状动脉粥样硬化性心脏病', '不稳定型心绞痛', '高血压病3级（极高危）'],
  treatment: '予以抗血小板聚集、调脂、扩冠、降压等综合治疗，患者症状明显好转后出院。',
  vitals: { bp: '162/94 mmHg', hr: '88 次/分', temp: '36.5°C', spo2: '97%' },
};

export default function CaseSearchPage() {
  const [caseId, setCaseId] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<typeof MOCK_CASE | null>(null);

  const handleSearch = () => {
    if (!caseId.trim()) return;
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      setResult(MOCK_CASE);
    }, 1000);
  };

  return (
    <div className="p-6 max-w-3xl mx-auto">
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-slate-800 flex items-center gap-2">
          <FileSearch size={22} className="text-blue-600" />
          病例检索
        </h2>
        <p className="text-sm text-slate-500 mt-1">通过患者ID或姓名查询历史病历记录</p>
      </div>

      <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-4 flex gap-3">
        <input
          type="text"
          value={caseId}
          onChange={(e) => setCaseId(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
          placeholder="输入患者ID或姓名，如：CASE-2024-003821"
          className="flex-1 border border-slate-200 rounded-lg px-3 py-2.5 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
        <button
          onClick={handleSearch}
          disabled={loading || !caseId.trim()}
          className="flex items-center gap-2 px-4 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 text-white rounded-lg text-sm font-medium transition-colors"
        >
          <Search size={15} className={loading ? 'animate-pulse' : ''} />
          {loading ? '查询中...' : '查询'}
        </button>
      </div>

      {result && (
        <div className="mt-6 bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
          <div className="px-5 py-4 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-slate-800">{result.name} · {result.gender} · {result.age}岁</p>
              <p className="text-xs text-slate-500 mt-0.5">病案号：{result.id} · 入院：{result.admissionDate}</p>
            </div>
          </div>

          <div className="p-5 space-y-4">
            <div>
              <p className="text-xs font-semibold text-slate-500 mb-1">主诉</p>
              <p className="text-sm text-slate-700">{result.chiefComplaint}</p>
            </div>

            <div>
              <p className="text-xs font-semibold text-slate-500 mb-1.5">入院诊断</p>
              <div className="flex flex-wrap gap-1.5">
                {result.diagnosis.map((d, i) => (
                  <span key={i} className="text-xs bg-red-50 text-red-700 border border-red-200 px-2.5 py-1 rounded-lg">{d}</span>
                ))}
              </div>
            </div>

            <div>
              <p className="text-xs font-semibold text-slate-500 mb-1.5">生命体征</p>
              <div className="grid grid-cols-2 gap-2">
                {Object.entries(result.vitals).map(([k, v]) => (
                  <div key={k} className="flex items-center justify-between bg-slate-50 rounded-lg px-3 py-2">
                    <span className="text-xs text-slate-500">{k.toUpperCase()}</span>
                    <span className="text-xs font-medium text-slate-800">{v}</span>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <p className="text-xs font-semibold text-slate-500 mb-1">治疗方案</p>
              <p className="text-sm text-slate-700 leading-relaxed">{result.treatment}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
