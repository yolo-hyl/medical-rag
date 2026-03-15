import { useState } from 'react';
import { BookOpen, Search, ExternalLink } from 'lucide-react';

const MOCK_RESULTS = [
  {
    id: 1,
    title: 'Association of Type 2 Diabetes and Cardiovascular Disease Risk: A Meta-Analysis',
    authors: 'Zhang L, Wang Y, Li H, et al.',
    journal: 'Journal of the American Medical Association',
    year: 2023,
    abstract: '本研究通过荟萃分析评估了2型糖尿病与心血管疾病风险的关联性。纳入42项前瞻性队列研究，共计1,234,567例患者...',
  },
  {
    id: 2,
    title: 'Clinical Efficacy of Metformin in Glycemic Control: Systematic Review',
    authors: 'Chen X, Liu J, Zhao M.',
    journal: 'The Lancet Diabetes & Endocrinology',
    year: 2023,
    abstract: '对二甲双胍在血糖控制中临床疗效的系统综述。通过检索PubMed、Embase数据库，纳入RCT研究共68项...',
  },
  {
    id: 3,
    title: 'Hypertension Management Guidelines Update 2024: Evidence-Based Recommendations',
    authors: 'Smith R, Johnson K, Brown A, et al.',
    journal: 'New England Journal of Medicine',
    year: 2024,
    abstract: '2024年高血压管理指南更新：基于循证证据的推荐意见。本指南整合了近五年最新临床试验数据...',
  },
];

export default function LiteraturePage() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  const handleSearch = () => {
    if (!query.trim()) return;
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      setSearched(true);
    }, 1200);
  };

  return (
    <div className="p-6 max-w-3xl mx-auto">
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-slate-800 flex items-center gap-2">
          <BookOpen size={22} className="text-blue-600" />
          文献检索
        </h2>
        <p className="text-sm text-slate-500 mt-1">检索 PubMed、知网等医学文献数据库</p>
      </div>

      <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-4 flex gap-3">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
          placeholder="输入关键词，如：糖尿病 二甲双胍 血糖控制"
          className="flex-1 border border-slate-200 rounded-lg px-3 py-2.5 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
        <button
          onClick={handleSearch}
          disabled={loading || !query.trim()}
          className="flex items-center gap-2 px-4 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 text-white rounded-lg text-sm font-medium transition-colors"
        >
          <Search size={15} className={loading ? 'animate-pulse' : ''} />
          {loading ? '检索中...' : '检索'}
        </button>
      </div>

      {searched && (
        <div className="mt-6 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-700">检索结果</h3>
            <span className="text-xs text-slate-500 bg-slate-100 px-2 py-1 rounded-full">{MOCK_RESULTS.length} 篇文献</span>
          </div>
          {MOCK_RESULTS.map((r) => (
            <div key={r.id} className="bg-white border border-slate-200 rounded-xl shadow-sm p-4 hover:border-blue-200 transition-colors">
              <div className="flex items-start justify-between gap-3">
                <h4 className="text-sm font-semibold text-blue-700 leading-snug flex-1">{r.title}</h4>
                <button className="p-1 text-slate-400 hover:text-blue-500 flex-shrink-0">
                  <ExternalLink size={14} />
                </button>
              </div>
              <p className="text-xs text-slate-500 mt-1">{r.authors} · {r.journal} · {r.year}</p>
              <p className="text-sm text-slate-700 mt-2 leading-relaxed line-clamp-2">{r.abstract}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
