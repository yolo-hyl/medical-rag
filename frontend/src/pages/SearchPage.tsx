import { useState } from 'react';
import { Search, FileText, AlertCircle } from 'lucide-react';
import { apiSearch } from '../api';
interface SearchResult {
  id?: string;
  score?: number;
  distance?: number;
  content?: string;
  content_preview?: string;
  source?: string;
  source_name?: string;
  summary?: string;
  metadata?: Record<string, unknown>;
  [key: string]: unknown;
}

export default function SearchPage() {
  const [query, setQuery] = useState('');
  const [limit, setLimit] = useState(5);
  const [filter, setFilter] = useState('');
  const [hybrid, setHybrid] = useState(false);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<SearchResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await apiSearch(query.trim(), limit, filter.trim() || undefined, hybrid);
      setResults(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : '检索失败');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') handleSearch();
  };

  return (
    <div className="p-6 max-w-3xl mx-auto">
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-slate-800 flex items-center gap-2">
          <Search size={22} className="text-blue-600" />
          文档检索
        </h2>
        <p className="text-sm text-slate-500 mt-1">在向量数据库中检索相关医疗文档</p>
      </div>

      <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-6 space-y-4">
        {/* Query input */}
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">检索查询</label>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="输入检索关键词或问题..."
            className="w-full border border-slate-200 rounded-lg px-3 py-2.5 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          {/* Limit */}
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              返回数量
            </label>
            <input
              type="number"
              value={limit}
              onChange={(e) => setLimit(Math.max(1, Math.min(50, parseInt(e.target.value) || 5)))}
              min={1}
              max={50}
              className="w-full border border-slate-200 rounded-lg px-3 py-2.5 text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          {/* Filter */}
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              过滤表达式 <span className="text-slate-400 font-normal">(可选)</span>
            </label>
            <input
              type="text"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder='例: category == "心血管"'
              className="w-full border border-slate-200 rounded-lg px-3 py-2.5 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
        </div>

        {/* Hybrid toggle */}
        <div className="flex items-center justify-between py-3 border border-slate-200 rounded-lg px-4">
          <div>
            <p className="text-sm font-medium text-slate-700">混合检索</p>
            <p className="text-xs text-slate-400 mt-0.5">结合向量语义检索与关键词全文检索</p>
          </div>
          <button
            onClick={() => setHybrid((v) => !v)}
            className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none ${
              hybrid ? 'bg-blue-600' : 'bg-slate-300'
            }`}
          >
            <span
              className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow transition-transform ${
                hybrid ? 'translate-x-4' : 'translate-x-1'
              }`}
            />
          </button>
        </div>

        {/* Submit */}
        <button
          onClick={handleSearch}
          disabled={loading || !query.trim()}
          className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 text-white rounded-lg text-sm font-medium transition-colors shadow-sm"
        >
          <Search size={15} className={loading ? 'animate-pulse' : ''} />
          {loading ? '检索中...' : '开始检索'}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="mt-4 flex items-start gap-3 p-4 bg-red-50 border border-red-200 rounded-xl">
          <AlertCircle size={18} className="text-red-500 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      {/* Results */}
      {results !== null && (
        <div className="mt-6">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-slate-700">
              检索结果
            </h3>
            <span className="text-xs text-slate-500 bg-slate-100 px-2 py-1 rounded-full">
              {results.length} 条结果
            </span>
          </div>

          {results.length === 0 ? (
            <div className="text-center py-12 bg-white border border-slate-200 rounded-xl">
              <FileText size={32} className="text-slate-300 mx-auto mb-3" />
              <p className="text-slate-500 text-sm">未找到相关文档</p>
              <p className="text-slate-400 text-xs mt-1">尝试调整查询词或减少过滤条件</p>
            </div>
          ) : (
            <div className="space-y-3">
              {results.map((doc, idx) => {
                const preview = doc.content_preview ?? doc.content ?? '';
                const srcName = doc.source_name ?? doc.source ?? '';
                // score: prefer explicit score, else derive from distance (similarity = 1 - distance)
                const score =
                  typeof doc.score === 'number'
                    ? doc.score
                    : typeof doc.distance === 'number'
                    ? 1 - doc.distance
                    : null;

                return (
                  <div
                    key={idx}
                    className="bg-white border border-slate-200 rounded-xl shadow-sm p-4 hover:border-blue-200 transition-colors"
                  >
                    <div className="flex items-start justify-between gap-3 mb-2">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-xs font-medium text-slate-500 bg-slate-100 px-2 py-0.5 rounded">
                          #{idx + 1}
                        </span>
                        {srcName && (
                          <span className="text-xs font-medium text-teal-700 bg-teal-50 border border-teal-200 px-2 py-0.5 rounded">
                            {srcName}
                          </span>
                        )}
                        {doc.id && (
                          <span className="text-xs text-slate-400 font-mono">
                            {String(doc.id)}
                          </span>
                        )}
                      </div>
                      {score !== null && (
                        <span
                          className={`text-xs font-semibold px-2 py-0.5 rounded-full flex-shrink-0 ${
                            score >= 0.8
                              ? 'bg-green-100 text-green-700'
                              : score >= 0.6
                              ? 'bg-yellow-100 text-yellow-700'
                              : 'bg-orange-100 text-orange-700'
                          }`}
                        >
                          {(score * 100).toFixed(1)}%
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-slate-700 leading-relaxed">{String(preview)}</p>
                    {doc.metadata && Object.keys(doc.metadata).length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1">
                        {Object.entries(doc.metadata).map(([k, v]) => (
                          <span key={k} className="text-xs text-slate-400 bg-slate-50 border border-slate-100 px-2 py-0.5 rounded">
                            {k}: {String(v)}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
