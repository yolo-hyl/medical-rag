import { useState } from 'react';
import { Database, Upload, CheckCircle2, XCircle, AlertTriangle } from 'lucide-react';
import { apiIngest } from '../api';
import type { IngestResult } from '../types';

const EXAMPLE_RECORDS = JSON.stringify(
  [
    {
      id: "doc001",
      content: "糖尿病是一种以高血糖为特征的代谢疾病，主要分为1型和2型。",
      source: "medical_textbook",
      source_name: "内科学教材",
      metadata: { category: "内分泌", topic: "糖尿病" },
    },
    {
      id: "doc002",
      content: "高血压是指体循环动脉血压持续升高，收缩压≥140mmHg和/或舒张压≥90mmHg。",
      source: "medical_textbook",
      source_name: "心血管病学",
      metadata: { category: "心血管", topic: "高血压" },
    },
  ],
  null,
  2
);

export default function IngestPage() {
  const [recordsText, setRecordsText] = useState(EXAMPLE_RECORDS);
  const [dropExisting, setDropExisting] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<IngestResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [parseError, setParseError] = useState<string | null>(null);

  const handleSubmit = async () => {
    setParseError(null);
    setError(null);

    let records: unknown[];
    try {
      const parsed = JSON.parse(recordsText);
      if (!Array.isArray(parsed)) {
        setParseError('请输入 JSON 数组格式');
        return;
      }
      records = parsed;
    } catch (e) {
      setParseError(e instanceof Error ? `JSON 解析错误: ${e.message}` : 'JSON 格式无效');
      return;
    }

    setLoading(true);
    try {
      const res = await apiIngest(records, dropExisting);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : '数据录入失败');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-3xl mx-auto">
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-slate-800 flex items-center gap-2">
          <Database size={22} className="text-blue-600" />
          数据录入
        </h2>
        <p className="text-sm text-slate-500 mt-1">向向量数据库中录入医疗文档</p>
      </div>

      <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-6 space-y-5">
        {/* Records textarea */}
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            文档记录 (JSON 数组)
          </label>
          <textarea
            value={recordsText}
            onChange={(e) => {
              setRecordsText(e.target.value);
              setParseError(null);
            }}
            rows={12}
            className={`w-full border rounded-lg px-3 py-3 text-xs font-mono text-slate-700 resize-y focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent scrollbar-thin ${
              parseError ? 'border-red-300 bg-red-50' : 'border-slate-200 bg-slate-50'
            }`}
            placeholder="输入 JSON 数组..."
          />
          {parseError && (
            <div className="flex items-center gap-2 mt-1.5">
              <AlertTriangle size={13} className="text-red-500" />
              <p className="text-xs text-red-600">{parseError}</p>
            </div>
          )}
        </div>

        {/* Drop existing toggle */}
        <div className="flex items-center justify-between py-3 border border-slate-200 rounded-lg px-4">
          <div>
            <p className="text-sm font-medium text-slate-700">清空现有集合</p>
            <p className="text-xs text-slate-400 mt-0.5">录入前删除数据库中的所有现有文档</p>
          </div>
          <button
            onClick={() => setDropExisting((v) => !v)}
            className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none ${
              dropExisting ? 'bg-red-500' : 'bg-slate-300'
            }`}
          >
            <span
              className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow transition-transform ${
                dropExisting ? 'translate-x-4' : 'translate-x-1'
              }`}
            />
          </button>
        </div>

        {dropExisting && (
          <div className="flex items-start gap-2 p-3 bg-orange-50 border border-orange-200 rounded-lg">
            <AlertTriangle size={15} className="text-orange-500 flex-shrink-0 mt-0.5" />
            <p className="text-xs text-orange-700">
              警告：将在录入前清空现有集合中的所有数据，此操作不可撤销。
            </p>
          </div>
        )}

        {/* Submit button */}
        <button
          onClick={handleSubmit}
          disabled={loading}
          className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white rounded-lg text-sm font-medium transition-colors shadow-sm"
        >
          <Upload size={15} className={loading ? 'animate-pulse' : ''} />
          {loading ? '正在录入...' : '开始录入'}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="mt-4 flex items-start gap-3 p-4 bg-red-50 border border-red-200 rounded-xl">
          <XCircle size={18} className="text-red-500 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-red-700">录入失败</p>
            <p className="text-xs text-red-600 mt-1">{error}</p>
          </div>
        </div>
      )}

      {/* Success result */}
      {result && (
        <div className="mt-4 space-y-3">
          <div className="flex items-start gap-3 p-4 bg-green-50 border border-green-200 rounded-xl">
            <CheckCircle2 size={18} className="text-green-500 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="text-sm font-medium text-green-700">录入成功</p>
              {result.message && (
                <p className="text-xs text-green-600 mt-1">{String(result.message)}</p>
              )}
              {result.count !== undefined && (
                <p className="text-xs text-green-600 mt-0.5">
                  共录入 <strong>{result.count}</strong> 条记录
                </p>
              )}
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-slate-100">
              <h3 className="text-sm font-semibold text-slate-700">响应详情</h3>
            </div>
            <pre className="p-4 text-xs text-slate-600 overflow-x-auto scrollbar-thin font-mono leading-relaxed">
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
