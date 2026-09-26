import { useState } from 'react';
import { ServerCog, Trash2, Download, CheckCircle } from 'lucide-react';

const MOCK_RECORDS = [
  { id: '1', source: 'qa', name: '糖尿病常见问答集', count: 2341, updated: '2024-03-10' },
  { id: '2', source: 'text', name: '内科学第九版', count: 18920, updated: '2024-02-28' },
  { id: '3', source: 'huatuo_qa', name: '华佗医学问答库', count: 15032, updated: '2024-03-01' },
  { id: '4', source: 'medical', name: '临床药学手册', count: 5678, updated: '2024-01-15' },
];

export default function DataManagePage() {
  const [records, setRecords] = useState(MOCK_RECORDS);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [toast, setToast] = useState<string | null>(null);

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 2000);
  };

  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const handleDelete = () => {
    setRecords((prev) => prev.filter((r) => !selected.has(r.id)));
    setSelected(new Set());
    showToast('删除成功');
  };

  const handleExport = () => {
    showToast('导出任务已提交，请稍候...');
  };

  const sourceLabel: Record<string, string> = { qa: 'QA', text: '文本', huatuo_qa: '华佗QA', medical: '医疗' };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="mb-6 flex items-start justify-between">
        <div>
          <h2 className="text-xl font-semibold text-slate-800 flex items-center gap-2">
            <ServerCog size={22} className="text-blue-600" />
            数据管理
          </h2>
          <p className="text-sm text-slate-500 mt-1">管理向量数据库中的知识文档集合</p>
        </div>
        {selected.size > 0 && (
          <div className="flex gap-2">
            <button
              onClick={handleExport}
              className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-blue-600 hover:bg-blue-50 border border-blue-200 rounded-lg transition-colors"
            >
              <Download size={14} />
              导出 ({selected.size})
            </button>
            <button
              onClick={handleDelete}
              className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-red-600 hover:bg-red-50 border border-red-200 rounded-lg transition-colors"
            >
              <Trash2 size={14} />
              删除 ({selected.size})
            </button>
          </div>
        )}
      </div>

      <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-slate-50 border-b border-slate-200">
              <th className="w-10 px-4 py-3 text-left">
                <input type="checkbox" onChange={(e) => {
                  setSelected(e.target.checked ? new Set(records.map(r => r.id)) : new Set());
                }} className="rounded" />
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600">数据集名称</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600">来源类型</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600">记录数</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600">更新时间</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {records.map((r) => (
              <tr key={r.id} className={`hover:bg-slate-50 transition-colors ${selected.has(r.id) ? 'bg-blue-50' : ''}`}>
                <td className="px-4 py-3">
                  <input
                    type="checkbox"
                    checked={selected.has(r.id)}
                    onChange={() => toggleSelect(r.id)}
                    className="rounded"
                  />
                </td>
                <td className="px-4 py-3 font-medium text-slate-800">{r.name}</td>
                <td className="px-4 py-3">
                  <span className="text-xs bg-emerald-100 text-emerald-700 px-2 py-0.5 rounded font-medium">
                    {sourceLabel[r.source] ?? r.source}
                  </span>
                </td>
                <td className="px-4 py-3 text-slate-600">{r.count.toLocaleString()}</td>
                <td className="px-4 py-3 text-slate-500">{r.updated}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {toast && (
        <div className="fixed bottom-6 right-6 flex items-center gap-2 px-4 py-3 bg-emerald-600 text-white rounded-xl shadow-lg text-sm font-medium animate-fade-in">
          <CheckCircle size={16} />
          {toast}
        </div>
      )}
    </div>
  );
}
