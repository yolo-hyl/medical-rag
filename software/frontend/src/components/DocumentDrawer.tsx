import { useState } from 'react';
import { X, FileText, ChevronRight } from 'lucide-react';
import type { SourceDoc } from '../types';

interface DocumentDrawerProps {
  docs: SourceDoc[];
}

function sourceLabel(source?: string): string {
  if (!source) return '未知来源';
  const map: Record<string, string> = {
    qa: 'QA',
    huatuo_qa: '华佗QA',
    text: '文本',
    medical: '医疗',
  };
  return map[source] ?? source;
}

function DrawerPanel({
  doc,
  onClose,
}: {
  doc: SourceDoc;
  onClose: () => void;
}) {
  const similarity =
    doc.distance !== undefined
      ? (1 - doc.distance).toFixed(4)
      : undefined;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/20 z-40"
        onClick={onClose}
      />
      {/* Panel */}
      <div className="fixed right-0 top-0 h-full w-[400px] max-w-full bg-white shadow-xl z-50 flex flex-col">
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-200">
          <h3 className="text-sm font-semibold text-slate-800 flex items-center gap-2">
            <FileText size={16} className="text-emerald-600" />
            文档详情
          </h3>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
          >
            <X size={16} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4 scrollbar-thin">
          {/* Meta */}
          <div className="space-y-2">
            {doc.source && (
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium text-slate-500 w-16 flex-shrink-0">来源类型</span>
                <span className="text-xs bg-emerald-100 text-emerald-700 px-2 py-0.5 rounded font-medium">
                  {sourceLabel(doc.source)}
                </span>
              </div>
            )}
            {doc.source_name && (
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium text-slate-500 w-16 flex-shrink-0">文档名称</span>
                <span className="text-xs text-slate-700">{doc.source_name}</span>
              </div>
            )}
            {similarity !== undefined && (
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium text-slate-500 w-16 flex-shrink-0">相似度</span>
                <div className="flex items-center gap-2">
                  <div className="w-24 h-1.5 bg-slate-200 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-emerald-500 rounded-full"
                      style={{ width: `${Math.max(0, Math.min(1, parseFloat(similarity))) * 100}%` }}
                    />
                  </div>
                  <span className="text-xs font-medium text-emerald-700">{similarity}</span>
                </div>
              </div>
            )}
          </div>

          <hr className="border-slate-200" />

          {/* Summary */}
          {doc.summary && (
            <div>
              <p className="text-xs font-semibold text-slate-600 mb-1.5">摘要</p>
              <p className="text-sm text-slate-700 leading-relaxed bg-slate-50 rounded-lg px-3 py-2.5">
                {doc.summary}
              </p>
            </div>
          )}

          {/* Full content */}
          <div>
            <p className="text-xs font-semibold text-slate-600 mb-1.5">正文</p>
            <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap bg-slate-50 rounded-lg px-3 py-2.5">
              {doc.content_preview}
            </p>
          </div>
        </div>
      </div>
    </>
  );
}

export default function DocumentDrawer({ docs }: DocumentDrawerProps) {
  const [activeDoc, setActiveDoc] = useState<SourceDoc | null>(null);

  if (!docs.length) return null;

  return (
    <div className="mt-2">
      <p className="text-xs text-slate-500 mb-1.5 flex items-center gap-1">
        <FileText size={11} />
        参考文档 ({docs.length})
      </p>
      <div className="border border-slate-200 rounded-lg overflow-hidden divide-y divide-slate-100">
        {docs.map((doc, i) => (
          <button
            key={i}
            onClick={() => setActiveDoc(doc)}
            className="w-full flex items-center gap-2 px-3 py-2 bg-white hover:bg-slate-50 text-left transition-colors group"
          >
            {doc.source && (
              <span className="text-[10px] bg-emerald-100 text-emerald-700 px-1.5 py-0.5 rounded font-medium flex-shrink-0">
                {sourceLabel(doc.source)}
              </span>
            )}
            <span className="text-xs text-slate-600 flex-1 line-clamp-2 leading-relaxed">
              {doc.source_name && (
                <span className="font-medium text-slate-700 mr-1">{doc.source_name}</span>
              )}
              {doc.summary || doc.content_preview}
            </span>
            <ChevronRight size={12} className="text-slate-300 group-hover:text-slate-500 flex-shrink-0" />
          </button>
        ))}
      </div>

      {activeDoc && (
        <DrawerPanel doc={activeDoc} onClose={() => setActiveDoc(null)} />
      )}
    </div>
  );
}
